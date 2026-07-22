"""
SessionService — session/message/plan persistence via db.py.

Conversation messages are written to the DB immediately on creation.
MealPlan / WeeklyMealPlan objects are kept in-memory and also persisted
as JSON blobs in the meal_plans table. The in-memory dict is a cache in
front of the DB (``_load_from_db`` repopulates it on demand); the DB row is
the source of truth, including the serialized clarification state.

Each plan type maintains an independent "canvas": a versioned lineage of
plans generated and refined within a session.  When the user asks to
switch plan types (e.g. "forget daily, let's do weekly"), the old canvas
is frozen and a fresh canvas for the new type is started.
"""

import json
import logging
import uuid
from datetime import datetime as dt, timezone as _tz
from typing import Dict, List, Optional


def _aware(value):
    """Coerce pre-M5 naive timestamps (assumed UTC) to aware — mixing naive
    and aware datetimes in sort keys raises TypeError."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=_tz.utc)
    return value

from db import (
    SessionLocal,
    db_create_session,
    db_get_session,
    db_delete_session,
    db_get_member_sessions,
    db_add_message,
    db_get_messages,
    db_update_canvases,
    db_update_state,
    db_update_profile,
    db_save_meal_plan,
    db_get_session_meal_plans,
    db_get_meal_plan,
    db_update_meal_plan_payload,
)
from models.recipe import CandidateRecipe
from models.session import (
    Message,
    MealPlan,
    MealCourse,
    PlanCanvas,
    Session,
    WeeklyMealPlan,
    MAX_MESSAGES_PER_SESSION,
)

logger = logging.getLogger(__name__)


class SessionService:
    """Hybrid session store: metadata + messages → SQLite; plan objects → in-memory + SQLite."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def create_session(self, member_id: str, user_profile: dict) -> Session:
        session = Session.create(member_id=member_id, user_profile=user_profile)
        self._sessions[session.session_id] = session

        db = SessionLocal()
        try:
            db_create_session(db, session.session_id, member_id, user_profile)
        finally:
            db.close()

        return session

    def get_session(self, session_id: str, member_id: Optional[str] = None) -> Optional[Session]:
        session = self._sessions.get(session_id)

        if session is None:
            session = self._load_from_db(session_id)

        if session is None:
            return None

        if member_id and session.member_id != member_id:
            logger.warning(
                "Session %s belongs to %s, not %s — access denied",
                session_id, session.member_id, member_id,
            )
            return None

        return session

    def _load_from_db(self, session_id: str) -> Optional[Session]:
        db = SessionLocal()
        try:
            row = db_get_session(db, session_id)
            if not row:
                return None

            user_profile = json.loads(row.user_profile)

            clarification = None
            if getattr(row, "clarification_state", None):
                clarification = json.loads(row.clarification_state)

            session = Session(
                session_id=row.session_id,
                member_id=row.member_id,
                user_profile=user_profile,
                state=row.state,
                clarification=clarification,
                max_messages=row.max_messages,
                created_at=_aware(row.created_at),
            )

            # Restore canvases
            if getattr(row, "daily_canvas", None):
                d = json.loads(row.daily_canvas)
                session.daily_canvas = PlanCanvas(
                    plan_type="daily",
                    current_id=d["current_id"],
                    root_id=d["root_id"],
                )
            if getattr(row, "weekly_canvas", None):
                w = json.loads(row.weekly_canvas)
                session.weekly_canvas = PlanCanvas(
                    plan_type="weekly",
                    current_id=w["current_id"],
                    root_id=w["root_id"],
                )

            # Reload persisted plans
            plan_rows = db_get_session_meal_plans(db, session_id)
            for pr in plan_rows:
                payload = json.loads(pr.payload)
                if pr.plan_type == "daily":
                    session.meal_plans.append(_deserialize_meal_plan(pr.id, payload))
                else:
                    session.weekly_meal_plans.append(_deserialize_weekly_plan(pr.id, payload))

            # Load recent messages
            msg_rows = db_get_messages(db, session_id, before_id=None, limit=MAX_MESSAGES_PER_SESSION)
            for m in msg_rows:
                session.conversation.append(
                    Message(
                        role=m.role,
                        content=m.content,
                        timestamp=_aware(m.timestamp),
                        intent=m.intent,
                        plan_id=m.plan_id,
                    )
                )

            self._sessions[session_id] = session
            return session
        finally:
            db.close()

    def delete_session(self, session_id: str, member_id: str) -> bool:
        db = SessionLocal()
        try:
            deleted = db_delete_session(db, session_id, member_id)
        finally:
            db.close()

        if deleted:
            self._sessions.pop(session_id, None)
        return deleted

    def get_member_sessions(self, member_id: str) -> list[Session]:
        db = SessionLocal()
        try:
            rows = db_get_member_sessions(db, member_id)
        finally:
            db.close()

        sessions = []
        for row in rows:
            s = self._sessions.get(row.session_id) or self._load_from_db(row.session_id)
            if s:
                sessions.append(s)
        return sessions

    def get_member_current_plans(self, member_id: str) -> Optional[Session]:
        """The member's session with the most recently updated plan canvas.

        Powers the dashboard's "Recent Meal Plans" widget: FoodChat plans are
        saved canvases (daily/weekly), not date-scheduled entries, so "what
        should this member cook today" means "their newest canvas". Returns
        None when no session has a plan yet.
        """
        best: Optional[Session] = None
        best_ts = None
        for session in self.get_member_sessions(member_id):
            ts = session.active_canvas_updated_at
            if ts is None:
                continue
            if best_ts is None or ts > best_ts:
                best, best_ts = session, ts
        return best

    def list_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    def persist_profile(self, session_id: str) -> None:
        """Write the session's (mutated) profile snapshot back to the DB.

        Called after accepted memories or diner changes so the snapshot a
        restarted replica loads matches what the conversation already knows.
        """
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        db = SessionLocal()
        try:
            db_update_profile(db, session_id, session.user_profile)
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Message management                                                   #
    # ------------------------------------------------------------------ #

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        plan_id: Optional[str] = None,
    ) -> Message:
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.is_at_message_limit:
            raise RuntimeError(
                f"Session {session_id} has reached the {session.max_messages}-message limit."
            )

        message = Message(role=role, content=content, intent=intent, plan_id=plan_id)
        session.conversation.append(message)

        db = SessionLocal()
        try:
            db_add_message(db, session_id, role, content, intent, plan_id)
        finally:
            db.close()

        return message

    def get_messages_page(
        self,
        session_id: str,
        before_id: Optional[int] = None,
        limit: int = 20,
    ) -> list[dict]:
        db = SessionLocal()
        try:
            rows = db_get_messages(db, session_id, before_id=before_id, limit=limit)
        finally:
            db.close()

        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "intent": r.intent,
                "plan_id": r.plan_id,
                "timestamp": r.timestamp,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Plan management                                                      #
    # ------------------------------------------------------------------ #

    def add_meal_plan(
        self,
        session_id: str,
        courses: List[CandidateRecipe],
        reasoning: str,
        metrics: dict | None = None,
    ) -> MealPlan:
        """Create a brand-new daily plan (version 1, no parent)."""
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        meal_plan = MealPlan.from_courses(courses, reasoning, metrics, version=1, parent_id=None)
        session.meal_plans.append(meal_plan)

        db = SessionLocal()
        try:
            db_save_meal_plan(
                db, meal_plan.id, session_id, "daily",
                _serialize_meal_plan(meal_plan),
                version=1, parent_id=None,
            )
        finally:
            db.close()

        # Start a fresh canvas for this plan
        session.daily_canvas = PlanCanvas(
            plan_type="daily",
            current_id=meal_plan.id,
            root_id=meal_plan.id,
        )
        self._persist_canvases(session_id, session)

        return meal_plan

    def refine_meal_plan(
        self,
        session_id: str,
        courses: List[CandidateRecipe],
        reasoning: str,
        metrics: dict | None = None,
    ) -> MealPlan:
        """Create a refined version of the current daily canvas plan."""
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.daily_canvas is None:
            # No existing canvas — treat as a fresh plan
            return self.add_meal_plan(session_id, courses, reasoning, metrics)

        parent = session.get_current_daily_plan()
        parent_id = parent.id if parent else None
        next_version = (parent.version + 1) if parent else 1

        meal_plan = MealPlan.from_courses(
            courses, reasoning, metrics,
            version=next_version, parent_id=parent_id,
        )
        session.meal_plans.append(meal_plan)

        db = SessionLocal()
        try:
            db_save_meal_plan(
                db, meal_plan.id, session_id, "daily",
                _serialize_meal_plan(meal_plan),
                version=next_version, parent_id=parent_id,
            )
        finally:
            db.close()

        # Advance canvas current pointer; root stays the same
        session.daily_canvas = PlanCanvas(
            plan_type="daily",
            current_id=meal_plan.id,
            root_id=session.daily_canvas.root_id,
        )
        self._persist_canvases(session_id, session)

        return meal_plan

    def add_weekly_meal_plan(
        self, session_id: str, plan_entries: List[dict],
        day_summaries: Optional[dict] = None,
        explainability: Optional[dict] = None,
    ) -> WeeklyMealPlan:
        """Create a brand-new weekly plan (version 1, no parent)."""
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        explainability = explainability or {}
        weekly_plan = WeeklyMealPlan(
            id=str(uuid.uuid4()),
            created_at=dt.now(_tz.utc),
            entries=plan_entries,
            version=1,
            parent_id=None,
            day_summaries=day_summaries or {},
            constraints_applied=explainability.get("constraints_applied") or [],
            personalization_summary=explainability.get("personalization_summary"),
            metrics=explainability.get("metrics") or {},
            reasoning=explainability.get("reasoning") or "",
        )
        session.weekly_meal_plans.append(weekly_plan)

        db = SessionLocal()
        try:
            db_save_meal_plan(
                db, weekly_plan.id, session_id, "weekly",
                _serialize_weekly_plan(weekly_plan),
                version=1, parent_id=None,
            )
        finally:
            db.close()

        session.weekly_canvas = PlanCanvas(
            plan_type="weekly",
            current_id=weekly_plan.id,
            root_id=weekly_plan.id,
        )
        self._persist_canvases(session_id, session)

        return weekly_plan

    def refine_weekly_meal_plan(
        self, session_id: str, plan_entries: List[dict],
        day_summaries: Optional[dict] = None,
        explainability: Optional[dict] = None,
    ) -> WeeklyMealPlan:
        """Create a refined version of the current weekly canvas plan."""
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.weekly_canvas is None:
            return self.add_weekly_meal_plan(
                session_id, plan_entries, day_summaries=day_summaries,
                explainability=explainability,
            )

        parent = session.get_current_weekly_plan()
        parent_id = parent.id if parent else None
        next_version = (parent.version + 1) if parent else 1

        explainability = explainability or {}
        weekly_plan = WeeklyMealPlan(
            id=str(uuid.uuid4()),
            created_at=dt.now(_tz.utc),
            entries=plan_entries,
            version=next_version,
            parent_id=parent_id,
            day_summaries=day_summaries or {},
            constraints_applied=explainability.get("constraints_applied") or [],
            personalization_summary=explainability.get("personalization_summary"),
            metrics=explainability.get("metrics") or {},
            reasoning=explainability.get("reasoning") or "",
        )
        session.weekly_meal_plans.append(weekly_plan)

        db = SessionLocal()
        try:
            db_save_meal_plan(
                db, weekly_plan.id, session_id, "weekly",
                _serialize_weekly_plan(weekly_plan),
                version=next_version, parent_id=parent_id,
            )
        finally:
            db.close()

        session.weekly_canvas = PlanCanvas(
            plan_type="weekly",
            current_id=weekly_plan.id,
            root_id=session.weekly_canvas.root_id,
        )
        self._persist_canvases(session_id, session)

        return weekly_plan

    # ------------------------------------------------------------------ #
    # Plan history                                                         #
    # ------------------------------------------------------------------ #

    def get_daily_plan_history(self, session_id: str) -> list[MealPlan]:
        """All daily plan versions for this session, oldest first."""
        session = self._sessions.get(session_id) or self._load_from_db(session_id)
        if not session:
            return []
        return sorted(session.meal_plans, key=lambda p: p.created_at)

    def get_weekly_plan_history(self, session_id: str) -> list[WeeklyMealPlan]:
        """All weekly plan versions for this session, oldest first."""
        session = self._sessions.get(session_id) or self._load_from_db(session_id)
        if not session:
            return []
        return sorted(session.weekly_meal_plans, key=lambda p: p.created_at)

    def resave_meal_plan(self, meal_plan: MealPlan) -> None:
        """Re-persist a daily plan after in-place enrichment/transparency."""
        db = SessionLocal()
        try:
            db_update_meal_plan_payload(db, meal_plan.id, _serialize_meal_plan(meal_plan))
        finally:
            db.close()

    def get_meal_plan_payload(self, plan_id: str) -> Optional[dict]:
        db = SessionLocal()
        try:
            row = db_get_meal_plan(db, plan_id)
            if not row:
                return None
            return {"plan_type": row.plan_type, "payload": json.loads(row.payload)}
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Canvas persistence                                                   #
    # ------------------------------------------------------------------ #

    def _persist_canvases(self, session_id: str, session: Session) -> None:
        daily_data = None
        if session.daily_canvas:
            daily_data = {
                "current_id": session.daily_canvas.current_id,
                "root_id": session.daily_canvas.root_id,
            }
        weekly_data = None
        if session.weekly_canvas:
            weekly_data = {
                "current_id": session.weekly_canvas.current_id,
                "root_id": session.weekly_canvas.root_id,
            }

        db = SessionLocal()
        try:
            db_update_canvases(db, session_id, daily_data, weekly_data)
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Clarification state (serialized dict — persisted, restart-safe)     #
    # ------------------------------------------------------------------ #

    def set_clarification_state(self, session_id: str, state_dict: dict) -> None:
        """Persist an in-progress clarification (serialized ClarificationState)."""
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        session.state = "clarifying"
        session.clarification = state_dict

        db = SessionLocal()
        try:
            db_update_state(db, session_id, "clarifying", clarification_state=json.dumps(state_dict))
        finally:
            db.close()

    def clear_clarification_state(self, session_id: str) -> None:
        session = self.get_session(session_id)  # load-through (replica-safe)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        session.state = "ready"
        session.clarification = None

        db = SessionLocal()
        try:
            db_update_state(db, session_id, "ready", clarification_state=None)
        finally:
            db.close()


# ------------------------------------------------------------------ #
# Serialization helpers                                                #
# ------------------------------------------------------------------ #

def _serialize_course(course: MealCourse) -> dict:
    return {
        "recipe_id": course.recipe_id,
        "title": course.title,
        "ingredients": course.ingredients,
        "directions": course.directions,
        "nutrition": course.nutrition,
        "image_url": course.image_url,
        "match_reasons": course.match_reasons,
    }


def _serialize_meal_plan(mp: MealPlan) -> dict:
    return {
        "id": mp.id,
        "created_at": mp.created_at.isoformat(),
        "version": mp.version,
        "parent_id": mp.parent_id,
        "reasoning": mp.reasoning,
        "llm_score": mp.llm_score,
        "llm_reasoning": mp.llm_reasoning,
        "fvs_count": mp.fvs_count,
        "fvs_reasoning": mp.fvs_reasoning,
        "diversity_llm_score": mp.diversity_llm_score,
        "diversity_llm_reasoning": mp.diversity_llm_reasoning,
        "guideline_adherence_score": mp.guideline_adherence_score,
        "guideline_adherence_reasoning": mp.guideline_adherence_reasoning,
        "constraints_applied": mp.constraints_applied,
        "personalization_summary": mp.personalization_summary,
        "breakfast": _serialize_course(mp.breakfast),
        "lunch": _serialize_course(mp.lunch),
        "dinner": _serialize_course(mp.dinner),
    }


def _serialize_weekly_plan(wp: WeeklyMealPlan) -> dict:
    return {
        "id": wp.id,
        "created_at": wp.created_at.isoformat(),
        "version": wp.version,
        "parent_id": wp.parent_id,
        "entries": wp.entries,
        "day_summaries": wp.day_summaries,
        "constraints_applied": wp.constraints_applied,
        "personalization_summary": wp.personalization_summary,
        "metrics": wp.metrics,
        "reasoning": wp.reasoning,
    }


def _deserialize_meal_plan(plan_id: str, payload: dict) -> MealPlan:
    def _course(d: dict) -> MealCourse:
        return MealCourse(
            recipe_id=d.get("recipe_id", ""),
            title=d.get("title", ""),
            ingredients=d.get("ingredients", ""),
            directions=d.get("directions", ""),
            nutrition=d.get("nutrition"),
            image_url=d.get("image_url"),
            match_reasons=d.get("match_reasons", []) or [],
        )

    return MealPlan(
        id=plan_id,
        created_at=_aware(dt.fromisoformat(payload["created_at"])),
        version=payload.get("version", 1),
        parent_id=payload.get("parent_id"),
        constraints_applied=payload.get("constraints_applied", []) or [],
        personalization_summary=payload.get("personalization_summary"),
        breakfast=_course(payload.get("breakfast", {})),
        lunch=_course(payload.get("lunch", {})),
        dinner=_course(payload.get("dinner", {})),
        reasoning=payload.get("reasoning", ""),
        llm_score=payload.get("llm_score", 0),
        llm_reasoning=payload.get("llm_reasoning", ""),
        fvs_count=payload.get("fvs_count", 0),
        fvs_reasoning=payload.get("fvs_reasoning", ""),
        diversity_llm_score=payload.get("diversity_llm_score", 0),
        diversity_llm_reasoning=payload.get("diversity_llm_reasoning", ""),
        guideline_adherence_score=payload.get("guideline_adherence_score", 0),
        guideline_adherence_reasoning=payload.get("guideline_adherence_reasoning", ""),
    )


def _deserialize_weekly_plan(plan_id: str, payload: dict) -> WeeklyMealPlan:
    # JSON round-trips dict keys as strings — restore the 1-7 day ints.
    day_summaries = {}
    for key, value in (payload.get("day_summaries") or {}).items():
        try:
            day_summaries[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return WeeklyMealPlan(
        id=plan_id,
        created_at=_aware(dt.fromisoformat(payload["created_at"])),
        version=payload.get("version", 1),
        parent_id=payload.get("parent_id"),
        entries=payload.get("entries", []),
        day_summaries=day_summaries,
        constraints_applied=payload.get("constraints_applied") or [],
        personalization_summary=payload.get("personalization_summary"),
        metrics=payload.get("metrics") or {},
        reasoning=payload.get("reasoning") or "",
    )
