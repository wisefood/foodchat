"""
SessionService — persists sessions and messages to SQLite via db.py.

Conversation messages are written to the DB immediately on creation.
MealPlan / WeeklyMealPlan objects are kept in-memory (they are large
structured blobs; add a plans table later if persistence is needed).

All session lookups that take a session_id also require the member_id
of the requesting user so access is always scoped to the owner.
"""

import json
import logging
from typing import Dict, Optional, List, Any

from db import (
    SessionLocal,
    db_create_session,
    db_get_session,
    db_get_session_scoped,
    db_delete_session,
    db_get_member_sessions,
    db_add_message,
    db_get_messages,
    db_update_active_context,
    db_update_state,
    db_save_meal_plan,
    db_get_session_meal_plans,
    db_get_meal_plan,
)
from models.session import (
    ActiveContext,
    Message,
    MealPlan,
    Session,
    WeeklyMealPlan,
    MAX_MESSAGES_PER_SESSION,
)

logger = logging.getLogger(__name__)


class SessionService:
    """Hybrid session store: metadata + messages → SQLite; plan objects → in-memory."""

    def __init__(self):
        # In-memory index of live Session objects (rebuilt from DB on get_session)
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
        """Return session if it exists and belongs to member_id (when provided)."""
        session = self._sessions.get(session_id)

        if session is None:
            # Try to rebuild from DB (e.g. after restart)
            session = self._load_from_db(session_id)

        if session is None:
            return None

        # Enforce user scoping when caller provides member_id
        if member_id and session.member_id != member_id:
            logger.warning(
                f"Session {session_id} belongs to {session.member_id}, "
                f"not {member_id} — access denied"
            )
            return None

        return session

    def _load_from_db(self, session_id: str) -> Optional[Session]:
        """Reconstruct a Session object from DB rows (messages loaded lazily)."""
        db = SessionLocal()
        try:
            row = db_get_session(db, session_id)
            if not row:
                return None

            user_profile = json.loads(row.user_profile)
            active_context_data = json.loads(row.active_context) if row.active_context else None

            session = Session(
                session_id=row.session_id,
                member_id=row.member_id,
                user_profile=user_profile,
                state=row.state,
                max_messages=row.max_messages,
                created_at=row.created_at,
            )

            if active_context_data:
                session.active_context = ActiveContext(
                    plan_type=active_context_data["plan_type"],
                    plan_id=active_context_data["plan_id"],
                )

            # Load recent messages into the in-memory conversation
            msg_rows = db_get_messages(db, session_id, before_id=None, limit=MAX_MESSAGES_PER_SESSION)
            for m in msg_rows:
                session.conversation.append(
                    Message(
                        role=m.role,
                        content=m.content,
                        timestamp=m.timestamp,
                        intent=m.intent,
                        plan_id=m.plan_id,
                    )
                )

            self._sessions[session_id] = session
            return session
        finally:
            db.close()

    def delete_session(self, session_id: str, member_id: str) -> bool:
        """Delete session only if it belongs to member_id."""
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

    def list_sessions(self) -> list[Session]:
        return list(self._sessions.values())

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
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.is_at_message_limit:
            raise RuntimeError(
                f"Session {session_id} has reached the {session.max_messages}-message limit. "
                "Please start a new session to continue."
            )

        message = Message(role=role, content=content, intent=intent, plan_id=plan_id)
        session.conversation.append(message)

        db = SessionLocal()
        try:
            db_add_message(db, session_id, role, content, intent, plan_id)
        finally:
            db.close()

        return message

    # Backward-compat aliases used by ChatService / WeeklyPlanService
    def add_weekly_message(self, session_id: str, role: str, content: str) -> Message:
        return self.add_message(session_id, role, content)

    def get_messages_page(
        self,
        session_id: str,
        before_id: Optional[int] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Cursor-based pagination for the conversation thread.

        Returns up to `limit` messages before `before_id` (the DB row id),
        oldest-first, so the UI can prepend them to the top of the scroll view.

        Each item: {"id": int, "role": str, "content": str, "intent": str|None,
                    "plan_id": str|None, "timestamp": datetime}
        """
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
    # Plan management (in-memory)                                          #
    # ------------------------------------------------------------------ #

    def add_meal_plan(
        self, session_id: str, meal_plan_tuple: tuple, reasoning: str, metrics: dict | None = None
    ) -> MealPlan:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        meal_plan = MealPlan.from_response(meal_plan_tuple, reasoning, metrics)
        session.meal_plans.append(meal_plan)

        db = SessionLocal()
        try:
            db_save_meal_plan(db, meal_plan.id, session_id, "daily", _serialize_meal_plan(meal_plan))
        finally:
            db.close()

        session.active_context = ActiveContext(plan_type="daily", plan_id=meal_plan.id)
        self._persist_active_context(session_id, session.active_context)

        return meal_plan

    def add_weekly_meal_plan(
        self, session_id: str, plan_entries: List[dict]
    ) -> WeeklyMealPlan:
        import uuid
        from datetime import datetime as dt
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        weekly_plan = WeeklyMealPlan(
            id=str(uuid.uuid4()),
            created_at=dt.now(),
            entries=plan_entries,
        )
        session.weekly_meal_plans.append(weekly_plan)

        db = SessionLocal()
        try:
            db_save_meal_plan(db, weekly_plan.id, session_id, "weekly", _serialize_weekly_plan(weekly_plan))
        finally:
            db.close()

        session.active_context = ActiveContext(plan_type="weekly", plan_id=weekly_plan.id)
        self._persist_active_context(session_id, session.active_context)

        return weekly_plan

    def _persist_active_context(self, session_id: str, ctx: Optional[ActiveContext]) -> None:
        db = SessionLocal()
        try:
            data = {"plan_type": ctx.plan_type, "plan_id": ctx.plan_id} if ctx else None
            db_update_active_context(db, session_id, data)
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Clarification state (in-memory only — generator not serialisable)   #
    # ------------------------------------------------------------------ #

    def set_clarification_state(self, session_id: str, generator, pending_data: dict) -> None:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        session.state = "clarifying"
        session.clarification_generator = generator
        session.pending_rag_data = pending_data

        db = SessionLocal()
        try:
            db_update_state(db, session_id, "clarifying")
        finally:
            db.close()

    def clear_clarification_state(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        session.state = "ready"
        session.clarification_generator = None
        session.pending_rag_data = None

        db = SessionLocal()
        try:
            db_update_state(db, session_id, "ready")
        finally:
            db.close()

    def get_meal_plan_payload(self, plan_id: str) -> Optional[dict]:
        """Return the raw JSON payload for a plan by ID, or None if not found."""
        db = SessionLocal()
        try:
            row = db_get_meal_plan(db, plan_id)
            if not row:
                return None
            import json
            return {"plan_type": row.plan_type, "payload": json.loads(row.payload)}
        finally:
            db.close()


# ------------------------------------------------------------------ #
# Serialization helpers                                                #
# ------------------------------------------------------------------ #

def _serialize_meal_plan(mp: MealPlan) -> dict:
    return {
        "id": mp.id,
        "created_at": mp.created_at.isoformat(),
        "reasoning": mp.reasoning,
        "llm_score": mp.llm_score,
        "llm_reasoning": mp.llm_reasoning,
        "fvs_count": mp.fvs_count,
        "fvs_reasoning": mp.fvs_reasoning,
        "diversity_llm_score": mp.diversity_llm_score,
        "diversity_llm_reasoning": mp.diversity_llm_reasoning,
        "guideline_adherence_score": mp.guideline_adherence_score,
        "guideline_adherence_reasoning": mp.guideline_adherence_reasoning,
        "breakfast": {
            "recipe_id": mp.breakfast.recipe_id,
            "title": mp.breakfast.title,
            "ingredients": mp.breakfast.ingredients,
            "directions": mp.breakfast.directions,
        },
        "lunch": {
            "recipe_id": mp.lunch.recipe_id,
            "title": mp.lunch.title,
            "ingredients": mp.lunch.ingredients,
            "directions": mp.lunch.directions,
        },
        "dinner": {
            "recipe_id": mp.dinner.recipe_id,
            "title": mp.dinner.title,
            "ingredients": mp.dinner.ingredients,
            "directions": mp.dinner.directions,
        },
    }


def _serialize_weekly_plan(wp: WeeklyMealPlan) -> dict:
    return {
        "id": wp.id,
        "created_at": wp.created_at.isoformat(),
        "entries": wp.entries,
    }
