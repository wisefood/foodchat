"""
OrchestratorService — unified conversational entry point.

The ONLY intent classification in the pipeline happens here (one
``OrchestratorAgent.classify`` call per turn), then the turn is routed:

  daily_plan         → ChatService.process_plan_request (fresh canvas)
  refine_plan        → ChatService / WeeklyPlanService with is_refinement=True,
                       targeting whichever canvas was most recently updated
  weekly_plan        → WeeklyPlanService (fresh canvas)
  switch_plan_type   → acknowledge, then fresh canvas of the target type
  edit_plan_slot     → EditService (one verified slot swap on the canvas)
  nutrition_question → FoodScholarService (evidence-based answer + attribution)
  plan_question      → PlanAnalyst (question ABOUT the active canvas — answered
                       from its nutrition data, never modifies the plan;
                       no canvas → falls through to FoodScholar)
  preference_update  → acknowledge a stated durable preference ("remember I
                       don't like chicken") — the durable write stays
                       consent-gated behind the M3 memory nudge
  chat               → ChatService.process_smalltalk

While a session is mid-clarification (session.state == "clarifying"), the
classifier is normally skipped — the user is answering our question. The
persisted clarification dict routes the turn: ``kind == "foodscholar"`` goes
back to FoodScholarService, anything else to ChatService (plan flow, whose
state carries the original intent). Both are restart-safe (data, not objects).
Edit-slot clarifications are the exception: when the reply still doesn't
resolve the slot it usually isn't an answer at all, so the turn falls back to
normal classification instead of re-interrogating.

Memory nudges (M3) run on EVERY turn — including clarification turns — so a
preference stated while answering a question is never silently dropped.

Returns a unified ChatTurn so the router needs one response model.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from agents import OrchestratorAgent, PlanAnalyst
from models.attribution import Attribution
from models.session import MealPlan, WeeklyMealPlan
from . import plan_parameters
from .edit_service import EditService
from .foodscholar_service import FoodScholarService
from .seed_service import SeedService
from .session_service import SessionService

logger = logging.getLogger(__name__)

# Words that count as accepting the favorites offer. Kept deliberately simple
# for M2 — anything else is treated as a decline and the original request
# proceeds unchanged, so a misread costs nothing but the boost.
_AFFIRMATIVE = re.compile(
    r"^\s*(yes|yeah|yep|sure|ok(ay)?|please|sounds good|do it|why not|go for it)\b",
    re.IGNORECASE,
)
# Favorites shown by title in the offer message (each needs a detail fetch).
MAX_OFFERED_FAVORITES = 3


@dataclass
class ChatTurn:
    role: str
    content: str
    intent: str
    needs_clarification: bool = False
    meal_plan: Optional[MealPlan] = None
    weekly_meal_plan: Optional[WeeklyMealPlan] = None
    at_message_limit: bool = False
    # Version metadata surfaced to the caller (canvas tracking in the UI)
    plan_version: Optional[int] = None
    plan_parent_id: Optional[str] = None
    # Provenance when the answer came from another WiseFood app (FoodScholar)
    attribution: Optional[Attribution] = None
    # Consent nudges ("remember this?") detected in the user's turn (M3)
    memory_suggestions: Optional[list] = None
    # Slot-edit proof (M4b): [{meal_type, day, old{title,kcal}, new{...}, directive, verified}]
    changed_slots: Optional[list] = None
    # Optional slider card (time/difficulty/goal) attached to fresh daily
    # plans; answered via POST /sessions/{id}/plan-parameters
    plan_parameters: Optional[dict] = None


class OrchestratorService:

    def __init__(
        self,
        session_service: SessionService,
        chat_service: Any,
        weekly_plan_service: Any,
        foodscholar_service: Optional[FoodScholarService] = None,
        memory_service: Any = None,
    ):
        self.session_service = session_service
        self.chat_service = chat_service
        self.weekly_plan_service = weekly_plan_service
        self.foodscholar_service = foodscholar_service or FoodScholarService(session_service)
        self.seed_service = SeedService()
        self.edit_service = EditService(session_service)
        self.memory_service = memory_service
        self.orchestrator = OrchestratorAgent()
        self.plan_analyst = PlanAnalyst()

    def process(self, session_id: str, member_id: str, message: str) -> ChatTurn:
        """Validate ownership, check the message cap, classify, and route."""
        session = self.session_service.get_session(session_id, member_id=member_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or access denied")

        if session.is_at_message_limit:
            return ChatTurn(
                role="assistant",
                content=(
                    f"This conversation has reached the {session.max_messages}-message limit. "
                    "Please start a new session to continue."
                ),
                intent="chat",
                at_message_limit=True,
            )

        # Mid-clarification turns usually bypass classification — the user is
        # answering our question. Handlers may still bounce the turn back to
        # normal routing when the reply clearly isn't an answer.
        if session.state == "clarifying":
            turn = self._handle_clarification_turn(session_id, message)
        else:
            turn = self._classify_and_route(session, session_id, message)

        # Consent nudges (M3): detect durable preferences in the user's turn
        # and ATTACH suggestions — durable writes happen only when the user
        # answers via POST /sessions/{id}/memory. Best-effort by design, and
        # runs on clarification turns too: a preference stated while
        # answering a question still counts.
        if self.memory_service is not None and not turn.at_message_limit:
            try:
                suggestions = self.memory_service.suggest(session, message)
                if suggestions:
                    turn.memory_suggestions = suggestions
            except Exception as e:
                logger.warning("[%s] Memory suggestion failed: %s", session_id, e)
        return turn

    def _classify_and_route(self, session, session_id: str, message: str) -> ChatTurn:
        """One classifier call, then dispatch — the only intent decision per turn."""
        history = [
            {"role": m.role, "content": m.content}
            for m in session.conversation[-12:]
        ]
        classification = self.orchestrator.classify(message, history)
        intent = classification["intent"]
        target_plan_type = classification.get("target_plan_type")
        logger.info("[%s] intent=%s target=%s", session_id, intent, target_plan_type)
        return self._route(session, session_id, message, intent, target_plan_type)

    def _route(self, session, session_id: str, message: str, intent: str,
               target_plan_type: Optional[str]) -> ChatTurn:

        if intent == "switch_plan_type":
            return self._handle_switch(session_id, message, target_plan_type)

        if intent in ("weekly_plan", "daily_plan"):
            # Named anchor dishes are extracted once per plan turn: they feed
            # pinned-slot planning AND gate the favorites offer (an explicit
            # dish request means the user already has a starting point).
            seeds = self.seed_service.extract_seeds(message)

            offer = self._maybe_offer_favorites(session, message, intent, seeds)
            if offer is not None:
                return offer

            if intent == "weekly_plan":
                return self._handle_weekly(session_id, message, intent, is_refinement=False, seeds=seeds)
            return self._handle_plan(session_id, message, intent, is_refinement=False, seeds=seeds)

        if intent == "refine_plan":
            canvas = session.active_canvas
            if canvas is None:
                # Nothing to refine yet — treat as a fresh daily plan.
                logger.info("[%s] refine_plan with no active canvas — fresh daily plan", session_id)
                return self._handle_plan(session_id, message, "daily_plan", is_refinement=False)
            if canvas.plan_type == "weekly":
                return self._handle_weekly(session_id, message, intent, is_refinement=True)
            return self._handle_plan(session_id, message, intent, is_refinement=True)

        if intent == "edit_plan_slot":
            return self._handle_edit(session_id, message)

        if intent == "nutrition_question":
            return self._handle_nutrition_question(session_id, message)

        if intent == "plan_question":
            return self._handle_plan_question(session, session_id, message)

        if intent == "preference_update":
            return self._handle_preference_update(session_id, message)

        # "chat" and any unexpected values
        return self._handle_smalltalk(session_id, message)

    # ------------------------------------------------------------------ #
    # Handlers                                                             #
    # ------------------------------------------------------------------ #

    def _maybe_offer_favorites(
        self, session, message: str, intent: str, seeds: list[dict]
    ) -> Optional[ChatTurn]:
        """One-time proactive offer to work the member's favorites into the plan.

        Fires only when ALL hold: first plan of the session (no canvases yet),
        the member has favorites, the request names no dishes itself, and no
        offer was made before in this session (offer messages are tagged with
        intent="favorites_offer" — the tag is the dedupe record, persisted
        with the message). Declining is safe: the original request proceeds
        unchanged on the next turn.
        """
        if seeds:
            return None
        if session.meal_plans or session.weekly_meal_plans:
            return None
        favorites = session.user_profile.get("favorite_recipe_ids") or []
        if not favorites:
            return None
        if any(m.intent == "favorites_offer" for m in session.conversation):
            return None

        # Show up to a few favorites by title (best-effort detail fetches).
        titles = []
        for recipe_id in favorites[:MAX_OFFERED_FAVORITES]:
            resolved = self.seed_service.client.fetch_recipe(recipe_id)
            if resolved and resolved.recipe.title:
                titles.append(resolved.recipe.title)
        named = ", ".join(f"“{t}”" for t in titles) if titles else "some favorite recipes"

        offer_text = (
            f"Before I plan — I noticed you've favorited {named}. "
            "Want me to work them into this plan? (yes / no)"
        )
        self.session_service.set_clarification_state(session.session_id, {
            "kind": "favorites_offer",
            "original_message": message,
            "origin_intent": intent,
        })
        self.session_service.add_message(
            session.session_id, "assistant", offer_text, intent="favorites_offer"
        )
        logger.info("[%s] Favorites offer made (%d favorites).", session.session_id, len(favorites))
        return ChatTurn(
            role="assistant", content=offer_text,
            intent="favorites_offer", needs_clarification=True,
        )

    def _handle_favorites_offer_reply(self, session_id: str, message: str) -> ChatTurn:
        """Consume the yes/no reply to the favorites offer and generate the plan."""
        session = self.session_service.get_session(session_id)
        pending = session.clarification or {}
        self.session_service.clear_clarification_state(session_id)

        original_message = pending.get("original_message", message)
        intent = pending.get("origin_intent", "daily_plan")
        accepted = bool(_AFFIRMATIVE.match(message or ""))
        logger.info("[%s] Favorites offer %s.", session_id, "accepted" if accepted else "declined")

        seeds: list[dict] = []
        if accepted:
            # Pin the favorites themselves as anchors (resolution re-checks
            # allergies/diet, so an unsafe favorite is skipped with a note).
            favorites = session.user_profile.get("favorite_recipe_ids") or []
            for recipe_id in favorites[:MAX_OFFERED_FAVORITES]:
                resolved = self.seed_service.client.fetch_recipe(recipe_id)
                if resolved and resolved.recipe.title:
                    seeds.append({"name": resolved.recipe.title})

        if intent == "weekly_plan":
            return self._handle_weekly(session_id, original_message, intent, is_refinement=False, seeds=seeds)
        return self._handle_plan(session_id, original_message, intent, is_refinement=False, seeds=seeds)

    def _handle_clarification_turn(self, session_id: str, message: str) -> ChatTurn:
        session = self.session_service.get_session(session_id)
        pending = (session.clarification or {}) if session else {}

        if pending.get("kind") == "favorites_offer":
            # The user is answering the favorites offer, not a plan question.
            self.session_service.add_message(session_id, "user", message)
            return self._handle_favorites_offer_reply(session_id, message)

        if pending.get("kind") == "edit_slot":
            # The user is (probably) telling us WHICH meal to swap (M4b).
            outcome = self.edit_service.continue_clarification(session_id, message)
            if outcome.unresolved:
                # The reply didn't answer the slot question — the trap is
                # cleared and nothing was logged, so route it as a fresh turn.
                session = self.session_service.get_session(session_id)
                return self._classify_and_route(session, session_id, message)
            return self._turn_from_edit(session_id, outcome)

        # FoodScholar clarifications are tagged with kind="foodscholar";
        # plan-flow states (ClarificationState.to_dict) have no "kind" key.
        if pending.get("kind") == FoodScholarService.CLARIFICATION_KIND:
            fs_turn = self.foodscholar_service.continue_clarification(session_id, message)
            self._tag_last_message(session_id, "nutrition_question", None)
            return ChatTurn(
                role="assistant",
                content=fs_turn.text,
                intent="nutrition_question",
                needs_clarification=fs_turn.needs_clarification,
                attribution=fs_turn.attribution,
            )

        response_text, needs_clarification, meal_plan, origin_intent = (
            self.chat_service.continue_clarification(session_id, message)
        )
        self._tag_last_message(session_id, origin_intent, meal_plan.id if meal_plan else None)
        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=origin_intent,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan,
            plan_version=meal_plan.version if meal_plan else None,
            plan_parent_id=meal_plan.parent_id if meal_plan else None,
            plan_parameters=self._parameter_card(session_id, origin_intent, meal_plan),
        )

    def _handle_edit(self, session_id: str, message: str) -> ChatTurn:
        """Targeted single-slot edit with verified directive (M4b)."""
        outcome = self.edit_service.process(session_id, message)
        return self._turn_from_edit(session_id, outcome)

    def _turn_from_edit(self, session_id: str, outcome) -> ChatTurn:
        plan = outcome.meal_plan or outcome.weekly_meal_plan
        self._tag_last_message(session_id, "edit_plan_slot", plan.id if plan else None)
        return ChatTurn(
            role="assistant",
            content=outcome.text,
            intent="edit_plan_slot",
            needs_clarification=outcome.needs_clarification,
            meal_plan=outcome.meal_plan,
            weekly_meal_plan=outcome.weekly_meal_plan,
            changed_slots=outcome.changed_slots or None,
            plan_version=plan.version if plan else None,
            plan_parent_id=plan.parent_id if plan else None,
        )

    def _handle_nutrition_question(self, session_id: str, message: str) -> ChatTurn:
        """Delegate a nutrition-science question to FoodScholar (M1 bridge)."""
        fs_turn = self.foodscholar_service.process_question(session_id, message)
        self._tag_last_message(session_id, "nutrition_question", None)
        return ChatTurn(
            role="assistant",
            content=fs_turn.text,
            intent="nutrition_question",
            needs_clarification=fs_turn.needs_clarification,
            attribution=fs_turn.attribution,
        )

    def _handle_plan_question(self, session, session_id: str, message: str) -> ChatTurn:
        """Answer a question ABOUT the active plan without touching it.

        Grounded in the serialized canvas (titles + nutrition enrichment) and
        recent conversation so references like "that" resolve to the guidance
        just discussed. No canvas → the question is really a nutrition
        question, so it falls through to FoodScholar.
        """
        summary = self._summarize_active_plan(session)
        if summary is None:
            return self._handle_nutrition_question(session_id, message)

        self.session_service.add_message(session_id, "user", message)
        history = [(m.role, m.content) for m in session.conversation[-8:]]
        try:
            answer = self.plan_analyst.answer(message, summary, history)
        except Exception as e:
            logger.warning("[%s] PlanAnalyst failed: %s", session_id, e)
            answer = (
                "I couldn't analyze the plan just now — try asking again, or "
                "tell me what you'd like changed and I'll take it from there."
            )
        self.session_service.add_message(
            session_id, "assistant", answer, intent="plan_question"
        )
        return ChatTurn(role="assistant", content=answer, intent="plan_question")

    def _summarize_active_plan(self, session) -> Optional[str]:
        """Serialize the active canvas for the analyst (None if no plan yet).

        Includes per-meal nutrition where M4 enrichment succeeded; missing
        data is marked so the analyst can be explicit about gaps.
        """

        def fmt_nutrition(n: Optional[dict]) -> str:
            if not n:
                return "(no nutrition data)"
            parts = []
            if n.get("kcal") is not None:
                parts.append(f"{n['kcal']} kcal")
            for key, unit in (("protein_g", "g protein"), ("carbs_g", "g carbs"), ("fat_g", "g fat")):
                if n.get(key) is not None:
                    parts.append(f"{n[key]}{unit}")
            return "(" + ", ".join(parts) + ")" if parts else "(no nutrition data)"

        canvas = session.active_canvas
        if canvas is None:
            return None
        if canvas.plan_type == "weekly":
            plan = session.get_current_weekly_plan()
            if plan is None:
                return None
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            by_day: dict[int, list] = {}
            for entry in plan.entries:
                by_day.setdefault(entry.get("day", 0), []).append(entry)
            lines = [f"7-day plan (version {plan.version}):"]
            for day_idx in sorted(by_day):
                label = day_names[day_idx] if day_idx < len(day_names) else f"Day {day_idx + 1}"
                lines.append(f"{label}:")
                for entry in sorted(by_day[day_idx], key=lambda e: e.get("meal_idx", 0)):
                    recipe = entry.get("recipe", {})
                    lines.append(
                        f"  {entry.get('meal_type', 'meal')}: {recipe.get('title', '?')} "
                        f"{fmt_nutrition(recipe.get('nutrition'))}"
                    )
            return "\n".join(lines)

        plan = session.get_current_daily_plan()
        if plan is None:
            return None
        lines = [f"Daily plan (version {plan.version}):"]
        for slot in ("breakfast", "lunch", "dinner"):
            course = getattr(plan, slot)
            lines.append(f"  {slot}: {course.title} {fmt_nutrition(course.nutrition)}")
        return "\n".join(lines)

    def _handle_smalltalk(self, session_id: str, message: str) -> ChatTurn:
        response_text, _, _ = self.chat_service.process_smalltalk(session_id, message)
        self._tag_last_message(session_id, "chat", None)
        return ChatTurn(role="assistant", content=response_text, intent="chat")

    def _handle_preference_update(self, session_id: str, message: str) -> ChatTurn:
        """Acknowledge a stated durable preference without interrogating (M3).

        The durable write stays consent-gated: process() attaches the memory
        nudge and the profile only changes via POST /sessions/{id}/memory —
        this handler just answers in the persona voice.
        """
        response_text, _, _ = self.chat_service.process_smalltalk(session_id, message)
        self._tag_last_message(session_id, "preference_update", None)
        return ChatTurn(role="assistant", content=response_text, intent="preference_update")

    def _handle_plan(
        self, session_id: str, message: str, intent: str, is_refinement: bool,
        seeds: Optional[list[dict]] = None, skip_clarification: bool = False,
    ) -> ChatTurn:
        response_text, needs_clarification, meal_plan = self.chat_service.process_plan_request(
            session_id, message, is_refinement=is_refinement, seeds=seeds,
            skip_clarification=skip_clarification,
        )
        self._tag_last_message(session_id, intent, meal_plan.id if meal_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan,
            plan_version=meal_plan.version if meal_plan else None,
            plan_parent_id=meal_plan.parent_id if meal_plan else None,
            plan_parameters=self._parameter_card(session_id, intent, meal_plan),
        )

    def _parameter_card(self, session_id: str, intent: str, meal_plan) -> Optional[dict]:
        """The slider card rides along with fresh daily plans only — showing
        it again on every text refinement would be noise (the apply flow
        re-attaches it explicitly with updated values)."""
        if intent != "daily_plan" or meal_plan is None:
            return None
        session = self.session_service.get_session(session_id)
        return plan_parameters.build_card(session.user_profile if session else {})

    def apply_plan_parameters(self, session_id: str, member_id: str, values: dict) -> ChatTurn:
        """Apply slider-card values (already sanitized by the router).

        Deterministic counterpart of a refinement turn: the values become a
        canonical query (no intent classification, no clarification LLM
        round), are stored on the profile so the card shows current settings,
        and land in the profile history so the reconciler treats them as
        known facts for the rest of the session.
        """
        session = self.session_service.get_session(session_id, member_id=member_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or access denied")
        if session.is_at_message_limit:
            return ChatTurn(
                role="assistant",
                content=(
                    f"This conversation has reached the {session.max_messages}-message limit. "
                    "Please start a new session to continue."
                ),
                intent="chat",
                at_message_limit=True,
            )

        applied = dict(session.user_profile.get("plan_parameters") or {})
        applied.update(values)
        session.user_profile["plan_parameters"] = applied
        history = session.user_profile.get("history", "") or ""
        line = plan_parameters.history_line(values)
        session.user_profile["history"] = f"{history}\n{line}" if history else line
        self.session_service.persist_profile(session_id)

        message = plan_parameters.describe(values)
        is_refinement = session.get_current_daily_plan() is not None
        intent = "refine_plan" if is_refinement else "daily_plan"
        logger.info("[%s] Applying plan parameters %s (%s).", session_id, values, intent)

        turn = self._handle_plan(
            session_id, message, intent,
            is_refinement=is_refinement, skip_clarification=True,
        )
        # Always return the card with its new current values so the UI stays
        # in sync, even on the refine path where _handle_plan skips it.
        turn.plan_parameters = plan_parameters.build_card(session.user_profile)
        return turn

    def _handle_weekly(
        self, session_id: str, message: str, intent: str, is_refinement: bool,
        seeds: Optional[list[dict]] = None,
    ) -> ChatTurn:
        response_text, weekly_plan = self.weekly_plan_service.process_message(
            session_id, message, is_refinement=is_refinement, seeds=seeds
        )
        self._tag_last_message(session_id, intent, weekly_plan.id if weekly_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            weekly_meal_plan=weekly_plan,
            plan_version=weekly_plan.version if weekly_plan else None,
            plan_parent_id=weekly_plan.parent_id if weekly_plan else None,
        )

    def _handle_switch(self, session_id: str, message: str, target_plan_type: Optional[str]) -> ChatTurn:
        """Freeze the current canvas type and start a fresh one of the target type.

        Both canvases stay retrievable — switching never destroys history.
        """
        target = target_plan_type if target_plan_type in ("daily", "weekly") else "weekly"
        logger.info("[%s] switch_plan_type → %s", session_id, target)

        if target == "weekly":
            ack = "Sure! I'm setting aside the daily plan and starting a fresh weekly plan for you."
            self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
            return self._handle_weekly(session_id, message, "weekly_plan", is_refinement=False)

        ack = "Sure! I'm setting aside the weekly plan and starting a fresh daily plan for you."
        self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
        return self._handle_plan(session_id, message, "daily_plan", is_refinement=False)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _tag_last_message(self, session_id: str, intent: str, plan_id: Optional[str]) -> None:
        """Back-fill intent and plan_id onto the last assistant message.

        The UI uses these tags to associate chat bubbles with canvas versions.
        """
        session = self.session_service.get_session(session_id)
        if not session or not session.conversation:
            return
        last = session.conversation[-1]
        if last.role == "assistant":
            last.intent = intent
            last.plan_id = plan_id
