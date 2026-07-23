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

    # Explicit FoodScholar consults bypass classification entirely: "can you
    # check with food scholar?" is a request to ask the expert, and the
    # classifier reliably filed it as plan_question — after which the
    # PlanAnalyst ROLE-PLAYED the consult ("I've checked with the Food
    # Scholar...") without the bridge ever running.
    _SCHOLAR_CONSULT_RE = re.compile(r"\bfood\s*scholar\b", re.IGNORECASE)

    def _compose_scholar_question(self, session, message: str) -> str:
        """The question FoodScholar should answer for an explicit consult.

        A bare consult ("check with food scholar?") carries no question of its
        own — reuse the member's previous question. Either way, attach the
        active plan's meals as context so the scholar answers about THESE
        dishes, not in the abstract.
        """
        stripped = self._SCHOLAR_CONSULT_RE.sub("", message)
        stripped = re.sub(
            r"\b(can|could|would|will)\s+you\s+(check|ask|consult|verify)\s*(with|the)?\b",
            "", stripped, flags=re.IGNORECASE,
        ).strip(" ?,.!-")
        if len(stripped.split()) >= 4:
            return message
        prior = next(
            (m.content for m in reversed(session.conversation) if m.role == "user"),
            None,
        )
        return prior or message

    def _with_plan_context(self, session, question: str) -> str:
        """Attach the active plan's meals so the scholar answers about THESE
        dishes rather than in the abstract. No plan → question unchanged."""
        summary = self._summarize_active_plan(session)
        if summary:
            return f"{question}\n\nContext — the meals under discussion:\n{summary}"
        return question

    def _classify_and_route(self, session, session_id: str, message: str) -> ChatTurn:
        """One classifier call, then dispatch — the only intent decision per turn."""
        if self._SCHOLAR_CONSULT_RE.search(message):
            logger.info("Explicit FoodScholar consult — routing to the M1 bridge")
            return self._handle_nutrition_question(
                session_id, message, session=session,
                question=self._compose_scholar_question(session, message),
            )

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
            return self._handle_nutrition_question(session_id, message, session=session)

        if intent == "plan_question":
            return self._handle_plan_question(session, session_id, message)

        if intent == "preference_update":
            return self._handle_preference_update(session_id, message)

        # "chat" and any unexpected values
        return self._handle_smalltalk(session_id, message)

    # ------------------------------------------------------------------ #
    # Handlers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_favorites(self, favorite_ids: list) -> list[tuple[str, str]]:
        """(recipe_id, title) for each resolvable favorite, junk-tolerant.

        Legacy favorite rows hold titles ("Leftover Turkey Casserole") or dead
        ids; a failed direct fetch retries through the seed path's tolerant
        autocomplete so those still resolve to a real recipe when one exists.
        """
        resolved_pairs: list[tuple[str, str]] = []
        for raw_id in list(favorite_ids)[:MAX_OFFERED_FAVORITES]:
            raw_id = str(raw_id).strip()
            if not raw_id:
                continue
            resolved = self.seed_service.client.fetch_recipe(raw_id)
            if resolved is None:
                suggestions = self.seed_service._autocomplete_tolerant(raw_id)
                if suggestions:
                    candidate_id, _title = suggestions[0]
                    resolved = self.seed_service.client.fetch_recipe(candidate_id)
            if resolved and resolved.recipe.title:
                resolved_pairs.append((resolved.recipe.recipe_id, resolved.recipe.title))
        return resolved_pairs

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

        # Resolve favorites ONCE, here — legacy rows may hold titles or dead
        # ids, so unresolvable direct fetches fall back to the same tolerant
        # autocomplete the seed path uses. The resolved pairs are stored in
        # the clarification state so acceptance anchors EXACTLY what was
        # offered: offer and accept previously read the favorites list
        # independently and could diverge, offering titles it then dropped.
        resolved_favorites = self._resolve_favorites(favorites)
        if not resolved_favorites:
            return None
        named = ", ".join(f"“{title}”" for _rid, title in resolved_favorites)

        offer_text = (
            f"Before I plan — I noticed you've favorited {named}. "
            "Want me to work them into this plan? (yes / no)"
        )
        self.session_service.set_clarification_state(session.session_id, {
            "kind": "favorites_offer",
            "original_message": message,
            "origin_intent": intent,
            "favorites": [
                {"recipe_id": rid, "title": title} for rid, title in resolved_favorites
            ],
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
            # Anchor exactly the favorites that were OFFERED (resolved pairs
            # stored with the offer). recipe_id makes resolution a DIRECT
            # fetch — no name round-trip that could land on another recipe.
            # Seed resolution still re-checks allergies, so an unsafe
            # favorite is skipped with a note.
            for fav in pending.get("favorites") or []:
                title = str(fav.get("title") or "").strip()
                if title:
                    seeds.append({"name": title, "recipe_id": fav.get("recipe_id")})
            if not seeds:
                # Offers made before this fix carry no stored pairs.
                for rid, title in self._resolve_favorites(
                    session.user_profile.get("favorite_recipe_ids") or []
                ):
                    seeds.append({"name": title, "recipe_id": rid})

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
            # Re-attach FRESH plan context: nutrition computed since the
            # thread started (backfill, recipe visits) must reach the scholar.
            fs_turn = self.foodscholar_service.continue_clarification(
                session_id, message,
                contextualize=lambda q: self._with_plan_context(session, q),
            )
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
        if plan is not None and outcome.changed_slots:
            # A verified swap of a hand-picked slot means the user changed
            # their mind — the old pick must not resurrect on the next refine.
            self._drop_manual_picks_for_slots(session_id, outcome.changed_slots)
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

    def _handle_nutrition_question(self, session_id: str, message: str,
                                   session=None, question: Optional[str] = None) -> ChatTurn:
        """Delegate a nutrition-science question to FoodScholar (M1 bridge).

        When the session has an active plan, its meals ride along as context —
        "is this good for heart health?" should be answered about the actual
        dishes on the member's plan. The transcript keeps the member's own
        words (``message``); only the question sent to FoodScholar is enriched.
        """
        base = question or message
        ask = self._with_plan_context(session, base) if session is not None else base
        fs_turn = self.foodscholar_service.process_question(
            session_id, message, question=ask, raw_question=base,
        )
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

        # Recipes may have been profiled AFTER the plan was stored (backfill,
        # recipe-page visits) — pull any now-available nutrition in before
        # serializing, so analysts never reason over stale "(no data)" gaps.
        try:
            from .plan_nutrition import refresh_plan_nutrition
            refresh_plan_nutrition(session, self.session_service)
        except Exception:  # noqa: BLE001
            logger.warning("Plan nutrition refresh failed", exc_info=True)

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
        if is_refinement:
            seeds = self._seeds_for_refinement(session_id, "daily", seeds)
        else:
            # A fresh plan starts a new lineage — old manual picks don't
            # follow it (compose re-stores its own picks after this call).
            self._store_manual_picks(session_id, "daily", [])
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

    def _parameter_card(self, session_id: str, intent: str, plan) -> Optional[dict]:
        """The slider card rides along with fresh plans only (daily AND
        weekly) — showing it again on every text refinement would be noise
        (the apply flow re-attaches it explicitly with updated values)."""
        if intent not in ("daily_plan", "weekly_plan") or plan is None:
            return None
        session = self.session_service.get_session(session_id)
        return plan_parameters.build_card(session.user_profile if session else {})

    # ------------------------------------------------------------------ #
    # Manual picks (compose mode) — survive refinements, die honestly      #
    # ------------------------------------------------------------------ #
    # Hand-picked dishes are a stronger signal than anything inferred: a
    # text refinement must not silently replace them. They are stored in
    # seed shape per plan type, re-injected on refinements (re-resolved and
    # safety-rechecked each time — diners may have changed), cleared by a
    # fresh plan request, and dropped per slot when a verified edit swaps
    # that slot (the user changed their mind about that pick).

    def _manual_picks(self, session, plan_type: str) -> list:
        store = session.user_profile.get("manual_picks") or {}
        return list(store.get(plan_type) or [])

    def _store_manual_picks(self, session_id: str, plan_type: str, picks: list) -> None:
        session = self.session_service.get_session(session_id)
        if not session:
            return
        store = dict(session.user_profile.get("manual_picks") or {})
        if picks:
            store[plan_type] = picks
        else:
            store.pop(plan_type, None)
        session.user_profile["manual_picks"] = store
        self.session_service.persist_profile(session_id)

    def _seeds_for_refinement(self, session_id: str, plan_type: str,
                              seeds: Optional[list]) -> Optional[list]:
        """Explicit seeds win; otherwise stored manual picks anchor the turn."""
        if seeds:
            return seeds
        session = self.session_service.get_session(session_id)
        if not session:
            return seeds
        picks = self._manual_picks(session, plan_type)
        if picks:
            logger.info(
                "[%s] Re-injecting %d manual pick(s) into %s refinement.",
                session_id, len(picks), plan_type,
            )
            return picks
        return seeds

    def _drop_manual_picks_for_slots(self, session_id: str, changed_slots: list) -> None:
        session = self.session_service.get_session(session_id)
        if not session:
            return
        for slot in changed_slots or []:
            day = slot.get("day") or None
            plan_type = "weekly" if day is not None else "daily"
            picks = self._manual_picks(session, plan_type)
            remaining = [
                p for p in picks
                if not (p.get("meal_type") == slot.get("meal_type")
                        and (p.get("day") or None) == day)
            ]
            if len(remaining) != len(picks):
                logger.info(
                    "[%s] Verified edit replaced a manual pick (%s%s) — unpinning it.",
                    session_id, slot.get("meal_type"), f" day {day}" if day else "",
                )
                self._store_manual_picks(session_id, plan_type, remaining)

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
        canvas = session.active_canvas
        if canvas is not None and canvas.plan_type == "weekly":
            # Weekly canvas active → the values refine THAT plan (the weekly
            # flow has no clarification round to skip).
            logger.info("[%s] Applying plan parameters %s (weekly refine).", session_id, values)
            turn = self._handle_weekly(
                session_id, message, "refine_plan", is_refinement=True,
            )
        else:
            is_refinement = session.get_current_daily_plan() is not None
            intent = "refine_plan" if is_refinement else "daily_plan"
            logger.info("[%s] Applying plan parameters %s (%s).", session_id, values, intent)
            turn = self._handle_plan(
                session_id, message, intent,
                is_refinement=is_refinement, skip_clarification=True,
            )
        # Always return the card with its new current values so the UI stays
        # in sync, even on the refine paths where the handlers skip it.
        turn.plan_parameters = plan_parameters.build_card(session.user_profile)
        return turn

    def compose_plan(
        self, session_id: str, member_id: str, picks: list[dict],
        plan_type: str = "daily", message: Optional[str] = None,
    ) -> ChatTurn:
        """Manual mode: the user hand-picked recipes on a blank canvas and
        asks FoodChat to fill out the rest (daily or weekly).

        Picks are already validated by the router ({meal_type, recipe_id,
        title?, day?}). They become seed anchors — resolved by id,
        allergy/diet re-checked (an unsafe pick is skipped with a note,
        never silently planned), pinned to their exact slots — and
        generation composes the rest deterministically: no intent
        classification, no clarification round.
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

        seeds = []
        for pick in picks:
            seed = {
                "recipe_id": pick["recipe_id"],
                "meal_type": pick["meal_type"],
                "name": pick.get("title") or "",
            }
            if pick.get("day") is not None:
                seed["day"] = pick["day"]
            seeds.append(seed)

        logger.info(
            "[%s] Manual compose (%s): %d pick(s) → %s",
            session_id, plan_type, len(seeds),
            [(s.get("day"), s["meal_type"], s["recipe_id"]) for s in seeds],
        )

        if plan_type == "weekly":
            text = (message or "").strip() or (
                "Complete my meal plan for this week around the dishes I picked."
            )
            turn = self._handle_weekly(
                session_id, text, "weekly_plan", is_refinement=False, seeds=seeds,
            )
        else:
            text = (message or "").strip() or (
                "Complete my meal plan for today around the dishes I picked."
            )
            turn = self._handle_plan(
                session_id, text, "daily_plan",
                is_refinement=False, seeds=seeds, skip_clarification=True,
            )

        # Persist the picks (seed shape) AFTER generation — the fresh-plan
        # path above just cleared the previous lineage's picks — so later
        # refinements keep anchoring what the user chose by hand.
        plan = turn.weekly_meal_plan if plan_type == "weekly" else turn.meal_plan
        if plan is not None:
            self._store_manual_picks(session_id, plan_type, seeds)
        return turn

    def _handle_weekly(
        self, session_id: str, message: str, intent: str, is_refinement: bool,
        seeds: Optional[list[dict]] = None,
    ) -> ChatTurn:
        if is_refinement:
            seeds = self._seeds_for_refinement(session_id, "weekly", seeds)
        else:
            self._store_manual_picks(session_id, "weekly", [])
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
            plan_parameters=self._parameter_card(session_id, intent, weekly_plan),
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
