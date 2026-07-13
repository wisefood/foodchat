"""
ChatService — daily-plan generation and small-talk handling.

Called exclusively by ``OrchestratorService`` after intent classification
(this service does NOT classify — the orchestrator is the single router):

    process_plan_request()      — "daily_plan" / "refine_plan" intents
    process_smalltalk()         — "chat" intent
    continue_clarification()    — any turn while session.state == "clarifying"

Plan flow: ClarificationManager (ask/skip questions, reconcile profile)
→ PlanningPipeline (RecipeWrangler candidates + LLM grading)
→ quality metrics (variety count, LLM diversity, guideline adherence)
→ SessionService (versioned canvas storage).

Clarification state is plain data persisted on the session row, so this flow
survives restarts and works across replicas (see ``services.clarification``).
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

from agents import GuidelineAdherenceGrader, MealDiversityGrader, ResponseWriter, SimpleChatBot
from models.recipe import CandidateRecipe, ScoredPlan
from models.session import MealPlan
from services.adapted_recipes import overlay_plan
from services.candidates_client import CANDIDATES
from services.clarification import ClarificationManager, ClarificationState
from services.feedback_service import FeedbackService
from services.planning_pipeline import PlanningPipeline
from services.seed_service import SeedService
from services.transparency import apply_transparency
from .session_service import SessionService

logger = logging.getLogger(__name__)

# National dietary guidelines used by the adherence grader. Optional: when the
# file is absent the grader scores against an empty context (logged once per call).
GUIDELINES_PATH = Path(__file__).resolve().parents[2] / "belgium_dietary_guidelines_augmentation.cypher"

NO_PLAN_APOLOGY = (
    "I'm sorry, I couldn't find enough recipes to build a "
    "complete meal plan matching your preferences. "
    "Could you try adjusting your requirements?"
)


def _format_plan_as_context(plan: MealPlan) -> str:
    """Serialize the current canvas plan into a text block for refinement prompts."""
    lines = [f"[Current daily meal plan — version {plan.version}]"]
    for name, course in [("Breakfast", plan.breakfast), ("Lunch", plan.lunch), ("Dinner", plan.dinner)]:
        if course.recipe_id:
            lines.append(
                f"{name}: {course.title}\n"
                f"  Ingredients: {course.ingredients}\n"
                f"  Directions: {course.directions}"
            )
    lines.append(f"Reasoning: {plan.reasoning}")
    return "\n".join(lines)


def _extract_ingredient_names(ingredients_text: str) -> list[str]:
    """Normalize a free-text ingredients blob into comparable item names."""
    if not isinstance(ingredients_text, str):
        return []
    cleaned = []
    for part in re.split(r"[\n,;•\-]+", ingredients_text):
        t = part.strip().lower()
        t = re.sub(r"\([^\)]*\)", "", t)
        t = re.sub(r"[^a-zA-Z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            cleaned.append(t)
    return cleaned


def _food_variety_score(plan: ScoredPlan) -> tuple[int, str]:
    """Count unique food items across the plan's three courses (FVS metric)."""
    items: list[str] = []
    for course in plan.courses:
        items.extend(_extract_ingredient_names(course.ingredients))
    unique_items = sorted(set(items))
    reasoning = (
        f"Unique food items across meals: {len(unique_items)} "
        f"(e.g., {', '.join(unique_items[:8])}{'...' if len(unique_items) > 8 else ''})"
    )
    return len(unique_items), reasoning


def _plan_as_text(plan: ScoredPlan) -> str:
    return "\n".join(
        f"{name}: {course.title}\nIngredients: {course.ingredients}\nDirections: {course.directions}\n"
        for name, course in (
            ("Breakfast", plan.breakfast), ("Lunch", plan.lunch), ("Dinner", plan.dinner),
        )
    )


class ChatService:
    """Daily-plan conversation flows (generation, refinement, clarification, small talk)."""

    def __init__(self, session_service: SessionService):
        self.session_service = session_service
        self.pipeline = PlanningPipeline()
        self.clarifier = ClarificationManager()
        self.chatbot = SimpleChatBot()
        self.seed_service = SeedService()
        self.feedback_service = FeedbackService()
        self.response_writer = ResponseWriter()
        self.diversity_grader = MealDiversityGrader()
        self.guideline_grader = GuidelineAdherenceGrader()
        logger.info("ChatService initialized.")

    # ------------------------------------------------------------------ #
    # Entry points (called by OrchestratorService)                         #
    # ------------------------------------------------------------------ #

    def process_smalltalk(self, session_id: str, message: str) -> Tuple[str, bool, Optional[MealPlan]]:
        """Handle a 'chat' intent turn — no plan generation."""
        session = self._get_session(session_id)
        self.session_service.add_message(session_id, "user", message)

        history = [(m.role, m.content) for m in session.conversation[:-1]]
        response = self.chatbot.chat(message, history)
        self.session_service.add_message(session_id, "assistant", response)
        return response, False, None

    def process_plan_request(
        self,
        session_id: str,
        message: str,
        is_refinement: bool = False,
        seeds: Optional[list[dict]] = None,
        skip_clarification: bool = False,
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        """Handle a 'daily_plan' or 'refine_plan' intent turn.

        ``seeds`` are named anchor dishes extracted upstream by the
        orchestrator (M2); they are resolved and pinned here so the pins
        survive an intervening clarification round-trip (they ride inside
        the persisted profile snapshot under ``_pinned_slots``).

        ``skip_clarification`` is for messages we composed ourselves (the
        plan-parameters card): they are explicit by construction, so the
        reconciler/specificity LLM round is pointless.

        Returns (response_text, needs_clarification, meal_plan|None).
        """
        session = self._get_session(session_id)
        logger.info(
            "[%s] Plan request (state=%s, refinement=%s): %.120s",
            session_id, session.state, is_refinement, message,
        )
        self.session_service.add_message(session_id, "user", message)

        # For refinements, prepend the current canvas plan so the pipeline
        # sees what it is being asked to change.
        effective_message = message
        if is_refinement and session.daily_canvas:
            current_plan = session.get_current_daily_plan()
            if current_plan:
                effective_message = (
                    f"{_format_plan_as_context(current_plan)}\n\n"
                    f"User refinement request: {message}"
                )
                logger.info(
                    "[%s] Refinement: injecting canvas plan v%d as context.",
                    session_id, current_plan.version,
                )

        # Resolve anchors before clarification and stash them in the profile
        # snapshot — the profile is what the clarification state persists, so
        # pins survive restarts mid-clarification.
        profile = dict(session.user_profile)
        if seeds:
            resolutions = self.seed_service.resolve_seeds(seeds, profile)
            pinned = self.seed_service.place_daily(resolutions)
            if pinned:
                profile["_pinned_slots"] = {
                    slot: {
                        "recipe_id": r.recipe_id, "title": r.title,
                        "ingredients": r.ingredients, "directions": r.directions,
                    }
                    for slot, r in pinned.items()
                }
            note = self.seed_service.describe(resolutions)
            if note:
                profile["_seed_note"] = note

        if skip_clarification:
            return self._generate_and_store(
                session_id, effective_message, profile, is_refinement
            )

        origin_intent = "refine_plan" if is_refinement else "daily_plan"
        outcome = self.clarifier.start(effective_message, profile, origin_intent)

        if outcome.needs_clarification:
            logger.info("[%s] Clarification needed — persisting clarification state.", session_id)
            self.session_service.set_clarification_state(session_id, outcome.state.to_dict())
            self.session_service.add_message(session_id, "assistant", outcome.question)
            return outcome.question, True, None

        return self._generate_and_store(
            session_id, outcome.final_query, outcome.profile, is_refinement
        )

    def continue_clarification(self, session_id: str, message: str) -> Tuple[str, bool, Optional[MealPlan], str]:
        """Consume a user answer while session.state == "clarifying".

        Returns (response_text, needs_clarification, meal_plan|None, origin_intent).
        The origin intent ("daily_plan"/"refine_plan") is restored from the
        persisted state so the orchestrator can tag the turn correctly.
        """
        session = self._get_session(session_id)
        self.session_service.add_message(session_id, "user", message)

        if not session.clarification:
            # Recoverable inconsistency (e.g. state row said "clarifying" but no
            # payload). Reset and treat the message as a fresh plan request.
            logger.warning("[%s] Clarifying state without payload — resetting.", session_id)
            self.session_service.clear_clarification_state(session_id)
            text, needs, plan = self.process_plan_request(session_id, message)
            return text, needs, plan, "daily_plan"

        state = ClarificationState.from_dict(session.clarification)
        origin_intent = state.origin_intent
        outcome = self.clarifier.step(state, message)

        if outcome.needs_clarification:
            self.session_service.set_clarification_state(session_id, outcome.state.to_dict())
            self.session_service.add_message(session_id, "assistant", outcome.question)
            return outcome.question, True, None, origin_intent

        self.session_service.clear_clarification_state(session_id)

        # Remember what the user just told us (session-scoped): the answers go
        # into the profile history so the reconciler's known-facts check stops
        # re-asking the same things for the rest of the session.
        if outcome.collected_facts:
            history = session.user_profile.get("history", "") or ""
            addition = " | ".join(
                fact.replace("\n", " ") for fact in outcome.collected_facts
            )
            session.user_profile["history"] = (
                f"{history}\n{addition}" if history else addition
            )
            self.session_service.persist_profile(session_id)

        is_refinement = origin_intent == "refine_plan"
        text, needs, plan = self._generate_and_store(
            session_id, outcome.final_query, outcome.profile, is_refinement
        )
        return text, needs, plan, origin_intent

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    def _generate_and_store(
        self,
        session_id: str,
        final_query: str,
        profile: dict,
        is_refinement: bool,
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        """Run the pipeline, compute quality metrics, and store the plan version."""
        self.session_service.clear_clarification_state(session_id)
        logger.info("[%s] Generating plan (query=%.120s)", session_id, final_query)

        # Anchor pins ride in the profile snapshot (see process_plan_request);
        # pop the transient keys so they never leak into grader prompts.
        pinned_raw = profile.pop("_pinned_slots", None) or {}
        seed_note = profile.pop("_seed_note", None)
        pinned = {
            slot: CandidateRecipe(**fields) for slot, fields in pinned_raw.items()
        }

        # Feedback finally drives recommendations (M3): downvoted recipes are
        # excluded and the rating history reaches the grader prompts.
        session = self._get_session(session_id)
        signals = self.feedback_service.get_signals(session.member_id)

        plans = self.pipeline.generate(
            final_query, profile, pinned=pinned,
            exclude_recipe_ids=signals.downvoted_recipe_ids,
            feedback_history=signals.history_text,
        )
        if not plans:
            logger.warning("[%s] No candidate plans — returning apology.", session_id)
            self.session_service.add_message(session_id, "assistant", NO_PLAN_APOLOGY)
            return NO_PLAN_APOLOGY, False, None

        best = plans[0]
        metrics = self._compute_metrics(session_id, best)

        if is_refinement:
            meal_plan = self.session_service.refine_meal_plan(
                session_id, best.courses, best.reasoning, metrics
            )
            logger.info(
                "[%s] Refined daily plan → %s (v%d, parent=%s).",
                session_id, meal_plan.id, meal_plan.version, meal_plan.parent_id,
            )
            fallback = (
                "Here's your updated meal plan! I've made the adjustments you asked for. "
                "Let me know if you'd like anything else changed."
            )
        else:
            meal_plan = self.session_service.add_meal_plan(
                session_id, best.courses, best.reasoning, metrics
            )
            logger.info("[%s] New daily plan %s stored (metrics=%s).", session_id, meal_plan.id, metrics)
            fallback = (
                "Here's your meal plan for today! "
                "I've picked out breakfast, lunch, and dinner based on your preferences. "
                "Let me know if you'd like to swap something out."
            )

        # M4 enrichment + transparency: nutrition/images per course, reason
        # chips, constraint ledger — then re-persist the enriched payload.
        pinned_ids = {r.recipe_id for r in pinned.values()}
        enrichment = CANDIDATES.fetch_details([c.recipe_id for c in best.courses])
        apply_transparency(
            meal_plan, profile, pinned_ids, enrichment,
            downvoted_count=len(signals.downvoted_recipe_ids),
            feedback_lines=len(signals.history_text.splitlines()) if signals.history_text else 0,
        )
        # Member-saved adapted recipes replace the originals as the starting
        # point (title/ingredients/nutrition; ids stay the original).
        adapted_count = overlay_plan(meal_plan, profile)
        if adapted_count:
            logger.info(
                "[%s] %d course(s) use the member's adapted version.",
                session_id, adapted_count,
            )
        self.session_service.resave_meal_plan(meal_plan)

        # Grounded response writer (M4c): prose from facts, canned fallback.
        facts = {
            "action": "refined_daily_plan" if is_refinement else "new_daily_plan",
            "meals": {
                "breakfast": meal_plan.breakfast.title,
                "lunch": meal_plan.lunch.title,
                "dinner": meal_plan.dinner.title,
            },
            "seed_note": seed_note,
            "cooking_for": profile.get("cooking_for_names") or [],
            "constraints_honored": [c["constraint"] for c in meal_plan.constraints_applied[:4]],
        }
        formatted = self.response_writer.write(
            facts, final_query, fallback=f"{fallback} {seed_note}".strip() if seed_note else fallback,
        )

        self.session_service.add_message(session_id, "assistant", formatted)
        return formatted, False, meal_plan

    def _compute_metrics(self, session_id: str, plan: ScoredPlan) -> dict:
        """Compute the four plan-quality metrics surfaced in the API response."""
        plan_text = _plan_as_text(plan)

        fvs_count, fvs_reasoning = _food_variety_score(plan)
        logger.info("[%s] FVS: %d unique ingredients.", session_id, fvs_count)

        diversity = self.diversity_grader.score(plan_text)

        guidelines_text = ""
        try:
            guidelines_text = GUIDELINES_PATH.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("[%s] Guidelines file unavailable (%s) — scoring without it.", session_id, e)
        adherence = self.guideline_grader.score(plan_text, guidelines_text)

        return {
            "llm_score": plan.score,
            "llm_reasoning": plan.reasoning,
            "fvs_count": fvs_count,
            "fvs_reasoning": fvs_reasoning,
            "diversity_llm_score": int(diversity.get("score", 0)),
            "diversity_llm_reasoning": str(diversity.get("reasoning", "")),
            "guideline_adherence_score": int(adherence.get("score", 0)),
            "guideline_adherence_reasoning": str(adherence.get("reasoning", "")),
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_session(self, session_id: str):
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session
