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
from typing import Optional, Tuple

from agents import (
    GuidelineAdherenceGrader,
    MealDiversityGrader,
    PlanStrategist,
    ResponseWriter,
    SimpleChatBot,
)
from models.plan_spec import PlanSpec
from models.recipe import CandidateRecipe, ScoredPlan
from models.session import MealPlan
from models.planning_state import PlanningStateDelta
from services.adapted_recipes import overlay_plan
from services import (
    guidelines_service,
    pantry_service,
    plan_parameters,
    plan_history,
    plan_quality,
    plan_repair,
    plan_verifier,
    turn_budget,
    turn_intake,
)
from services import diet_intent
from services.candidates_client import CANDIDATES
from services.clarification import ClarificationManager, ClarificationState
from services.feedback_service import FeedbackService
from services.planning_pipeline import PlanningPipeline
from services.seed_service import SeedService
from services import transparency
from services.transparency import apply_transparency, split_ledger
from .session_service import SessionService

logger = logging.getLogger(__name__)

def no_plan_message(profile: dict) -> str:
    """The empty-plan answer, naming what stood in the way.

    The old text — "Could you try adjusting your requirements?" — named no
    requirement, so a member whose profile combination was the cause had
    nothing to adjust and nowhere to start. If we know the standing
    constraints, say them; an apology that teaches nothing is just a shrug
    with manners.

    But it has to name the constraints that were APPLIED. It listed the raw
    profile, so a member who asked for something vegetarian was told the
    blocker was "diet: omnivore" — a value dropped before the request as
    non-restrictive, and no mention of the vegetarian filter that actually
    narrowed the search. Naming a constraint that was never sent points the
    member at the wrong thing to relax.
    """
    constraints = []
    diet_line = diet_intent.describe_applied(
        profile.get("_diet_tags") or (), profile.get("diet")
    )
    if diet_line:
        constraints.append("diet — " + diet_line)
    allergies = profile.get("allergies") or []
    if allergies:
        constraints.append("allergens excluded: " + ", ".join(sorted(map(str, allergies))))
    dislikes = profile.get("food_dislikes") or []
    if dislikes:
        constraints.append("avoiding: " + ", ".join(sorted(map(str, dislikes))))

    if constraints:
        return (
            "I couldn't find enough recipes for a complete plan with your "
            "current constraints (" + "; ".join(constraints) + "). "
            "Tell me which one to relax, or name a dish you'd like and "
            "I'll plan around it."
        )
    return (
        "I couldn't find enough recipes to build a complete meal plan for "
        "that request. Try being a little broader, or name a dish you'd "
        "like and I'll plan around it."
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


# These moved to `services/plan_quality.py` so the weekly service can reach
# them too — it produces 21 meals and had no variety score, no diversity
# judgement and no guideline adherence, because the graders were instance
# attributes on this class. Kept as aliases: the names are used by tests and by
# readers who know where they were.
_extract_ingredient_names = plan_quality.extract_ingredient_names
_food_variety_score = plan_quality.food_variety
_plan_as_text = plan_quality.as_text
scored_plan_from = plan_quality.scored_from_plan


class ChatService:
    """Daily-plan conversation flows (generation, refinement, clarification, small talk)."""

    def __init__(self, session_service: SessionService):
        self.session_service = session_service
        self.pipeline = PlanningPipeline()
        self.clarifier = ClarificationManager()
        self.chatbot = SimpleChatBot()
        self.seed_service = SeedService()
        self.feedback_service = FeedbackService()
        # Used to re-resolve anchors carried from earlier turns.
        self.client = CANDIDATES
        self.response_writer = ResponseWriter()
        # Decides HOW to search; the pipeline still executes and the
        # verifier still checks. Constructed here rather than per turn
        # so the pooled Groq client is shared like every other agent's.
        self.strategist = PlanStrategist()
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

        # Standing constraints: what the member has already told us.
        #
        # Read before anything this turn says, merged with this turn's delta,
        # and written back. Previously each turn started from the profile alone
        # and rebuilt the request from a rewritten query, so "no favourites",
        # an anchored dish and "salads on the side" all evaporated the moment
        # the next message arrived.
        # Shape, pantry, diet and facets, all from the RAW message — never the
        # refinement context, which quotes the plan on screen, and the chicken
        # in a recipe the member is looking at is not a request for chicken.
        #
        # This lived here as four sequential extractions, which is why only the
        # daily path heard all four. It is now one fanned-out pass that runs in
        # front of the router, so every kind of turn records what was said; the
        # call here is the same pass, memoised, and costs nothing the second
        # time.
        state = turn_intake.intake(
            session_id, message, session_service=self.session_service,
        )

        # A fresh daily request is one day unless this turn said otherwise —
        # a standing seven-day horizon from "plan my week" is not a request
        # for seven days. The rule and its reasons live with the intake.
        state = turn_intake.plan_horizon(state, is_refinement=is_refinement)

        if seeds:
            resolutions = self.seed_service.resolve_seeds(seeds, profile)
            pinned, dropped = self.seed_service.place_daily(resolutions)
            if pinned:
                profile["_pinned_slots"] = {
                    slot: {
                        "recipe_id": r.recipe_id, "title": r.title,
                        "ingredients": r.ingredients, "directions": r.directions,
                    }
                    for slot, r in pinned.items()
                }
                # An anchor the member named is a standing choice, not a
                # property of this one message. Saying "add a salad" next turn
                # must not silently drop the apple pie they asked for.
                state = state.merge(
                    PlanningStateDelta(
                        anchors={slot: r.recipe_id for slot, r in pinned.items()}
                    )
                )
            note = self.seed_service.describe(resolutions, dropped)
            if note:
                profile["_seed_note"] = note

        self._apply_standing_state(session_id, profile, state)

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

    def _apply_standing_state(self, session_id: str, profile: dict, state) -> None:
        """Stash everything standing onto the profile snapshot the pipeline reads.

        Extracted because it had ONE caller and two callers needed it. A member
        answering a clarifying question with "just a dinner" had that answer
        heard by nobody: `continue_clarification` went straight to
        `_generate_and_store`, which reads `profile["_plan_spec"]` and found
        nothing there, so the shape fell back to the default three meals and
        the member who asked for one meal got a whole day. The same silence
        swallowed a diet, a pantry item, a cooking time and a facet stated in
        that answer — every underscore key below.
        """
        # Anchors set in earlier turns, re-pinned for this one.
        if state.anchors:
            carried = self._pin_from_anchors(state, profile)
            if carried:
                profile.setdefault("_pinned_slots", {}).update(carried)

        if state.use_favorites is False:
            # The member said no. It has to keep meaning no — a favourite that
            # reappears one turn after being declined reads as not listening.
            profile["favorite_recipe_ids"] = []
            profile["_favorites_declined"] = True

        if state.excluded_recipe_ids:
            profile["_excluded_recipe_ids"] = list(state.excluded_recipe_ids)

        # Serialized, not the dataclass: this snapshot is json.dumps-ed onto
        # the session row whenever the turn asks a clarifying question, and a
        # live PlanSpec made that write raise ("not JSON serializable") on
        # exactly those turns — intermittently, since clarification is an LLM
        # decision. `_generate_and_store` coerces it back.
        profile["_plan_spec"] = state.spec.to_dict()
        # "Keep it under 20 minutes" and the cooking-time slider are one
        # constraint. This puts the spoken one where the slider's value already
        # lives, so all seven fetch sites filter on it without a seventh place
        # to remember.
        plan_parameters.apply_state(profile, state)
        if state.pantry:
            # Rides the profile snapshot like the other underscore keys, so
            # the pantry survives an intervening clarification round-trip.
            profile["_pantry"] = list(state.pantry)
        if state.diet_tags:
            # Read by candidates_client.effective_diet at EVERY fetch site, so
            # unlike "_pantry" it is never popped — the base pool, the pantry
            # fan-out and a seed lookup all have to agree on the diet.
            profile["_diet_tags"] = list(state.diet_tags)
        if state.claim_tags:
            # Read at every fetch site like the other underscore keys.
            profile["_claim_tags"] = list(state.claim_tags)
        facets = state.facets()
        if facets:
            # Same convention, same reason: read at every fetch site, never
            # popped, so the pools cannot disagree about what was asked for.
            profile["_facets"] = facets
        self.session_service.set_planning_state(session_id, state)
        logger.info("[%s] Standing plan state: %s", session_id, state.describe())

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
        # Read before step() — it advances the phase, and only the conflict
        # phase's answer can retract a stated diet.
        was_conflict = state.phase == "conflict"
        outcome = self.clarifier.step(state, message)

        if was_conflict and diet_intent.is_conflict_refusal(message):
            # "No, follow my profile." Until now the answer to this question
            # was recorded as prose and nothing acted on it, so the only
            # reachable outcome was the one the member had just declined.
            planning = self.session_service.get_planning_state(session_id)
            if planning.diet_tags:
                self.session_service.set_planning_state(
                    session_id, planning.merge(PlanningStateDelta(diet_clear=True))
                )
                logger.info(
                    "[%s] Dietary conflict declined — stated diet retracted.",
                    session_id,
                )

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

        # Hear the ANSWER, not only the question it answered.
        #
        # "Just a dinner" is a plan shape. "I'm coeliac" is a diet. "I have
        # spinach to use up" is a pantry. All of them arrive as clarification
        # answers, and this path used to walk past the intake entirely — so the
        # member watched a full day appear after asking for one meal, and read
        # it, correctly, as not being listened to.
        state = turn_intake.intake(
            session_id, message, session_service=self.session_service,
        )
        state = turn_intake.plan_horizon(state, is_refinement=is_refinement)
        self._apply_standing_state(session_id, outcome.profile, state)

        text, needs, plan = self._generate_and_store(
            session_id, outcome.final_query, outcome.profile, is_refinement
        )
        return text, needs, plan, origin_intent

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    def _pin_from_anchors(self, state, profile: dict) -> dict:
        """Re-resolve anchors held from earlier turns into pinned slots.

        Stored as ids rather than snapshots, so the title and ingredients are
        whatever they are now — including the member's own adapted version,
        which a snapshot taken three turns ago would not have.
        """
        pinned: dict = {}
        already = set((profile.get("_pinned_slots") or {}).keys())
        for slot, recipe_id in state.anchors.items():
            if slot in already:
                continue  # this turn named one explicitly; it wins
            resolved = self.client.fetch_recipe(recipe_id) if self.client else None
            if resolved is None:
                logger.info("Anchor %s for %s no longer resolvable", recipe_id, slot)
                continue
            recipe = resolved.recipe
            pinned[slot] = {
                "recipe_id": recipe.recipe_id, "title": recipe.title,
                "ingredients": recipe.ingredients, "directions": recipe.directions,
            }
        return pinned

    def _avoid_for(self, session_id: str, is_refinement: bool) -> list[str]:
        """Dishes this plan should not serve, because they were just served.

        A soft preference the fetch gives up rather than empty a slot — see
        `PlanningPipeline.generate`.

        The two cases differ in WHICH plan to avoid, not in whether to:

        * **Fresh plan** — the last few plans of the session. Without it, "plan
          my day" twice returns the same day: RecipeWrangler's order is
          deterministic and the grader runs at temperature 0.
        * **Refinement** — the plan being refined. This used to pass nothing,
          on the reasoning that "a refinement is a request to change the plan
          on screen, so keeping its unchanged slots is the whole point". That
          reasoning was wrong about this path: it regenerates every slot, and
          slot preservation belongs to `edit_service`, which handles a
          single-slot swap. So "make it lighter" refetched the same pool, took
          the same head of it, and handed back the same three dishes.

        Anchors need no special case. A pinned slot's pool is replaced by its
        anchor outright, so excluding the anchor from the fetch cannot lose it.
        """
        session = self.session_service.get_session(session_id)
        if not is_refinement:
            return plan_history.recently_served(session)
        current = session.get_current_daily_plan() if session else None
        return plan_history.plan_recipe_ids(current) if current else []

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
        # Read before the pipeline pops "_pantry" from this same dict — the
        # coverage badges below need to know what was asked for.
        pantry = pantry_service.normalize_items(profile.get("_pantry") or [])

        # Feedback finally drives recommendations (M3): downvoted recipes are
        # excluded and the rating history reaches the grader prompts.
        session = self._get_session(session_id)
        signals = self.feedback_service.get_signals(session.member_id)

        # The standing plan shape. A non-default spec — extra meals, several
        # days, a multi-plate lunch — goes through the structured path, which
        # can express it; the classic three-slot grader cannot. This is the
        # dispatch DYNAMIC_MEALS_PLAN.md Phase 2 was built for: the engine and
        # the state both existed, and nothing connected them, so "add salads
        # as side dishes" extracted a spec, stored it, and then generated
        # exactly the plan it would have generated anyway.
        spec_raw = profile.pop("_plan_spec", None)
        spec = PlanSpec.coerce(spec_raw) if spec_raw is not None else None
        if spec is not None and not spec.is_default:
            return self._generate_structured(
                session_id, final_query, profile, spec, pinned, seed_note,
                signals, is_refinement,
            )

        # A recipe the member rejected in conversation ("not that one") must not
        # come back on a regeneration. `_excluded_recipe_ids` reached only the
        # structured path, so on the classic path — the default — the standing
        # exclusion was recorded, persisted, and then ignored at the fetch.
        # The brief, on the path that actually gets used.
        #
        # `PlanBrief` -> `PlanStrategist` -> `plan_verifier` were wired into the
        # structured path only — the one a member reaches by asking for an
        # unusual SHAPE. A plain "plan my day" comes here, and here had no
        # brief, no strategist, and not one measured constraint: it reported
        # the request back as though it were a result, which is the whole
        # failure the verifier was built to end.
        brief = self._brief_for(final_query, profile)

        plans = self.pipeline.generate(
            final_query, profile, pinned=pinned,
            exclude_recipe_ids=list(signals.downvoted_recipe_ids or [])
            + list(profile.get("_excluded_recipe_ids") or []),
            # What the member was just served — the session's recent plans on a
            # fresh request, the plan on screen on a refinement.
            avoid_recent=self._avoid_for(session_id, is_refinement),
            # Recipes come back in pages. Exclusion narrows the window; this
            # MOVES it, which is what keeps the pool full instead of shrinking
            # it toward empty as a session goes on.
            window_offset=plan_history.window_offset(
                self.session_service.get_session(session_id)
            ),
            feedback_history=signals.history_text,
        )
        if not plans:
            logger.warning("[%s] No candidate plans — returning apology.", session_id)
            apology = no_plan_message(profile)
            self.session_service.add_message(session_id, "assistant", apology)
            return apology, False, None

        best = plans[0]
        # Scores describe a plan that has already been chosen — they make the
        # card richer and change nothing about what the member eats. First
        # thing to drop when the turn is running late.
        metrics = (
            {} if turn_budget.skip("quality metrics", turn_budget.COST_METRICS)
            else self._compute_metrics(session_id, best, profile)
        )

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
        # Pantry coverage: UI badges per course + a ledger row, computed by
        # the deterministic matcher AFTER transparency/overlay so the chips
        # land on what the member will actually see. Before the resave, so
        # they persist.
        pantry_facts = pantry_service.annotate_daily_plan(meal_plan, pantry)
        pantry_note = pantry_service.describe_coverage(pantry_facts)

        # Measure what came back, not what was asked for. The declarative rows
        # above carry `source` — which diner a constraint is there for — which
        # a measurement cannot know; these carry evidence, which a declaration
        # cannot have. Both, in that order.
        report = plan_verifier.verify(meal_plan, brief.to_requested(), enrichment)
        logger.info("[%s] Verified: %s", session_id, plan_verifier.describe(report))
        if report.checks:
            meal_plan.constraints_applied = (
                list(meal_plan.constraints_applied or []) + report.as_ledger_rows()
            )

        # One repair pass. The verifier names the plates that failed a hard
        # check; until this existed `report.offenders` had no consumer anywhere
        # in the codebase, so a plan that failed its own vegetarian check was
        # rendered with a red chip and handed over. Bounded to one pass on
        # purpose (see plan_repair), and it re-verifies — a repair that did not
        # work must not be announced as one.
        repair_note = None
        if report.blocking and not turn_budget.skip("plan repair", turn_budget.COST_FETCH):
            outcome = plan_repair.repair(meal_plan, brief, report, profile)
            if outcome.changed:
                report = outcome.report
                # The measured rows describe the plan the member is GIVEN, so
                # the pre-repair ones are replaced rather than appended to. The
                # declarative rows stay: they carry `source`, which a
                # measurement cannot know.
                meal_plan.constraints_applied = [
                    row for row in (meal_plan.constraints_applied or [])
                    if row.get("source") != "measured on the plan"
                ] + report.as_ledger_rows()
            repair_note = plan_repair.describe(outcome)

        self.session_service.resave_meal_plan(meal_plan)

        # Grounded response writer (M4c): prose from facts, canned fallback.
        # The ledger is split by status, never sliced: a "relaxed" row handed
        # over as honoured let the reply claim a goal was met while the ledger
        # beside it said it wasn't. Both halves are passed, so "couldn't honour
        # X" is a fact the writer has rather than a silence.
        honored, not_honored = split_ledger(meal_plan.constraints_applied)
        facts = {
            "action": "refined_daily_plan" if is_refinement else "new_daily_plan",
            "meals": {
                "breakfast": meal_plan.breakfast.title,
                "lunch": meal_plan.lunch.title,
                "dinner": meal_plan.dinner.title,
            },
            "seed_note": seed_note,
            "cooking_for": profile.get("cooking_for_names") or [],
            "constraints_honored": honored,
            "constraints_not_honored": not_honored,
        }
        # Why this plan is worth eating, alongside what it was allowed to be.
        #
        # Without this the facts were five parts constraint bookkeeping to zero
        # parts health, so every reply came out as a compliance result: what was
        # permitted, what was swapped, what fell short. The measurements were
        # already being taken and shown as a collapsed panel of scores out of
        # five.
        facts["plan_value"] = transparency.plan_value(
            meal_plan, profile,
            metrics=metrics if isinstance(metrics, dict) else None,
            kcal_target=brief.kcal_target,
            pantry_facts=pantry_facts,
        )
        if pantry_facts:
            # The writer may only phrase what the matcher measured — used
            # AND unused items both reach the member.
            facts["pantry"] = {
                "used": pantry_facts["used"],
                "unused": pantry_facts["unused"],
                "note": pantry_note,
            }
        # A failed check the reply does not mention is a failure the member
        # discovers by eating it.
        if report.failed:
            facts["verified_problems"] = [
                {"constraint": c.name, "detail": c.detail} for c in report.failed
            ]
        if brief.rationale:
            facts["strategy"] = brief.rationale
        if repair_note:
            facts["repair"] = repair_note
        fallback_extras = " ".join(
            p for p in (seed_note, pantry_note, repair_note) if p
        )
        canned = f"{fallback} {fallback_extras}".strip() if fallback_extras else fallback
        # The writer phrases facts that are already a usable sentence. Last
        # optional stage to go, because a plainer reply is a smaller loss than
        # any of the others — and every caller already keeps this fallback for
        # the case where the writer fails outright.
        formatted = (
            canned if turn_budget.skip("response writer", turn_budget.COST_WRITER)
            else self.response_writer.write(facts, final_query, fallback=canned)
        )

        self.session_service.add_message(session_id, "assistant", formatted)
        return formatted, False, meal_plan

    def _generate_structured(
        self,
        session_id: str,
        final_query: str,
        profile: dict,
        spec,
        pinned: dict,
        seed_note: Optional[str],
        signals,
        is_refinement: bool,
    ) -> Tuple[str, bool, Optional[MealPlan]]:
        """Generate and store a plan whose shape the classic path cannot say.

        Everything the member reads comes from what actually happened: the
        shape that was built, the concerns the spec raised (three desserts in
        a day gets a sentence, not silent compliance), and the anchors that
        were honoured. The writer phrases; it does not invent.
        """
        excluded = list(signals.downvoted_recipe_ids or []) + list(
            profile.get("_excluded_recipe_ids") or []
        )
        # Before plan_structured pops "_pantry" — coverage badges need it.
        pantry = pantry_service.normalize_items(profile.get("_pantry") or [])

        # The brief: what this plan is trying to do, written down before a
        # single recipe is fetched. Deterministic on its own; the strategist
        # only ever adds to it, and only values the corpus actually carries.
        brief = self._brief_for(final_query, profile, spec)

        meal_plan = self.pipeline.plan_structured(
            profile, spec, exclude_recipe_ids=excluded, pinned=pinned,
            avoid_recent=self._avoid_for(session_id, is_refinement),
            # Same page walk as the classic path. A shaped plan has more plates
            # to fill, so it exhausts a window sooner, not later.
            window_offset=plan_history.window_offset(
                self.session_service.get_session(session_id)
            ),
            # The member's words. This path had no query parameter at all, so
            # the shape was honoured and the request was not — and the reply
            # was then phrased around something that reached nothing.
            query=final_query,
        )
        if meal_plan is None:
            logger.warning("[%s] Structured plan came back empty — apology.", session_id)
            apology = no_plan_message(profile)
            self.session_service.add_message(session_id, "assistant", apology)
            return apology, False, None

        # Everything below ran on the classic path and not on this one, so a
        # multi-plate or multi-day plan arrived with no nutrition on any plate,
        # no reason chips, an empty constraints ledger, no personalization
        # summary, and the member's own adapted recipes ignored.
        all_plates = [
            plate
            for day in meal_plan.day_plans
            for meal in day.meals
            for plate in meal.plates
            if getattr(plate, "recipe_id", "")
        ]
        pinned_ids = {r.recipe_id for r in pinned.values()}
        enrichment = CANDIDATES.fetch_details([p.recipe_id for p in all_plates])
        apply_transparency(
            meal_plan, profile, pinned_ids, enrichment,
            downvoted_count=len(signals.downvoted_recipe_ids or []),
            feedback_lines=(
                len(signals.history_text.splitlines()) if signals.history_text else 0
            ),
        )
        adapted_count = overlay_plan(meal_plan, profile)
        if adapted_count:
            logger.info(
                "[%s] %d plate(s) use the member's adapted version.",
                session_id, adapted_count,
            )

        # Pantry coverage badges + ledger row, AFTER transparency so the chips
        # are appended to the ones it built rather than overwritten.
        pantry_facts = pantry_service.annotate_daily_plan(meal_plan, pantry)
        pantry_note = pantry_service.describe_coverage(pantry_facts)

        # Quality metrics, which this path has never had.
        #
        # `_compute_metrics` needed a `ScoredPlan` and a `ScoredPlan` could only
        # be three named courses, so a multi-day or multi-plate plan arrived
        # with every score at zero — indistinguishable, in the UI, from a plan
        # that had been judged and found wanting.
        #
        # There is no `llm_score` here and there deliberately is not one:
        # `plan_meals` returns ONE recipe per slot, not a pool, so there is no
        # combination to rank and a fabricated ranking would be worse than an
        # absent one. Variety, diversity and guideline adherence all judge a
        # produced plan, which is exactly what this is.
        if turn_budget.skip("quality metrics", turn_budget.COST_METRICS):
            structured_metrics: dict = {}
        else:
            structured_metrics = self._compute_metrics(
                session_id, scored_plan_from(meal_plan, reasoning=meal_plan.reasoning),
                profile,
                # An N-day plan is judged by the rules a week of meals can show.
                plan_type="weekly" if len(meal_plan.day_plans) > 1 else "daily",
            )
            for key, value in structured_metrics.items():
                # `llm_score`/`llm_reasoning` carry the plan's own reasoning
                # through; the rest are measured here.
                if key not in ("llm_score", "llm_reasoning"):
                    setattr(meal_plan, key, value)

        # Now measure. Everything above reports what was REQUESTED; this reads
        # the plates that came back and says what is actually true of them.
        report = plan_verifier.verify(meal_plan, brief.to_requested(), enrichment)
        logger.info("[%s] Verified: %s", session_id, plan_verifier.describe(report))
        if report.checks:
            # Measured rows sit alongside the declarative ledger rather than
            # replacing it: the declarative rows carry `source` (which diner a
            # constraint is for), which a measurement cannot know, and the
            # measured rows carry evidence, which a declaration cannot have.
            meal_plan.constraints_applied = (
                list(meal_plan.constraints_applied or []) + report.as_ledger_rows()
            )

        # One repair pass. The verifier names the plates that failed a hard
        # check; until this existed `report.offenders` had no consumer anywhere
        # in the codebase, so a plan that failed its own vegetarian check was
        # rendered with a red chip and handed over. Bounded to one pass on
        # purpose (see plan_repair), and it re-verifies — a repair that did not
        # work must not be announced as one.
        repair_note = None
        if report.blocking and not turn_budget.skip("plan repair", turn_budget.COST_FETCH):
            outcome = plan_repair.repair(meal_plan, brief, report, profile)
            if outcome.changed:
                report = outcome.report
                # The measured rows describe the plan the member is GIVEN, so
                # the pre-repair ones are replaced rather than appended to. The
                # declarative rows stay: they carry `source`, which a
                # measurement cannot know.
                meal_plan.constraints_applied = [
                    row for row in (meal_plan.constraints_applied or [])
                    if row.get("source") != "measured on the plan"
                ] + report.as_ledger_rows()
            repair_note = plan_repair.describe(outcome)

        if is_refinement:
            # `is_refinement` was accepted and never used: every refinement
            # called add_prepared_meal_plan, which starts a fresh canvas, so
            # each "make it lighter" became version 1 of a new lineage and the
            # member's history was silently discarded.
            meal_plan = self.session_service.refine_prepared_meal_plan(
                session_id, meal_plan
            )
            logger.info(
                "[%s] Refined structured plan → %s (v%d, parent=%s).",
                session_id, meal_plan.id, meal_plan.version, meal_plan.parent_id,
            )
        else:
            meal_plan = self.session_service.add_prepared_meal_plan(
                session_id, meal_plan
            )
            logger.info(
                "[%s] Structured plan %s stored (%s).",
                session_id, meal_plan.id, spec.describe(),
            )

        concerns = spec.concerns()
        facts = {
            "action": "structured_plan",
            "shape": spec.describe(),
            "day_one": {
                meal.meal_type: [plate.title for plate in meal.plates]
                for meal in (meal_plan.days[0].meals if meal_plan.days else [])
            },
            "seed_note": seed_note,
            # The spec's own reservations — the assistant must say them, not
            # build three desserts silently. The member asked for guidance as
            # well as obedience.
            "concerns": concerns,
            "notes": meal_plan.reasoning[:300],
        }
        # The ledger this path now builds, split by status like the classic
        # path — so a relaxed constraint cannot be announced as honoured, and
        # the reply can say plainly what could not be met.
        honored, not_honored = split_ledger(meal_plan.constraints_applied)
        facts["constraints_honored"] = honored
        facts["constraints_not_honored"] = not_honored
        facts["plan_value"] = transparency.plan_value(
            meal_plan, profile,
            metrics=structured_metrics if isinstance(structured_metrics, dict) else None,
            kcal_target=brief.kcal_target,
            pantry_facts=pantry_facts,
        )
        # And the rating history, which this path dropped: the classic path
        # feeds it to the grader, and with no grader here it belongs in the
        # facts so the reply is at least written in light of it.
        if signals.history_text:
            facts["feedback_history"] = signals.history_text
        # What the measurement found, as facts the writer may phrase. A failed
        # check the reply does not mention is a failure the member discovers by
        # eating it.
        if report.failed:
            facts["verified_problems"] = [
                {"constraint": c.name, "detail": c.detail} for c in report.failed
            ]
        if brief.rationale:
            facts["strategy"] = brief.rationale
        if pantry_facts:
            facts["pantry"] = {
                "used": pantry_facts["used"],
                "unused": pantry_facts["unused"],
                "note": pantry_note,
            }
        if repair_note:
            facts["repair"] = repair_note
        fallback_parts = [f"Here's your plan — {spec.describe()}."]
        if seed_note:
            fallback_parts.append(seed_note)
        if pantry_note:
            fallback_parts.append(pantry_note)
        if repair_note:
            fallback_parts.append(repair_note)
        fallback_parts.extend(concerns)
        canned = " ".join(fallback_parts)
        formatted = (
            canned if turn_budget.skip("response writer", turn_budget.COST_WRITER)
            else self.response_writer.write(facts, final_query, fallback=canned)
        )
        self.session_service.add_message(session_id, "assistant", formatted)
        return formatted, False, meal_plan

    def _brief_for(self, query: str, profile: dict, spec=None):
        """The brief for this request: deterministic, then adjusted by reasoning.

        The order is the safety property. `PlanBrief.build` produces a working
        plan from the profile and the standing state with no model involved;
        the strategist can only add facets and claim tags the corpus carries,
        cannot touch allergens or diet, and a failure leaves the deterministic
        brief untouched. The plan that used to be built is the floor.
        """
        from models.plan_brief import PlanBrief

        state = None
        try:
            state = self.session_service.get_planning_state(
                profile.get("_session_id") or ""
            )
        except Exception:  # noqa: BLE001 - the profile already carries the stash
            state = None

        brief = PlanBrief.build(profile, state, spec)
        try:
            vocab = CANDIDATES.vocabularies() or {}
            if vocab:
                brief = brief.with_strategy(
                    self.strategist.propose(query, brief, vocab), vocab
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Strategy step skipped: %s", exc)
        logger.info("Brief: %s", brief.describe())
        return brief

    def _compute_metrics(
        self, session_id: str, plan: ScoredPlan, profile: dict, plan_type: str = "daily",
    ) -> dict:
        """The four plan-quality metrics surfaced in the API response.

        Delegates to `plan_quality`, which the weekly service uses too — one
        implementation, so a change to how a plan is scored cannot land on one
        path and not the other. Guideline adherence is judged against the
        member's rules from the data catalog for this `plan_type`.
        """
        result = plan_quality.metrics(
            plan, guidelines=guidelines_service.guidelines_text(plan_type, profile),
        )
        logger.info("[%s] FVS: %d unique ingredients.", session_id, result["fvs_count"])
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_session(self, session_id: str):
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session
