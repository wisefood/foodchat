import logging
from typing import Optional, Tuple

from .feedback_service import FeedbackService
from .seed_service import SeedService
from .session_service import SessionService
from .weekly_planner.action_adapter import RecipeActionSpace
from .weekly_planner.reward_logic import RewardCalculator
from .weekly_planner.environment import WeeklyMealPlanEnv
from .weekly_planner.day_summary import build_day_summaries
from .weekly_planner.explainability import build_weekly_explainability
from .transparency import split_ledger
from .weekly_planner.planner import (
    PlanGenerationError,
    WeeklyPlanner,
    build_preference_scorer,
)
from .adapted_recipes import overlay_weekly_entries
from .candidates_client import CANDIDATES
from agents import ResponseWriter
from models.session import WeeklyMealPlan

logger = logging.getLogger(__name__)


def _as_meal_plan(plan_entries: list[dict]):
    """Weekly entries in the shape `plan_verifier` reads.

    The verifier walks `day_plans` -> meals -> plates, which every daily plan
    already is. Weekly is a flat list of `{day, meal_idx, meal_type, recipe}`
    dicts, so rather than teach the verifier a second shape — and have two
    definitions of "every plate in a plan" drift apart — the entries are
    adapted into the one it knows.

    Nothing is stored from this. It exists to be measured.
    """
    from models.session import DayPlan, Meal, MealCourse, MealPlan

    by_day: dict[int, dict[str, list]] = {}
    for entry in sorted(plan_entries, key=lambda e: (e.get("day", 0), e.get("meal_idx", 0))):
        recipe = entry.get("recipe") or {}
        recipe_id = str(recipe.get("recipe_id") or "").strip()
        if not recipe_id:
            continue
        day = int(entry.get("day") or 1)
        slot = str(entry.get("meal_type") or "meal")
        by_day.setdefault(day, {}).setdefault(slot, []).append(MealCourse(
            recipe_id=recipe_id,
            title=str(recipe.get("recipe_title") or recipe.get("title") or ""),
            ingredients=str(
                recipe.get("recipe_ingredients") or recipe.get("ingredients") or ""
            ),
            directions=str(
                recipe.get("recipe_directions") or recipe.get("directions") or ""
            ),
            nutrition=recipe.get("nutrition"),
            # Carried so the verifier and the repair pass keep a side a side.
            # Two entries sharing a slot already became two plates here; what
            # they lacked was which plate each one was.
            role=str(entry.get("role") or "main"),
        ))

    days = [
        DayPlan(day=day, meals=[Meal(slot, plates) for slot, plates in slots.items()])
        for day, slots in sorted(by_day.items())
    ]
    return MealPlan.from_days(days, "weekly") if days else None


def _apply_repairs(plan_entries: list[dict], adapted, outcome) -> int:
    """Write a repair made on the adapted plan back into the weekly entries.

    The repair runs against a `MealPlan` because that is the one shape the
    verifier and the repair both understand. Weekly stores entry dicts, so the
    swap has to be carried across — and it is carried by RECIPE ID, not by
    title or position: two dishes in a week can share a title, and an entry's
    index shifts if anything upstream ever reorders.

    Returns how many entries were rewritten, so a mismatch is visible rather
    than a plan that silently kept the dish the repair thought it removed.
    """
    replacements = {
        row["was_id"]: row["now_id"]
        for row in outcome.repaired
        if row.get("was_id") and row.get("now_id")
    }
    if not replacements:
        return 0

    # The repaired plates, by their new id, so the entry can be rebuilt from
    # the same MealCourse the verifier just re-measured.
    fresh = {
        plate.recipe_id: plate
        for day in adapted.day_plans for meal in day.meals for plate in meal.plates
        if plate.recipe_id
    }

    applied = 0
    for entry in plan_entries:
        recipe = entry.get("recipe") or {}
        old_id = str(recipe.get("recipe_id") or "")
        new_id = replacements.get(old_id)
        plate = fresh.get(new_id) if new_id else None
        if plate is None:
            continue
        recipe["recipe_id"] = plate.recipe_id
        recipe["recipe_title"] = plate.title
        recipe["recipe_ingredients"] = plate.ingredients
        recipe["recipe_directions"] = plate.directions
        # The old nutrition and image belonged to the dish that was removed.
        recipe["nutrition"] = plate.nutrition
        recipe["image_url"] = plate.image_url
        recipe["match_reasons"] = list(plate.match_reasons or [])
        entry["recipe"] = recipe
        applied += 1

    if applied != len(replacements):
        logger.warning(
            "Repair wrote back %d of %d swaps — an entry could not be matched",
            applied, len(replacements),
        )
    return applied

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _format_weekly_plan_as_context(plan: WeeklyMealPlan) -> str:
    """Serialise the current canvas weekly plan into a compact text for the planner."""
    lines = [f"[Current weekly meal plan — version {plan.version}]"]
    by_day: dict[int, list] = {}
    for entry in plan.entries:
        day = entry.get("day", 0)
        by_day.setdefault(day, []).append(entry)

    for day_idx in sorted(by_day):
        # Days are 1-based (1=Monday … 7=Sunday) throughout the weekly planner.
        label = DAY_NAMES[day_idx - 1] if 1 <= day_idx <= len(DAY_NAMES) else f"Day {day_idx}"
        lines.append(f"\n{label}:")
        for entry in sorted(by_day[day_idx], key=lambda e: e.get("meal_idx", 0)):
            recipe = entry.get("recipe", {})
            lines.append(f"  {entry.get('meal_type', '')}: {recipe.get('title', '')}")
    return "\n".join(lines)


class WeeklyPlanService:
    """Service to orchestrate weekly meal planning."""

    def __init__(self, session_service: SessionService):
        self.session_service = session_service
        self.reward_calculator = RewardCalculator()
        self.seed_service = SeedService()
        self.feedback_service = FeedbackService()
        self.response_writer = ResponseWriter()
        logger.info("WeeklyPlanService initialized.")

    def process_message(
        self,
        session_id: str,
        content: str,
        is_refinement: bool = False,
        seeds: Optional[list[dict]] = None,
    ) -> Tuple[str, Optional[WeeklyMealPlan]]:
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        logger.info(
            "[%s] Weekly plan requested (refinement=%s): %.120s",
            session_id, is_refinement, content,
        )
        self.session_service.add_message(session_id, "user", content)

        # Inject existing canvas plan as context for the planner when refining
        effective_query = content
        if is_refinement and session.weekly_canvas:
            current_plan = session.get_current_weekly_plan()
            if current_plan:
                plan_context = _format_weekly_plan_as_context(current_plan)
                effective_query = (
                    f"{plan_context}\n\n"
                    f"User refinement request: {content}"
                )
                logger.info(
                    "[%s] Refinement: injecting canvas weekly plan v%d as context.",
                    session_id, current_plan.version,
                )

        # What the member said, on the same pass every other kind of turn now
        # runs. This path used to extract two of the four things — pantry, and
        # its own copy of the diet extraction — and it filed nutrition claims
        # under `notes`, which is read only by `describe()` and only logged. So
        # "a high-protein week" was heard, stored, and dropped. Intake puts the
        # claim on `claim_tags`, which every fetch site reads, and adds the two
        # extractions weekly never had: facets ("a Thai week") and shape.
        from services import (
            pantry_service, plan_history, plan_parameters, turn_intake,
        )

        state = turn_intake.intake(
            session_id, content, session_service=self.session_service,
        )
        # A cooking-time ceiling stated in words belongs where the slider's
        # value lives — `RecipeActionSpace`, the pantry fan-out and the brief
        # all read it from there.
        session.user_profile = plan_parameters.apply_state(
            dict(session.user_profile or {}), state,
        )

        pantry = state.pantry
        if pantry:
            logger.info("[%s] Pantry to use up: %s", session_id, ", ".join(pantry))
        # Union of every diet stated this session, plus the profile's own inside
        # RecipeActionSpace.
        standing_diet = list(state.diet_tags)
        if standing_diet:
            logger.info("[%s] Diet in force: %s", session_id, ", ".join(standing_diet))

        # Standing seeds (M3): dishes the user consented to "always include"
        # auto-anchor into fresh weekly plans when no explicit seeds compete.
        if not seeds and not is_refinement:
            standing = session.user_profile.get("standing_seeds") or []
            if standing:
                seeds = [{"name": s["name"]} for s in standing[:3] if s.get("name")]
                logger.info("[%s] Auto-seeding %d standing dish(es).", session_id, len(seeds))

        # Resolve user-named anchor dishes into pinned (day, meal) slots (M2).
        seed_note = ""
        pinned: dict = {}
        if seeds:
            resolutions = self.seed_service.resolve_seeds(seeds, session.user_profile)
            placements, dropped = self.seed_service.place_weekly(resolutions)
            pinned = {
                slot: {
                    "recipe_id": r.recipe_id, "recipe_title": r.title,
                    "recipe_ingredients": r.ingredients, "recipe_directions": r.directions,
                }
                for slot, r in placements.items()
            }
            seed_note = self.seed_service.describe(resolutions, dropped)

        # "No thanks" to the favourites offer is a standing answer, and it was
        # honoured on the daily path only — weekly kept adding +5 per favourite
        # and putting them in the plan. A member who says no and sees their
        # favourite anyway has been told their answer does not matter.
        if state.use_favorites is False and session.user_profile.get("favorite_recipe_ids"):
            session.user_profile = {
                **session.user_profile, "favorite_recipe_ids": [],
            }
            logger.info("[%s] Favourites declined — not used for this week.", session_id)

        logger.info("[%s] Initializing action space and environment.", session_id)
        action_space = RecipeActionSpace(
            session.user_profile, additional_diet=standing_diet, pantry=pantry,
            # The standing shape, so a week that was asked for with a salad
            # beside dinner keeps it through a refinement. Without this the
            # action space had no notion of a plate and every refinement
            # flattened a multi-plate week back to single dishes.
            spec=state.spec,
            # What the member was served on their last few plans, so a second
            # "plan my week" is a different week. Fresh plans only: a
            # refinement is a request to change the week on screen.
            avoid_recent=(
                [] if is_refinement
                else plan_history.recently_served(session)
            ),
        )
        # Anchored recipes must never repeat elsewhere in the week.
        for entry in pinned.values():
            action_space.mark_selected(entry["recipe_id"])
        # Downvoted recipes never come back (M3 feedback loop).
        signals = self.feedback_service.get_signals(session.member_id)
        for recipe_id in signals.downvoted_recipe_ids:
            action_space.mark_selected(recipe_id)
        # Nor does anything the member rejected in conversation. This used the
        # same channel as downvotes and simply was not connected to it, so
        # "not that one" held on the daily canvas and not the weekly one.
        for recipe_id in state.excluded_recipe_ids:
            if recipe_id:
                action_space.mark_selected(recipe_id)
        # The shape the member asked for, if they asked for one. The weekly
        # walk was a hardcoded 7 days x three meals, so "two weeks" and "a week
        # with a snack" were both unbuildable here — `PlanSpec` expresses
        # exactly that and never reached this path.
        env = WeeklyMealPlanEnv(
            user_profile=session.user_profile,
            action_space=action_space,
            reward_calculator=self.reward_calculator,
            user_query=effective_query,
            stated_diet=standing_diet,
            spec=state.spec,
        )
        planner = WeeklyPlanner(env)

        logger.info(
            "[%s] Generating %d-day plan (%d meals: %s, %d pinned).",
            session_id, env.num_days, env.total_slots,
            ", ".join(env.meal_types), len(pinned),
        )
        try:
            plan_entries = planner.generate_full_plan(
                user_query=effective_query, pinned=pinned,
                scorer=build_preference_scorer(session.user_profile, pantry=pantry),
            )
        except PlanGenerationError as exc:
            # An unfillable slot is an answer, not a server fault. Name the
            # slot and the standing constraints, because the member cannot fix
            # what they cannot see — "adjust your requirements" with no noun
            # was the old daily-path message, and it taught nobody anything.
            logger.warning("[%s] Weekly plan unfillable: %s", session_id, exc)
            constraints = []
            if action_space.diet:
                constraints.append("diet: " + ", ".join(sorted(map(str, action_space.diet))))
            if action_space.allergens:
                constraints.append(
                    "allergens excluded: " + ", ".join(sorted(map(str, action_space.allergens)))
                )
            because = (
                " with your current constraints (" + "; ".join(constraints) + ")"
                if constraints else ""
            )
            message = (
                f"I couldn't build the full week — I found no recipes for "
                f"{exc.meal_type} on day {exc.day}{because}. "
                "I can try a shorter plan, relax one of the constraints, or "
                "you can name a dish you'd like there and I'll plan around it."
            )
            self.session_service.add_message(session_id, "assistant", message)
            return message, None
        logger.info("[%s] Plan generation complete — %d entries.", session_id, len(plan_entries))

        # M4 enrichment: one batch details call covers all 21 recipes —
        # nutrition chips, images, and diet tags for the weekly canvas.
        entry_ids = [str(e.get("recipe", {}).get("recipe_id", "")) for e in plan_entries]
        enrichment = CANDIDATES.fetch_details(entry_ids)
        for entry in plan_entries:
            rich = enrichment.get(str(entry.get("recipe", {}).get("recipe_id", "")))
            if rich:
                entry["recipe"]["nutrition"] = rich.nutrition_dict()
                entry["recipe"]["image_url"] = rich.image_url
                entry["recipe"]["tags"] = rich.tags or []
                entry["recipe"]["dish_types"] = rich.dish_types or []
        # Member-saved adapted recipes replace the originals as the starting
        # point (title/ingredients/nutrition; ids stay the original).
        adapted_count = overlay_weekly_entries(plan_entries, session.user_profile)
        if adapted_count:
            logger.info(
                "[%s] %d weekly slot(s) use the member's adapted version.",
                session_id, adapted_count,
            )

        # Per-day headline summaries (M6) — computed after enrichment and
        # after the adapted-recipe overlay so they describe what the member
        # will actually see.
        day_summaries = build_day_summaries(plan_entries)

        # Explainability (M7) — attaches per-entry match_reasons in place
        # and builds the measured ledger, weekly metrics (meat count,
        # calorie budget, guideline checklist), per-day breakdown, and the
        # whole-week justification. LLM-free; selection events were
        # recorded by the planner at decision time.
        history_text = getattr(signals, "history_text", "") or ""
        explainability = build_weekly_explainability(
            plan_entries, session.user_profile,
            selection_events=env.selection_events,
            day_summaries=day_summaries,
            downvoted_count=len(signals.downvoted_recipe_ids),
            feedback_lines=len(history_text.splitlines()) if history_text else 0,
        )

        # Pantry coverage: per-entry UI badges + a ledger row, appended AFTER
        # explainability so its chips are extended, not overwritten. Measured
        # by the deterministic matcher — the reply may not claim more.
        pantry_facts = pantry_service.annotate_weekly_entries(
            plan_entries, pantry, explainability=explainability,
        )
        pantry_note = pantry_service.describe_coverage(pantry_facts)

        # Measure the week, the same way the daily paths measure a day.
        #
        # Weekly reported the declarative ledger and nothing else: `vegetarian`
        # rendered as satisfied because the word had been sent, across 21 meals
        # rather than three. The verifier reads the plates that came back.
        report = None
        repair_note = None
        try:
            from models.plan_brief import PlanBrief
            from services import plan_quality, plan_repair, plan_verifier, turn_budget

            adapted = _as_meal_plan(plan_entries)
            if adapted is not None:
                brief = PlanBrief.build(session.user_profile)
                report = plan_verifier.verify(
                    adapted, brief.to_requested(), enrichment,
                )
                logger.info(
                    "[%s] Weekly verified: %s",
                    session_id, plan_verifier.describe(report),
                )

                # One repair pass, same as the daily paths. Across 21 meals a
                # hard-constraint failure is more likely than on three, and
                # weekly was the path where nothing acted on it.
                if report.blocking and not turn_budget.skip(
                    "weekly repair", turn_budget.COST_FETCH,
                ):
                    outcome = plan_repair.repair(
                        adapted, brief, report, session.user_profile,
                    )
                    if outcome.changed:
                        applied = _apply_repairs(plan_entries, adapted, outcome)
                        if applied:
                            report = outcome.report
                            repair_note = plan_repair.describe(outcome)
                            logger.info(
                                "[%s] Weekly repair: %d entry(ies) rewritten",
                                session_id, applied,
                            )

                if report.checks:
                    explainability["constraints_applied"] = (
                        list(explainability.get("constraints_applied") or [])
                        + report.as_ledger_rows()
                    )

                # Quality metrics, which weekly has never had: the graders were
                # instance attributes on ChatService, so the path that produces
                # 21 meals said the least about them. Measured across the whole
                # week — judging it on Monday reports the variety of a Monday.
                if not turn_budget.skip("weekly quality", turn_budget.COST_METRICS):
                    explainability.setdefault("metrics", {})
                    explainability["metrics"]["quality"] = plan_quality.metrics(
                        plan_quality.scored_from_plan(_as_meal_plan(plan_entries)),
                    )
        except Exception as exc:  # noqa: BLE001
            # All of this describes a plan that already exists. Losing any of
            # it costs the measured rows or the scores, never the week.
            logger.warning("[%s] Weekly verification failed: %s", session_id, exc)

        if is_refinement:
            weekly_plan = self.session_service.refine_weekly_meal_plan(
                session_id, plan_entries, day_summaries=day_summaries,
                explainability=explainability,
            )
            logger.info(
                "[%s] Refined weekly plan → %s (v%d, parent=%s).",
                session_id, weekly_plan.id, weekly_plan.version, weekly_plan.parent_id,
            )
            fallback = (
                "Here's your updated weekly meal plan! "
                "I've adjusted it based on what you asked for — take a look and let me know if you'd like any other tweaks."
            )
        else:
            weekly_plan = self.session_service.add_weekly_meal_plan(
                session_id, plan_entries, day_summaries=day_summaries,
                explainability=explainability,
            )
            logger.info("[%s] Weekly meal plan %s stored.", session_id, weekly_plan.id)
            fallback = (
                "Here's your 7-day meal plan! "
                "I've picked out breakfast, lunch, and dinner for each day based on your profile. "
                "Let me know if you'd like to swap anything out or adjust it."
            )

        pinned_titles = [p.get("recipe_title", "") for p in pinned.values()]
        weekly_honored, weekly_not_honored = split_ledger(
            explainability["constraints_applied"]
        )
        # A failed check the reply does not mention is a failure the member
        # discovers by eating it — 21 chances of that on a week.
        if repair_note:
            explainability.setdefault("metrics", {})["repair"] = repair_note
        weekly_problems = (
            [{"constraint": c.name, "detail": c.detail} for c in report.failed]
            if report is not None else []
        )
        facts = {
            "action": "refined_weekly_plan" if is_refinement else "new_weekly_plan",
            "days": 7, "meals": 21,
            "verified_problems": weekly_problems,
            "repair": repair_note,
            "anchored_dishes": pinned_titles,
            "seed_note": seed_note,
            "cooking_for": session.user_profile.get("cooking_for_names") or [],
            "day_summaries": [
                f"{DAY_NAMES[d - 1] if 1 <= d <= 7 else f'Day {d}'}: {s}"
                for d, s in sorted(day_summaries.items()) if s
            ],
            # M7: same fact keys the daily flow uses, plus the measured week
            # summary ("kept within your 3-meat-meal limit, 96% of your
            # calorie budget") so the reply can mention it. Split by status,
            # never sliced — the weekly ledger also carries "violated", and
            # announcing a violated meat limit as honoured is the one thing
            # this ledger exists to prevent.
            "constraints_honored": weekly_honored,
            "constraints_not_honored": weekly_not_honored,
            "week_summary": explainability["reasoning"],
        }
        if pantry_facts:
            facts["pantry"] = {
                "used": pantry_facts["used"],
                "unused": pantry_facts["unused"],
                "note": pantry_note,
            }
        fallback_extras = " ".join(p for p in (seed_note, pantry_note) if p)
        response_text = self.response_writer.write(
            facts, content,
            fallback=f"{fallback} {fallback_extras}".strip() if fallback_extras else fallback,
        )

        self.session_service.add_message(session_id, "assistant", response_text)

        return response_text, weekly_plan
