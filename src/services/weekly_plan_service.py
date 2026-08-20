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
from agents import DietaryIntentExtractor, ResponseWriter
from models.session import WeeklyMealPlan

logger = logging.getLogger(__name__)

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
        self.diet_extractor = DietaryIntentExtractor()
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

        # Extract dietary requirements from the user query to filter recipes correctly
        query_diet_tags = self.diet_extractor.extract(content)
        if query_diet_tags:
            logger.info("[%s] Extracted diet tags from query: %s", session_id, query_diet_tags)

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

        logger.info("[%s] Initializing action space and environment.", session_id)
        action_space = RecipeActionSpace(session.user_profile, additional_diet=query_diet_tags)
        # Anchored recipes must never repeat elsewhere in the week.
        for entry in pinned.values():
            action_space.mark_selected(entry["recipe_id"])
        # Downvoted recipes never come back (M3 feedback loop).
        signals = self.feedback_service.get_signals(session.member_id)
        for recipe_id in signals.downvoted_recipe_ids:
            action_space.mark_selected(recipe_id)
        env = WeeklyMealPlanEnv(
            user_profile=session.user_profile,
            action_space=action_space,
            reward_calculator=self.reward_calculator,
            user_query=effective_query,
        )
        planner = WeeklyPlanner(env)

        logger.info(
            "[%s] Generating 7-day plan (21 meals, %d pinned).", session_id, len(pinned)
        )
        try:
            plan_entries = planner.generate_full_plan(
                user_query=effective_query, pinned=pinned,
                scorer=build_preference_scorer(session.user_profile),
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
        facts = {
            "action": "refined_weekly_plan" if is_refinement else "new_weekly_plan",
            "days": 7, "meals": 21,
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
        response_text = self.response_writer.write(
            facts, content,
            fallback=f"{fallback} {seed_note}".strip() if seed_note else fallback,
        )

        self.session_service.add_message(session_id, "assistant", response_text)

        return response_text, weekly_plan
