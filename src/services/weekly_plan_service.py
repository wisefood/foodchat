import logging
from typing import Optional, Tuple

from .feedback_service import FeedbackService
from .seed_service import SeedService
from .session_service import SessionService
from .weekly_planner.action_adapter import RecipeActionSpace
from .weekly_planner.reward_logic import RewardCalculator
from .weekly_planner.environment import WeeklyMealPlanEnv
from .weekly_planner.planner import WeeklyPlanner, build_preference_scorer
from .candidates_client import CANDIDATES
from agents import DietaryIntentExtractor, ResponseWriter
from models.session import WeeklyMealPlan

logger = logging.getLogger(__name__)


def _format_weekly_plan_as_context(plan: WeeklyMealPlan) -> str:
    """Serialise the current canvas weekly plan into a compact text for the planner."""
    lines = [f"[Current weekly meal plan — version {plan.version}]"]
    by_day: dict[int, list] = {}
    for entry in plan.entries:
        day = entry.get("day", 0)
        by_day.setdefault(day, []).append(entry)

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day_idx in sorted(by_day):
        label = day_names[day_idx] if day_idx < len(day_names) else f"Day {day_idx + 1}"
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
    ) -> Tuple[str, WeeklyMealPlan]:
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
            placements = self.seed_service.place_weekly(resolutions)
            pinned = {
                slot: {
                    "recipe_id": r.recipe_id, "recipe_title": r.title,
                    "recipe_ingredients": r.ingredients, "recipe_directions": r.directions,
                }
                for slot, r in placements.items()
            }
            seed_note = self.seed_service.describe(resolutions)

        logger.info("[%s] Initializing action space and environment.", session_id)
        action_space = RecipeActionSpace(session.user_profile, additional_diet=query_diet_tags)
        # Anchored recipes must never repeat elsewhere in the week.
        for entry in pinned.values():
            action_space.mark_selected(entry["recipe_id"])
        # Downvoted recipes never come back (M3 feedback loop).
        for recipe_id in self.feedback_service.get_signals(session.member_id).downvoted_recipe_ids:
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
        plan_entries = planner.generate_full_plan(
            user_query=effective_query, pinned=pinned,
            scorer=build_preference_scorer(session.user_profile),
        )
        logger.info("[%s] Plan generation complete — %d entries.", session_id, len(plan_entries))

        # M4 enrichment: one batch details call covers all 21 recipes —
        # nutrition chips and images for the weekly canvas.
        entry_ids = [str(e.get("recipe", {}).get("recipe_id", "")) for e in plan_entries]
        enrichment = CANDIDATES.fetch_details(entry_ids)
        for entry in plan_entries:
            rich = enrichment.get(str(entry.get("recipe", {}).get("recipe_id", "")))
            if rich:
                entry["recipe"]["nutrition"] = rich.nutrition_dict()
                entry["recipe"]["image_url"] = rich.image_url

        if is_refinement:
            weekly_plan = self.session_service.refine_weekly_meal_plan(session_id, plan_entries)
            logger.info(
                "[%s] Refined weekly plan → %s (v%d, parent=%s).",
                session_id, weekly_plan.id, weekly_plan.version, weekly_plan.parent_id,
            )
            fallback = (
                "Here's your updated weekly meal plan! "
                "I've adjusted it based on what you asked for — take a look and let me know if you'd like any other tweaks."
            )
        else:
            weekly_plan = self.session_service.add_weekly_meal_plan(session_id, plan_entries)
            logger.info("[%s] Weekly meal plan %s stored.", session_id, weekly_plan.id)
            fallback = (
                "Here's your 7-day meal plan! "
                "I've picked out breakfast, lunch, and dinner for each day based on your profile. "
                "Let me know if you'd like to swap anything out or adjust it."
            )

        pinned_titles = [p.get("recipe_title", "") for p in pinned.values()]
        facts = {
            "action": "refined_weekly_plan" if is_refinement else "new_weekly_plan",
            "days": 7, "meals": 21,
            "anchored_dishes": pinned_titles,
            "seed_note": seed_note,
            "cooking_for": session.user_profile.get("cooking_for_names") or [],
        }
        response_text = self.response_writer.write(
            facts, content,
            fallback=f"{fallback} {seed_note}".strip() if seed_note else fallback,
        )

        self.session_service.add_message(session_id, "assistant", response_text)

        return response_text, weekly_plan
