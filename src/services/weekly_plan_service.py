import logging
from typing import List, Tuple, Optional

from .session_service import SessionService
from .weekly_planner.action_adapter import RecipeActionSpace
from .weekly_planner.reward_logic import RewardCalculator
from .weekly_planner.environment import WeeklyMealPlanEnv
from .weekly_planner.planner import WeeklyPlanner
from agents import DietaryIntentExtractor
from models.session import WeeklyMealPlan, Message

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
        logger.info("WeeklyPlanService initialized.")

    def process_message(
        self,
        session_id: str,
        content: str,
        is_refinement: bool = False,
    ) -> Tuple[str, WeeklyMealPlan]:
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        logger.info(
            "[%s] Weekly plan requested (refinement=%s): %.120s",
            session_id, is_refinement, content,
        )
        self.session_service.add_weekly_message(session_id, "user", content)

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

        logger.info("[%s] Initializing action space and environment.", session_id)
        action_space = RecipeActionSpace(session.user_profile, additional_diet=query_diet_tags)
        env = WeeklyMealPlanEnv(
            user_profile=session.user_profile,
            action_space=action_space,
            reward_calculator=self.reward_calculator,
            user_query=effective_query,
        )
        planner = WeeklyPlanner(env)

        logger.info("[%s] Generating 7-day plan (21 meals).", session_id)
        plan_entries = planner.generate_full_plan(user_query=effective_query)
        logger.info("[%s] Plan generation complete — %d entries.", session_id, len(plan_entries))

        if is_refinement:
            weekly_plan = self.session_service.refine_weekly_meal_plan(session_id, plan_entries)
            logger.info(
                "[%s] Refined weekly plan → %s (v%d, parent=%s).",
                session_id, weekly_plan.id, weekly_plan.version, weekly_plan.parent_id,
            )
            response_text = (
                f"Here's your updated weekly meal plan! "
                f"I've adjusted it based on what you asked for — take a look and let me know if you'd like any other tweaks."
            )
        else:
            weekly_plan = self.session_service.add_weekly_meal_plan(session_id, plan_entries)
            logger.info("[%s] Weekly meal plan %s stored.", session_id, weekly_plan.id)
            response_text = (
                "Here's your 7-day meal plan! "
                "I've picked out breakfast, lunch, and dinner for each day based on your profile. "
                "Let me know if you'd like to swap anything out or adjust it."
            )

        self.session_service.add_weekly_message(session_id, "assistant", response_text)

        return response_text, weekly_plan

    def get_weekly_messages(self, session_id: str) -> List[Message]:
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session.weekly_messages

    def get_weekly_meal_plans(self, session_id: str) -> List[WeeklyMealPlan]:
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session.weekly_meal_plans
