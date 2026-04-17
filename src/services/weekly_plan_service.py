import logging
from typing import List, Dict, Any, Tuple, Optional
from src.services.session_service import SessionService
from src.services.weekly_planner.action_adapter import RecipeActionSpace
from src.services.weekly_planner.reward_logic import RewardCalculator
from src.services.weekly_planner.environment import WeeklyMealPlanEnv
from src.services.weekly_planner.planner import WeeklyPlanner
from src.models.session import WeeklyMealPlan, Message

logger = logging.getLogger(__name__)


class WeeklyPlanService:
    """Service to orchestrate weekly meal planning."""

    def __init__(self, session_service: SessionService):
        self.session_service = session_service
        self.reward_calculator = RewardCalculator()
        logger.info("WeeklyPlanService initialized.")

    def process_message(self, session_id: str, content: str) -> Tuple[str, WeeklyMealPlan]:
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        logger.info("[%s] Weekly plan requested: %.120s", session_id, content)
        self.session_service.add_weekly_message(session_id, "user", content)

        logger.info("[%s] Initializing action space and environment.", session_id)
        action_space = RecipeActionSpace(session.user_profile)
        env = WeeklyMealPlanEnv(
            user_profile=session.user_profile,
            action_space=action_space,
            reward_calculator=self.reward_calculator,
            user_query=content
        )
        planner = WeeklyPlanner(env)

        logger.info("[%s] Generating 7-day plan (21 meals).", session_id)
        plan_entries = planner.generate_full_plan(user_query=content)
        logger.info("[%s] Plan generation complete — %d entries.", session_id, len(plan_entries))

        weekly_plan = self.session_service.add_weekly_meal_plan(session_id, plan_entries)
        logger.info("[%s] Weekly meal plan %s stored.", session_id, weekly_plan.id)

        response_text = f"I've generated a personalized 7-day meal plan for you based on your request: '{content}'. The plan includes 21 meals tailored to your preferences and nutritional goals."
        self.session_service.add_weekly_message(session_id, "assistant", response_text)

        return response_text, weekly_plan

    def get_weekly_messages(self, session_id: str) -> List[Message]:
        """Get the weekly message history for a session."""
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session.weekly_messages

    def get_weekly_meal_plans(self, session_id: str) -> List[WeeklyMealPlan]:
        """Get all weekly meal plans for a session."""
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session.weekly_meal_plans
