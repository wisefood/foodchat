from typing import List, Dict, Any, Tuple, Optional
from src.services.session_service import SessionService
from src.services.weekly_planner.action_adapter import RecipeActionSpace
from src.services.weekly_planner.reward_logic import RewardCalculator
from src.services.weekly_planner.environment import WeeklyMealPlanEnv
from src.services.weekly_planner.planner import WeeklyPlanner
from src.models.session import WeeklyMealPlan, Message

class WeeklyPlanService:
    """Service to orchestrate weekly meal planning."""

    def __init__(self, session_service: SessionService):
        self.session_service = session_service
        self.reward_calculator = RewardCalculator()

    def process_message(self, session_id: str, content: str) -> Tuple[str, WeeklyMealPlan]:
        """Process a user message for weekly planning and generate a plan."""
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # 1. Store user message in weekly history
        self.session_service.add_weekly_message(session_id, "user", content)

        # 2. Initialize components, environment and planner
        action_space = RecipeActionSpace(session.user_profile)
        env = WeeklyMealPlanEnv(
            user_profile=session.user_profile,
            action_space=action_space,
            reward_calculator=self.reward_calculator,
            user_query=content
        )
        planner = WeeklyPlanner(env)

        # 3. Generate the full 7-day plan
        plan_entries = planner.generate_full_plan(user_query=content)

        # 4. Store the generated plan
        weekly_plan = self.session_service.add_weekly_meal_plan(session_id, plan_entries)

        # 5. Format response message
        response_text = f"I've generated a personalized 7-day meal plan for you based on your request: '{content}'. The plan includes 21 meals tailored to your preferences and nutritional goals."
        
        # 6. Store assistant message in weekly history
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
