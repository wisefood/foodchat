import logging
from typing import Dict, Any, List, Tuple, Optional
from src.models.session import MealCourse
from src.services.weekly_planner.state_tracking import WeeklyNutritionalTracker
from src.services.weekly_planner.action_adapter import RecipeActionSpace
from src.services.weekly_planner.reward_logic import RewardCalculator

logger = logging.getLogger(__name__)

class WeeklyMealPlanEnv:
    """
    An environment for the 7-day meal planning MDP.
    Each step represents choosing a recipe for a specific meal (Breakfast, Lunch, Dinner).
    Total steps: 21 (7 days * 3 meals).
    """

    def __init__(
        self, 
        user_profile: Dict[str, Any], 
        action_space: RecipeActionSpace, 
        reward_calculator: RewardCalculator,
        user_query: Optional[str] = None
    ):
        """
        Initialize the environment with user preferences and components.
        
        Args:
            user_profile: Dict containing user preferences and constraints.
            action_space: The RecipeActionSpace instance to fetch candidate recipes.
            reward_calculator: The RewardCalculator instance to evaluate actions.
            user_query: Optional user query to guide LLM evaluation during planning.
        """
        self.user_profile = user_profile
        self.action_space = action_space
        self.reward_calculator = reward_calculator
        self.preferences = user_profile.get("preferences", [])
        self.user_query = user_query
        
        self.tracker = WeeklyNutritionalTracker(user_profile)
        self.current_day = 1
        self.current_meal_idx = 0  # 0: Breakfast, 1: Lunch, 2: Dinner
        self.meal_types = ["breakfast", "lunch", "dinner"]
        self.done = False
        self.plan = [] # To store the generated plan details

    def reset(self, user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset the environment to start a new 7-day planning cycle.
        
        Args:
            user_query: Optional update to the user query for the new cycle.
        """
        self.tracker = WeeklyNutritionalTracker(self.user_profile)
        self.current_day = 1
        self.current_meal_idx = 0
        self.done = False
        self.plan = []
        if user_query is not None:
            self.user_query = user_query
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """Returns the current state of the environment."""
        return {
            "day": self.current_day,
            "meal_idx": self.current_meal_idx,
            "meal_type": self.meal_types[self.current_meal_idx],
            "tracker_status": self.tracker.get_status(),
            "done": self.done
        }

    def step(self, chosen_recipe: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        Calculates the reward, updates the tracker, and advances the state.
        
        Args:
            chosen_recipe: The recipe dictionary selected from candidate actions.
            
        Returns:
            A tuple of (next_state, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Environment is done. Please reset.")

        # 1. Update Tracker
        # Convert dictionary to MealCourse model for the tracker update logic (meat detection, etc.)
        meal = MealCourse(
            recipe_id=str(chosen_recipe.get("recipe_id", "")),
            title=chosen_recipe.get("recipe_title", ""),
            ingredients=chosen_recipe.get("recipe_ingredients", ""),
            directions=chosen_recipe.get("recipe_directions", "")
        )
        
        # update_tracker handles meat detection and cumulative nutritional totals
        # In this MDP, nutrition_info is optional; we pass it if present in chosen_recipe
        self.tracker.update_tracker(meal, nutrition_info=chosen_recipe.get("nutrition"))
        
        # 2. Calculate Reward
        # reward_logic.py: calculate_step_reward(action, tracker, preferences, user_query)
        # We calculate the reward AFTER updating the tracker to include constraint penalties
        reward = self.reward_calculator.calculate_step_reward(
            chosen_recipe, 
            self.tracker, 
            self.preferences,
            user_query=self.user_query
        )
        
        # Store step in the plan list
        self.plan.append({
            "day": self.current_day,
            "meal_idx": self.current_meal_idx,
            "meal_type": self.meal_types[self.current_meal_idx],
            "recipe": chosen_recipe,
            "reward": reward
        })

        # 3. Advance to the next meal/day
        status = self.tracker.get_status()
        cumulative = status["cumulative"]
        targets = status["targets"]
        remaining = status["remaining"]
        
        logger.info(f"--- Tracker Status after Day {self.current_day}, {self.meal_types[self.current_meal_idx].capitalize()} ---")
        logger.info(f"Calories: {cumulative['calories']:.1f}/{targets['calories']:.1f} (Remaining: {remaining['calories']:.1f})")
        logger.info(f"Meat Meals: {cumulative['meat_meals']}/{targets['meat_limit']} (Limit Left: {remaining['meat_limit_left']})")
        logger.info(f"Protein: {cumulative['protein']:.1f}g, Carbs: {cumulative['carbs']:.1f}g, Fat: {cumulative['fat']:.1f}g")
        logger.info("-" * 40)

        self.current_meal_idx += 1
        if self.current_meal_idx >= 3:
            self.current_meal_idx = 0
            self.current_day += 1
            
        if self.current_day > 7:
            self.done = True
            
        return self._get_state(), reward, self.done, {}
