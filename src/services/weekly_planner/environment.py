import logging
from typing import Dict, Any, List, Tuple, Optional
from models.session import MealCourse
from .state_tracking import WeeklyNutritionalTracker
from .action_adapter import RecipeActionSpace
from .reward_logic import RewardCalculator

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
        # Selection events recorded while picking (M7 explainability) —
        # meat-pool prunes and limit relaxations appended by the planner,
        # sanctioned repeats appended below, and the action space's own
        # sourcing decisions (M9). One ledger, so `metrics.selection_events`
        # carries the whole selection story in the order it happened.
        self.selection_events: List[Dict[str, Any]] = []
        if hasattr(action_space, "selection_events"):
            action_space.selection_events = self.selection_events

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
        # Cleared in place: the action space holds a reference to this same
        # list (see __init__), and rebinding it here would silently orphan
        # every event it records after a reset.
        self.selection_events.clear()
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
        
        # update_tracker handles meat detection and cumulative nutritional totals.
        # Candidates carry nutrition/tags from the per-day enrichment (M6);
        # both stay optional so bare recipes (pins, fakes) still work.
        self.tracker.update_tracker(
            meal,
            nutrition_info=chosen_recipe.get("nutrition"),
            tags=chosen_recipe.get("tags"),
        )
        
        # 2. Calculate Reward
        # reward_logic.py: calculate_step_reward(action, tracker, preferences, user_query)
        # We calculate the reward AFTER updating the tracker to include constraint penalties
        reward = self.reward_calculator.calculate_step_reward(
            chosen_recipe, 
            self.tracker, 
            self.preferences,
            user_query=self.user_query
        )
        
        # Register the commitment so later fetches know what the week has
        # already served. `mark_committed` carries the day and the slot, which
        # is what the repeat policy needs; an action space that predates it
        # (the fakes in the tests) falls back to the old never-again call and
        # behaves exactly as before.
        recipe_id = str(chosen_recipe.get("recipe_id", ""))
        commit = getattr(self.action_space, "mark_committed", None)
        if callable(commit):
            commit(recipe_id, self.current_day, self.meal_types[self.current_meal_idx])
        else:
            self.action_space.mark_selected(recipe_id)

        # A repeat is a decision, so it is recorded where the other selection
        # decisions are, at the moment it is taken. `source` is the whole point:
        # a second serving the member starred and one the planner chose are
        # different claims, and only this event knows which happened.
        if chosen_recipe.get("repeat_of_day"):
            self.selection_events.append({
                "type": "repeat_allowed",
                "day": self.current_day,
                "meal_type": self.meal_types[self.current_meal_idx],
                "recipe_id": recipe_id,
                "recipe_title": chosen_recipe.get("recipe_title", ""),
                "repeat_of_day": chosen_recipe["repeat_of_day"],
                "source": chosen_recipe.get("repeat_source", "plan"),
            })

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
