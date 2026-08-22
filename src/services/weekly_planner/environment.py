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
        user_query: Optional[str] = None,
        stated_diet: Optional[list] = None,
        spec: Optional[object] = None,
    ):
        """
        Initialize the environment with user preferences and components.
        
        Args:
            user_profile: Dict containing user preferences and constraints.
            action_space: The RecipeActionSpace instance to fetch candidate recipes.
            reward_calculator: The RewardCalculator instance to evaluate actions.
            user_query: Optional user query to guide LLM evaluation during planning.
            stated_diet: Diet tags the member stated in chat this session. The
                tracker needs them for the meat limit and for whether fish
                counts as meat — reading only the stored profile meant a
                stated vegetarian was still budgeted three meat meals.
        """
        self.user_profile = user_profile
        self.action_space = action_space
        self.reward_calculator = reward_calculator
        self.preferences = user_profile.get("preferences", [])
        self.user_query = user_query
        self.stated_diet = list(stated_diet or [])

        # The shape being planned.
        #
        # This walk was a hardcoded 7 days x ["breakfast", "lunch", "dinner"],
        # with the 7 and the 3 written as literals in `step()`. So "plan me
        # three days" and "a week with a snack" were both unbuildable on this
        # path — `PlanSpec` exists to express exactly that and never reached
        # here. The default is the old shape, so every existing caller is
        # unchanged.
        self.num_days = 7
        self.meal_types = ["breakfast", "lunch", "dinner"]
        if spec is not None:
            days = int(getattr(spec, "num_days", 0) or 0)
            slots = [str(m) for m in (getattr(spec, "meals", ()) or ())]
            # `> 1`, not `> 0`. `PlanSpec.default()` is ONE day, and the
            # default is what a spec looks like when the extractor found no
            # shape in the message — so honouring it here would turn "plan my
            # week" into a single day whenever the words did not happen to
            # name a number. The extractor does set num_days=7 for weekly
            # language, so a real request still lands.
            if days > 1:
                self.num_days = days
            if slots:
                self.meal_types = slots

        self.tracker = WeeklyNutritionalTracker(
            user_profile, self.stated_diet, num_days=self.num_days,
        )
        self.current_day = 1
        self.current_meal_idx = 0
        self.done = False
        self.plan = [] # To store the generated plan details
        # Selection events recorded while picking (M7 explainability) —
        # meat-pool prunes and limit relaxations, appended by the planner.
        self.selection_events: List[Dict[str, Any]] = []

    @property
    def total_slots(self) -> int:
        """Every slot this plan will fill. Was the constant `TOTAL_SLOTS = 21`."""
        return self.num_days * len(self.meal_types)

    def reset(self, user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset the environment to start a new planning cycle.
        
        Args:
            user_query: Optional update to the user query for the new cycle.
        """
        self.tracker = WeeklyNutritionalTracker(
            self.user_profile, self.stated_diet, num_days=self.num_days,
        )
        self.current_day = 1
        self.current_meal_idx = 0
        self.done = False
        self.plan = []
        self.selection_events = []
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
        
        # Register selected recipe so future fetches exclude it
        self.action_space.mark_selected(str(chosen_recipe.get("recipe_id", "")))

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
        # `>= 3` and `> 7` were literals here, which is what made the shape
        # fixed no matter what the member asked for.
        if self.current_meal_idx >= len(self.meal_types):
            self.current_meal_idx = 0
            self.current_day += 1

        if self.current_day > self.num_days:
            self.done = True
            
        return self._get_state(), reward, self.done, {}
