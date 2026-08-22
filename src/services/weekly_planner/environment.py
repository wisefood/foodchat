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

    @property
    def slots_filled(self) -> int:
        """Meals committed so far — NOT rows in `self.plan`.

        The two were the same number until a meal could be more than one plate.
        `self.plan` now holds one row per plate, so a caller measuring progress
        by its length reads a two-plate week as twice as far along as it is —
        and the calorie scorer divides the remaining budget by "slots left",
        which would go negative before the week was half planned.
        """
        return len({(row.get("day"), row.get("meal_idx")) for row in self.plan})

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

    def _entries_for(
        self, chosen_recipe: Dict[str, Any], plates: List[Dict[str, Any]], reward: float
    ) -> List[Dict[str, Any]]:
        """The plan rows this meal becomes — one per plate, or one for a dish.

        The reward belongs to the MEAL, and it is written onto the first plate
        only. Repeating it on every plate would make a two-plate dinner look
        twice as well-scored as a single-plate one to anything that sums the
        column.
        """
        base = {
            "day": self.current_day,
            "meal_idx": self.current_meal_idx,
            "meal_type": self.meal_types[self.current_meal_idx],
        }
        if not plates:
            return [{**base, "recipe": chosen_recipe, "reward": reward}]

        rows = []
        for index, plate in enumerate(plates):
            recipe = {k: v for k, v in plate.items() if k != "role"}
            if chosen_recipe.get("pinned"):
                recipe["pinned"] = True
            rows.append({
                **base,
                "role": plate.get("role") or "main",
                "recipe": recipe,
                "reward": reward if index == 0 else 0.0,
            })
        return rows

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
        #
        # One update per MEAL, not per plate. `nutrition` on a composed action
        # already totals every plate and `meal_ingredients` already concatenates
        # them, so the tracker counts the whole table once — a main-plus-side
        # meal counted twice would double the day against its calorie budget,
        # and counted as the main alone would let every side through unmeasured.
        meal = MealCourse(
            recipe_id=str(chosen_recipe.get("recipe_id", "")),
            title=chosen_recipe.get("recipe_title", ""),
            ingredients=(
                chosen_recipe.get("meal_ingredients")
                or chosen_recipe.get("recipe_ingredients", "")
            ),
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
        
        # Register every recipe in the meal so future fetches exclude all of
        # them — marking the main alone would let a week serve the same salad
        # on Tuesday as a side and on Friday as a main.
        plates = list(chosen_recipe.get("plates") or [])
        for recipe_id in (
            [str(p.get("recipe_id", "")) for p in plates] or
            [str(chosen_recipe.get("recipe_id", ""))]
        ):
            self.action_space.mark_selected(recipe_id)

        # Store the step: ONE ENTRY PER PLATE.
        #
        # That is the shape the weekly plan is already read through — the
        # renderer groups entries by `(day, meal_type)` and draws one cell each,
        # `_as_meal_plan` collects them into a meal's plates, and the API model
        # is a flat list. So a two-plate dinner is two entries sharing a day, a
        # slot and a `meal_idx`; sorting is stable, so they stay in plate order.
        # Nothing downstream needed a new shape — this walk was simply the only
        # thing that could not produce the one that existed.
        self.plan.extend(self._entries_for(chosen_recipe, plates, reward))

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
