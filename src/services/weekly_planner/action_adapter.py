from typing import List, Dict, Any, Union
from src.models.session import MealCourse
from KG_neo4j.query_kg import get_filtered_recipe_ids

class RecipeActionSpace:
    """
    Action space for recipe selection in the weekly MDP.
    Fetches fresh candidates per day using exclude_ids to guarantee
    no recipe repeats across the 7-day plan.
    """

    def __init__(self, user_profile: Dict[str, Any], additional_diet: List[str] = None):
        self.user_profile = user_profile
        self.allergens = user_profile.get("allergies", [])

        profile_diet = user_profile.get("diet", [])
        if isinstance(profile_diet, str):
            profile_diet = [profile_diet]
        self.diet = list(set(profile_diet + (additional_diet or [])))

        # Per-day candidate pool; refreshed each time get_candidate_actions is called
        # for a new day. Keyed by day index.
        self._day_cache: Dict[int, Dict[str, List]] = {}
        # IDs of recipes already committed to the plan — passed as exclude_ids each fetch
        self._selected_ids: List[str] = []

    def get_candidate_actions(self, meal_type: Union[str, int], current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        current_day = current_state.get("day", 1)

        # Fetch a fresh pool for this day if not already done
        if current_day not in self._day_cache:
            self._day_cache[current_day] = get_filtered_recipe_ids(
                self.allergens,
                self.diet,
                limit=10,
                exclude_ids=list(self._selected_ids),
                randomize=True,
            )

        if isinstance(meal_type, int):
            course_map = {0: "breakfast", 1: "lunch", 2: "dinner"}
            meal_type = course_map.get(meal_type, "lunch")

        candidates_raw = self._day_cache[current_day].get(str(meal_type).lower(), [])

        actions = []
        for r in candidates_raw:
            meal_course = MealCourse.from_list(r)
            actions.append({
                "recipe_id": meal_course.recipe_id,
                "recipe_title": meal_course.title,
                "recipe_ingredients": meal_course.ingredients,
                "recipe_directions": meal_course.directions,
            })
        return actions

    def mark_selected(self, recipe_id: str) -> None:
        """Called by the environment after a recipe is committed to the plan."""
        if recipe_id and recipe_id not in self._selected_ids:
            self._selected_ids.append(recipe_id)
