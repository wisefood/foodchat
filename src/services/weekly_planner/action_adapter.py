"""
Action space for the weekly-plan MDP — candidate recipes per meal slot.

Fetches a fresh candidate pool from RecipeWrangler once per plan day
(``services.candidates_client``), excluding every recipe already committed to
the plan so a 7-day plan never repeats a recipe.
"""

from typing import Any, Dict, List, Union

from services.candidates_client import CANDIDATES

# Candidates fetched per slot per day. One fetch serves all three slots of a
# day, so this stays small to keep the RecipeWrangler payloads light.
DAILY_POOL_LIMIT = 10


class RecipeActionSpace:
    """Per-day candidate pools for the weekly planner."""

    def __init__(self, user_profile: Dict[str, Any], additional_diet: List[str] = None):
        self.user_profile = user_profile
        self.allergens = user_profile.get("allergies", [])

        profile_diet = user_profile.get("diet", [])
        if isinstance(profile_diet, str):
            profile_diet = [profile_diet]
        # Query-level diet tags (extracted from the user message) tighten the
        # profile diet for this plan only.
        self.diet = list(set(profile_diet + (additional_diet or [])))

        # Per-day candidate pool cache, keyed by day index.
        self._day_cache: Dict[int, Dict[str, list]] = {}
        # Recipes already committed to the plan — excluded from every fetch.
        self._selected_ids: List[str] = []

    def get_candidate_actions(
        self, meal_type: Union[str, int], current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return candidate action dicts for the given meal slot and state."""
        current_day = current_state.get("day", 1)

        if current_day not in self._day_cache:
            self._day_cache[current_day] = CANDIDATES.fetch_candidates(
                allergens=self.allergens,
                diet=self.diet,
                exclude_recipe_ids=list(self._selected_ids),
                favorite_recipe_ids=self.user_profile.get("favorite_recipe_ids") or [],
                limit_per_slot=DAILY_POOL_LIMIT,
                randomize=True,
            )

        if isinstance(meal_type, int):
            meal_type = {0: "breakfast", 1: "lunch", 2: "dinner"}.get(meal_type, "lunch")

        candidates = self._day_cache[current_day].get(str(meal_type).lower(), [])
        return [
            {
                "recipe_id": c.recipe_id,
                "recipe_title": c.title,
                "recipe_ingredients": c.ingredients,
                "recipe_directions": c.directions,
            }
            for c in candidates
        ]

    def mark_selected(self, recipe_id: str) -> None:
        """Called by the environment after a recipe is committed to the plan."""
        if recipe_id and recipe_id not in self._selected_ids:
            self._selected_ids.append(recipe_id)
