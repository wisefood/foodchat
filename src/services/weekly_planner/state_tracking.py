"""
Cumulative weekly nutrition/constraint state for the 7-day planner.

The tracker is updated once per committed slot (``environment.step``) and
read by the constraint logic in ``reward_logic`` BEFORE each pick (M6 — it
used to be write-only: nutrition never arrived during generation and the
resulting penalties were only logged).

Targets come from the member profile: calorie/macro targets are parsed from
preference strings (``profile_service._build_preferences`` format), and the
weekly meat limit is diet-aware instead of a hardcoded 3 — vegetarian/vegan
profiles get 0, pescatarian profiles stop counting fish toward the limit,
and an explicit "meat limit N" / "N meat meals" preference wins outright.
"""

import re
from typing import Any, Dict, List, Optional

from models.session import MealCourse

from .day_summary import is_meat_meal

DEFAULT_WEEKLY_MEAT_LIMIT = 3


class WeeklyNutritionalTracker:
    """
    Tracks cumulative weekly macros and constraint limits for a user's meal plan.
    Relies on existing user profile schemas and MealCourse models.
    """

    def __init__(self, user_profile: Dict[str, Any]):
        """
        Initialize the tracker with user preferences and constraints.

        Args:
            user_profile: Dict containing 'diet', 'allergies', 'preferences', etc.
                         Expected to follow the structure from ProfileService._map_profile.
        """
        self.user_profile = user_profile
        self.weekly_calories = 0.0
        self.weekly_protein = 0.0
        self.weekly_carbs = 0.0
        self.weekly_fat = 0.0
        self.meat_meals_count = 0

        diet = user_profile.get("diet") or []
        if isinstance(diet, str):
            diet = [diet]
        diet_set = {str(d).lower() for d in diet}
        # Pescatarians would have every fish meal counted as "meat" otherwise.
        self.counts_fish_as_meat = not (diet_set & {"pescatarian", "pescatarian_safe"})

        self.targets = self._extract_targets(
            user_profile.get("preferences", []) or [], diet_set
        )

    def _extract_targets(self, preferences: List[str], diet: set) -> Dict[str, float]:
        """Extract numeric targets from preference strings + diet."""
        targets = {
            "calories": 2000.0 * 7,  # Default weekly
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "meat_limit": 0 if diet & {"vegetarian", "vegan"} else DEFAULT_WEEKLY_MEAT_LIMIT,
        }

        for pref in preferences:
            pref_lower = pref.lower()
            if "calories target" in pref_lower:
                try:
                    val = float(pref_lower.split()[0])
                    targets["calories"] = val * 7
                except ValueError:
                    pass
            elif "high protein" in pref_lower:
                # e.g., "high protein (150g)"
                match = re.search(r"\((\d+)g\)", pref_lower)
                if match:
                    targets["protein"] = float(match.group(1)) * 7
            elif "g carbs" in pref_lower:
                try:
                    targets["carbs"] = float(pref_lower.split("g")[0].strip()) * 7
                except ValueError:
                    pass
            elif "g fat" in pref_lower:
                try:
                    targets["fat"] = float(pref_lower.split("g")[0].strip()) * 7
                except ValueError:
                    pass
            elif "meat" in pref_lower:
                # "meat limit 2", "max 2 meat meals", "2 meat meals a week"
                match = re.search(r"(\d+)\s*meat|meat[^\d]{0,12}(\d+)", pref_lower)
                if match:
                    targets["meat_limit"] = int(match.group(1) or match.group(2))

        return targets

    def update_tracker(
        self,
        meal: MealCourse,
        nutrition_info: Dict[str, float] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Update cumulative totals with a new meal.

        Args:
            meal: The MealCourse object added to the plan.
            nutrition_info: Optional per-serving dict. Accepts both the
                           RecipeWrangler enrichment keys (kcal/protein_g/
                           carbs_g/fat_g) and the generic calories/protein/
                           carbs/fat keys.
            tags: Optional RecipeWrangler tags for the recipe — a
                  vegetarian/vegan tag overrides keyword meat detection.
        """
        if nutrition_info:
            def _num(*keys: str) -> float:
                for key in keys:
                    value = nutrition_info.get(key)
                    if isinstance(value, (int, float)):
                        return float(value)
                return 0.0

            self.weekly_calories += _num("kcal", "calories")
            self.weekly_protein += _num("protein_g", "protein")
            self.weekly_carbs += _num("carbs_g", "carbs")
            self.weekly_fat += _num("fat_g", "fat")

        if is_meat_meal(
            meal.title, meal.ingredients,
            tags=tags, count_fish=self.counts_fish_as_meat,
        ):
            self.meat_meals_count += 1

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status vs targets."""
        return {
            "cumulative": {
                "calories": self.weekly_calories,
                "protein": self.weekly_protein,
                "carbs": self.weekly_carbs,
                "fat": self.weekly_fat,
                "meat_meals": self.meat_meals_count
            },
            "targets": self.targets,
            "remaining": {
                "calories": self.targets["calories"] - self.weekly_calories,
                "meat_limit_left": self.targets["meat_limit"] - self.meat_meals_count
            }
        }
