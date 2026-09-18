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
from services import reference_intake

from .day_summary import is_meat_meal

DEFAULT_WEEKLY_MEAT_LIMIT = 3


class WeeklyNutritionalTracker:
    """
    Tracks cumulative weekly macros and constraint limits for a user's meal plan.
    Relies on existing user profile schemas and MealCourse models.
    """

    def __init__(
        self,
        user_profile: Dict[str, Any],
        stated_diet: Optional[List[str]] = None,
        num_days: int = 7,
    ):
        """
        Initialize the tracker with user preferences and constraints.

        Args:
            user_profile: Dict containing 'diet', 'allergies', 'preferences', etc.
                         Expected to follow the structure from ProfileService._map_profile.
            stated_diet: Diet stated in chat, which outranks the stored profile.
            num_days: How many days this plan covers. Defaults to 7 — the only
                horizon that existed when this was written — so every existing
                caller is unchanged.
        """
        self.user_profile = user_profile
        self.weekly_calories = 0.0
        self.weekly_protein = 0.0
        self.weekly_carbs = 0.0
        self.weekly_fat = 0.0
        self.meat_meals_count = 0

        # A diet the member stated in chat counts as much as a stored one. It
        # used to be invisible here, so someone who said "vegetarian" this
        # session still got a meat budget of 3 and had every fish meal counted
        # against it — the tracker was reading a profile that disagreed with
        # the plan being built from it.
        diet = user_profile.get("diet") or []
        if isinstance(diet, str):
            diet = [diet]
        diet_set = {str(d).lower() for d in diet}
        diet_set |= {str(d).lower() for d in (stated_diet or [])}
        # Pescatarians would have every fish meal counted as "meat" otherwise.
        self.counts_fish_as_meat = not (diet_set & {"pescatarian", "pescatarian_safe"})

        # How many days this plan covers. Every target below is a DAILY figure
        # multiplied by it — a 3-day plan given a 7-day budget would never
        # think it was near the limit, so the tracker's calorie and meat
        # steering would do nothing at all for the whole plan.
        self.num_days = max(1, int(num_days or 7))

        self.targets = self._extract_targets(
            user_profile.get("preferences", []) or [], diet_set
        )

    def _extract_targets(self, preferences: List[str], diet: set) -> Dict[str, Any]:
        """Extract numeric targets from preference strings + diet."""
        days = self.num_days
        # The daily figure this week is measured against, and whose it is.
        #
        # It was a flat 2000, reported by the ledger as `source: "calorie
        # target"` with the words "over your target" — the meat limit's bug
        # exactly, on the row beside it. `reference_for` returns the number
        # AND its provenance, so a member who stated a sex gets the reference
        # for an adult of that sex rather than the label figure, and nobody is
        # told a number is theirs when it is not.
        # `None` — nobody for whom a single figure would mean anything, which
        # today is anyone who is not an adult. The week is still STEERED by the
        # label figure, as it always was, because the planner needs some
        # per-meal sense of size; what changes is that `calories_basis` stays
        # empty and the ledger then says nothing rather than measuring a child
        # against an adult's day.
        reference = reference_intake.reference_for(self.user_profile)
        daily = reference.kcal if reference else reference_intake.EU_REFERENCE_INTAKE
        targets = {
            "calories": daily * days,
            # Whether the number above is the MEMBER's or a population figure.
            "calories_explicit": bool(reference and reference.chosen),
            "calories_basis": reference.basis if reference else "",
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            # The meat limit is a WEEKLY figure, so it scales with the horizon
            # rather than being handed whole to a shorter plan: three meat
            # meals across three days is every dinner, which is not the limit
            # anyone meant. Never below 1 for a diet that allows any at all —
            # rounding a short plan down to zero would silently turn it
            # vegetarian.
            "meat_limit": (
                0 if diet & {"vegetarian", "vegan"}
                else max(1, round(DEFAULT_WEEKLY_MEAT_LIMIT * days / 7))
            ),
            # Whether the number above is the MEMBER's or ours.
            #
            # The plan reported it either way as `source: "dietary preference"`
            # and apologised — "Your weekly meat limit (3) couldn't be fully
            # honored" — to a member who never set one. A default is a fine
            # thing to have and a lie to attribute.
            "meat_limit_explicit": False,
        }

        for pref in preferences:
            pref_lower = pref.lower()
            if "calories target" in pref_lower:
                # `reference_for` already read this — it parses the same
                # string — but a profile may carry a form this loop handles
                # and that parser does not, so the explicit reading still
                # wins where it finds one.
                try:
                    val = float(pref_lower.split()[0])
                    targets["calories"] = val * days
                    targets["calories_explicit"] = True
                    targets["calories_basis"] = (
                        "the daily calorie target on your profile"
                    )
                except ValueError:
                    pass
            elif "high protein" in pref_lower:
                # e.g., "high protein (150g)"
                match = re.search(r"\((\d+)g\)", pref_lower)
                if match:
                    targets["protein"] = float(match.group(1)) * days
            elif "g carbs" in pref_lower:
                try:
                    targets["carbs"] = float(pref_lower.split("g")[0].strip()) * days
                except ValueError:
                    pass
            elif "g fat" in pref_lower:
                try:
                    targets["fat"] = float(pref_lower.split("g")[0].strip()) * days
                except ValueError:
                    pass
            elif "meat" in pref_lower:
                # "meat limit 2", "max 2 meat meals", "2 meat meals a week"
                match = re.search(r"(\d+)\s*meat|meat[^\d]{0,12}(\d+)", pref_lower)
                if match:
                    targets["meat_limit"] = int(match.group(1) or match.group(2))
                    targets["meat_limit_explicit"] = True

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
