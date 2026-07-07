from typing import Dict, Any, List
from models.session import MealCourse

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
        
        # Extract targets from user_profile preferences if available
        self.targets = self._extract_targets(user_profile.get("preferences", []))
        
    def _extract_targets(self, preferences: List[str]) -> Dict[str, float]:
        """Extract numeric targets from preference strings."""
        targets = {
            "calories": 2000.0 * 7,  # Default weekly
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "meat_limit": 3 # Default from MealPlanEnv
        }
        
        for pref in preferences:
            pref_lower = pref.lower()
            if "calories target" in pref_lower:
                try:
                    val = float(pref_lower.split()[0])
                    targets["calories"] = val * 7
                except: pass
            elif "high protein" in pref_lower:
                # e.g., "high protein (150g)"
                import re
                match = re.search(r"\((\d+)g\)", pref_lower)
                if match:
                    targets["protein"] = float(match.group(1)) * 7
            elif "g carbs" in pref_lower:
                try:
                    targets["carbs"] = float(pref_lower.split("g")[0].strip()) * 7
                except: pass
            elif "g fat" in pref_lower:
                try:
                    targets["fat"] = float(pref_lower.split("g")[0].strip()) * 7
                except: pass
        
        return targets

    def update_tracker(self, meal: MealCourse, nutrition_info: Dict[str, float] = None):
        """
        Update cumulative totals with a new meal.
        
        Args:
            meal: The MealCourse object added to the plan.
            nutrition_info: Optional dict with 'calories', 'protein', 'carbs', 'fat'.
                           In a real scenario, this might be extracted from recipe directions/ingredients
                           or passed from a nutrition service.
        """
        if nutrition_info:
            self.weekly_calories += nutrition_info.get("calories", 0.0)
            self.weekly_protein += nutrition_info.get("protein", 0.0)
            self.weekly_carbs += nutrition_info.get("carbs", 0.0)
            self.weekly_fat += nutrition_info.get("fat", 0.0)
            
        # Check for meat (simplified detection similar to weekly_meal_planner.py)
        MEAT_KEYWORDS = {
            'beef', 'pork', 'chicken', 'lamb', 'turkey', 'fish', 'seafood', 'steak', 'bacon', 'ham', 'sausage',
            'mutton', 'veal', 'venison', 'duck', 'goose', 'quail', 'rabbit', 'shrimp', 'crab', 'lobster', 'mussel',
            'oyster', 'scallop', 'clam', 'anchovy', 'sardine', 'tuna', 'salmon', 'trout', 'cod', 'haddock', 'halibut',
            'meat', 'pepperoni', 'salami', 'gelatin'
        }
        
        meal_text = (meal.title + " " + meal.ingredients).lower()
        if any(keyword in meal_text for keyword in MEAT_KEYWORDS):
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
