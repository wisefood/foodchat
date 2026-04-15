from typing import List, Dict, Any, Union
from src.models.session import MealCourse
from KG_neo4j.query_kg import get_filtered_recipe_ids

class RecipeActionSpace:
    """
    Action space for recipe selection in the MDP.
    Connects the MDP to the existing Neo4j/API recipe querying service.
    """

    def __init__(self, user_profile: Dict[str, Any]):
        """
        Initialize the action space with user preferences.
        
        Args:
            user_profile: Dict containing 'diet', 'allergies', etc.
        """
        self.user_profile = user_profile
        self.allergens = user_profile.get("allergies", [])
        # 'diet' can be a list or string, get_filtered_recipe_ids handles it
        self.diet = user_profile.get("diet", [])
        self._recipes_cache = None

    def get_candidate_actions(self, meal_type: Union[str, int], current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch recipes from the query service and format them as MDP actions.
        
        Args:
            meal_type: String ('breakfast', 'lunch', 'dinner') or integer (0, 1, 2).
            current_state: The current state of the MDP environment.
            
        Returns:
            A list of lightweight dictionaries representing available recipe actions.
        """
        # Fetch recipes if not already cached (lazy loading)
        # We cache to avoid redundant expensive DB/API calls during MDP solving
        if self._recipes_cache is None:
            # We use the existing get_filtered_recipe_ids dispatcher from query_kg.py
            self._recipes_cache = get_filtered_recipe_ids(self.allergens, self.diet, limit=15)

        # Handle integer meal_type (common in MDP states like MealPlanEnv)
        if isinstance(meal_type, int):
            course_map = {0: 'breakfast', 1: 'lunch', 2: 'dinner'}
            meal_type = course_map.get(meal_type, 'lunch')
        
        meal_type_key = str(meal_type).lower()
        candidates = self._recipes_cache.get(meal_type_key, [])
        
        # Format results into a lightweight dictionary that the MDP can use
        # We leverage the existing MealCourse dataclass from session models
        actions = []
        for r in candidates:
            # Each r is expected to be [recipe_id, title, ingredients, directions]
            # as returned by get_filtered_recipe_ids_neo4j or get_filtered_recipe_ids_api
            meal_course = MealCourse.from_list(r)
            
            # Convert to dictionary to provide a lightweight representation
            # using keys compatible with MealPlanEnv and WeeklyPlanService
            actions.append({
                "recipe_id": meal_course.recipe_id,
                "recipe_title": meal_course.title,
                "recipe_ingredients": meal_course.ingredients,
                "recipe_directions": meal_course.directions
            })
            
        return actions
