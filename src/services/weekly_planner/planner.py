import random
from typing import List, Dict, Any
from src.services.weekly_planner.environment import WeeklyMealPlanEnv

class WeeklyPlanner:
    """
    Orchestration pipeline to generate a full 7-day meal plan.
    Strictly additive and does not impact existing standalone daily meal plan services.
    """

    def __init__(self, env: WeeklyMealPlanEnv):
        """
        Initialize the planner with a configured environment.
        
        Args:
            env: The WeeklyMealPlanEnv to operate on.
        """
        self.env = env

    def generate_full_plan(self, user_query: str = None) -> List[Dict[str, Any]]:
        """
        Runs a 'while not done' loop over the environment to generate a full 7-day plan.
        
        Args:
            user_query: Optional user query to guide the planning process.
            
        Returns:
            A list of 21 generated meal entries containing the day, meal type, recipe, and reward.
        """
        state = self.env.reset(user_query=user_query)
        done = False
        
        while not done:
            # 1. Get candidate actions for the current state from the action space
            # The action space uses the meal_type and state to filter candidates
            candidates = self.env.action_space.get_candidate_actions(
                state["meal_type"], 
                state
            )
            
            if not candidates:
                # If no recipes are found for the specific course, we signal a failure.
                # In a robust system, we might relax constraints here.
                raise ValueError(f"No candidate recipes found for {state['meal_type']} on day {state['day']}")
            
            # 2. Strategy: Select a recipe from candidates.
            # To keep the plan generation efficient (given LLM scoring in env.step),
            # we pick a random candidate. For a more advanced version, we could 
            # implement a greedy lookahead or MCTS.
            chosen_recipe = random.choice(candidates)
            
            # 3. Advance the environment state
            # This method updates the tracker, calculates the reward (with LLM feedback),
            # and moves to the next time step.
            state, reward, done, info = self.env.step(chosen_recipe)
            
        return self.env.plan
