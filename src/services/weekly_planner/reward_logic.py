import json
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from src.backend.groq import GROQ_CHAT
from src.prompts import GRADER_SYSTEM_INSTRUCTIONS, GRADER_USER_INSTRUCTIONS
from src.schemas import ScoringSchema

if TYPE_CHECKING:
    from .state_tracking import WeeklyNutritionalTracker

logger = logging.getLogger(__name__)

class RewardCalculator:
    """
    Calculates rewards for the weekly meal planner MDP.
    Integrates LLM-based feedback and hard constraint penalties.
    """

    def __init__(self, model: str = None, temperature: float = 0.0):
        """
        Initialize the reward calculator with an LLM client.
        
        Args:
            model: Optional model name override.
            temperature: LLM temperature (default 0.0 for consistent scoring).
        """
        self.llm = GROQ_CHAT.get_client(
            model=model,
            temperature=temperature,
            format=ScoringSchema.model_json_schema()
        )

    def get_llm_feedback(self, recipe: Dict[str, Any], preferences: List[str], user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Prompt the judge (LLM) to evaluate how well a recipe matches user preferences.
        
        Args:
            recipe: Dict containing 'recipe_title', 'recipe_ingredients', etc.
            preferences: List of user preference strings.
            user_query: Optional user-specific query to guide evaluation.
            
        Returns:
            Dict containing 'reasoning' and 'score' (1-5).
        """
        # Format recipe for the prompt
        recipe_text = f"Title: {recipe.get('recipe_title', 'N/A')}\n"
        recipe_text += f"Ingredients: {recipe.get('recipe_ingredients', 'N/A')}\n"
        recipe_text += f"Directions: {recipe.get('recipe_directions', 'N/A')}"

        # Prepare messages using existing prompt templates
        query_to_use = user_query if user_query else "Evaluate this recipe based on my preferences."
        
        user_content = GRADER_USER_INSTRUCTIONS.format(
            query=query_to_use,
            preferences=", ".join(preferences),
            feedback_history="No prior feedback provided.",
            daily_plan=recipe_text
        )

        messages = [
            SystemMessage(content=GRADER_SYSTEM_INSTRUCTIONS),
            HumanMessage(content=user_content)
        ]

        try:
            response = self.llm.invoke(messages)
            # ChatGroq with format='json' (or schema) might return a string or dict
            if isinstance(response.content, str):
                return json.loads(response.content)
            return response.content
        except Exception as e:
            return {
                "reasoning": f"Failed to get LLM feedback: {str(e)}",
                "score": 3  # Neutral score on failure
            }

    def calculate_constraint_penalty(self, tracker: "WeeklyNutritionalTracker", preferences: List[str]) -> float:
        """
        Calculate penalties for violating nutritional or dietary constraints.
        
        Args:
            tracker: The WeeklyNutritionalTracker containing current cumulative state.
            preferences: List of user preference strings (for additional context).
            
        Returns:
            A non-negative penalty value (to be subtracted from reward).
        """
        status = tracker.get_status()
        remaining = status["remaining"]
        
        penalty = 0.0
        
        # 1. Meat limit penalty
        if remaining["meat_limit_left"] < 0:
            # Significant penalty for exceeding the meat limit
            penalty += abs(remaining["meat_limit_left"]) * 15.0
            
        # 2. Calorie constraint penalty
        if remaining["calories"] < 0:
            # Penalty increases with the amount of excess calories
            penalty += abs(remaining["calories"]) * 0.1
            
        return penalty

    def calculate_step_reward(self, action: Dict[str, Any], tracker: "WeeklyNutritionalTracker", preferences: List[str], user_query: Optional[str] = None) -> float:
        """
        Combine LLM feedback and constraint penalties into a single scalar reward.
        
        Args:
            action: The recipe action taken.
            tracker: The updated nutritional tracker.
            preferences: List of user preferences.
            user_query: Optional user-specific query to guide LLM evaluation.
            
        Returns:
            The total reward for the current step.
        """
        # Get LLM feedback score (1-5)
        llm_feedback = self.get_llm_feedback(action, preferences, user_query=user_query)
        llm_score = float(llm_feedback.get("score", 3))
        llm_reasoning = llm_feedback.get("reasoning", "No reasoning provided.")
        
        # Center LLM score (e.g., 3 becomes 0, 5 becomes 10, 1 becomes -10)
        base_reward = (llm_score - 3.0) * 5.0
        
        # Subtract constraint penalties
        penalty = self.calculate_constraint_penalty(tracker, preferences)
        
        total_reward = base_reward - penalty
        
        # Log reward calculation details
        logger.info(f"--- Reward Calculation for Recipe: {action.get('recipe_title', 'Unknown')} ---")
        logger.info(f"LLM Score: {llm_score}/5")
        logger.info(f"LLM Reasoning: {llm_reasoning}")
        logger.info(f"Base Reward (Centered): {base_reward:.2f}")
        logger.info(f"Constraint Penalty: {penalty:.2f}")
        logger.info(f"Total Step Reward: {total_reward:.2f}")
        
        return total_reward
