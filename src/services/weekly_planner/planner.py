import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .environment import WeeklyMealPlanEnv

logger = logging.getLogger(__name__)


def build_preference_scorer(user_profile: Dict[str, Any]) -> Callable:
    """Heuristic candidate scorer (M4): preference-aware selection at zero
    LLM cost. Favorites dominate, liked ingredients boost, similarity to
    meals already in the plan penalizes (variety).

    scorer(candidate, chosen_titles) -> float
    """
    favorites = {str(f) for f in (user_profile.get("favorite_recipe_ids") or [])}
    likes = [str(l).lower() for l in (user_profile.get("food_likes") or [])]

    def scorer(candidate: Dict[str, Any], chosen_titles: List[str]) -> float:
        score = 0.0
        if str(candidate.get("recipe_id", "")) in favorites:
            score += 5.0
        ingredients = str(candidate.get("recipe_ingredients", "")).lower()
        score += sum(1.0 for like in likes if like and like in ingredients)

        # Variety: penalize title-token overlap with meals already planned.
        title_tokens = {
            t for t in str(candidate.get("recipe_title", "")).lower().split() if len(t) > 3
        }
        for chosen in chosen_titles:
            overlap = title_tokens & {t for t in chosen.lower().split() if len(t) > 3}
            if overlap:
                score -= 2.0 * len(overlap)
        return score

    return scorer


class WeeklyPlanner:
    """
    Orchestration pipeline to generate a full 7-day meal plan.

    Supports pinned anchor slots (seeded planning, M2): a pinned (day,
    meal_idx) slot always receives its anchor recipe; only free slots are
    filled from the action space.
    """

    def __init__(self, env: WeeklyMealPlanEnv):
        self.env = env

    def generate_full_plan(
        self,
        user_query: str = None,
        pinned: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
        scorer: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the 21-step (7 days × 3 meals) planning loop.

        Args:
            user_query: Optional user query to guide reward evaluation.
            pinned: {(day, meal_idx): recipe-action dict} anchor slots.
            scorer: optional preference scorer (build_preference_scorer) —
                    argmax selection with random tiebreak among equals.
                    Without it, selection is uniformly random.

        Returns:
            The 21 generated entries with day, meal type, recipe, and reward.
        """
        pinned = pinned or {}
        state = self.env.reset(user_query=user_query)
        done = False
        chosen_titles: List[str] = []

        while not done:
            slot_key = (state["day"], state["meal_idx"])
            if slot_key in pinned:
                # User-requested anchor — bypass candidate selection entirely.
                chosen_recipe = {**pinned[slot_key], "pinned": True}
                logger.info(
                    "Day %d %s pinned to %r",
                    state["day"], state["meal_type"], chosen_recipe.get("recipe_title"),
                )
            else:
                candidates = self.env.action_space.get_candidate_actions(
                    state["meal_type"], state
                )
                if not candidates:
                    # No recipes for this course — fail loudly rather than
                    # produce a plan with holes. (Constraint relaxation is a
                    # planned improvement.)
                    raise ValueError(
                        f"No candidate recipes found for {state['meal_type']} on day {state['day']}"
                    )
                if scorer is not None:
                    # Preference-aware argmax; random tiebreak keeps variety.
                    best_score = max(scorer(c, chosen_titles) for c in candidates)
                    top = [c for c in candidates if scorer(c, chosen_titles) == best_score]
                    chosen_recipe = random.choice(top)
                else:
                    chosen_recipe = random.choice(candidates)

            chosen_titles.append(str(chosen_recipe.get("recipe_title", "")))
            # Advance the environment (updates tracker, computes reward).
            state, reward, done, info = self.env.step(chosen_recipe)

        return self.env.plan
