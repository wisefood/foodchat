import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .environment import WeeklyMealPlanEnv
from .reward_logic import apply_hard_constraints, constraint_score

logger = logging.getLogger(__name__)

TOTAL_SLOTS = 21  # 7 days x 3 meals


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
                # Hard constraints prune the pool BEFORE the pick (M6):
                # e.g. meat candidates leave once the weekly limit is spent.
                # Prunes/relaxations are recorded on env.selection_events at
                # decision time (M7 explainability).
                candidates = apply_hard_constraints(
                    candidates, self.env.tracker,
                    events=self.env.selection_events,
                    slot={"day": state["day"], "meal_type": state["meal_type"]},
                )

                # Preference score + soft constraint score (calorie budget)
                # rank the pool; random tiebreak among equals keeps variety.
                # Without a scorer and without nutrition data every score is
                # 0.0 and selection stays uniformly random, as before.
                slots_remaining = TOTAL_SLOTS - len(self.env.plan)
                scored = [
                    (
                        (scorer(c, chosen_titles) if scorer is not None else 0.0)
                        + constraint_score(c, self.env.tracker, slots_remaining),
                        c,
                    )
                    for c in candidates
                ]
                best_score = max(s for s, _ in scored)
                top = [c for s, c in scored if s == best_score]
                chosen_recipe = random.choice(top)

            chosen_titles.append(str(chosen_recipe.get("recipe_title", "")))
            # Advance the environment (updates tracker, computes reward).
            state, reward, done, info = self.env.step(chosen_recipe)

        return self.env.plan
