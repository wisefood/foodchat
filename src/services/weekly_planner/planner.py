import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .environment import WeeklyMealPlanEnv
from .reward_logic import apply_hard_constraints, constraint_score

logger = logging.getLogger(__name__)

TOTAL_SLOTS = 21  # 7 days x 3 meals


class PlanGenerationError(RuntimeError):
    """A slot could not be filled and the plan cannot be completed.

    Carries which slot failed so the caller can *say* it. This surfaced to
    members as an HTTP 500 with a stack trace — a member asking for a week of
    meals got an internal error page because day 1's breakfast came back
    empty. An unfillable slot is a conversational answer ("here is what I
    couldn't do, and why"), not a server fault.
    """

    def __init__(self, meal_type: str, day: int):
        self.meal_type = str(meal_type)
        self.day = int(day)
        super().__init__(f"No candidate recipes found for {meal_type} on day {day}")


# Ingredients that sit in every kitchen and never go to waste. Reuse scoring
# ignores them: two meals sharing "olive oil" have not saved anyone a wilting
# vegetable, and counting staples would reward exactly nothing while drowning
# the signal from the ingredients that do spoil.
_PANTRY_STAPLES: frozenset = frozenset({
    "salt", "pepper", "water", "sugar", "flour", "oil", "olive", "butter",
    "vinegar", "garlic", "onion", "onions", "stock", "sauce", "soy", "honey",
    "mustard", "cumin", "paprika", "oregano", "thyme", "cinnamon", "vanilla",
    "baking", "powder", "cornflour", "cornstarch", "rice", "pasta", "milk",
    "eggs", "cheese", "lemon", "juice", "cloves", "seeds", "chilli", "chili",
    "ginger", "parsley", "coriander", "basil", "extract", "yeast", "breadcrumbs",
})

# Measurement noise that survives naive tokenising of an ingredients blob.
_UNITS: frozenset = frozenset({
    "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
    "tbsp", "tsp", "gram", "grams", "kg", "ml", "litre", "liter", "large",
    "small", "medium", "fresh", "dried", "chopped", "sliced", "diced",
    "grated", "finely", "roughly", "optional", "taste", "extra", "plus",
})


def perishable_tokens(ingredients_text: Any) -> set:
    """The words in an ingredients blob that name things that can spoil."""
    words = str(ingredients_text or "").lower().replace(",", " ").replace(";", " ").split()
    return {
        w.strip("().:-")
        for w in words
        if len(w) > 3
        and w.strip("().:-").isalpha()
        and w.strip("().:-") not in _PANTRY_STAPLES
        and w.strip("().:-") not in _UNITS
    }


def build_preference_scorer(user_profile: Dict[str, Any]) -> Callable:
    """Heuristic candidate scorer (M4): preference-aware selection at zero
    LLM cost. Favorites dominate, liked ingredients boost, similarity to
    meals already in the plan penalizes (variety), and — when the member has
    switched it on — sharing perishable ingredients with meals already chosen
    scores positively (food waste).

    scorer(candidate, chosen_titles, chosen_ingredients=frozenset()) -> float

    The waste weights sit deliberately between the variety penalty (−2 per
    shared title token) and the favourites bonus (+5): `reuse` nudges, it does
    not override a favourite or flatten variety; `strict` is allowed to beat
    the variety penalty, because that is what the member asked for — a
    smaller shopping list at some cost to sameness. This scorer is the only
    place the weekly planner can honour the setting at all: it selects
    without an LLM, so a preference that never becomes a number here is a
    preference that does not exist.
    """
    favorites = {str(f) for f in (user_profile.get("favorite_recipe_ids") or [])}
    likes = [str(l).lower() for l in (user_profile.get("food_likes") or [])]

    from services import plan_parameters  # local import; avoids a cycle at module load
    waste = plan_parameters.waste_mode(user_profile.get("plan_parameters") or {})
    waste_weight = {"off": 0.0, "reuse": 0.8, "strict": 1.6}[waste]

    def scorer(
        candidate: Dict[str, Any],
        chosen_titles: List[str],
        chosen_ingredients: frozenset = frozenset(),
    ) -> float:
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

        # Food waste: shared perishables with the plan so far. Capped so one
        # candidate cannot ride a single busy ingredient list to the top.
        if waste_weight and chosen_ingredients:
            shared = perishable_tokens(ingredients) & chosen_ingredients
            score += waste_weight * min(len(shared), 4)
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
        # Scorers written before the food-waste axis take (candidate, titles);
        # ours takes a third perishables argument. Checked once here, not
        # guessed per call with a try/except that would also swallow a
        # scorer's own TypeErrors.
        import inspect
        takes_perishables = (
            scorer is not None
            and len(inspect.signature(scorer).parameters) >= 3
        )
        state = self.env.reset(user_query=user_query)
        done = False
        chosen_titles: List[str] = []
        # Perishables already committed to the plan — the food-waste scorer's
        # working set. Grows as slots fill, so later days lean toward what
        # earlier days already put in the basket.
        chosen_perishables: set = set()

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
                    # produce a plan with holes. Typed so the service can turn
                    # it into a sentence instead of a 500. (Constraint
                    # relaxation is a planned improvement.)
                    raise PlanGenerationError(state["meal_type"], state["day"])
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
                        (
                            scorer(c, chosen_titles, frozenset(chosen_perishables))
                            if takes_perishables
                            else scorer(c, chosen_titles)
                            if scorer is not None
                            else 0.0
                        )
                        + constraint_score(c, self.env.tracker, slots_remaining),
                        c,
                    )
                    for c in candidates
                ]
                best_score = max(s for s, _ in scored)
                top = [c for s, c in scored if s == best_score]
                chosen_recipe = random.choice(top)

            chosen_titles.append(str(chosen_recipe.get("recipe_title", "")))
            chosen_perishables |= perishable_tokens(
                chosen_recipe.get("recipe_ingredients", "")
            )
            # Advance the environment (updates tracker, computes reward).
            state, reward, done, info = self.env.step(chosen_recipe)

        return self.env.plan
