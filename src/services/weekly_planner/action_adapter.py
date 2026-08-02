"""
Action space for the weekly-plan MDP — candidate recipes per meal slot.

Fetches a fresh candidate pool from RecipeWrangler once per plan day
(``services.candidates_client``), excluding every recipe already committed to
the plan so a 7-day plan never repeats a recipe.

M6: each day's pool is enriched with one batch details call (nutrition,
diet tags) at fetch time, so the nutritional tracker and the constraint
filter see real numbers DURING selection — previously nutrition only
existed after the full plan was generated, which made the calorie
constraint structurally inert. Enrichment stays best-effort: a failed
details call leaves the candidates bare and constraints degrade to
neutral, never blocking the plan.
"""

from typing import Any, Dict, List, Union

from services import plan_parameters
from services.candidates_client import CANDIDATES, normalize_diet_tags

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
        # recipe_id -> RecipeEnrichment for every fetched pool (M6).
        self._enrichment: Dict[str, Any] = {}

    def get_candidate_actions(
        self, meal_type: Union[str, int], current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return candidate action dicts for the given meal slot and state."""
        current_day = current_state.get("day", 1)

        if current_day not in self._day_cache:
            # Same preference split as the daily pipeline. The weekly planner
            # selects without an LLM, so a cuisine preference it cannot express
            # as a filter is a preference it cannot honour at all — there is no
            # grader downstream to compensate.
            cuisines, liked_ingredients = CANDIDATES.split_cuisines(
                self.user_profile.get("food_likes") or []
            )
            # Sourced from `/api/v2/tools/plan_meals`, like the daily pipeline.
            #
            # This planner matters most for the switch: it selects without an
            # LLM, so a preference it cannot express as a filter is one it
            # cannot honour at all. The v1 endpoint queried a store holding no
            # cuisine, mood or flavour, and no `planning_tier` — so a week of
            # meals could include recipes explicitly withdrawn from automated
            # planning, and no amount of downstream scoring would notice.
            pool = _fetch_candidate_pool(
                profile=self.user_profile,
                allergens=self.allergens,
                # Normalised: an unknown tag ANDs to zero candidates,
                # and diet is never relaxed.
                diet=normalize_diet_tags(self.diet),
                cuisines=cuisines,
                exclude_recipe_ids=list(self._selected_ids),
                limit_per_slot=DAILY_POOL_LIMIT,
            )
            self._day_cache[current_day] = pool
            # One batch details call enriches the whole day's pool (M6) —
            # nutrition/tags feed the tracker and constraint filter during
            # selection. Best-effort: {} on failure.
            day_ids = [c.recipe_id for slot in pool.values() for c in slot]
            self._enrichment.update(CANDIDATES.fetch_details(day_ids))

        if isinstance(meal_type, int):
            meal_type = {0: "breakfast", 1: "lunch", 2: "dinner"}.get(meal_type, "lunch")

        candidates = self._day_cache[current_day].get(str(meal_type).lower(), [])
        actions = []
        for c in candidates:
            action = {
                "recipe_id": c.recipe_id,
                "recipe_title": c.title,
                "recipe_ingredients": c.ingredients,
                "recipe_directions": c.directions,
            }
            rich = self._enrichment.get(c.recipe_id)
            if rich:
                nutrition = rich.nutrition_dict()
                if nutrition:
                    action["nutrition"] = nutrition
                action["tags"] = rich.tags or []
                action["dish_types"] = rich.dish_types or []
            actions.append(action)
        return actions

    def mark_selected(self, recipe_id: str) -> None:
        """Called by the environment after a recipe is committed to the plan."""
        if recipe_id and recipe_id not in self._selected_ids:
            self._selected_ids.append(recipe_id)


def _fetch_candidate_pool(
    *,
    profile: dict,
    allergens: list,
    diet: list,
    cuisines: list,
    exclude_recipe_ids: list,
    limit_per_slot: int,
) -> dict:
    """A per-slot candidate pool from the planning endpoint.

    Shares the daily pipeline's reasoning: `plan_meals` asked for N recipes per
    slot is a candidate source whose pool already respects the member's
    preferences and already excludes anything withdrawn from planning.

    Liked ingredients are deliberately not forwarded — `plan_meals` treats
    `include_ingredients` as a requirement, so a member who likes chickpeas
    would demand chickpeas in every breakfast and empty the slot.

    Returns `{}` on failure; the MDP treats an empty pool as a day it cannot
    fill, which is what it already did when the old endpoint failed.
    """
    from services import plan_parameters
    from services.plan_client import PLANNER

    try:
        envelope = PLANNER.plan_meals(
            days=1,
            count_per_slot=limit_per_slot,
            allergens=allergens,
            diet=normalize_diet_tags(diet),
            cuisines=cuisines,
            exclude_ingredients=profile.get("food_dislikes") or [],
            exclude_recipe_ids=exclude_recipe_ids,
            favorite_recipe_ids=profile.get("favorite_recipe_ids") or [],
            max_minutes=plan_parameters.max_duration_minutes(
                profile.get("plan_parameters") or {}
            ),
            min_nutri_score=profile.get("min_nutri_score"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("weekly candidate pool fetch failed: %s", exc)
        return {}

    return PLANNER.to_candidates(envelope, allergens=allergens)
