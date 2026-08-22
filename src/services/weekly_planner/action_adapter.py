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

import logging
from typing import Any, Dict, List, Optional, Union

from services import intent_facets
from services.candidates_client import CANDIDATES, normalize_diet_tags

# The failure path below logs; without this the except clause itself raised
# NameError, turning a degradable fetch failure into a crashed plan.
logger = logging.getLogger(__name__)

# Candidates fetched per slot per day. One fetch serves all three slots of a
# day, so this stays small to keep the RecipeWrangler payloads light.
DAILY_POOL_LIMIT = 10


class RecipeActionSpace:
    """Per-day candidate pools for the weekly planner."""

    def __init__(
        self,
        user_profile: Dict[str, Any],
        additional_diet: List[str] = None,
        pantry: tuple = (),
        spec: Optional[object] = None,
    ):
        self.user_profile = user_profile
        self.allergens = user_profile.get("allergies", [])
        # The plan's shape, and the reason this class can now produce a meal
        # rather than a dish.
        #
        # A weekly dinner served as a main and a salad had no producer: this
        # fetched ONE pool per meal slot and handed back single recipes, so the
        # concept of a plate did not exist here at all. The renderer has been
        # plate-aware for a while and had nothing to render.
        #
        # None, or a spec whose meals are all single-plate, keeps the old
        # single-dish path exactly — including the request it sends — so a
        # normal week is unaffected by any of this.
        self.spec = spec
        self.multi_plate = bool(
            spec is not None
            and any(
                len(spec.roles_for(slot)) > 1
                for slot in (getattr(spec, "meals", ()) or ())
            )
        )

        profile_diet = user_profile.get("diet", [])
        if isinstance(profile_diet, str):
            profile_diet = [profile_diet]
        # Query-level diet tags (extracted from the user message) tighten the
        # profile diet for this plan only.
        self.diet = list(set(profile_diet + (additional_diet or [])))
        # On-hand ingredients to use up (food waste). Pantry-matching recipes
        # are folded into every day's pool so the scorer's boost has something
        # to boost — a preference the pool never contains cannot be honoured.
        self.pantry = tuple(pantry or ())

        # Per-day candidate pool cache, keyed by day index.
        self._day_cache: Dict[int, Dict[str, list]] = {}
        # Per-day ROLE-scoped pools, keyed by day — `{(slot, role): [...]}`.
        # Only populated on the multi-plate path.
        self._role_cache: Dict[int, Dict[tuple, list]] = {}
        # Recipes already committed to the plan — excluded from every fetch.
        self._selected_ids: List[str] = []
        # recipe_id -> RecipeEnrichment for every fetched pool (M6).
        self._enrichment: Dict[str, Any] = {}

    def get_candidate_actions(
        self, meal_type: Union[str, int], current_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return candidate action dicts for the given meal slot and state.

        On a multi-plate spec an "action" is a whole MEAL — a list of plates —
        rather than a dish. The MDP still steps once per meal, which is what
        keeps this change small: the loop, the tracker, the reward and the
        preference scorer all see one action per slot exactly as before, and the
        action simply describes more of the table.
        """
        current_day = current_state.get("day", 1)

        if self.multi_plate:
            return self._composed_actions(str(meal_type).lower(), current_day)

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
            if self.pantry and pool:
                # Same Tier-A fan-out as the daily pipeline: single-item hard
                # includes, merged coverage-first, pool size unchanged. The
                # allergen/diet constraints ride inside the fetch.
                from services import pantry_service, plan_parameters

                pantry_pools = pantry_service.fetch_pantry_candidates(
                    self.user_profile, self.pantry,
                    exclude_recipe_ids=list(self._selected_ids),
                    per_item=2,
                    # Profile + query-level tags, tightened. Normalised like
                    # every other call site (idempotent inside the fetch).
                    diet=normalize_diet_tags(self.diet),
                    # Parity with the base pool above, which applies both.
                    # The merge ranks coverage-first, so a constraint missing
                    # here surfaces at the top of the day's pool rather than
                    # merely appearing in it.
                    cuisines=cuisines,
                    max_minutes=plan_parameters.max_duration_minutes(
                        self.user_profile.get("plan_parameters") or {}
                    ),
                )
                pool = pantry_service.merge_pantry_pool(
                    pool, pantry_pools, self.pantry, DAILY_POOL_LIMIT,
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

    # ------------------------------------------------------------------ #
    # Multi-plate: a pool per plate, composed into whole meals             #
    # ------------------------------------------------------------------ #

    def _composed_actions(self, slot: str, day: int) -> List[Dict[str, Any]]:
        """Whole meals for this slot, best composition first.

        Deliberately LLM-free, and that is a decision rather than an omission.
        This loop used to make one Groq call per committed slot — 21 per week —
        to grade a recipe that was already locked in: pure cost with no effect
        on the output, and it was removed for that reason. Composition here is
        the arithmetic half only: no duplicate dish, no repeated ingredient
        across plates, and each plate near its share of the meal. The judgement
        half — whether a side SUITS a main — runs on the structured path, which
        is where a fresh multi-plate request is routed, and it runs once for the
        whole plan rather than once per meal.

        What this path exists for is the case the structured path cannot serve:
        a multi-plate week that already exists and is being refined. Before
        this, a refinement flattened it back to single dishes and the member
        watched the shape they asked for disappear.
        """
        from services import meal_composer

        if day not in self._role_cache:
            pools, notes = meal_composer.role_pools(
                self.user_profile, self.spec,
                exclude_recipe_ids=list(self._selected_ids),
                boost_ids=(
                    list(self.user_profile.get("favorite_recipe_ids") or [])
                    if self.user_profile.get("use_favorites") is not False else []
                ),
                # One day at a time, because the exclusion list is what stops a
                # week repeating a dish and it only exists once earlier days are
                # committed.
                days=1,
            )
            if notes:
                logger.info("day %s pool relaxations: %s", day, "; ".join(notes))
            # `role_pools` asks for a single day, so the response has one.
            self._role_cache[day] = next(iter(pools.values()), {})
            ids = [
                c.recipe_id
                for pool in self._role_cache[day].values() for c in pool
            ]
            self._enrichment.update(CANDIDATES.fetch_details(ids))

        roles = tuple(self.spec.roles_for(slot))
        compositions = meal_composer.compose(
            slot, roles, self._role_cache[day],
            kcal_split=self.spec.kcal_split(slot),
            meal_kcal_target=self._meal_kcal_target(),
            exclude_ids=set(self._selected_ids),
            enrichment=self._enrichment,
            # Every viable composition, not the top three: the MDP's own scorer
            # (favourites, likes, variety, pantry, calorie budget) has to rank
            # them, and handing it three would silently overrule preferences
            # this module knows nothing about.
            limit=DAILY_POOL_LIMIT,
        )
        return [self._as_action(slot, c) for c in compositions]

    def _meal_kcal_target(self) -> Optional[float]:
        """One meal's share of the day's budget, or None when nobody set one."""
        from models.plan_brief import PlanBrief

        target = PlanBrief.build(self.user_profile).kcal_target
        if not target:
            return None
        meals = len(getattr(self.spec, "meals", ()) or ()) or 1
        return float(target) / meals

    def _as_action(self, slot: str, composition) -> Dict[str, Any]:
        """One composition as the action dict the MDP already understands.

        The top-level `recipe_*` fields are the MAIN plate, so every existing
        consumer — the preference scorer, the pinning check, the stored entry —
        reads what it always read. What is added is the rest of the table:

        * `nutrition` totals the WHOLE meal, because that is what the member
          eats and what the weekly calorie tracker is counting. Scoring a
          main-plus-side meal on the main alone would let every side through
          the budget unmeasured.
        * `meal_ingredients` concatenates every plate, so the meat-limit check
          sees the bacon in the salad. A meat limit that only looks at mains is
          a meat limit with a hole in it.
        """
        plates = composition.plates
        main = next((p for p in plates if p.role == "main"), plates[0])
        totals: Dict[str, float] = {}
        counted = 0
        for plate in plates:
            nutrition = getattr(plate.candidate, "nutrition", None) or {}
            rich = self._enrichment.get(plate.recipe_id)
            if not nutrition and rich is not None:
                nutrition = rich.nutrition_dict() or {}
            if not nutrition:
                continue
            counted += 1
            for key in ("kcal", "calories", "protein_g", "carbs_g", "fat_g"):
                value = nutrition.get(key)
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)

        tags: List[str] = []
        for plate in plates:
            rich = self._enrichment.get(plate.recipe_id)
            for tag in (getattr(rich, "tags", None) or []):
                if tag not in tags:
                    tags.append(tag)

        action: Dict[str, Any] = {
            "recipe_id": main.recipe_id,
            "recipe_title": main.title,
            "recipe_ingredients": str(getattr(main.candidate, "ingredients", "") or ""),
            "recipe_directions": str(getattr(main.candidate, "directions", "") or ""),
            "meal_ingredients": " ; ".join(
                str(getattr(p.candidate, "ingredients", "") or "") for p in plates
            ),
            "tags": tags,
            "composition_score": composition.score,
            "composition_findings": list(composition.findings),
            "plates": [
                {
                    "role": plate.role,
                    "recipe_id": plate.recipe_id,
                    "recipe_title": plate.title,
                    "recipe_ingredients": str(
                        getattr(plate.candidate, "ingredients", "") or ""
                    ),
                    "recipe_directions": str(
                        getattr(plate.candidate, "directions", "") or ""
                    ),
                    "nutrition": (
                        getattr(plate.candidate, "nutrition", None)
                        or (
                            self._enrichment[plate.recipe_id].nutrition_dict()
                            if plate.recipe_id in self._enrichment else None
                        )
                    ),
                    "image_url": getattr(plate.candidate, "image_url", None),
                }
                for plate in plates
            ],
        }
        if totals:
            # Said explicitly, like every other total in this codebase: a meal
            # whose side carries no macros must not report a figure that looks
            # like the whole meal.
            totals["complete"] = counted == len(plates)
            action["nutrition"] = totals
        return action


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
            **intent_facets.facet_kwargs(profile, cuisines),
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
