"""
Action space for the weekly-plan MDP — candidate recipes per meal slot.

Fetches a fresh candidate pool from RecipeWrangler once per plan day
(``services.candidates_client``), excluding recipes already committed to the
plan.

M9 retires the absolute no-repeat contract for ONE slot. A 7-day plan used to
exclude every committed id from every later fetch, so repeats were impossible
at the source rather than merely disfavoured — and nobody eats seven different
breakfasts. Breakfast now has a **slot-scoped cooldown** (see the repeat policy
below) while lunch and dinner keep the old rule exactly. A repeat is always
labelled with the day it repeats and *why* it was allowed, so a week that
repeated because the pool was thin can never be presented as one the member
asked for.

M9 also adds the sourcing half of cross-day ingredient reuse: `WeeklyPlanner`
offers this action space the ingredients the week has already bought
(``offer_derived_pantry``), and when the member's food-waste slider is at
``strict`` those are searched for like a stated pantry. Scoring alone can only
reorder the pool it is handed; this is what puts the second dill recipe in it.

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

from services.candidates_client import CANDIDATES, normalize_diet_tags

# The failure path below logs; without this the except clause itself raised
# NameError, turning a degradable fetch failure into a crashed plan.
logger = logging.getLogger(__name__)

# Candidates fetched per slot per day. One fetch serves all three slots of a
# day, so this stays small to keep the RecipeWrangler payloads light.
DAILY_POOL_LIMIT = 10


# --- Repeat policy (M9) -----------------------------------------------------
#
# Real households eat the same breakfast most mornings and treat dinner as the
# place variety belongs. The planner used to enforce the opposite: 21 recipes,
# each seen exactly once, which reads as 21 shopping lists rather than a week.
#
# Every rule below is deliberately narrow, because the failure mode is not "too
# few repeats" — it is a thin candidate pool quietly producing a repetitive week
# that the UI then describes as a feature. So: one slot, a real gap, a hard cap,
# and a recorded reason for every repeat that happens.

# Slots where a recipe may come back at all. Lunch and dinner keep the original
# never-repeat rule; component 3 of the natural-planning plan (day N's dinner
# becoming day N+1's lunch) is a different mechanism and is not this.
REPEATABLE_SLOTS: frozenset = frozenset({"breakfast"})

# Days between two servings of the same recipe. The same gap the ingredient
# spacing in ``planner`` uses, for the same reason: back-to-back is repetition,
# every other day is a routine.
REPEAT_MIN_GAP_DAYS = 2

# Servings of one recipe in a week. Two is a routine; more is the planner
# running out of candidates and calling it a preference.
MAX_APPEARANCES = 2

# Why a repeat was allowed — carried on the candidate as ``repeat_source`` and
# from there onto the stored plan entry, so the ledger and the chips can tell
# the two apart. They are never merged: crediting the member for the planner's
# thin pool is the single thing this labelling exists to prevent.
REPEAT_MEMBER_REQUEST = "member_request"  # the member starred this recipe
REPEAT_PLAN = "plan"                      # the planner's own doing

# Ingredients from earlier days forwarded to per-item candidate search. Each
# one is an extra `plan_meals` round-trip per day, so this is the whole latency
# budget of the sourcing half — kept at three, and only spent at ``strict``.
DERIVED_PANTRY_ITEMS = 3


class RecipeActionSpace:
    """Per-day candidate pools for the weekly planner."""

    def __init__(
        self,
        user_profile: Dict[str, Any],
        additional_diet: List[str] = None,
        pantry: tuple = (),
    ):
        self.user_profile = user_profile
        self.allergens = user_profile.get("allergies", [])

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
        # Recipes that may never appear (again): member-pinned anchors and
        # downvoted dishes, registered by the service through `mark_selected`.
        # Distinct from `_commitments` below — this list has no way back.
        self._selected_ids: List[str] = []
        # recipe_id -> [(day, meal_type)] the plan actually served it at (M9).
        # A commitment is not automatically an exclusion any more; whether it
        # is depends on the slot, the gap and the cap. See `_repeatable_from`.
        self._commitments: Dict[str, List[tuple]] = {}
        self._favorites = {
            str(f) for f in (user_profile.get("favorite_recipe_ids") or [])
        }
        # recipe_id -> RecipeEnrichment for every fetched pool (M6).
        self._enrichment: Dict[str, Any] = {}
        # Ingredients the week has already bought, offered by the planner
        # before each new day's fetch (M9 sourcing half).
        self._derived_pantry: tuple = ()
        # Whether the "we had something to reuse and did not search for it"
        # note has been recorded. Once per plan, not once per day: six near
        # identical events would bloat every stored plan on the default
        # setting to say one thing.
        self._sourcing_skip_noted = False
        # (day, meal_type) slots whose repeat offer has already been recorded.
        self._repeat_offers_noted: set = set()
        # Selection events. The environment replaces this with its own list at
        # construction, so sourcing decisions taken here land in the same
        # ledger as the planner's prunes and reach `metrics.selection_events`
        # — one place to read the whole selection story. Owned here so the
        # action space is still usable, and still records, without an env.
        self.selection_events: List[Dict[str, Any]] = []
        from services import plan_parameters  # local import; avoids a cycle

        self.waste_mode = plan_parameters.waste_mode(
            user_profile.get("plan_parameters") or {}
        )

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
                exclude_recipe_ids=self._fetch_exclusions(current_day),
                limit_per_slot=DAILY_POOL_LIMIT,
            )
            # Cross-day reuse, sourcing half (M9). Merged BEFORE the member's
            # own pantry below, deliberately: `merge_pantry_pool` sorts
            # coverage-first, so whichever merge runs last decides the top of
            # the pool. What the member told us they have must outrank what
            # the plan inferred, every time.
            derived = self._derived_items()
            if derived and pool and self.waste_mode == "strict":
                from services import pantry_service, plan_parameters

                derived_pools = pantry_service.fetch_pantry_candidates(
                    self.user_profile, derived,
                    exclude_recipe_ids=self._fetch_exclusions(current_day),
                    per_item=2,
                    diet=normalize_diet_tags(self.diet),
                    # Threaded for the same reason as the pantry call below:
                    # this pool is merged and sorted coverage-first, so a
                    # constraint omitted here does not merely appear in the
                    # day's pool, it appears at the top of it.
                    cuisines=cuisines,
                    max_minutes=plan_parameters.max_duration_minutes(
                        self.user_profile.get("plan_parameters") or {}
                    ),
                )
                pool = pantry_service.merge_pantry_pool(
                    pool, derived_pools, derived, DAILY_POOL_LIMIT,
                )
                self.selection_events.append({
                    "type": "derived_pantry_sourced",
                    "day": current_day,
                    "items": list(derived),
                })
            elif (
                derived
                and self.waste_mode != "strict"
                and not self._sourcing_skip_noted
            ):
                # The mechanism had something to offer and the SETTING said
                # no. Recorded once, because "why did the week not search for
                # anything it already buys" is a question someone should be
                # able to answer without reading this file.
                #
                # The setting is re-checked rather than inferred from falling
                # through: a `strict` member whose day came back with an empty
                # pool would otherwise be recorded as having chosen to skip a
                # search they had in fact asked for.
                self._sourcing_skip_noted = True
                self.selection_events.append({
                    "type": "derived_pantry_skipped",
                    "day": current_day,
                    "waste_mode": self.waste_mode,
                    "items": list(derived),
                })
            if self.pantry and pool:
                # Same Tier-A fan-out as the daily pipeline: single-item hard
                # includes, merged coverage-first, pool size unchanged. The
                # allergen/diet constraints ride inside the fetch.
                from services import pantry_service, plan_parameters

                pantry_pools = pantry_service.fetch_pantry_candidates(
                    self.user_profile, self.pantry,
                    exclude_recipe_ids=self._fetch_exclusions(current_day),
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

        meal_type = str(meal_type).lower()
        candidates = self._day_cache[current_day].get(meal_type, [])
        actions = []
        # Recipes this slot COULD have repeated, recorded whether or not one is
        # chosen. Without it, "the week repeated nothing" is indistinguishable
        # from "the source never offered anything to repeat" — and those have
        # opposite fixes, one in this file and one at RecipeWrangler.
        offered: List[str] = []
        for c in candidates:
            # The day's pool was fetched with the loosest exclusion any of its
            # three slots needs (see `_fetch_exclusions`), so the per-slot rule
            # is applied here. Costs no extra HTTP, and it is the only place
            # that knows which slot is being filled.
            repeat_of_day = None
            if c.recipe_id in self._commitments or c.recipe_id in self._selected_ids:
                repeat_of_day = self._repeatable_from(c.recipe_id, current_day, meal_type)
                if repeat_of_day is None:
                    continue
            action = {
                "recipe_id": c.recipe_id,
                "recipe_title": c.title,
                "recipe_ingredients": c.ingredients,
                "recipe_directions": c.directions,
            }
            if repeat_of_day is not None:
                # Rides on the candidate, so it survives selection onto the
                # stored plan entry: the scorer reads it (a sanctioned repeat
                # is not charged for resembling itself), the environment logs
                # it, and the explainability layer labels it. One flag, set
                # once, at the only point that can actually justify it.
                action["repeat_of_day"] = repeat_of_day
                action["repeat_source"] = (
                    REPEAT_MEMBER_REQUEST if c.recipe_id in self._favorites
                    else REPEAT_PLAN
                )
            if repeat_of_day is not None:
                offered.append(c.recipe_id)
            rich = self._enrichment.get(c.recipe_id)
            if rich:
                nutrition = rich.nutrition_dict()
                if nutrition:
                    action["nutrition"] = nutrition
                action["tags"] = rich.tags or []
                action["dish_types"] = rich.dish_types or []
            actions.append(action)

        if offered and (current_day, meal_type) not in self._repeat_offers_noted:
            self._repeat_offers_noted.add((current_day, meal_type))
            self.selection_events.append({
                "type": "repeat_offered",
                "day": current_day,
                "meal_type": meal_type,
                "count": len(offered),
                "recipe_ids": offered,
            })
        return actions

    def mark_selected(self, recipe_id: str) -> None:
        """Bar a recipe from the plan entirely — no slot, no day, no cooldown.

        The service's two callers mean exactly that: a member-pinned anchor
        must not turn up a second time elsewhere in the week, and a downvoted
        dish must not turn up at all. Ordinary commitments go through
        `mark_committed`, which leaves the repeat policy a say.
        """
        if recipe_id and recipe_id not in self._selected_ids:
            self._selected_ids.append(recipe_id)

    def mark_committed(self, recipe_id: str, day: int, meal_type: str) -> None:
        """Record that the plan served this recipe, on this day, in this slot.

        Called by the environment as each slot commits. The day and slot are
        what the repeat policy needs — without them a commitment can only mean
        "never again", which is the rule M9 exists to loosen.
        """
        if not recipe_id:
            return
        self._commitments.setdefault(str(recipe_id), []).append(
            (int(day), str(meal_type).lower())
        )

    def offer_derived_pantry(self, items: List[str], day: int) -> None:
        """Ingredients the week has already bought, offered as a search source.

        Called by `WeeklyPlanner` before each new day's pool is fetched, with
        the ingredients its basket says are still worth another meal. An offer
        is not a decision: whether these are actually searched for depends on
        the member's food-waste setting, and that call is made at fetch time
        in `get_candidate_actions` so it can be recorded against the day.
        """
        self._derived_pantry = tuple(items or ())

    def _derived_items(self) -> List[str]:
        """The offered ingredients worth spending a round-trip on.

        Anything the member already named is dropped: it has its own fan-out
        a few lines below, and searching for it twice would buy nothing but
        latency. Capped at `DERIVED_PANTRY_ITEMS` — this is the whole extra
        cost of the sourcing half, and it is paid once per day.
        """
        if not self._derived_pantry:
            return []
        items = list(self._derived_pantry)
        if self.pantry:
            from services.pantry_service import matched_items

            items = [item for item in items if not matched_items(item, self.pantry)]
        return items[:DERIVED_PANTRY_ITEMS]

    def _repeatable_from(
        self, recipe_id: str, day: int, meal_type: Optional[str] = None
    ) -> Optional[int]:
        """The earlier day this recipe may repeat from, or None if it may not.

        ``meal_type=None`` asks the looser question — "could ANY slot on this
        day legally repeat it?" — which is what a per-day fetch needs, since
        one fetch serves all three slots and must not exclude an id that one
        of them could still use.
        """
        if recipe_id in self._selected_ids:
            return None  # pinned or downvoted: no way back, by design
        uses = self._commitments.get(recipe_id) or []
        if not uses or len(uses) >= MAX_APPEARANCES:
            return None
        slots = {slot for _day, slot in uses}
        if not slots & REPEATABLE_SLOTS:
            return None
        if meal_type is not None and str(meal_type).lower() not in slots:
            # A breakfast may come back as breakfast. Moving it to dinner is a
            # different feature and a different claim.
            return None
        last = max(used_day for used_day, _slot in uses)
        if int(day) - last < REPEAT_MIN_GAP_DAYS:
            return None
        return last

    def _fetch_exclusions(self, day: int) -> List[str]:
        """Ids to keep out of a day's fetch — the loosest exclusion it needs.

        One fetch serves all three of the day's slots, so excluding anything a
        single slot might legally repeat would make the cooldown unreachable.
        The per-slot rule is applied afterwards in `get_candidate_actions`,
        which is why this costs no extra requests.
        """
        excluded = list(self._selected_ids)
        for recipe_id in self._commitments:
            if recipe_id in excluded:
                continue
            if self._repeatable_from(recipe_id, day) is None:
                excluded.append(recipe_id)
        return excluded


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
