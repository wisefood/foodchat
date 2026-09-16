"""
Action space for the weekly-plan MDP — candidate recipes per meal slot.

Fetches a fresh candidate pool from RecipeWrangler once per plan day
(``services.candidates_client``), excluding recipes already committed to the
plan.

M9 retired the absolute no-repeat contract for ONE slot. A 7-day plan used to
exclude every committed id from every later fetch, so repeats were impossible
at the source rather than merely disfavoured — and nobody eats seven different
breakfasts. Breakfast got a **slot-scoped cooldown** (see the repeat policy
below). A repeat is always labelled with the day it repeats and *why* it was
allowed, so a week that repeated because the pool was thin can never be
presented as one the member asked for.

M10 makes the SCOPE of that cooldown the member's to set, and adds the
cross-slot case M9 explicitly excluded:

- ``plan_parameters.repeat_meals`` — one ordered control, ``off`` (21 distinct
  recipes, the pre-M9 rule) → ``breakfast`` (M9's behaviour, and still the
  default) → ``all`` (lunch and dinner too) → ``leftovers``. How often a
  *dinner* may recur before a week reads as lazy rather than familiar is a
  household question, not one this file can answer, so it is asked rather than
  guessed. The gap and the cap stay here: they are what keeps a thin pool from
  cashing the member's setting in for monotony.
- **Leftovers** — at ``leftovers``, day N's dinner may be served as day N+1's
  lunch. It is a *repeat with a slot transition*, not a new mechanism: the same
  commitments, the same cap, the same labelling, and one extra rule (yesterday
  only, at most ``MAX_LEFTOVER_MEALS`` a week). The candidate is rebuilt from
  what was actually committed rather than re-fetched, so it costs no request
  and cannot drift from the dish on the plate.

  A leftover entry holds the **whole recipe**, not a reference to another slot:
  the member really does eat that dish, so its nutrition, ingredients and card
  are the dish's own. What it does NOT do is buy anything — the ingredient
  basket, and everything downstream that measures the shopping list, skip it.
  No portion arithmetic is claimed anywhere: nothing in this service records
  quantities, so the honest reading of this feature is "eat the same dinner
  again tomorrow at noon", and that is what the wording says.

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

from services import intent_facets
from services.candidates_client import CANDIDATES, MEAL_SLOTS, normalize_diet_tags

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

# Slots where a recipe may come back at all, at the DEFAULT setting. Kept as a
# named constant because it is the M9 contract other modules (and their tests)
# read the policy from; `plan_parameters.repeat_meals` is what widens it, and
# `_slot_repeats_allowed` is the one place that resolves the two.
REPEATABLE_SLOTS: frozenset = frozenset({"breakfast"})

# Every slot the loosest setting opens up. Not a free-for-all list: a slot
# absent from here can never repeat however the control is set.
ALL_REPEATABLE_SLOTS: frozenset = frozenset({"breakfast", "lunch", "dinner"})

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
REPEAT_LEFTOVER = "leftover"              # yesterday's dinner, eaten at lunch


# --- Leftovers (M10) --------------------------------------------------------
#
# "Cook once, eat twice" — the most common real-world weekly pattern, and the
# one the planner could not express at all.
#
# Deliberately the narrowest possible version. Yesterday's dinner, today's
# lunch, nothing else: not the day before yesterday (that is a fridge claim,
# and nothing here records when anything was cooked or how long it keeps), not
# dinner-to-dinner (that is an ordinary repeat and already has a rule), and not
# breakfast (nobody saves half a dinner for tomorrow's breakfast).

# {slot being filled: slot it may take yesterday's dish from}. A dict rather
# than a pair, so the transition is stated in one direction only — lunch may
# take from dinner; dinner may never take from lunch.
LEFTOVER_FROM_SLOT: dict = {"lunch": "dinner"}

# Exactly one day, not "at least one". Two days later is not a leftover, it is
# a repeat, and it has its own rule and its own honest wording.
LEFTOVER_GAP_DAYS = 1

# Leftover lunches in a week. Six would mean lunch is never cooked, which is a
# different product; three is "about half the week", and is the number to turn
# if members ask for more.
MAX_LEFTOVER_MEALS = 3


# --- Offering a repeat rather than waiting for one (M10) --------------------
#
# The cooldown decides whether a recipe MAY come back. Until now, whether one
# ever got the chance was RecipeWrangler's: a day's pool is a fresh fetch, and
# an eligible earlier dish only reappeared if the source happened to rank it
# into the day's top `DAILY_POOL_LIMIT`. On a real week it usually did not.
# Observed on a live plan with the control set to "cook once, eat twice": the
# policy allowed a repeat at all four remaining breakfast slots and the source
# offered one at none of them, so a member who had asked for repeated
# breakfasts got exactly one, for a reason nothing in the plan could name.
#
# So an eligible dish is now PUT in the pool, rebuilt from what was committed
# — the same trick the leftover uses, for the same price of zero requests.
#
# Only when the member set the control themselves (`repeat_mode_is_explicit`).
# The default must keep waiting for the source: injecting on a default would
# change every existing member's week to satisfy a preference none of them
# expressed, which is the difference between honouring a request and inventing
# one.
#
# Injected candidates stay COUNTED SEPARATELY in the `repeat_offered` event.
# That event exists to distinguish "the week repeated nothing" from "the
# source never offered anything to repeat", and injecting would erase exactly
# that signal if the two were merged.

# Eligible earlier dishes added to one slot's pool. Everything eligible would
# be legal, but a slot whose pool is a third old dishes is a repetitive week
# arriving by the back door rather than by the member's setting.
INJECTED_REPEATS_PER_SLOT = 2

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
        spec: Optional[object] = None,
        avoid_recent: Optional[List[str]] = None,
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
        # Recipes that may never appear (again): member-pinned anchors and
        # downvoted dishes, registered by the service through `mark_selected`.
        # Distinct from `_commitments` below — this list has no way back.
        self._selected_ids: List[str] = []
        # recipe_id -> [(day, meal_type)] the plan actually served it at (M9).
        # A commitment is not automatically an exclusion any more; whether it
        # is depends on the slot, the gap and the cap. See `_repeatable_from`.
        self._commitments: Dict[str, List[tuple]] = {}
        # (day, meal_type) -> the action dict actually committed there (M10).
        # A leftover is rebuilt from this rather than re-fetched: the dish it
        # claims to be is the dish on the plate, by construction, and it costs
        # no request. Only what a leftover needs is kept — the whole action, so
        # the copy carries the same nutrition and tags the source card shows.
        self._served: Dict[tuple, Dict[str, Any]] = {}
        # Leftover lunches taken so far, against `MAX_LEFTOVER_MEALS`.
        self._leftovers_taken = 0
        self._favorites = {
            str(f) for f in (user_profile.get("favorite_recipe_ids") or [])
        }
        # What the member was served on their LAST few plans. A soft ask: it
        # rides the same exclusion list, and is dropped for the whole plan if
        # the first day cannot be filled with it. Without it, a second "plan my
        # week" returned the same week — RecipeWrangler's order is
        # deterministic and this loop has no model to vary the pick.
        self._avoid_recent: List[str] = [
            r for r in (avoid_recent or []) if r
        ]
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
        # The member's repeat setting, resolved once. Read through
        # `plan_parameters` so an unrecognised stored value degrades to the
        # default rather than to "no repeats at all" (M10).
        self.repeat_mode = plan_parameters.repeat_mode(
            user_profile.get("plan_parameters") or {}
        )
        # Whether the member SET that, or merely inherited it. Only an
        # explicit setting earns an injected repeat — see the policy note
        # above `INJECTED_REPEATS_PER_SLOT`.
        self.repeats_explicit = plan_parameters.repeat_mode_is_explicit(
            user_profile.get("plan_parameters") or {}
        )

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
                exclude_recipe_ids=self._fetch_exclusions(current_day),
                limit_per_slot=DAILY_POOL_LIMIT,
                slots=tuple(self._slots()),
            )
            if self._avoid_recent and not all(pool.get(s) for s in self._slots()):
                # Nothing new left for some slot. Give the history up for the
                # whole plan rather than per slot: a week is 21 picks, and
                # dropping it once keeps the pools consistent across days
                # instead of a different exclusion set on every fetch.
                logger.info(
                    "Nothing new left for every slot — planning this week "
                    "without the recently-served exclusion",
                )
                self._avoid_recent = []
                pool = _fetch_candidate_pool(
                    profile=self.user_profile,
                    allergens=self.allergens,
                    diet=normalize_diet_tags(self.diet),
                    cuisines=cuisines,
                    exclude_recipe_ids=self._fetch_exclusions(current_day),
                    limit_per_slot=DAILY_POOL_LIMIT,
                    slots=tuple(self._slots()),
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
            # Indexed into the plan's OWN slots. The literal map here was the
            # default three, so on a four-meal day index 3 answered "lunch".
            slots = self._slots()
            meal_type = (
                slots[meal_type] if 0 <= meal_type < len(slots) else slots[0]
            )

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

        # Eligible earlier dishes the source did not return, put in the pool
        # rather than waited for (M10). Only on an explicit setting, and only
        # after the fetched pool has been read, so a dish the source DID
        # return is never added twice.
        injected = self._injected_repeats(
            current_day, meal_type, {a["recipe_id"] for a in actions}
        )
        for action in injected:
            actions.append(action)
            offered.append(action["recipe_id"])

        leftover = self._leftover_action(current_day, meal_type)
        if leftover is not None:
            # Appended, never substituted. The member asked that leftovers be
            # POSSIBLE, not that lunch stop being planned: the scorer weighs
            # this against the day's real candidates, hard constraints still
            # prune it (a leftover that would break the meat limit is dropped
            # like any other meat dish), and if it loses, nothing is lost.
            actions.append(leftover)
            offered.append(leftover["recipe_id"])

        if offered and (current_day, meal_type) not in self._repeat_offers_noted:
            self._repeat_offers_noted.add((current_day, meal_type))
            event = {
                "type": "repeat_offered",
                "day": current_day,
                "meal_type": meal_type,
                "count": len(offered),
                "recipe_ids": offered,
            }
            if injected:
                # Kept apart from `count`, never folded into it. This event's
                # whole job is to separate "the week repeated nothing" from
                # "the source never offered anything to repeat", and once the
                # plan puts dishes in the pool itself, that question can only
                # be answered by a number that says how many it put there.
                event["injected"] = len(injected)
            self.selection_events.append(event)
        return actions

    def _slots(self) -> List[str]:
        """The meal slots this plan needs a pool for."""
        return [
            str(m).lower() for m in (getattr(self.spec, "meals", None) or ())
        ] or ["breakfast", "lunch", "dinner"]

    def mark_selected(self, recipe_id: str) -> None:
        """Bar a recipe from the plan entirely — no slot, no day, no cooldown.

        The service's two callers mean exactly that: a member-pinned anchor
        must not turn up a second time elsewhere in the week, and a downvoted
        dish must not turn up at all. Ordinary commitments go through
        `mark_committed`, which leaves the repeat policy a say.
        """
        if recipe_id and recipe_id not in self._selected_ids:
            self._selected_ids.append(recipe_id)

    def mark_committed(
        self,
        recipe_id: str,
        day: int,
        meal_type: str,
        action: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record that the plan served this recipe, on this day, in this slot.

        Called by the environment as each slot commits. The day and slot are
        what the repeat policy needs — without them a commitment can only mean
        "never again", which is the rule M9 exists to loosen.

        ``action`` is the committed candidate itself, kept so a leftover lunch
        can be built from the dinner that was actually served (M10). Optional,
        and the whole feature simply does not fire without it: an environment
        that predates this passes three arguments and gets M9's behaviour.
        """
        if not recipe_id:
            return
        slot = str(meal_type).lower()
        self._commitments.setdefault(str(recipe_id), []).append((int(day), slot))
        if action is not None:
            self._served[(int(day), slot)] = dict(action)
            if action.get("leftover_of"):
                self._leftovers_taken += 1

    def _second_serving(
        self,
        served: Dict[str, Any],
        source_day: int,
        source_slot: str,
        source: str,
    ) -> Dict[str, Any]:
        """A committed dish, rebuilt as a candidate for a second serving.

        Shared by the two ways one can come back — an injected repeat and a
        leftover — so the pair cannot drift in what they carry or, more
        importantly, in what they drop. The source's own labels describe the
        source's slot, and `pinned` would credit the member for a dish the
        plan chose to serve again.
        """
        action = {
            key: value for key, value in served.items()
            if key not in ("repeat_of_day", "repeat_source", "leftover_of",
                           "pinned", "match_reasons")
        }
        action["repeat_of_day"] = int(source_day)
        action["repeat_source"] = source
        if source == REPEAT_LEFTOVER:
            action["leftover_of"] = {
                "day": int(source_day), "meal_type": str(source_slot),
            }
        return action

    def _injected_repeats(
        self, day: int, meal_type: str, already: set
    ) -> List[Dict[str, Any]]:
        """Eligible earlier dishes for this slot that the pool did not contain.

        The cooldown says a recipe may come back; without this, whether it ever
        got the chance was RecipeWrangler's ranking. Rebuilt from `_served`, so
        this costs no request and offers the dish that was actually eaten.

        Gated on an EXPLICIT setting: a member who never touched the card keeps
        waiting for the source, exactly as before. Every other rule is the
        ordinary one — `_repeatable_from` decides eligibility, so the slot, the
        gap, the cap and `mark_selected` all still apply, and a dish the source
        already returned is skipped rather than duplicated.

        Ordered by how recently the dish was eaten, newest first: a routine is
        built out of what the week is already in the habit of.
        """
        if not self.repeats_explicit or not self._served:
            return []
        slot = str(meal_type).lower()
        if not self._slot_repeats_allowed(slot):
            return []

        eligible: List[tuple] = []
        for (served_day, served_slot), served in self._served.items():
            if served_slot != slot:
                continue
            recipe_id = str(served.get("recipe_id") or "")
            if not recipe_id or recipe_id in already:
                continue
            if served.get("leftover_of"):
                # A leftover is a dish already on its second serving; the cap
                # below catches it, but skipping here says why.
                continue
            if self._repeatable_from(recipe_id, day, slot) is None:
                continue
            eligible.append((-served_day, recipe_id, served, served_slot))

        eligible.sort(key=lambda row: (row[0], row[1]))
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for _order, recipe_id, served, served_slot in eligible:
            if recipe_id in seen:
                continue
            seen.add(recipe_id)
            out.append(
                self._second_serving(served, -_order, served_slot, REPEAT_PLAN)
            )
            if len(out) >= INJECTED_REPEATS_PER_SLOT:
                break
        return out

    def _leftover_action(
        self, day: int, meal_type: str
    ) -> Optional[Dict[str, Any]]:
        """Yesterday's dinner, offered as today's lunch — or ``None``.

        Built from ``_served``, so the candidate IS the committed dish rather
        than a fresh search that resembles it. Every gate that stops an
        ordinary repeat stops this too, plus two of its own:

        - ``MAX_LEFTOVER_MEALS`` — six leftover lunches is a week where lunch
          is never cooked, which is a different product from the one asked for;
        - a pinned or downvoted dish stays barred. ``mark_selected`` means
          "no way back" and a leftover is a way back; a member's anchor turning
          up twice is the thing that call exists to prevent, and honouring the
          contract matters more than the extra leftover it costs.

        Returns a candidate carrying BOTH ``repeat_of_day`` and
        ``leftover_of``. The first is deliberate: a leftover is a sanctioned
        second serving, so every rule M9 already wrote for one — the scorer's
        own-title exemption, the ingredient-axis exclusion, the cap, the
        measured ledger row — applies without a parallel code path. The second
        is what lets the chip, the ledger and the prose say *dinner at lunch*
        instead of "the same lunch as Monday", which would be false.
        """
        from services import plan_parameters  # local import; avoids a cycle

        slot = str(meal_type).lower()
        source_slot = LEFTOVER_FROM_SLOT.get(slot)
        if source_slot is None:
            return None
        if not plan_parameters.leftovers_allowed(
            self.user_profile.get("plan_parameters") or {}
        ):
            return None
        if self._leftovers_taken >= MAX_LEFTOVER_MEALS:
            return None
        source_day = int(day) - LEFTOVER_GAP_DAYS
        if source_day < 1:
            return None
        served = self._served.get((source_day, source_slot))
        if not served or served.get("leftover_of"):
            return None
        recipe_id = str(served.get("recipe_id") or "")
        if not recipe_id or recipe_id in self._selected_ids:
            return None
        if len(self._commitments.get(recipe_id) or []) >= MAX_APPEARANCES:
            return None

        return self._second_serving(
            served, source_day, source_slot, REPEAT_LEFTOVER
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

    def _slot_repeats_allowed(self, meal_type: Optional[str]) -> bool:
        """Whether the member's setting lets ``meal_type`` serve a dish twice.

        ``None`` asks the per-day question a fetch needs: may ANY slot repeat?
        Resolved here rather than at each call site so the control and the
        hard list stay in one place — a slot outside `ALL_REPEATABLE_SLOTS`
        can never repeat, whatever the setting says.
        """
        from services import plan_parameters  # local import; avoids a cycle

        values = self.user_profile.get("plan_parameters") or {}
        if meal_type is None:
            return any(
                plan_parameters.repeats_allowed(values, slot)
                for slot in ALL_REPEATABLE_SLOTS
            )
        slot = str(meal_type).lower()
        if slot not in ALL_REPEATABLE_SLOTS:
            return False
        return plan_parameters.repeats_allowed(values, slot)

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
        # Which slots may repeat is the member's setting (M10); which slots
        # *could* is the hard list. A dish only qualifies through the slot it
        # was actually served in — a breakfast the member allowed to repeat is
        # a repeatable breakfast, not a repeatable recipe.
        if not any(self._slot_repeats_allowed(slot) for slot in slots):
            return None
        if meal_type is not None:
            slot = str(meal_type).lower()
            # A breakfast may come back as breakfast. Moving it to dinner is a
            # different feature and a different claim — the one cross-slot move
            # this planner makes is the leftover, which has its own rule,
            # its own gap and its own label (`_leftover_action`).
            if slot not in slots or not self._slot_repeats_allowed(slot):
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
        # `_avoid_recent` rides here too — what the member was served on their
        # LAST few plans, which is what stops a second "plan my week" returning
        # the same week. It is a SOFT ask and the caller drops it wholesale
        # when a slot cannot be filled with it, so it belongs in the same list
        # rather than at one of the two call sites.
        excluded = list(self._selected_ids) + list(self._avoid_recent)
        for recipe_id in self._commitments:
            if recipe_id in excluded:
                continue
            if self._repeatable_from(recipe_id, day) is None:
                excluded.append(recipe_id)
        return excluded

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
                exclude_recipe_ids=self._fetch_exclusions(day),
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
    slots: tuple = MEAL_SLOTS,
) -> dict:
    """A per-slot candidate pool from the planning endpoint.

    `slots` is the plan's own shape, and it used to be absent — so this always
    fetched breakfast, lunch and dinner. The environment honours a spec's meals
    (`self.meal_types = slots`), so a week with a snack stepped through a snack
    slot whose pool had never been asked for: `get_candidate_actions` returned
    nothing and the planner raised `PlanGenerationError("snack")`. The member
    asked for a week with a snack and got told the week could not be built.

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
            # No `course_types` override: RecipeWrangler maps each slot to its
            # own courses (snack → snacks, dessert → desserts), which is what
            # keeps a snack slot from filling with main dishes.
            slots=tuple(slots) or MEAL_SLOTS,
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
