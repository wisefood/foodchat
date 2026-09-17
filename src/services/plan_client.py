"""Client for RecipeWrangler's `/api/v2/tools` surface.

The v1 candidate endpoint returns a *pool per slot* and leaves FoodChat to
assemble a plan from it — which is why the daily pipeline grades 8³ combinations
with an LLM and the weekly planner runs an MDP over per-day pools.

`plan_meals` returns an assembled plan instead. It fills named slots across N
days, never repeats a recipe, honours cuisine/mood/flavour/food-group and time
preferences, drops soft preferences one at a time rather than returning an empty
slot, and reports every drop. Allergens and diet are never relaxed.

That does not make the grader redundant. The two answer different questions:

- `plan_meals` answers "which recipes satisfy these constraints, varied and
  ranked deterministically" — reproducible, cheap, no model call.
- the grader answers "which combination best matches what this person just
  asked for in prose" — which no filter can express.

So this is wired as a *candidate source and a fallback*, not a replacement. It
is most useful exactly where the grader is weakest: when the model is
unavailable, when the query carries no preference the grader could act on, and
for the weekly planner, which has no grader at all.

Kept separate from `candidates_client` because it speaks a different API
version with a different response shape, and folding two contracts into one
module is how the next person ends up unsure which endpoint a change affects.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from models.plan_spec import PlanSpec
from models.recipe import CandidateRecipe

logger = logging.getLogger(__name__)

RECIPEWRANGLER_API_URL = os.getenv("RECIPEWRANGLER_API_URL", "http://recipewrangler:8001")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("RECIPEWRANGLER_TIMEOUT", "60"))

# The slots FoodChat plans. `plan_meals` supports more (brunch, snack, dessert,
# side, drink); these are the three the product actually renders today.
MEAL_SLOTS = ("breakfast", "lunch", "dinner")


class PlanClient:
    """RecipeWrangler `/api/v2/tools` — meal planning and capability discovery."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or RECIPEWRANGLER_API_URL).rstrip("/")

    # ------------------------------------------------------------------ #
    # Capability discovery
    # ------------------------------------------------------------------ #
    _manifest_cache: Optional[dict] = None

    def manifest(self) -> dict:
        """The service's self-description: corpus, vocabularies, limitations.

        Cached for the process lifetime. Worth having beyond the vocabularies:
        it states what the corpus can and cannot do, including that allergen
        data is advisory — which is a thing a planner should be able to read
        rather than assume.
        """
        if PlanClient._manifest_cache is not None:
            return PlanClient._manifest_cache
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/v2/tools")
                response.raise_for_status()
                manifest = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load RecipeWrangler tool manifest: %s", exc)
            manifest = {}
        PlanClient._manifest_cache = manifest
        return manifest

    def vocabularies(self) -> dict:
        return self.manifest().get("vocabularies") or {}

    def accepts(self, tool: str, field: str) -> bool:
        """Whether the live service takes this request field on this tool.

        The planning surface rejects unknown fields with a 422 rather than
        ignoring them — deliberately, so a misremembered name is reported
        instead of silently widening a query. The consequence for a caller is
        that there is no safe way to try an optional parameter and see: it
        either knows the field exists or it must not send it.

        So the manifest says. Each tool entry carries `accepts`, generated from
        its own request model, and FoodChat asks here before sending anything
        newer than the oldest deployment it might be talking to.

        Absent `accepts` — an older RecipeWrangler — is a NO. Guessing yes
        would turn every plan request into a 422 the moment the two services
        drifted, which is the outage this method exists to prevent.
        """
        for entry in self.manifest().get("tools") or []:
            if entry.get("name") == tool:
                return field in (entry.get("accepts") or [])
        return False

    def planning_options(self) -> dict[str, Any]:
        """What a user can actually ask for, from the live service.

        The chat agent needs this to answer "what can you do?" without either
        guessing or reciting a hardcoded list that drifts from the corpus. Every
        value comes from RecipeWrangler's own manifest, so a cuisine added by a
        re-annotation shows up here without a FoodChat deploy.

        Falls back to the mirrored constants when the manifest is unreachable —
        an agent that cannot describe its options is better than one that
        claims it has none.
        """
        from models.plan_spec import (
            KNOWN_SLOTS,
            MAX_MEALS_PER_DAY,
            MAX_PLATES_PER_MEAL,
            ROLES,
        )

        vocab = self.vocabularies()
        return {
            "meal_slots": vocab.get("meal_slots") or list(KNOWN_SLOTS),
            "course_types": vocab.get("course_types") or [],
            "plate_roles": list(ROLES),
            "cuisines": vocab.get("cuisines") or [],
            "moods": vocab.get("moods") or [],
            "flavor_profiles": vocab.get("flavor_profiles") or [],
            "food_groups": vocab.get("food_groups") or [],
            "max_days": 14,
            "max_meals_per_day": MAX_MEALS_PER_DAY,
            "max_plates_per_meal": MAX_PLATES_PER_MEAL,
            "supports_multi_course_meals": True,
            "supports_nutri_score_floor": True,
            # No macro targeting: `plan_meals` has no calorie or protein
            # parameter, and advertising one told the agent it could promise
            # something the endpoint cannot do. Flips to True when the
            # planning surface grows nutrition targets.
            "supports_macro_targets": False,
        }

    def describe_options(self) -> str:
        """The same, as prose the agent can say or fold into a prompt."""
        options = self.planning_options()
        cuisines = options["cuisines"]
        return (
            "I can plan up to {days} days, with up to {meals} meals a day drawn from "
            "{slots}. A meal can be a single dish or several courses "
            "(a main, plus a side, dessert or drink) — so a dinner can be a main "
            "and a salad, or a main and a dessert. I can steer by cuisine "
            "({n_cuisines} available, "
            "e.g. {cuisine_examples}), by mood, flavour or food group, cap the "
            "cooking time and set a minimum Nutri-Score. I cannot hit calorie "
            "or protein targets yet. Allergens and dietary requirements are "
            "never relaxed."
        ).format(
            days=options["max_days"],
            meals=options["max_meals_per_day"],
            slots=", ".join(options["meal_slots"]),
            n_cuisines=len(cuisines),
            cuisine_examples=", ".join(cuisines[:5]) if cuisines else "Italian, Thai",
        )

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def plan_meals(
        self,
        *,
        days: int = 1,
        slots: tuple[str, ...] = MEAL_SLOTS,
        count_per_slot: int = 1,
        # Spend `count_per_slot` only on plates of meals that have more than
        # one. A single-plate meal has nothing to compose, so a deeper pool for
        # it is recipes fetched and ranked to arrive at the first one anyway.
        deepen_multiplate_only: bool = False,
        # Where each slot's window starts. The ranking is deterministic, so
        # every request without this draws from the same top of the same list —
        # which is why asking for a second plan returned the first one.
        offset: int = 0,
        spec: Optional["PlanSpec"] = None,
        allergens: Optional[list[str]] = None,
        diet: Optional[list[str]] = None,
        cuisines: Optional[list[str]] = None,
        moods: Optional[list[str]] = None,
        flavor_profiles: Optional[list[str]] = None,
        food_groups: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        include_ingredients: Optional[list[str]] = None,
        exclude_ingredients: Optional[list[str]] = None,
        exclude_recipe_ids: Optional[list[str]] = None,
        favorite_recipe_ids: Optional[list[str]] = None,
        max_minutes: Optional[int] = None,
        min_nutri_score: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ask RecipeWrangler to assemble a plan.

        Returns the raw envelope — `days`, `applied`, `relaxations`,
        `rejected_options`, `total_recipes_used` — because the relaxation
        record is the interesting part and collapsing it into recipes would
        throw away the service's own account of what it could not honour.

        Raises `httpx.HTTPError` on transport failure, matching
        `candidates_client.fetch_candidates`: callers already know how to
        degrade from that, and inventing a second failure convention for the
        same kind of problem would mean two sets of error handling.
        """
        # A spec supersedes `slots` and `days`. Those remain for the simple
        # case — three meals, one recipe each — because most callers want
        # exactly that and should not have to construct an object to say so.
        #
        # `count_per_slot` is NOT superseded: it is recipes per plate, which is
        # orthogonal to the shape. It used to be silently ignored whenever a
        # spec was passed, so a caller asking a shaped plan for four candidates
        # per plate got one — and a composer handed one candidate per plate has
        # nothing to compose. A parameter that reaches nothing is worse than one
        # that does not exist.
        if spec is not None:
            request_slots = spec.to_request_slots(
                count=max(1, int(count_per_slot)),
                only_multiplate=deepen_multiplate_only,
            )
            days = spec.num_days
        else:
            request_slots = [
                {"slot": slot, "count": max(1, int(count_per_slot))} for slot in slots
            ]

        payload: dict[str, Any] = {
            "days": max(1, int(days)),
            "slots": request_slots,
            "diet": diet or [],
            "exclude_allergens": allergens or [],
            "exclude_ingredients": exclude_ingredients or [],
            "include_ingredients": include_ingredients or [],
            "cuisines": cuisines or [],
            "moods": moods or [],
            "flavor_profiles": flavor_profiles or [],
            "food_groups": food_groups or [],
            "exclude_recipe_ids": exclude_recipe_ids or [],
            # Soft boost — favourites float within their slot, hard filters
            # still decide eligibility.
            "favorite_recipe_ids": favorite_recipe_ids or [],
            "allow_relaxation": True,
        }
        # Claim tags (high_protein, high_fibre, low_calorie, …) ride a
        # parameter that older RecipeWrangler deployments do not have, and its
        # request model rejects unknown fields with a 422 rather than ignoring
        # them. So the key is only added when the live manifest advertises the
        # vocabulary — the vocabulary IS the capability flag, and it is already
        # cached, so this costs nothing per call.
        if tags:
            from services.candidates_client import CANDIDATES

            if CANDIDATES.vocabularies().get("tags"):
                payload["tags"] = list(tags)
            else:
                logger.info(
                    "Not sending tags=%s — this RecipeWrangler does not "
                    "advertise the vocabulary", tags,
                )
        if max_minutes:
            payload["max_minutes"] = int(max_minutes)
        if min_nutri_score:
            payload["min_nutri_score"] = str(min_nutri_score).upper()
        # Gated on the manifest, like `tags`, and for the same reason: the
        # request model rejects unknown fields with a 422, so sending this to a
        # RecipeWrangler that predates it would break every plan rather than
        # degrade. Without it the pool is page one, every time.
        if offset > 0:
            if self.accepts("plan_meals", "offset"):
                payload["offset"] = int(offset)
            else:
                logger.info(
                    "Not sending offset=%d — this RecipeWrangler does not "
                    "advertise it, so every plan draws from the same window",
                    offset,
                )

        logger.info(
            "plan_meals days=%d slots=%s cuisines=%s max_minutes=%s",
            payload["days"],
            [s.get("slot") for s in request_slots],
            payload["cuisines"],
            max_minutes,
        )
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{self.base_url}/api/v2/tools/plan_meals", json=payload
            )
            response.raise_for_status()
            return response.json()

    def find_recipes(
        self,
        query: str,
        *,
        limit: int = 5,
        allergens: Optional[list[str]] = None,
        diet: Optional[list[str]] = None,
        exclude_ingredients: Optional[list[str]] = None,
        course_types: Optional[list[str]] = None,
        exclude_recipe_ids: Optional[list[str]] = None,
        favorite_recipe_ids: Optional[list[str]] = None,
        max_minutes: Optional[int] = None,
        min_nutri_score: Optional[str] = None,
    ) -> list[CandidateRecipe]:
        """Search the corpus by free text, honouring the member's constraints.

        The counterpart to `plan_meals` for "I want *that* dish". Autocomplete
        answers a different question — it prefix-matches titles for a type-ahead
        box, with no analysis and no fuzziness, which is why seed resolution had
        to retry with progressively truncated queries to survive a trailing
        typo. This is an analysed `multi_match` with relevance ranking, so
        "bolognesse" finds bolognese without the workaround.

        More importantly it takes the member's allergens and diet. Resolving a
        named dish without them means offering someone a seed they cannot eat
        and discovering it one step later, which is how "I'd love that" becomes
        an apology. It also takes the Nutri-Score floor and the cooking-time
        slider, which every other fetch applied and this one did not — so a
        seed could be anchored that the planner would have refused.

        Returns `[]` on failure — a seed that cannot be resolved is simply not
        anchored, which the caller already handles.
        """
        text = (query or "").strip()
        if not text:
            return []

        payload: dict[str, Any] = {
            "q": text,
            "limit": max(1, int(limit)),
            "slots": [],
            "exclude_allergens": allergens or [],
            "diet": diet or [],
            "exclude_ingredients": exclude_ingredients or [],
            "course_types": course_types or [],
            "exclude_recipe_ids": exclude_recipe_ids or [],
            "favorite_recipe_ids": favorite_recipe_ids or [],
        }
        # The member's numeric constraints, which this call used to skip. A
        # named dish resolved without them can be anchored into a plan the
        # planner itself would never have chosen it for — the Nutri-Score floor
        # and the cooking-time slider applied to every other fetch and not to
        # this one.
        if max_minutes:
            payload["max_minutes"] = int(max_minutes)
        if min_nutri_score:
            payload["min_nutri_score"] = str(min_nutri_score).upper()
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.base_url}/api/v2/tools/find_recipes", json=payload
                )
                response.raise_for_status()
                found = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("find_recipes failed for %r: %s", text, exc)
            return []

        out: list[CandidateRecipe] = []
        for recipe in found.get("results") or []:
            recipe_id = str(recipe.get("recipe_id") or "").strip()
            if not recipe_id:
                continue
            out.append(
                CandidateRecipe(
                    recipe_id=recipe_id,
                    title=str(recipe.get("title") or ""),
                    ingredients=str(recipe.get("ingredients") or ""),
                    directions=str(
                        recipe.get("directions") or recipe.get("instructions") or ""
                    ),
                    nutrition=_meal_nutrition(recipe),
                    nutri_score=recipe.get("default_nutri_score"),
                    source=recipe.get("source"),
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # Adapting the envelope to FoodChat's shapes
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_candidates(
        envelope: dict[str, Any], allergens: Optional[list[str]] = None
    ) -> dict[str, list[CandidateRecipe]]:
        """Flatten a plan envelope into the `{slot: [CandidateRecipe]}` the
        existing pipeline already consumes.

        `plan_meals` returns rich cards; `CandidateRecipe` carries the text the
        grader reads plus nutrition. The annotations (cuisine, mood, flavour)
        are deliberately *not* carried: handing the grader a nested blob would
        change what the model sees for every existing prompt, and those facets
        have already done their work as filters by this point.

        Recipe text comes through too. `plan_meals` carries `ingredients` and
        `instructions` flattened to strings, which is what made this usable as
        the pipeline's candidate source: the grader reads them to judge a
        combination, and without them it was ranking titles.

        `allergens` applies the client-side backstop that used to live inside
        `candidates_client.fetch_candidates`. It stays because the reason for it
        stands: the corpus's allergen tags have been wrong in production —
        almond dishes tagged `nut_free` — and a service-side filter cannot be
        the only thing between a member and an allergen. The server filter got
        considerably better, which is a reason to keep checking, not to stop.
        """
        from services.candidates_client import allergen_conflict

        by_slot: dict[str, list[CandidateRecipe]] = {}
        dropped = 0
        for day in envelope.get("days") or []:
            for slot in day.get("slots") or []:
                name = slot.get("slot")
                if not name:
                    continue
                bucket = by_slot.setdefault(name, [])
                for recipe in slot.get("recipes") or []:
                    recipe_id = str(recipe.get("recipe_id") or "").strip()
                    if not recipe_id:
                        continue
                    candidate = CandidateRecipe(
                        recipe_id=recipe_id,
                        title=str(recipe.get("title") or ""),
                        # The grader reads both. They were empty here while
                        # the planning surface returned cards only, which is
                        # what stopped it being usable as a candidate source:
                        # a model asked to judge a meal with no ingredients
                        # and no method is judging a title.
                        ingredients=str(recipe.get("ingredients") or ""),
                        directions=str(
                            recipe.get("directions")
                            or recipe.get("instructions")
                            or ""
                        ),
                        nutrition=_meal_nutrition(recipe),
                        nutri_score=recipe.get("default_nutri_score"),
                        image_url=recipe.get("image_url"),
                        source=recipe.get("source"),
                    )

                    conflict = allergen_conflict(
                        f"{candidate.title} {candidate.ingredients}", allergens or []
                    )
                    if conflict:
                        dropped += 1
                        logger.warning(
                            "Dropping %r from %s — mentions %r despite the "
                            "server-side allergen filter",
                            candidate.title, name, conflict,
                        )
                        continue
                    bucket.append(candidate)

        if dropped:
            logger.warning("Allergen backstop dropped %d candidate(s)", dropped)
        return by_slot

    @staticmethod
    def to_role_pools(
        envelope: dict[str, Any],
        spec: "PlanSpec",
        allergens: Optional[list[str]] = None,
    ) -> dict[int, dict[tuple[str, str], list[CandidateRecipe]]]:
        """`{day: {(slot, role): [CandidateRecipe]}}` — a pool per PLATE.

        `to_candidates` buckets by slot name alone, which is right for the
        three-single-plate case and wrong the moment a meal has two plates: a
        main and a salad both land in `by_slot["lunch"]`, mixed together, with
        nothing left to say which was which. That is the whole reason multi-plate
        planning had no producer — the renderer was ready, the request was
        already one entry per plate, and the reply was being read back through a
        function that threw the distinction away.

        **Paired entry-wise, not recipe-wise.** `to_request_slots` emits one
        entry per plate and `role_sequence` regenerates the same order, so the
        Nth entry of the response is the Nth plate — whatever number of recipes
        it contains. `plan_structured` zipped the FLATTENED recipe list against
        the role sequence instead, which is correct only while every plate
        returns exactly one recipe: a plate the corpus could not fill shifts
        every role after it by one, so a two-plate lunch with an empty main
        rendered the salad as the main and the next slot's main as a side. Here
        the count per plate is deliberately greater than one — there is no
        composition to make without a choice — so recipe-wise pairing is not
        merely fragile, it is wrong.
        """
        from services.candidates_client import allergen_conflict

        sequence = spec.role_sequence()
        out: dict[int, dict[tuple[str, str], list[CandidateRecipe]]] = {}
        dropped = 0

        for index, day_payload in enumerate(envelope.get("days") or []):
            day = int(day_payload.get("day") or index + 1)
            entries = day_payload.get("slots") or []
            if len(entries) != len(sequence):
                # Worth a line rather than a silent truncation: the pairing is
                # positional, so a response that does not echo one entry per
                # requested plate means the plates below this point are being
                # matched to the wrong roles.
                logger.warning(
                    "day %s returned %d slot entries for %d requested plates — "
                    "pairing what lines up and dropping the rest",
                    day, len(entries), len(sequence),
                )
            pools: dict[tuple[str, str], list[CandidateRecipe]] = {}
            for (expected_slot, role), entry in zip(sequence, entries):
                # The slot name comes from the response so a service that
                # reorders is still read correctly; the role comes from the
                # request because roles are FoodChat's vocabulary and the
                # response has never carried them.
                slot = str(entry.get("slot") or expected_slot)
                bucket = pools.setdefault((slot, role), [])
                for recipe in entry.get("recipes") or []:
                    recipe_id = str(recipe.get("recipe_id") or "").strip()
                    if not recipe_id:
                        continue
                    candidate = CandidateRecipe(
                        recipe_id=recipe_id,
                        title=str(recipe.get("title") or ""),
                        ingredients=str(recipe.get("ingredients") or ""),
                        directions=str(
                            recipe.get("directions") or recipe.get("instructions") or ""
                        ),
                        nutrition=_meal_nutrition(recipe),
                        nutri_score=recipe.get("default_nutri_score"),
                        image_url=recipe.get("image_url"),
                    )
                    # Same backstop, same reason: the corpus's allergen tags
                    # have been wrong in production, and a server-side filter
                    # cannot be the only thing between a member and an
                    # allergen.
                    conflict = allergen_conflict(
                        f"{candidate.title} {candidate.ingredients}", allergens or [],
                    )
                    if conflict:
                        dropped += 1
                        logger.warning(
                            "Dropping %r from %s %s — mentions %r despite the "
                            "server-side allergen filter",
                            candidate.title, slot, role, conflict,
                        )
                        continue
                    bucket.append(candidate)
            out[day] = pools

        if dropped:
            logger.warning("Allergen backstop dropped %d candidate(s)", dropped)
        return out

    @staticmethod
    def describe_relaxations(envelope: dict[str, Any]) -> list[str]:
        """Human-readable lines for anything the service could not honour.

        Surfaced so the chat layer can say "I couldn't find Japanese breakfasts,
        so I widened it" instead of silently producing a plan that ignores what
        was asked. A preference dropped without a word is indistinguishable from
        one that was never understood.
        """
        lines = []
        for entry in envelope.get("relaxations") or []:
            dropped = entry.get("dropped")
            slot = entry.get("slot")
            day = entry.get("day")
            if not dropped:
                continue
            label = {
                "moods": "the mood",
                "flavor_profiles": "the flavour",
                "cuisines": "the cuisine",
                "food_groups": "the food group",
                "max_minutes": "the time limit",
            }.get(dropped, dropped)
            lines.append(f"day {day} {slot}: relaxed {label} to fill the slot")
        for option in envelope.get("rejected_options") or []:
            lines.append(f"ignored unrecognised option: {option}")
        return lines


PLANNER = PlanClient()


def _meal_nutrition(recipe: dict[str, Any]) -> Optional[dict[str, Any]]:
    """RecipeWrangler's macros, in the key names FoodChat and the UI read.

    The two disagree on one word: RecipeWrangler says `calories`, everything
    here says `kcal` — `MealCourse.nutrition`, the meal-card chips, the day
    totals. Passing the dict through unchanged left every consumer reading
    `kcal` from a dict that has none, so plans rendered with 0 kcal against
    recipes whose calories were right there in the response.

    Renamed once, at the boundary, rather than teaching every reader to check
    both spellings forever.
    """
    macros = recipe.get("nutrition") or {}
    if not macros:
        return None
    return {
        "kcal": macros.get("calories"),
        "protein_g": macros.get("protein_g"),
        "carbs_g": macros.get("carbs_g"),
        "fat_g": macros.get("fat_g"),
        "fiber_g": macros.get("fiber_g"),
        "nutri_score_label": recipe.get("default_nutri_score"),
    }
