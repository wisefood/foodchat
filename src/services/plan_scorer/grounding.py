"""
Plan scorer, step 2 — ground each pasted dish against RecipeWrangler.

    DishGrounder(seed_service).ground(plan, profile) → list[GroundedMeal]

Every dish ends in exactly one state, and the state decides what later steps
may treat as known:

    matched      title similarity >= MATCHED_SIMILARITY, measured on the
                 recipe actually fetched, AND the same dish by name: the
                 recipe's name keeps every part the member named
                 (``same_dish``) and adds only descriptive words ("Roasted",
                 "Smoked", "Vegetarian"), never a food ("Chicken Caesar
                 salad" is not "caesar salad"). Scored as the catalogue
                 recipe: its nutrition and image, and its ingredients and
                 tags unless the member wrote their own ingredients — what
                 they wrote is the dish.
    approximate  a catalogue hit below that, but >= APPROXIMATE_SIMILARITY.
                 NEVER lends its ingredients or tags: only the member's own
                 words are read. It lends nutrition figures only when its
                 name is a more generic form of the member's ("Lasagna" for
                 "vegetable lasagna"), never when it names something the
                 member did not write ("Apple strudel" for "an apple") or
                 leaves out a part they did ("Roast potatoes" for "roast
                 chicken with potatoes", "Hummus" for "houmous and pitta").
    unresolved   nothing close enough. Title and whatever ingredients the
                 member wrote.

A recipe's calories below MIN_MEAL_KCAL (a main meal) or MIN_SNACK_KCAL are
set aside: the catalogue holds whole meals at 7–97 kcal, and taking those as
measurements scored real days as starvation.

Calories, in order: the matched recipe; the closest recipe when its name is a
more generic form of the member's; then, for what is still missing, a small
model writes a typical serving's ingredients (keeping the member's own) and
RecipeWrangler's profiler looks them up in food composition tables; and only
when the profiler cannot give reliable figures — or its figure is more than
double off that guess — the model's own calorie guess.
``GroundedMeal.nutrition_source`` says which, so an estimate is never shown as
a measurement. Profiling runs in parallel with a short timeout and stops at the
first timeout — a stalled pipeline once held a turn for two minutes. Titles
are searched in their other spellings too ("lasagne" as well as "lasagna").

Why an approximate match lends so little: the first live run read "vegetable
lasagna" as a beef lasagna and failed a vegetarian profile on meat nobody
ate, read "peanut noodles" as a salmon noodle recipe and reported fish, and
counted an apple as apple strudel. An allergen found only in the closest
recipe is recorded with ``closest_recipe`` evidence — a warning, not a verdict.

Lookup reuses ``SeedService.find_dish`` — the seed path's search and tolerant
autocomplete — but NOT its filters or its allergy gate. Seed resolution narrows
to what the member can eat because a seed is about to be planned. A pasted dish
has already been eaten: filtering would swap "peanut noodles" for a peanut-free
recipe and hide the one fact the score most needs to report. So a dish that
conflicts with an allergy is grounded like any other, and the conflict is
recorded on it (``allergen_conflicts``) for the ledger to report.

The threshold is the part seed resolution never needed. For seeding a near
miss is harmless — the member sees the pinned dish and can object. For scoring
a near miss silently changes the number, so the matched title is returned with
every dish and approximate matches are named in the reply.

Best-effort like every RecipeWrangler call: a lookup failure leaves the dish
unresolved, never fails the turn. One lookup per distinct title, so a week of
the same breakfast costs one search.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from models.pasted_plan import (
    APPROXIMATE,
    AS_WRITTEN,
    CLOSEST_RECIPE,
    FROM_RECIPE,
    MATCHED,
    NUTRITION_FROM_CLOSEST,
    NUTRITION_FROM_RECIPE,
    NUTRITION_MODEL_ESTIMATE,
    MAIN_SLOTS,
    NUTRITION_TYPICAL,
    UNKNOWN,
    UNRESOLVED,
    GroundedMeal,
    PastedMeal,
    PastedPlan,
)
from models.recipe import CandidateRecipe, ResolvedRecipe
from services.candidates_client import ProfilingTimeout, allergen_conflict

from .parsing import PLANT_DAIRY, content_tokens, dish_heads, spelling_variants

logger = logging.getLogger(__name__)

# Dice coefficient over content words of the two titles. "Berry oatmeal" vs
# "oatmeal with berries" is 1.0; "lentil soup" vs "Fakes (Greek lentil soup)"
# is 0.67 — the same dish, but a claim worth showing; "chicken curry" vs
# "Thai green curry with chicken" is 0.67 too, and that one is not the same dish.
MATCHED_SIMILARITY = 0.75
APPROXIMATE_SIMILARITY = 0.4
CANDIDATES_PER_DISH = 5
# How many distinct dishes of one plan get a typical serving from the one
# batched model call: a week's worth. Dishes missing calories come first.
MAX_ESTIMATED_DISHES = 21
# How many of those are profiled. A pasted week of unknown dishes would
# otherwise be twenty-one calls to a pipeline whose cost is not ours to spend;
# the rest keep the model's own guess.
MAX_PROFILE_CALLS = 10
# Seconds one profiling call may take. A healthy pipeline answers in 1–7 s; a
# stalled one held the first live turn for two minutes at the 60 s default.
PROFILE_TIMEOUT_SECONDS = 15.0
PROFILE_WORKERS = 4
# A per-serving calorie guess outside this range is not an estimate.
MODEL_KCAL_RANGE = (20.0, 2500.0)
# A profiled figure more than this factor above or below the model's own guess
# is not used. Live: pho profiled at 2,025 kcal (guess 600), minestrone at
# 1,138 (guess 500), both with full coverage.
MAX_PROFILE_DISAGREEMENT = 2.0
# A catalogue recipe's per-serving calories below these are not a serving of
# the meal. Live: "Quick Chili" 12 kcal, "Tomato Basil Soup" 24, "Pancakes" 97.
MIN_MEAL_KCAL = 120.0
MIN_SNACK_KCAL = 40.0
# Words a recipe name may add without becoming another dish: how it is cooked,
# how it is sold. A food word ("chicken", "ricotta") is not here, and neither is
# a diet or cuisine word: "Vegan caesar salad" swaps the dressing for cashews,
# "Mexican Caesar salad" adds chorizo.
DESCRIPTIVE_WORDS = frozenset({
    "roast", "roasted", "baked", "grilled", "chargrilled", "fried", "pan", "seared", "steamed",
    "boiled", "poached", "smoked", "braised", "slow", "cooked", "stir", "toasted", "barbecue",
    "barbecued", "bbq", "sauteed", "oven", "crispy", "crunchy", "spicy", "spiced", "hot", "mild",
    "warm", "cold", "chunky", "hearty", "rustic", "light", "lighter", "healthy", "healthier",
    "classic", "traditional", "authentic", "basic", "best", "perfect", "ultimate", "favourite",
    "favorite", "family", "weeknight", "speedy", "super", "tasty", "delicious", "one", "pot",
    "tray", "sheet", "mini", "big", "little", "individual", "style", "recipe", "leftover",
})


def title_similarity(given: str, candidate: str) -> float:
    """Dice over content words. Amount words are ignored in the member's title
    only — in a catalogue title they name the dish (see parsing.MEASURE_WORDS)."""
    a = set(content_tokens(given))
    b = set(content_tokens(candidate, keep_measures=True))
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def covers_parts(given: str, candidate: str) -> bool:
    """Every part of the member's dish is named in the candidate's title."""
    words = set(content_tokens(candidate, keep_measures=True))
    return all(head in words for head in dish_heads(given))


def same_dish(given: str, candidate: str) -> bool:
    """A close title that is the member's dish, not a neighbour of it.

    It must keep every part the member named, and whatever it adds must be
    descriptive: "Roasted vegetable lasagne" is "vegetable lasagne", "Chicken
    Caesar salad" is not "caesar salad" and "Roast potatoes" is not "roast
    chicken with potatoes".
    """
    extra = set(content_tokens(candidate, keep_measures=True)) - set(content_tokens(given))
    return covers_parts(given, candidate) and extra <= DESCRIPTIVE_WORDS


def borrowable(given: str, candidate: str) -> bool:
    """The candidate's name says nothing the member's does not, and leaves out
    no part of it — the same dish named more generically ("Lasagna" for
    "vegetable lasagna"), so its figures are a fair estimate."""
    words = set(content_tokens(candidate, keep_measures=True))
    return bool(words) and words <= set(content_tokens(given)) and covers_parts(given, candidate)


def plausible_recipe_kcal(nutrition: Optional[dict], slot: str) -> bool:
    """Whether a catalogue recipe's calories could be a serving of this meal."""
    try:
        kcal = float((nutrition or {}).get("kcal") or 0)
    except (TypeError, ValueError):
        return False
    return kcal >= (MIN_MEAL_KCAL if slot in MAIN_SLOTS else MIN_SNACK_KCAL)


def _match_rank(title: str, candidate_title: str) -> tuple:
    """Rank hits: a hit that is the same dish first, then by similarity."""
    similarity = title_similarity(title, candidate_title)
    return (similarity >= MATCHED_SIMILARITY and same_dish(title, candidate_title), similarity)


def dish_request(meal: GroundedMeal) -> dict:
    """What the ingredient estimator is told about a dish: the member's words."""
    return {
        "title": (meal.title_given or "").strip(),
        "ingredients": meal.ingredients if meal.ingredients_source == AS_WRITTEN else None,
        "quantity": meal.quantity_note,
    }


def estimate_key(meal: GroundedMeal) -> str:
    """One estimate per distinct dish as written — a repeated breakfast is one."""
    request = dish_request(meal)
    if not request["title"]:
        return ""
    return "|".join(str(request[k] or "").strip().lower() for k in ("title", "ingredients", "quantity"))


def typical_recipe_text(title: str, ingredients: list) -> str:
    """Recipe text for the profiler: one serving, one ingredient per line.

    "Serves 1" is stated because the profiler otherwise guesses the servings,
    and a guessed two would halve the dish.
    """
    lines = [title, "Serves 1"]
    lines.extend(f"{quantity} {name}".strip() for quantity, name in ingredients)
    return "\n".join(lines)


def plausible_kcal(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        kcal = float(value)
    except (TypeError, ValueError):
        return None
    low, high = MODEL_KCAL_RANGE
    return kcal if low <= kcal <= high else None


def disagrees(profiled_kcal, guess_kcal: Optional[float]) -> bool:
    """The profiler and the model are more than MAX_PROFILE_DISAGREEMENT apart."""
    try:
        profiled = float(profiled_kcal or 0)
    except (TypeError, ValueError):
        return False
    if not guess_kcal or profiled <= 0:
        return False
    ratio = profiled / guess_kcal
    return ratio > MAX_PROFILE_DISAGREEMENT or ratio < 1 / MAX_PROFILE_DISAGREEMENT


def grounding_state(similarity: float, given: str = "", candidate: str = "") -> str:
    if similarity >= MATCHED_SIMILARITY and (not given or same_dish(given, candidate)):
        return MATCHED
    if similarity >= APPROXIMATE_SIMILARITY:
        return APPROXIMATE
    return UNRESOLVED


def allergen_conflicts(
    allergies: list,
    *,
    written_text: str,
    recipe_text: str = "",
    recipe_allergens: tuple = (),
    recipe_evidence: str = FROM_RECIPE,
) -> list[dict]:
    """Every profile allergen found in the dish, with where it was found.

    ``as_written`` when the member's own words name it, ``recipe`` when only
    the catalogue recipe does (its allergen tags or ingredients). Uses the
    same synonym-expanded matcher as the seed gate, so "tree nuts" catches an
    almond dish with no allergen tags.
    """
    tagged = {str(a).strip().lower() for a in recipe_allergens or ()}
    found: list[dict] = []
    for allergy in allergies or []:
        key = str(allergy).strip().lower()
        if not key:
            continue
        written, recipe = written_text, recipe_text
        if key in ("dairy", "lactose", "milk"):
            # "coconut milk" and "peanut butter" are not dairy.
            written, recipe = PLANT_DAIRY.sub(" ", written), PLANT_DAIRY.sub(" ", recipe)
        if allergen_conflict(written, [key]):
            found.append({"allergen": key, "evidence": AS_WRITTEN})
        elif key in tagged or (recipe and allergen_conflict(recipe, [key])):
            found.append({"allergen": key, "evidence": recipe_evidence})
    return found


@dataclass
class _Lookup:
    candidate: Optional[CandidateRecipe] = None
    resolved: Optional[ResolvedRecipe] = None
    similarity: float = 0.0


class DishGrounder:
    """Resolve pasted dishes to catalogue recipes, recording what is known."""

    def __init__(
        self, seed_service=None, details_client=None, profile_client=None,
        estimator=None, profile_workers: int = PROFILE_WORKERS,
    ):
        if seed_service is None:
            from services.seed_service import SeedService

            seed_service = SeedService()
        self.seed_service = seed_service
        self.details_client = details_client or seed_service.client
        self.profile_client = profile_client if profile_client is not None else self.details_client
        # Writes typical ingredients for dishes nothing else gave calories.
        # None means no estimates at all, and no model call: callers wire one
        # in (the orchestrator does), tests that do not want it leave it out.
        self.estimator = estimator
        self.profile_workers = max(1, int(profile_workers))

    def ground(self, plan: PastedPlan, profile: dict) -> list[GroundedMeal]:
        """One GroundedMeal per pasted dish, in the plan's day and text order."""
        allergies = list((profile or {}).get("allergies") or [])
        lookups: dict[str, _Lookup] = {}
        grounded: list[GroundedMeal] = []
        for day in plan.days:
            for meal in day.meals:
                key = " ".join(sorted(set(content_tokens(meal.title)))) or meal.title.lower()
                if key not in lookups:
                    lookups[key] = self._lookup(meal.title)
                grounded.append(self._ground_one(day.day, meal, lookups[key], allergies))
        self._enrich(grounded)
        self._estimate_missing(grounded)
        return grounded

    def _search(self, title: str) -> list:
        """Catalogue hits for the title, and for its other spellings.

        The alternative spellings are only searched when the first query found
        nothing close enough to be a match — one extra request for "lasagne"
        beats missing "Roasted vegetable lasagne" entirely.
        """
        try:
            candidates = list(self.seed_service.find_dish(title, limit=CANDIDATES_PER_DISH) or [])
        except Exception as exc:  # noqa: BLE001 — best-effort lookup
            logger.warning("Dish lookup failed for %r: %s", title, exc)
            return []
        if any(_match_rank(title, c.title)[0] for c in candidates):
            return candidates
        for variant in spelling_variants(title):
            try:
                candidates.extend(self.seed_service.find_dish(variant, limit=CANDIDATES_PER_DISH) or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Dish lookup failed for %r: %s", variant, exc)
        return candidates

    def _lookup(self, title: str) -> _Lookup:
        candidates = self._search(title)

        best: Optional[CandidateRecipe] = None
        best_similarity = 0.0
        best_rank: tuple = (False, 0.0)
        for candidate in candidates or []:
            rank = _match_rank(title, candidate.title)
            if rank > best_rank:
                best, best_similarity, best_rank = candidate, rank[1], rank
        if best is None or best_similarity < APPROXIMATE_SIMILARITY:
            return _Lookup(similarity=best_similarity)

        resolved = None
        try:
            resolved = self.seed_service.client.fetch_recipe(best.recipe_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recipe detail fetch failed for %s: %s", best.recipe_id, exc)
        if resolved is not None and resolved.recipe.title and resolved.recipe.title != best.title:
            # The similarity has to describe the recipe that is actually used:
            # a search hit and its detail record can carry different titles,
            # and the fetched one is what the member is shown and scored on.
            best_similarity = title_similarity(title, resolved.recipe.title)
            if best_similarity < APPROXIMATE_SIMILARITY:
                return _Lookup(similarity=best_similarity)
        return _Lookup(candidate=best, resolved=resolved, similarity=best_similarity)

    @staticmethod
    def _ground_one(day, meal: PastedMeal, lookup: _Lookup, allergies: list) -> GroundedMeal:
        written = (meal.ingredients or "").strip()
        recipe = lookup.resolved.recipe if lookup.resolved else lookup.candidate
        state = (
            grounding_state(lookup.similarity, meal.title, recipe.title or "")
            if lookup.candidate and recipe is not None else UNRESOLVED
        )
        recipe_ingredients = ((recipe.ingredients if recipe else "") or "").strip()
        has_recipe = state in (MATCHED, APPROXIMATE) and recipe is not None
        matched = has_recipe and state == MATCHED
        borrows = has_recipe and not matched and borrowable(meal.title, recipe.title)

        # What the member wrote is the dish, even when a recipe matches its
        # name: "Banana oat pancakes (bananas, eggs, oat flour)" is not the
        # catalogue's "Banana Pancakes" with milk.
        if written:
            ingredients, source = written, AS_WRITTEN
        elif matched and recipe_ingredients:
            ingredients, source = recipe_ingredients, FROM_RECIPE
        else:
            ingredients, source = "", UNKNOWN
        recipe_speaks = has_recipe and not written

        conflicts = allergen_conflicts(
            allergies,
            written_text=f"{meal.title} {written}",
            recipe_text=f"{recipe.title} {recipe_ingredients}" if recipe_speaks else "",
            recipe_allergens=(
                tuple(lookup.resolved.allergens) if recipe_speaks and lookup.resolved else ()
            ),
            recipe_evidence=FROM_RECIPE if matched else CLOSEST_RECIPE,
        )

        # The search hit's figures belong to the search hit: used only when the
        # fetched record is that same recipe.
        candidate = lookup.candidate
        same_recipe = candidate is not None and (
            lookup.resolved is None or lookup.resolved.recipe.recipe_id == candidate.recipe_id
        )
        nutrition = (
            dict(candidate.nutrition)
            if (matched or borrows) and same_recipe and candidate.nutrition else None
        )
        rejected = None
        if nutrition and not plausible_recipe_kcal(nutrition, meal.slot):
            rejected = {"title": recipe.title, "kcal": float(nutrition.get("kcal") or 0)}
            nutrition = None
        # Its own name: `source` above says where the INGREDIENTS came from,
        # and reusing it here emptied that on every dish.
        nutrition_from = ""
        if nutrition:
            nutrition_from = NUTRITION_FROM_RECIPE if matched else NUTRITION_FROM_CLOSEST
        return GroundedMeal(
            day=day,
            slot=meal.slot,
            title_given=meal.title,
            state=state,
            recipe_id=str(recipe.recipe_id) if has_recipe else None,
            title_matched=(recipe.title or None) if has_recipe else None,
            similarity=round(lookup.similarity, 3),
            ingredients=ingredients,
            ingredients_source=source,
            nutrition=nutrition,
            tags=list(lookup.resolved.tags) if matched and not written and lookup.resolved else [],
            dish_types=list(lookup.resolved.dish_types) if matched and lookup.resolved else [],
            allergen_conflicts=conflicts,
            quantity_note=meal.quantity_note,
            borrows_nutrition=bool(borrows),
            nutrition_source=nutrition_from,
            rejected_recipe_kcal=rejected,
        )

    def _enrich(self, grounded: list[GroundedMeal]) -> None:
        """One batch details call: nutrition the search did not carry, tags,
        and the card image of a MATCHED dish — an approximate match's photo
        would show a different dish."""
        ids = list(dict.fromkeys(
            g.recipe_id for g in grounded
            if g.recipe_id and (g.state == MATCHED or g.borrows_nutrition)
        ))
        if not ids:
            return
        try:
            details = self.details_client.fetch_details(ids) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recipe details fetch failed (%d ids): %s", len(ids), exc)
            return
        for meal in grounded:
            rich = details.get(meal.recipe_id) if meal.recipe_id else None
            if rich is None:
                continue
            if not (meal.state == MATCHED or meal.borrows_nutrition):
                continue
            if not meal.nutrition and not meal.rejected_recipe_kcal:
                nutrition = rich.nutrition_dict()
                if nutrition and not plausible_recipe_kcal(nutrition, meal.slot):
                    meal.rejected_recipe_kcal = {
                        "title": meal.title_matched, "kcal": float(nutrition.get("kcal") or 0),
                    }
                elif nutrition:
                    meal.nutrition = dict(nutrition)
                    meal.nutrition_source = (
                        NUTRITION_FROM_RECIPE if meal.state == MATCHED else NUTRITION_FROM_CLOSEST
                    )
            if meal.state == MATCHED and meal.ingredients_source == FROM_RECIPE and not meal.tags and rich.tags:
                meal.tags = list(rich.tags)
            if meal.state == MATCHED and getattr(rich, "image_url", None):
                meal.image_url = rich.image_url

    def _estimate_missing(self, grounded: list[GroundedMeal]) -> None:
        """Calories for dishes no recipe gave any, ingredients for dishes
        nobody listed them for.

        1. One batched call to the small model writes a typical single serving,
           with quantities, for every such dish, keeping the member's own
           ingredients and amount when they gave them. It also guesses the
           calories. The serving is kept on the dish as
           ``typical_ingredients``; the variety count reads it when the dish
           has no other ingredient list.
        2. Each ingredient list goes to RecipeWrangler's profiler, which looks
           the lines up in food composition tables. Reliable figures (enough
           ingredients matched) become the dish's nutrition,
           ``typical_ingredients``.
        3. Otherwise — or when the profiled calories are more than double off
           the model's guess — the guess is used when it is a plausible
           number, ``model_estimate``, calories only. A discarded profiled
           figure is kept on the dish so its remark can say so.

        Grounded numbers first; a guess only when there is nothing else.
        """
        if self.estimator is None:
            return
        wanted: dict[str, list[GroundedMeal]] = {}
        for meal in grounded:
            if meal.nutrition and meal.ingredients:
                continue
            key = estimate_key(meal)
            if key:
                wanted.setdefault(key, []).append(meal)
        if not wanted:
            return

        def needs_calories(key: str) -> bool:
            return any(not meal.nutrition for meal in wanted[key])

        # sorted() is stable: text order within each group.
        keys = sorted(wanted, key=lambda key: not needs_calories(key))[:MAX_ESTIMATED_DISHES]
        if len(wanted) > len(keys):
            logger.info("Estimating only %d of %d dish(es) without nutrition", len(keys), len(wanted))
        asks = [dish_request(wanted[key][0]) for key in keys]
        try:
            estimates = self.estimator.estimate(asks) or {}
        except Exception as exc:  # noqa: BLE001 — an estimate is best-effort
            logger.warning("Ingredient estimate failed: %s", exc)
            return

        texts = {}
        for index, key in enumerate(keys):
            ingredients = (estimates.get(index) or {}).get("ingredients") or []
            for meal in wanted[key]:
                meal.typical_ingredients = [
                    {"name": name, "quantity": quantity} for quantity, name in ingredients
                ]
            if ingredients and needs_calories(key) and len(texts) < MAX_PROFILE_CALLS:
                texts[key] = typical_recipe_text(asks[index]["title"], ingredients)
        profiled = self._profile_all(texts)

        for index, key in enumerate(keys):
            if not needs_calories(key):
                continue
            found = profiled.get(key)
            guess = plausible_kcal((estimates.get(index) or {}).get("kcal"))
            discarded = None
            if found is not None and found.reliable and not disagrees(found.nutrition.get("kcal"), guess):
                nutrition, source = dict(found.nutrition), NUTRITION_TYPICAL
            else:
                if guess is None:
                    continue
                if found is not None and found.reliable:
                    discarded = float(found.nutrition.get("kcal") or 0)
                nutrition, source = {"kcal": guess}, NUTRITION_MODEL_ESTIMATE
            for meal in wanted[key]:
                if not meal.nutrition:
                    meal.nutrition = dict(nutrition)
                    meal.nutrition_source = source
                    meal.discarded_profile_kcal = discarded

    def _profile_all(self, texts: dict[str, str]) -> dict:
        """Profile every text concurrently; stop asking after the first timeout.

        Calls already in flight finish (within the timeout); calls not yet
        started are skipped once one has timed out.
        """
        profiler = getattr(self.profile_client, "profile_recipe", None)
        if profiler is None or not texts:
            return {}
        stalled = threading.Event()

        def profile(item):
            key, text = item
            if stalled.is_set():
                return key, None
            try:
                return key, profiler(text, timeout=PROFILE_TIMEOUT_SECONDS)
            except ProfilingTimeout as exc:
                stalled.set()
                logger.warning("Recipe profiling timed out; skipping the remaining dishes: %s", exc)
            except Exception as exc:  # noqa: BLE001 — an estimate is best-effort
                logger.warning("Recipe profiling failed for %.60r: %s", text, exc)
            return key, None

        workers = min(self.profile_workers, len(texts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return {
                key: found for key, found in pool.map(profile, texts.items()) if found is not None
            }
