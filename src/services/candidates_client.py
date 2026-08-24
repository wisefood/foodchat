"""
RecipeWrangler candidate-recipe client.

This is FoodChat's ONLY recipe source. It wraps RecipeWrangler's
``POST /api/v1/recipes/foodchat_candidates`` endpoint, which applies the
hard constraints server-side (allergen exclusion via the FoodOn taxonomy,
diet-tag matching, ingredient include/exclude, recipe-id exclusion) and
returns candidates already grouped by meal slot (breakfast/lunch/dinner).

Upstream contract: RecipeWrangler-Backend
``src/recipe_wrangler/api/routers/recipes.py`` (``FoodChatRequest`` model).
Downstream consumers: ``services.planning_pipeline`` (daily plans) and
``services.weekly_planner.action_adapter`` (weekly plans).

Replaces the pre-M0 ``KG_neo4j`` module, which carried a direct-Neo4j
fallback (``RECIPE_SOURCE`` switch) that is no longer deployed anywhere.
"""

import logging
import os
import re
from typing import Optional

import httpx

from models.recipe import CandidateRecipe, RecipeEnrichment, ResolvedRecipe

logger = logging.getLogger(__name__)

RECIPEWRANGLER_API_URL = os.getenv("RECIPEWRANGLER_API_URL", "http://recipewrangler:8001")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("RECIPEWRANGLER_TIMEOUT", "60"))

# Retained for `fetch_details` batching and the response readers below. The
# candidate fetch that used to live here is gone: every slot pool now comes
# from `/api/v2/tools/plan_meals` via `services.plan_client`, which is the only
# surface that knows the corpus's annotations and its planning tier.
MEAL_SLOTS = ("breakfast", "lunch", "dinner")

# Dietary tags actually carried by RecipeWrangler recipes, censused against the
# corpus dump (dumps/*/elastic-recipes.ndjson.gz, n=4500) rather than assumed:
#
#   nut_free 3757 · dairy_free 2815 · pescatarian_safe 2717 · gluten_free 2605
#   vegetarian_or_vegan 2437 · vegetarian 2436 · pescatarian 2141 · vegan 1336
#   gluten_free_option 1316
#
# RW ANDs diet tags and never relaxes them, so a tag no recipe carries is not a
# narrow filter — it is a guaranteed-empty one.
VALID_RW_DIET_TAGS = {
    "gluten_free", "gluten_free_option", "pescatarian", "pescatarian_safe",
    "vegan", "vegetarian", "vegetarian_or_vegan", "dairy_free", "nut_free",
}

# Nutrition CLAIMS, not diets. "low-carb", "low-fat" and "high-protein" were in
# the valid-tag set and mapped straight through as hard diet filters — but the
# census above finds them on ZERO recipes (they live in the separate `tags`
# claim field). So "I want a low-carb week" filtered every slot to nothing and
# came back as "I couldn't find enough recipes", which is how a stated
# preference became an outage. They are routed to the grader as soft signals
# instead, and become real numeric targets when RW grows nutrition_targets.
NUTRITION_CLAIM_TAGS = {"low-carb", "low-fat", "high-protein"}

# Common user-profile diet values → RecipeWrangler tag. Values mapped to None are
# non-restrictive labels (they would produce empty result sets if sent as filters).
DIET_TAG_MAP = {
    "gluten_free": "gluten_free",
    "gluten-free": "gluten_free",
    "gluten_free_option": "gluten_free_option",
    "pescatarian": "pescatarian",
    "pescatarian_safe": "pescatarian_safe",
    "vegan": "vegan",
    "dairy_free": "dairy_free",
    "dairy-free": "dairy_free",
    "nut_free": "nut_free",
    "nut-free": "nut_free",
    "vegetarian": "vegetarian",
    "vegetarian_or_vegan": "vegetarian_or_vegan",
    # Claims — deliberately not diet filters. See NUTRITION_CLAIM_TAGS.
    "high_protein": None,
    "high-protein": None,
    "low_carb": None,
    "low-carb": None,
    "low_fat": None,
    "low-fat": None,
    # Non-restrictive labels.
    "omnivore": None,
    "mediterranean": None,
    "balanced": None,
    "healthy": None,
    "flexitarian": None,
}


def split_diet_intent(tags) -> tuple[list[str], list[str]]:
    """Split extracted diet words into (hard filters, soft nutrition claims).

    The extractor is allowed to say "low-carb" — it is a real thing a member
    wants. What it must not do is become a `diet` filter, because no recipe
    carries that tag. The claims come back separately so the caller can carry
    them as grader signals rather than dropping them silently.
    """
    filterable: list[str] = []
    claims: list[str] = []
    for tag in tags or ():
        key = str(tag).strip().lower()
        if not key:
            continue
        mapped = DIET_TAG_MAP.get(key, key if key in VALID_RW_DIET_TAGS else None)
        canonical = key.replace("_", "-")
        if mapped:
            if mapped not in filterable:
                filterable.append(mapped)
        elif canonical in NUTRITION_CLAIM_TAGS and canonical not in claims:
            claims.append(canonical)
    return filterable, claims


# --------------------------------------------------------------------------- #
# Defense-in-depth allergen screening (added after a live incident where the
# recipe graph tagged an almond dish "nut_free" with no allergen edges — see
# CHANGES.md). RecipeWrangler's hard filters remain the primary gate; this is
# a client-side ingredient-text backstop so poisoned tags can't reach a plan.
# --------------------------------------------------------------------------- #

ALLERGEN_SYNONYMS = {
    "tree nuts": ["almond", "walnut", "cashew", "pecan", "hazelnut", "pistachio",
                  "macadamia", "brazil nut", "pine nut", "chestnut"],
    "nuts": ["almond", "walnut", "cashew", "pecan", "hazelnut", "pistachio",
             "macadamia", "brazil nut", "pine nut", "peanut"],
    "peanuts": ["peanut"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "mussel", "oyster",
                  "scallop", "clam", "crayfish", "squid", "octopus"],
    "fish": ["salmon", "tuna", "cod", "haddock", "trout", "sardine", "anchovy",
             "mackerel", "halibut", "sea bass", "tilapia"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "ghee"],
    "lactose": ["milk", "cheese", "cream", "yogurt", "yoghurt"],
    "eggs": ["egg"],
    "gluten": ["wheat", "flour", "barley", "rye", "semolina", "couscous"],
    "soy": ["soy", "soya", "tofu", "edamame"],
    "sesame": ["sesame", "tahini"],
}

# The gateway's dietary_groups enum carries free-from values that RecipeWrangler
# has no diet tag for, so they were dropped and nothing filtered on them — while
# the ledger still announced them as satisfied hard constraints. They cannot
# become RW filters, but they CAN reach the client-side allergen backstop, which
# is the same defence that exists because the corpus has tagged almond dishes
# `nut_free` in production. Mapped to the allergen names above rather than new
# term lists, so there is one place to maintain.
FREE_FROM_TO_ALLERGEN = {
    "peanut_free": "peanuts",
    "nut_free": "tree nuts",
    "tree_nut_free": "tree nuts",
    "egg_free": "eggs",
    "dairy_free": "dairy",
    "lactose_free": "lactose",
    "gluten_free": "gluten",
    "soy_free": "soy",
    "sesame_free": "sesame",
    "shellfish_free": "shellfish",
    "fish_free": "fish",
}


def free_from_allergens(diet) -> list[str]:
    """Allergen names implied by free-from diet slugs on a profile.

    Returned so callers can union them into `allergies` before screening: a
    member who selected `peanut_free` gets the peanut backstop even though no
    RW diet tag exists for it.
    """
    raw = diet if isinstance(diet, list) else ([diet] if diet else [])
    out: list[str] = []
    for value in raw:
        allergen = FREE_FROM_TO_ALLERGEN.get(str(value).strip().lower())
        if allergen and allergen not in out:
            out.append(allergen)
    return out


def _allergen_terms(allergies: list[str]) -> list[str]:
    """Expand profile allergen names into matchable ingredient terms."""
    terms: list[str] = []
    for allergen in allergies or []:
        key = str(allergen).strip().lower()
        if not key:
            continue
        terms.append(key)
        terms.extend(ALLERGEN_SYNONYMS.get(key, []))
    return list(dict.fromkeys(terms))


def allergen_conflict(text: str, allergies: list[str]) -> Optional[str]:
    """First allergen term found in the text (word-boundary match), or None."""
    haystack = (text or "").lower()
    for term in _allergen_terms(allergies):
        if re.search(r"\b" + re.escape(term) + r"s?\b", haystack):
            return term
    return None


# Values that are deliberately not filters: they describe an absence of
# restriction, so forwarding one would empty every slot for no reason.
NON_RESTRICTIVE = {"omnivore", "mediterranean", "balanced", "healthy", "flexitarian"}


def screening_allergens(profile: dict) -> list[str]:
    """Allergen names to screen a plate against: stated allergies PLUS the ones
    implied by free-from diet slugs.

    A member who set `peanut_free` in their dietary groups had no filter and no
    backstop — the slug is not an RW diet tag and not an allergy entry. Unioning
    here means every call site that already screens gets the cover, in one
    place, without each one learning about diet slugs.
    """
    allergies = profile.get("allergies") or []
    allergies = [allergies] if isinstance(allergies, str) else list(allergies)
    implied = free_from_allergens(profile.get("diet"))
    known = {str(a).strip().lower() for a in allergies}
    return list(allergies) + [a for a in implied if a not in known]


def classify_diet_tags(diet) -> tuple[list[str], list[str]]:
    """Split profile diet values into (filterable, unsupported).

    The second list is the point. 26 of the gateway's 37 dietary groups have no
    RecipeWrangler diet tag — `peanut_free`, `halal`, `kosher`, `keto`,
    `low_sodium` and the rest — and they used to be dropped with a log line
    while `transparency.constraints_ledger` still rendered every raw profile
    diet value as a hard constraint with status "satisfied". A member who
    selected `peanut_free` was shown a plan asserting a peanut-free guarantee
    that nothing had enforced.

    Nothing here can invent a filter that does not exist upstream. What it can
    do is refuse to pretend: the caller gets the unsupported values back and
    says so, and free-from slugs additionally reach the allergen backstop via
    `free_from_allergens`.
    """
    raw = diet if isinstance(diet, list) else ([diet] if diet else [])
    tags: list[str] = []
    unsupported: list[str] = []
    for d in raw:
        key = str(d).lower().strip()
        if not key:
            continue
        if key in DIET_TAG_MAP:
            mapped = DIET_TAG_MAP[key]
            if mapped is not None:
                if mapped not in tags:
                    tags.append(mapped)
            elif key not in NON_RESTRICTIVE and key not in unsupported:
                # Mapped to None but still a real restriction the member chose
                # (the nutrition claims) — not filterable, not nothing.
                unsupported.append(key)
        elif key in VALID_RW_DIET_TAGS:
            if key not in tags:
                tags.append(key)
        elif key not in NON_RESTRICTIVE and key not in unsupported:
            unsupported.append(key)
    if unsupported:
        logger.warning(
            "Diet values with no RecipeWrangler filter: %s — reported to the "
            "member as unenforced rather than dropped", unsupported,
        )
    return tags, unsupported


def normalize_diet_tags(diet) -> list[str]:
    """The filterable diet tags only. See `classify_diet_tags` for the rest."""
    return classify_diet_tags(diet)[0]


def effective_diet(profile: dict) -> list[str]:
    """The diet to filter on: what the member SAID plus what their profile says.

    A diet stated in chat outranks a stored setting, and must not be lost to
    it. `profile["_diet_tags"]` is the transient stash chat_service fills from
    `PlanningState.diet_tags` (same underscore convention as `_pantry`).

    Before this existed every fetch site read `profile["diet"]` alone, so "I
    need something vegetarian" reached the grader as prose over a pool that had
    never been filtered — and the reply then blamed `diet: omnivore`, a value
    `normalize_diet_tags` had already dropped as non-restrictive.

    Union rather than override: a vegetarian request from a member whose profile
    says `gluten_free` must satisfy both. `omnivore` drops out on its own — it
    is not in the tag map, so it was never a constraint to override.
    """
    stated = list(profile.get("_diet_tags") or [])
    stored = profile.get("diet") or []
    stored = [stored] if isinstance(stored, str) else list(stored)
    return normalize_diet_tags(stated + stored)


class RecipeCandidatesClient:
    """Thin, stateless HTTP client for RecipeWrangler candidate generation."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or RECIPEWRANGLER_API_URL).rstrip("/")

    # Vocabulary cache. RecipeWrangler owns these lists and they change when the
    # corpus is re-annotated, so hardcoding them here would drift silently — a
    # cuisine added upstream would keep being sent as an ingredient forever.
    _vocab_cache: Optional[dict] = None

    def vocabularies(self) -> dict:
        """Closed vocabularies from RecipeWrangler's tool manifest.

        Cached for the process lifetime. Failure returns an empty dict rather
        than raising: every caller degrades to "treat nothing as a cuisine",
        which is exactly the behaviour that existed before this was added.
        """
        if RecipeCandidatesClient._vocab_cache is not None:
            return RecipeCandidatesClient._vocab_cache
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/v2/tools")
                response.raise_for_status()
                vocab = response.json().get("vocabularies") or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load RecipeWrangler vocabularies: %s", exc)
            vocab = {}
        RecipeCandidatesClient._vocab_cache = vocab
        return vocab

    # The four facet families RecipeWrangler annotates and relaxes. Cuisine was
    # the only one FoodChat ever sent, even though its own client declared all
    # four and the persona promised them — so "something comforting", "light and
    # fresh" and "more vegetables" reached the grader as prose over a pool that
    # had never been shaped by them.
    FACET_FAMILIES = ("cuisines", "moods", "flavor_profiles", "food_groups")

    def split_preferences(self, words: list[str]) -> dict[str, list[str]]:
        """Sort free-text preference words into the facet family each belongs to.

        Returns ``{"cuisines": [...], "moods": [...], "flavor_profiles": [...],
        "food_groups": [...], "ingredients": [...]}`` — everything unrecognised
        falls through to ``ingredients``, which is the pre-existing behaviour.

        Generalises `split_cuisines`, and keeps its two properties: the
        vocabulary is fetched LIVE from RecipeWrangler's manifest, so a value
        that only becomes a recognised mood next month starts working then; and
        the sort happens at READ time, so existing profiles are fixed with no
        migration.

        A word in two families goes to the first that claims it, in
        FACET_FAMILIES order — cuisine is the most specific signal and the one
        RecipeWrangler relaxes last.
        """
        vocab = self.vocabularies()
        known = {
            family: {str(v).lower() for v in (vocab.get(family) or [])}
            for family in self.FACET_FAMILIES
        }
        out: dict[str, list[str]] = {f: [] for f in self.FACET_FAMILIES}
        out["ingredients"] = []

        for raw in words or []:
            value = str(raw or "").strip().lower()
            if not value:
                continue
            slug = value.replace("-", "_").replace(" ", "_")
            for family in self.FACET_FAMILIES:
                if slug in known[family]:
                    if slug not in out[family]:
                        out[family].append(slug)
                    break
            else:
                out["ingredients"].append(raw)
        return out

    def split_cuisines(self, likes: list[str]) -> tuple[list[str], list[str]]:
        """Separate cuisines from ingredients in a member's `food_likes`.

        The profile stores both in one list — `apply_memory` folds a `cuisine`
        memory into `food_likes` alongside a `like` — so "greek" arrived at
        RecipeWrangler as an ingredient to search for, matching nothing, while
        the cuisine filter it should have driven went unused.

        Splitting here rather than at write time means existing profiles are
        fixed too, without a migration, and a member whose stored "thai" only
        becomes a recognised cuisine next month starts benefiting then.

        Returns ``(cuisines, remaining_ingredients)``.
        """
        known = {c.lower() for c in (self.vocabularies().get("cuisines") or [])}
        if not known:
            return [], list(likes or [])

        cuisines, ingredients = [], []
        for raw in likes or []:
            value = str(raw or "").strip().lower()
            if not value:
                continue
            slug = value.replace("-", "_").replace(" ", "_")
            if slug in known:
                cuisines.append(slug)
            else:
                ingredients.append(raw)
        return cuisines, ingredients

    def slot_candidates(
        self, profile: dict, meal_type: str, exclude_ids: list[str], limit: int = 8
    ) -> list[CandidateRecipe]:
        """Candidates for one slot, from `/api/v2/tools/plan_meals`.

        Lives on this client so the services that need recipes keep a single
        dependency, even though the pool now comes from the v2 planning surface
        rather than the v1 candidate endpoint this class was built around.

        Only the requested slot is asked for: a single-slot swap does not touch
        the other meals, and asking for them would spend the exclusion budget on
        recipes nobody will look at.
        """
        from services import intent_facets
        from services.plan_client import PLANNER

        cuisines, _ = self.split_cuisines(profile.get("food_likes") or [])
        try:
            envelope = PLANNER.plan_meals(
                days=1,
                slots=(meal_type,),
                count_per_slot=limit,
                allergens=screening_allergens(profile),
                # Normalised: an unknown tag ANDs to zero candidates.
                diet=effective_diet(profile),
                **intent_facets.facet_kwargs(profile, cuisines),
                exclude_ingredients=profile.get("food_dislikes") or [],
                exclude_recipe_ids=list(exclude_ids),
                favorite_recipe_ids=profile.get("favorite_recipe_ids") or [],
                min_nutri_score=profile.get("min_nutri_score"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("slot candidate fetch failed for %s: %s", meal_type, exc)
            return []

        return PLANNER.to_candidates(
            envelope, allergens=screening_allergens(profile)
        ).get(meal_type, [])

    def autocomplete(self, name: str, limit: int = 5) -> list[tuple[str, str]]:
        """Resolve a dish name to (recipe_id, title) suggestions.

        Wraps ``GET /api/v1/recipes/autocomplete`` (ES prefix search on
        titles). Returns [] on any failure — resolution is best-effort.
        """
        query = (name or "").strip()
        if len(query) < 2:
            return []
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = client.get(
                    f"{self.base_url}/api/v1/recipes/autocomplete",
                    params={"q": query, "limit": limit},
                )
                response.raise_for_status()
                suggestions = response.json().get("suggestions", {})
        except httpx.HTTPError as e:
            logger.warning("Autocomplete failed for %r: %s", name, e)
            return []
        return [(rid, title) for rid, title in suggestions.items()]

    def fetch_recipe(self, recipe_id: str) -> Optional[ResolvedRecipe]:
        """Fetch full recipe detail; None on failure (best-effort)."""
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = client.get(f"{self.base_url}/api/v1/recipes/{recipe_id}")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            logger.warning("Recipe detail fetch failed for %s: %s", recipe_id, e)
            return None

        # Detail ingredients are structured dicts; flatten to the text form
        # the rest of the pipeline (grading, plan storage) works with.
        ingredient_names = []
        for item in data.get("ingredients") or []:
            if isinstance(item, dict):
                text = item.get("name") or item.get("original") or ""
            else:
                text = str(item)
            if text:
                ingredient_names.append(text)

        return ResolvedRecipe(
            recipe=CandidateRecipe(
                recipe_id=str(data.get("recipe_id") or recipe_id),
                title=data.get("title") or "",
                ingredients=", ".join(ingredient_names),
                directions="\n".join(data.get("instructions") or []),
            ),
            dish_types=[str(d).lower() for d in (data.get("dish_types") or [])],
            allergens=[str(a).lower() for a in (data.get("allergens") or [])],
            tags=[str(t).lower() for t in (data.get("tags") or [])],
        )


    def fetch_details(self, recipe_ids: list[str]) -> dict[str, RecipeEnrichment]:
        """Batch nutrition/image/tag details (M4 enrichment + edit predicates).

        Wraps ``POST /api/v1/recipes/details``. Best-effort: failures return
        {} and unknown ids are simply absent — enrichment must never block a
        plan response.
        """
        ids = [str(r) for r in recipe_ids if r]
        if not ids:
            return {}
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{self.base_url}/api/v1/recipes/details",
                    json={"recipe_ids": ids[:30]},
                )
                response.raise_for_status()
                results = response.json().get("results", {})
        except httpx.HTTPError as e:
            logger.warning("Batch details fetch failed (%d ids): %s", len(ids), e)
            return {}

        enriched: dict[str, RecipeEnrichment] = {}
        for rid, r in results.items():
            enriched[str(rid)] = RecipeEnrichment(
                recipe_id=str(r.get("recipe_id") or rid),
                title=r.get("title") or "",
                image_url=r.get("image_url"),
                duration=r.get("duration"),
                kcal=r.get("kcal_per_serving"),
                protein_g=r.get("protein_g_per_serving"),
                carbs_g=r.get("carbs_g_per_serving"),
                fat_g=r.get("fat_g_per_serving"),
                nutri_score_label=r.get("nutri_score_label"),
                tags=[str(t).lower() for t in (r.get("tags") or [])],
                dish_types=[str(d).lower() for d in (r.get("dish_types") or [])],
                diet_tags=[str(d).lower() for d in (r.get("diet_tags") or [])],
                allergens=[str(a).lower() for a in (r.get("allergens") or [])],
            )
        return enriched


# Module-level singleton — the client is stateless, so sharing is safe.
CANDIDATES = RecipeCandidatesClient()
