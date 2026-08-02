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

from models.recipe import CandidateRecipe, CandidatesBySlot, RecipeEnrichment, ResolvedRecipe

logger = logging.getLogger(__name__)

RECIPEWRANGLER_API_URL = os.getenv("RECIPEWRANGLER_API_URL", "http://recipewrangler:8001")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("RECIPEWRANGLER_TIMEOUT", "60"))

# Retained for `fetch_details` batching and the response readers below. The
# candidate fetch that used to live here is gone: every slot pool now comes
# from `/api/v2/tools/plan_meals` via `services.plan_client`, which is the only
# surface that knows the corpus's annotations and its planning tier.
MEAL_SLOTS = ("breakfast", "lunch", "dinner")

# Dietary tags as stored on RecipeWrangler recipe nodes (confirmed against the API).
VALID_RW_DIET_TAGS = {
    "gluten_free", "high-protein", "low-carb", "low-fat",
    "pescatarian", "pescatarian_safe", "vegan", "dairy_free", "nut_free", "vegetarian",
}

# Common user-profile diet values → RecipeWrangler tag. Values mapped to None are
# non-restrictive labels (they would produce empty result sets if sent as filters).
DIET_TAG_MAP = {
    "gluten_free": "gluten_free",
    "gluten-free": "gluten_free",
    "high_protein": "high-protein",
    "high-protein": "high-protein",
    "low_carb": "low-carb",
    "low-carb": "low-carb",
    "low_fat": "low-fat",
    "low-fat": "low-fat",
    "pescatarian": "pescatarian",
    "pescatarian_safe": "pescatarian_safe",
    "vegan": "vegan",
    "dairy_free": "dairy_free",
    "dairy-free": "dairy_free",
    "nut_free": "nut_free",
    "nut-free": "nut_free",
    "vegetarian": "vegetarian",
    "omnivore": None,
    "mediterranean": None,
    "balanced": None,
    "healthy": None,
}


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


def normalize_diet_tags(diet) -> list[str]:
    """Convert user-profile diet values to valid RecipeWrangler diet tags.

    Unknown or non-restrictive values (e.g. 'omnivore', 'mediterranean') are
    dropped rather than forwarded, because RW treats diet tags as hard filters
    (ALL must match) and an unknown tag would return zero candidates.
    """
    raw = diet if isinstance(diet, list) else ([diet] if diet else [])
    tags: list[str] = []
    for d in raw:
        key = str(d).lower().strip()
        if key in DIET_TAG_MAP:
            mapped = DIET_TAG_MAP[key]
            if mapped is not None:
                tags.append(mapped)
        elif key in VALID_RW_DIET_TAGS:
            tags.append(key)
        else:
            logger.warning("Dropping unrecognized diet tag %r — not in RecipeWrangler schema", d)
    return tags


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
        from services.plan_client import PLANNER

        cuisines, _ = self.split_cuisines(profile.get("food_likes") or [])
        try:
            envelope = PLANNER.plan_meals(
                days=1,
                slots=(meal_type,),
                count_per_slot=limit,
                allergens=profile.get("allergies") or [],
                diet=profile.get("diet") or [],
                cuisines=cuisines,
                exclude_ingredients=profile.get("food_dislikes") or [],
                exclude_recipe_ids=list(exclude_ids),
                favorite_recipe_ids=profile.get("favorite_recipe_ids") or [],
                min_nutri_score=profile.get("min_nutri_score"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("slot candidate fetch failed for %s: %s", meal_type, exc)
            return []

        return PLANNER.to_candidates(
            envelope, allergens=profile.get("allergies") or []
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
                allergens=[str(a).lower() for a in (r.get("allergens") or [])],
            )
        return enriched


# Module-level singleton — the client is stateless, so sharing is safe.
CANDIDATES = RecipeCandidatesClient()
