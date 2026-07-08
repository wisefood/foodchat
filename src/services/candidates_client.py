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

    def fetch_candidates(
        self,
        allergens: Optional[list[str]] = None,
        diet=None,
        include_ingredients: Optional[list[str]] = None,
        exclude_ingredients: Optional[list[str]] = None,
        exclude_recipe_ids: Optional[list[str]] = None,
        favorite_recipe_ids: Optional[list[str]] = None,
        nutrition_profile: Optional[dict] = None,
        limit_per_slot: int = 5,
        randomize: bool = True,
    ) -> CandidatesBySlot:
        """Fetch candidate recipes grouped by meal slot.

        ``favorite_recipe_ids`` is a soft ranking boost applied server-side —
        favorites float to the top of their slot but hard filters (allergens,
        diet, exclusions) always win.

        Raises ``httpx.HTTPError`` on transport/HTTP failures — callers decide
        how to degrade (the chat layer apologizes; it must never 500 silently).
        """
        payload = {
            "user_profile": {
                "allergies": allergens or [],
                "diet": normalize_diet_tags(diet),
            },
            "constraints": {
                "include_ingredients": include_ingredients or [],
                "exclude_ingredients": exclude_ingredients or [],
                "exclude_recipe_ids": exclude_recipe_ids or [],
                "favorite_recipe_ids": favorite_recipe_ids or [],
                "nutrition_profile": nutrition_profile,
            },
            "quotas": {slot: limit_per_slot for slot in MEAL_SLOTS},
            "randomize": randomize,
        }

        logger.info(
            "Fetching RecipeWrangler candidates (limit=%d/slot, %d allergens, diet=%s)",
            limit_per_slot, len(payload["user_profile"]["allergies"]),
            payload["user_profile"]["diet"],
        )
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{self.base_url}/api/v1/recipes/foodchat_candidates", json=payload
            )
            response.raise_for_status()
            data = response.json()

        slot_results = data.get("results", {}) if isinstance(data, dict) else {}
        candidates: CandidatesBySlot = {slot: [] for slot in MEAL_SLOTS}
        dropped = 0
        for slot in MEAL_SLOTS:
            for r in slot_results.get(slot, []):
                candidate = CandidateRecipe(
                    recipe_id=str(r.get("recipe_id", "")),
                    title=r.get("title", ""),
                    ingredients=r.get("ingredients", ""),
                    directions=r.get("directions", ""),
                )
                # Backstop: never trust upstream tags with safety data.
                conflict = allergen_conflict(
                    f"{candidate.title} {candidate.ingredients}", allergens or []
                )
                if conflict:
                    dropped += 1
                    logger.warning(
                        "Allergen backstop dropped %r (%s) — contains %r despite "
                        "passing upstream filters", candidate.title, slot, conflict,
                    )
                    continue
                candidates[slot].append(candidate)
            logger.info("Got %d candidates for %s", len(candidates[slot]), slot)
        if dropped:
            logger.warning("Allergen backstop dropped %d candidate(s) in total — "
                           "upstream tagging needs attention", dropped)
        return candidates

    # ------------------------------------------------------------------ #
    # Recipe resolution (seeded planning, M2)                              #
    # ------------------------------------------------------------------ #

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
