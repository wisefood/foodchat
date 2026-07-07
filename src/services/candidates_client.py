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
from typing import Optional

import httpx

from models.recipe import CandidateRecipe, CandidatesBySlot

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
        nutrition_profile: Optional[dict] = None,
        limit_per_slot: int = 5,
        randomize: bool = True,
    ) -> CandidatesBySlot:
        """Fetch candidate recipes grouped by meal slot.

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
        for slot in MEAL_SLOTS:
            for r in slot_results.get(slot, []):
                candidates[slot].append(
                    CandidateRecipe(
                        recipe_id=str(r.get("recipe_id", "")),
                        title=r.get("title", ""),
                        ingredients=r.get("ingredients", ""),
                        directions=r.get("directions", ""),
                    )
                )
            logger.info("Got %d candidates for %s", len(candidates[slot]), slot)
        return candidates


# Module-level singleton — the client is stateless, so sharing is safe.
CANDIDATES = RecipeCandidatesClient()
