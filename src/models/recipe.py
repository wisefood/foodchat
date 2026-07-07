"""
Recipe domain models shared across layers.

Layering rule: ``models`` has no imports from ``agents`` or ``services`` —
both of those import from here. Keep this module dependency-free.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CandidateRecipe:
    """One recipe candidate as returned by RecipeWrangler (text fields only).

    Nutrition/image enrichment is fetched separately (planned M4 batch
    endpoint) — candidates stay lightweight for LLM grading.
    """

    recipe_id: str
    title: str
    ingredients: str
    directions: str


# Slot name ("breakfast"/"lunch"/"dinner") → candidates for that slot.
CandidatesBySlot = dict[str, list[CandidateRecipe]]


@dataclass(frozen=True)
class RecipeEnrichment:
    """Per-recipe card data from RecipeWrangler's batch details endpoint (M4).

    Used to enrich plan payloads (nutrition chips, images in the UI) and to
    VERIFY edit directives ("lighter" ⇒ kcal comparison) — macros are
    per-serving from the nutrition store, None when unknown.
    """

    recipe_id: str
    title: str
    image_url: Optional[str] = None
    duration: Optional[float] = None
    kcal: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    nutri_score_label: Optional[str] = None
    tags: list[str] = None
    dish_types: list[str] = None
    allergens: list[str] = None

    def nutrition_dict(self) -> Optional[dict]:
        """Compact nutrition payload for API responses (None if all unknown)."""
        if self.kcal is None and self.protein_g is None:
            return None
        return {
            "kcal": self.kcal,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
            "nutri_score_label": self.nutri_score_label,
        }


@dataclass(frozen=True)
class ResolvedRecipe:
    """A recipe resolved by name/id from RecipeWrangler's detail endpoint.

    Carries the metadata needed to place and safety-check a user-requested
    anchor dish (seeded planning, M2): dish types for slot placement and
    allergens for hard-constraint checks.
    """

    recipe: CandidateRecipe
    dish_types: list[str]
    allergens: list[str]
    tags: list[str]


@dataclass(frozen=True)
class ScoredPlan:
    """One LLM-graded daily-plan combination."""

    breakfast: CandidateRecipe
    lunch: CandidateRecipe
    dinner: CandidateRecipe
    score: int
    reasoning: str

    @property
    def courses(self) -> list[CandidateRecipe]:
        return [self.breakfast, self.lunch, self.dinner]
