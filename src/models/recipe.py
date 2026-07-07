"""
Recipe domain models shared across layers.

Layering rule: ``models`` has no imports from ``agents`` or ``services`` —
both of those import from here. Keep this module dependency-free.
"""

from dataclasses import dataclass


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
