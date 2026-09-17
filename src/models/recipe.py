"""
Recipe domain models shared across layers.

Layering rule: ``models`` has no imports from ``agents`` or ``services`` —
both of those import from here. Keep this module dependency-free.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CandidateRecipe:
    """One recipe candidate as returned by RecipeWrangler.

    The text fields are what the LLM grader reads; the rest are optional
    metadata the candidate endpoints already return.

    `nutrition` and `nutri_score` were previously discarded on arrival —
    RecipeWrangler sends per-serving macros with every candidate, and FoodChat
    dropped them and then made a second `/recipes/details` round trip to fetch
    the same numbers back. Keeping them costs nothing, since they are already
    in the response, and lets a plan be checked against a calorie target
    without a second call.

    All default to None so every existing construction site keeps working and
    the grader's view of a candidate is unchanged.
    """

    recipe_id: str
    title: str
    ingredients: str
    directions: str
    # Per serving: calories, protein_g, carbs_g, fat_g. None when the recipe
    # has no stored profile — which is a real state, not an error.
    nutrition: Optional[dict] = None
    # Letter grade, from the v2 planning surface only.
    nutri_score: Optional[str] = None
    # Card image, also from the v2 planning surface. Carried for the same
    # reason as the macros: it is already in the response, and a plate composed
    # straight from a pool would otherwise render blank until something made a
    # second details call for a URL it had already been handed.
    image_url: Optional[str] = None


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
    # The recipe's OWN dietary tags, straight from the corpus. The difference
    # from `profile["diet"]` matters: that is what was asked for, this is what
    # arrived, and only the second one can verify the first.
    diet_tags: list[str] = None
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


# Display order for slots, mirroring `utils/planMeals.ts` in the UI. Anything
# unlisted sorts after, alphabetically, so an unknown slot is placed rather
# than dropped.
SLOT_ORDER: tuple[str, ...] = (
    "breakfast", "brunch", "lunch", "snack", "dinner", "supper",
    "side", "dessert", "drink",
)


def slot_sort_key(slot: str) -> int:
    """Position in eating order; everything unknown ties at the end.

    Returns the index alone, not `(index, name)`, so Python's stable sort keeps
    unknown slots in the order they were inserted. That matters for labels the
    planner builds rather than the corpus — "day 2 dinner (side)" is not in
    SLOT_ORDER, and sorting those alphabetically would put every day 2 dinner
    before its lunch.
    """
    name = str(slot or "").lower()
    try:
        return SLOT_ORDER.index(name)
    except ValueError:
        return len(SLOT_ORDER)


@dataclass(frozen=True)
class ScoredPlan:
    """One LLM-graded day of a plan.

    Held as `slots`, a mapping, rather than three named fields. The three
    fields were why the grader could only ever score breakfast/lunch/dinner —
    and therefore why a multi-plate or four-meal day could not be ranked at
    all, and the structured path shipped with no grading and no quality
    metrics.

    The three names remain as properties, and the keyword constructor still
    accepts them, because every existing caller and test addresses them that
    way and a day with those three slots is still the common case. What
    changes is that it is no longer the only case.
    """

    score: int
    reasoning: str
    slots: dict[str, CandidateRecipe] = field(default_factory=dict)

    def __init__(
        self,
        score: int,
        reasoning: str,
        slots: Optional[dict] = None,
        breakfast: Optional[CandidateRecipe] = None,
        lunch: Optional[CandidateRecipe] = None,
        dinner: Optional[CandidateRecipe] = None,
    ):
        merged = dict(slots or {})
        for name, course in (("breakfast", breakfast), ("lunch", lunch),
                             ("dinner", dinner)):
            if course is not None:
                merged.setdefault(name, course)
        # frozen dataclass: __setattr__ is blocked, so assign through object.
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "reasoning", reasoning)
        object.__setattr__(self, "slots", merged)

    @property
    def breakfast(self) -> Optional[CandidateRecipe]:
        return self.slots.get("breakfast")

    @property
    def lunch(self) -> Optional[CandidateRecipe]:
        return self.slots.get("lunch")

    @property
    def dinner(self) -> Optional[CandidateRecipe]:
        return self.slots.get("dinner")

    @property
    def slot_names(self) -> list[str]:
        """Every slot on this plan, in the order a person eats them."""
        return sorted(self.slots, key=slot_sort_key)

    @property
    def courses(self) -> list[CandidateRecipe]:
        """The recipes, in slot order.

        Callers use this to build a plan and to fetch enrichment. It used to
        return exactly three; it now returns however many the day has, which
        is what makes `refine_meal_plan` and `_compute_metrics` work on a day
        that is not three meals.
        """
        return [self.slots[name] for name in self.slot_names]

    @property
    def is_classic(self) -> bool:
        """The three-slot shape `MealPlan.from_courses` can store directly."""
        return self.slot_names == ["breakfast", "lunch", "dinner"]


# Below this share of ingredients matched in the composition tables, the
# profiler's figures describe part of a dish, not the dish.
MIN_PROFILE_COVERAGE = 0.6


@dataclass(frozen=True)
class ProfiledNutrition:
    """Per-serving nutrition RecipeWrangler's profiler computed for free text.

    ``nutrition`` holds ``kcal`` and whichever of ``protein_g``, ``carbs_g``
    and ``fat_g`` it reported, always per serving. ``coverage`` is the share of
    ingredients it matched in its composition tables.
    """

    nutrition: dict
    coverage: Optional[float] = None
    low_coverage: bool = False

    @property
    def reliable(self) -> bool:
        if self.low_coverage:
            return False
        return self.coverage is None or self.coverage >= MIN_PROFILE_COVERAGE
