"""
Pasted meal plans — what a member wrote, as data (plan scorer, steps 1–3).

A member can paste a daily or weekly plan they wrote themselves and have
FoodChat score it (``score_plan`` intent). These are the shapes that plan
passes through before any metric runs:

    PastedMeal    one dish the member listed, in their own words
    PastedDay     the dishes under one day heading (a daily plan has one)
    PastedPlan    the whole parse: plan type, days, and every line nobody
                  could place — returned to the member verbatim, never dropped
    GroundedMeal  a PastedMeal after RecipeWrangler lookup, in exactly one of
                  three states: matched | approximate | unresolved

Nothing here is stored on a canvas. A pasted plan is scored, not adopted.

Layering rule: like every module in ``models``, no imports from ``agents``
or ``services``. ``to_dict``/``from_dict`` exist because a parsed plan rides
inside the persisted clarification state when FoodChat has to ask the member
how many days the text covers.
"""

from dataclasses import dataclass, field
from typing import Optional

MAIN_SLOTS = ("breakfast", "lunch", "dinner")
# `other` is brunch, a drink, anything the member named that is not one of
# the three meals or a snack. Kept rather than coerced: calling brunch a
# breakfast would be a claim about the member's day nobody made.
SLOTS = MAIN_SLOTS + ("snack", "other")
SLOT_INDEX = {slot: index for index, slot in enumerate(SLOTS)}

PLAN_TYPES = ("daily", "weekly")
MAX_DAYS = 7

# Grounding states. The response says which one each dish is in, because they
# make different claims: a matched dish is scored as the catalogue recipe, an
# unresolved one only from what the member wrote.
MATCHED = "matched"
APPROXIMATE = "approximate"
UNRESOLVED = "unresolved"

# Where the ingredient text a metric reads came from. Every LLM judge is shown
# this label, so a grounded list is never presented as the member's own words.
FROM_RECIPE = "recipe"
AS_WRITTEN = "as_written"
UNKNOWN = "unknown"
# Allergen evidence found only in the closest catalogue recipe of an
# APPROXIMATE dish — a possibility to warn about, not a fact about the dish.
CLOSEST_RECIPE = "closest_recipe"

# Where a dish's nutrition figures come from, most to least grounded:
#   recipe               the matched catalogue recipe
#   closest_recipe       a more generically named recipe ("Lasagna")
#   typical_ingredients  a small model wrote a typical serving's ingredients and
#                        RecipeWrangler's profiler looked them up in food
#                        composition tables
#   model_estimate       the small model's own calorie guess, used only when
#                        the profiler could not give reliable figures
# Neither estimate is ever presented as a measurement of a known recipe.
NUTRITION_FROM_RECIPE = "recipe"
NUTRITION_FROM_CLOSEST = "closest_recipe"
NUTRITION_TYPICAL = "typical_ingredients"
NUTRITION_MODEL_ESTIMATE = "model_estimate"
NUTRITION_NONE = ""


@dataclass
class PastedMeal:
    slot: str                                  # one of SLOTS
    title: str
    ingredients: Optional[str] = None          # only when the member wrote them
    quantity_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "title": self.title,
            "ingredients": self.ingredients,
            "quantity_note": self.quantity_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PastedMeal":
        return cls(
            slot=str(data.get("slot") or "other"),
            title=str(data.get("title") or ""),
            ingredients=data.get("ingredients"),
            quantity_note=data.get("quantity_note"),
        )


@dataclass
class PastedDay:
    day: Optional[int] = None                  # 1 = Monday / "Day 1" … 7
    label: Optional[str] = None                # the heading as written
    meals: list = field(default_factory=list)  # list[PastedMeal]

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "label": self.label,
            "meals": [meal.to_dict() for meal in self.meals],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PastedDay":
        day = data.get("day")
        return cls(
            day=int(day) if isinstance(day, int) else None,
            label=data.get("label"),
            meals=[PastedMeal.from_dict(m) for m in data.get("meals") or []],
        )


@dataclass
class PastedPlan:
    plan_type: str = "daily"                   # one of PLAN_TYPES
    days: list = field(default_factory=list)   # list[PastedDay]
    unparsed: list = field(default_factory=list)  # verbatim lines
    warnings: list = field(default_factory=list)  # what the parse had to assume
    # What the member wrote around the listing — preamble and questions. Their
    # own words about the plan ("trying to eat less meat"), for the fit judge.
    notes: list = field(default_factory=list)

    @property
    def meals(self) -> list:
        return [meal for day in self.days for meal in day.meals]

    @property
    def meal_count(self) -> int:
        return len(self.meals)

    @property
    def main_meal_count(self) -> int:
        return sum(1 for meal in self.meals if meal.slot in MAIN_SLOTS)

    @property
    def is_empty(self) -> bool:
        return self.meal_count == 0

    def to_dict(self) -> dict:
        return {
            "plan_type": self.plan_type,
            "days": [day.to_dict() for day in self.days],
            "unparsed": list(self.unparsed),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PastedPlan":
        plan_type = data.get("plan_type")
        return cls(
            plan_type=plan_type if plan_type in PLAN_TYPES else "daily",
            days=[PastedDay.from_dict(d) for d in data.get("days") or []],
            unparsed=[str(line) for line in data.get("unparsed") or []],
            warnings=[str(w) for w in data.get("warnings") or []],
            notes=[str(n) for n in data.get("notes") or []],
        )


@dataclass
class GroundedMeal:
    """One pasted dish after lookup, carrying only what is actually known."""

    day: Optional[int]
    slot: str
    title_given: str
    state: str                                 # MATCHED | APPROXIMATE | UNRESOLVED
    recipe_id: Optional[str] = None            # the catalogue recipe, when one was used
    title_matched: Optional[str] = None
    similarity: float = 0.0
    ingredients: str = ""                      # the text metrics will read
    ingredients_source: str = UNKNOWN          # FROM_RECIPE | AS_WRITTEN | UNKNOWN
    nutrition: Optional[dict] = None
    tags: list = field(default_factory=list)
    dish_types: list = field(default_factory=list)
    # [{"allergen": "peanuts", "evidence": AS_WRITTEN | FROM_RECIPE}]
    allergen_conflicts: list = field(default_factory=list)
    quantity_note: Optional[str] = None
    image_url: Optional[str] = None            # MATCHED dishes only
    # An APPROXIMATE dish whose nutrition figures are its closest recipe's —
    # only when that recipe's name is a more generic form of the member's.
    borrows_nutrition: bool = False
    # recipe | closest_recipe | typical_ingredients | model_estimate | "" (nothing known)
    nutrition_source: str = NUTRITION_NONE
    # A small model's typical single serving, [{"name", "quantity"}], written
    # for dishes missing calories or ingredients. A guess: it feeds the calorie
    # estimate and, when ``ingredients`` is empty, the variety count — never
    # the allergy, diet or dislike checks, which read ``ingredients`` only.
    typical_ingredients: list = field(default_factory=list)
    # A catalogue recipe's calories too low to be a serving of this meal
    # ("Quick Chili", 12 kcal), set aside rather than used: {"title", "kcal"}.
    rejected_recipe_kcal: Optional[dict] = None
    # A typical serving's profiled calories, not used because they were more
    # than double off the model's own guess (pho at 2,025 kcal).
    discarded_profile_kcal: Optional[float] = None

    @property
    def guessed_ingredients(self) -> bool:
        """The variety count reads the typical serving: nothing better is known."""
        return not self.ingredients and bool(self.typical_ingredients)

    @property
    def variety_ingredients(self) -> str:
        """The ingredient text the variety counts read."""
        if self.guessed_ingredients:
            return ", ".join(str(item.get("name") or "") for item in self.typical_ingredients)
        return self.ingredients

    @property
    def kcal(self) -> Optional[int]:
        """Calories per serving, rounded; None when unknown."""
        try:
            value = float((self.nutrition or {}).get("kcal") or 0)
        except (TypeError, ValueError):
            return None
        return round(value) if value > 0 else None

    def guess_remarks(self) -> list[str]:
        """What in this row is a guess rather than a recipe or the member's words."""
        remarks = []
        if self.rejected_recipe_kcal:
            remarks.append(
                f"The catalogue recipe “{self.rejected_recipe_kcal['title']}” lists "
                f"{self.rejected_recipe_kcal['kcal']:.0f} kcal a serving, which is too little "
                "for this meal to be right, so that figure was not used."
            )
        if self.nutrition_source == NUTRITION_TYPICAL:
            remarks.append(
                "The calories are an estimate: no recipe gave them, so they were computed "
                "from a typical serving's ingredients, not from this exact dish."
            )
        elif self.nutrition_source == NUTRITION_MODEL_ESTIMATE and self.discarded_profile_kcal:
            remarks.append(
                "The calories are a rough guess by a language model: no recipe gave them, and "
                f"the composition-table figure for a typical serving "
                f"({self.discarded_profile_kcal:,.0f} kcal) was more than double off that guess, "
                "so neither is certain."
            )
        elif self.nutrition_source == NUTRITION_MODEL_ESTIMATE:
            remarks.append(
                "The calories are a rough guess by a language model: no recipe gave them "
                "and a typical serving could not be profiled."
            )
        if self.guessed_ingredients:
            remarks.append(
                "The ingredients are a guess at a typical serving, not what you wrote or a "
                "recipe. They count towards food variety but are not checked against your "
                "allergies, diet or dislikes."
            )
        return remarks

    def to_row(self) -> dict:
        """The grounding row returned to the UI."""
        return {
            "day": self.day,
            "slot": self.slot,
            "title_given": self.title_given,
            "title_matched": self.title_matched,
            "recipe_id": self.recipe_id,
            "state": self.state,
            "ingredients_source": self.ingredients_source,
            "has_nutrition": bool(self.nutrition),
            "kcal": self.kcal,
            "borrows_nutrition": self.borrows_nutrition,
            "nutrition_source": self.nutrition_source,
            "typical_ingredients": [dict(item) for item in self.typical_ingredients],
            "guess_remarks": self.guess_remarks(),
            "allergen_conflicts": [dict(c) for c in self.allergen_conflicts],
        }
