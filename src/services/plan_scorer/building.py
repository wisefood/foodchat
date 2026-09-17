"""
Plan scorer, step 3 — build the objects the existing scorers already read.

    build_scoring_input(plan, grounded) → DailyScoringInput | WeeklyScoringInput

Weekly: entry dicts ``{day, meal_idx, meal_type, recipe, reward}`` in exactly
the shape ``WeeklyMealPlanEnv`` produces, so ``build_weekly_explainability``
runs on a pasted week unchanged. Snacks and "other" dishes are kept apart in
``extras``: the weekly guideline checklist counts *meals* ("most meals
plant-based"), and an apple is not a meal.

Daily: catalogue-shaped ``CandidateRecipe`` courses per slot. ``ScoredPlan``
is left as it is — ``DocumentGrader``, ``PlanningPipeline`` and the chat
metrics all construct or read it as one dish per slot for a complete day — so
``as_scored_plan()`` returns one only when the pasted day is exactly that, and
``None`` for a partial or multi-dish day.

Two identity rules decide what later metrics count as "the same dish":

- a **matched** dish is the catalogue recipe, and carries its id, tags and image;
- an **approximate** or **unresolved** dish carries a stable ``pasted:`` id
  built from the member's own title. "Chicken curry" on Monday and "Thai
  green curry" on Wednesday may both land near one catalogue recipe; they
  are still two different things the member wrote, not a repeat. Their
  catalogue tags are not carried either: a category or diet check reads the
  member's words, not the tags of a recipe that merely resembles them.

Repeats are the member's own. A dish written on two days is labelled
``repeat_source = REPEAT_BY_AUTHOR`` with ``repeat_of_day`` the most recent
earlier day it appeared, so the weekly variety metric reads it as planned and
the ledger says the repeats are theirs — not "unexplained", and not the
planner's cooldown. The same dish twice on ONE day cannot carry a
``repeat_of_day`` and is still reported as an unexplained duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from models.pasted_plan import MAIN_SLOTS, MATCHED, GroundedMeal, PastedPlan
from models.recipe import CandidateRecipe, ScoredPlan
from services.weekly_planner.explainability import REPEAT_BY_AUTHOR

from .parsing import content_tokens

MEAL_INDEX = {"breakfast": 0, "lunch": 1, "dinner": 2}
EXTRA_MEAL_INDEX = 3
PASTED_ID_PREFIX = "pasted:"


def entry_recipe_id(meal: GroundedMeal) -> str:
    if meal.state == MATCHED and meal.recipe_id:
        return str(meal.recipe_id)
    key = " ".join(sorted(set(content_tokens(meal.title_given))))
    return PASTED_ID_PREFIX + (key or meal.title_given.strip().lower())


def recipe_dict(meal: GroundedMeal) -> dict:
    """A weekly-entry recipe dict. ``title`` is always the member's words."""
    return {
        "recipe_id": entry_recipe_id(meal),
        "title": meal.title_given,
        "recipe_title": meal.title_matched or meal.title_given,
        "ingredients": meal.ingredients,
        "recipe_ingredients": meal.ingredients,
        "recipe_directions": "",
        "nutrition": dict(meal.nutrition) if meal.nutrition else None,
        "tags": list(meal.tags) if meal.state == MATCHED else [],
        "image_url": meal.image_url if meal.state == MATCHED else None,
        # Provenance, so no consumer mistakes a pasted dish for a planned one.
        "pasted": True,
        "grounding": meal.state,
        "catalogue_recipe_id": meal.recipe_id,
        "ingredients_source": meal.ingredients_source,
    }


def course(meal: GroundedMeal) -> CandidateRecipe:
    return CandidateRecipe(
        recipe_id=entry_recipe_id(meal),
        title=meal.title_given,
        ingredients=meal.ingredients,
        directions="",
        nutrition=dict(meal.nutrition) if meal.nutrition else None,
    )


@dataclass
class DailyScoringInput:
    courses: dict = field(default_factory=dict)   # slot -> list[CandidateRecipe]
    extras: list = field(default_factory=list)    # list[CandidateRecipe]
    meals: list = field(default_factory=list)     # list[GroundedMeal], text order

    plan_type = "daily"

    @property
    def all_courses(self) -> list:
        """Every dish, main slots first — what the variety count reads."""
        return [c for slot in MAIN_SLOTS for c in self.courses.get(slot, [])] + list(self.extras)

    @property
    def missing_slots(self) -> list[str]:
        return [slot for slot in MAIN_SLOTS if not self.courses.get(slot)]

    def as_scored_plan(self) -> Optional[ScoredPlan]:
        """A ``ScoredPlan`` for a complete one-dish-per-slot day, else None."""
        if not all(len(self.courses.get(slot) or []) == 1 for slot in MAIN_SLOTS):
            return None
        return ScoredPlan(
            breakfast=self.courses["breakfast"][0],
            lunch=self.courses["lunch"][0],
            dinner=self.courses["dinner"][0],
            score=0,
            reasoning="",
        )


@dataclass
class WeeklyScoringInput:
    entries: list = field(default_factory=list)   # main-slot entry dicts
    extras: list = field(default_factory=list)    # snack / other entry dicts
    meals: list = field(default_factory=list)     # list[GroundedMeal]
    days: int = 0
    author_repeats: int = 0

    plan_type = "weekly"


ScoringInput = Union[DailyScoringInput, WeeklyScoringInput]


def entry_dicts(meals: list[GroundedMeal], default_day: int = 1) -> list[dict]:
    """Weekly-entry-shaped dicts for any list of dishes, in text order.

    The shape ``nutrition_metrics`` and ``attach_match_reasons`` read, so a
    pasted day is measured by the same functions as a week.
    """
    return [
        {
            "day": meal.day if isinstance(meal.day, int) else default_day,
            "meal_idx": MEAL_INDEX.get(meal.slot, EXTRA_MEAL_INDEX),
            "meal_type": meal.slot,
            "recipe": recipe_dict(meal),
            "reward": 0.0,
        }
        for meal in meals
    ]


def build_daily(grounded: list[GroundedMeal]) -> DailyScoringInput:
    built = DailyScoringInput(meals=list(grounded))
    for meal in grounded:
        if meal.slot in MAIN_SLOTS:
            built.courses.setdefault(meal.slot, []).append(course(meal))
        else:
            built.extras.append(course(meal))
    return built


def label_author_repeats(entries: list[dict]) -> int:
    """Mark every dish served on an earlier day as the member's own repeat.

    Expects entries sorted by day. Returns how many were labelled.
    """
    last_day: dict[str, int] = {}
    labelled = 0
    for entry in entries:
        recipe = entry["recipe"]
        recipe_id = recipe["recipe_id"]
        day = entry["day"]
        earlier = last_day.get(recipe_id)
        if earlier is not None and earlier < day:
            recipe["repeat_of_day"] = earlier
            recipe["repeat_source"] = REPEAT_BY_AUTHOR
            labelled += 1
        last_day[recipe_id] = day
    return labelled


def build_weekly(grounded: list[GroundedMeal]) -> WeeklyScoringInput:
    entries: list[dict] = []
    extras: list[dict] = []
    for meal in grounded:
        entry = {
            "day": meal.day if isinstance(meal.day, int) else 1,
            "meal_type": meal.slot,
            "recipe": recipe_dict(meal),
            "reward": 0.0,
        }
        if meal.slot in MEAL_INDEX:
            entry["meal_idx"] = MEAL_INDEX[meal.slot]
            entries.append(entry)
        else:
            entry["meal_idx"] = EXTRA_MEAL_INDEX
            extras.append(entry)
    # Stable: two dinners on one day keep the order the member wrote them in.
    entries.sort(key=lambda e: (e["day"], e["meal_idx"]))
    extras.sort(key=lambda e: e["day"])
    return WeeklyScoringInput(
        entries=entries,
        extras=extras,
        meals=list(grounded),
        days=len({e["day"] for e in entries + extras}),
        author_repeats=label_author_repeats(entries),
    )


def build_scoring_input(plan: PastedPlan, grounded: list[GroundedMeal]) -> ScoringInput:
    if plan.plan_type == "weekly":
        return build_weekly(grounded)
    return build_daily(grounded)
