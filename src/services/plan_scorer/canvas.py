"""
Scoring the plan FoodChat itself built, rather than one the member pasted.

    from_canvas(plan, plan_type) -> (PastedPlan, [GroundedMeal])

"score my plan" answered *"There's no weekly plan in this conversation yet —
ask for a weekly plan first"* to a member with a daily plan open on the screen
in front of them. Two separate refusals produced that one sentence:

* ``is_explicit_score_request`` requires a **meal listing** in the message,
  because the scorer was written for text a member pasted. "score my plan"
  lists nothing, so the scoring route was never taken;
* the turn then fell to the tool selector, which picked a weekly reader, and
  the weekly reader said the only true thing it knew.

The scorer itself was never the problem. Every metric it runs — the guideline
checklist, food variety, the nutrition totals, the Likert judges — reads
``GroundedMeal``, and a plan on the canvas is *better* grounded than a pasted
one: the recipes are catalogue recipes, so there is no lookup to get wrong, no
similarity threshold, and no estimated serving. What was missing was the
adapter.

**Everything here is MATCHED.** These dishes are not a member's words that
resembled a recipe; they are the recipe. ``ingredients_source`` is
``FROM_RECIPE`` and ``nutrition_source`` is ``NUTRITION_FROM_RECIPE`` for the
same reason — and a plate the corpus gave no nutrition for is left with none,
rather than estimated, because the honest answer to "how many calories is
this" is the one the catalogue has or does not have.
"""

from __future__ import annotations

from typing import Any, Optional

from models.pasted_plan import (
    FROM_RECIPE,
    MATCHED,
    NUTRITION_FROM_RECIPE,
    NUTRITION_NONE,
    SLOTS,
    GroundedMeal,
    PastedDay,
    PastedMeal,
    PastedPlan,
)
from models.plan_spec import slot_kind

# Slots the scorer counts as meals. Everything else is `other` — kept, never
# coerced, exactly as the pasted path treats brunch.
_SCORER_SLOTS = frozenset(SLOTS)


def scorer_slot(slot: str) -> str:
    """A plan slot in the scorer's five-word vocabulary.

    By KIND, so a day's second snack is a snack. The scorer's weekly checklist
    counts meals and sets snacks aside; `snack_2` falling through to `other`
    would have put an afternoon apple in with the brunches.
    """
    kind = slot_kind(slot)
    return kind if kind in _SCORER_SLOTS else "other"


def _grounded(day: Optional[int], slot: str, plate: Any) -> Optional[GroundedMeal]:
    """One plate of a canvas plan as a grounded dish, or None if it is empty."""
    title = str(getattr(plate, "title", None) or "").strip()
    recipe_id = str(getattr(plate, "recipe_id", None) or "").strip()
    if not title or not recipe_id:
        return None
    nutrition = getattr(plate, "nutrition", None)
    return GroundedMeal(
        day=day,
        slot=scorer_slot(slot),
        title_given=title,
        state=MATCHED,
        recipe_id=recipe_id,
        title_matched=title,
        # It IS the recipe, so the similarity is not a measurement that could
        # have come out lower.
        similarity=1.0,
        ingredients=str(getattr(plate, "ingredients", None) or ""),
        ingredients_source=FROM_RECIPE,
        nutrition=nutrition if isinstance(nutrition, dict) and nutrition else None,
        nutrition_source=(
            NUTRITION_FROM_RECIPE
            if isinstance(nutrition, dict) and nutrition
            else NUTRITION_NONE
        ),
        image_url=getattr(plate, "image_url", None),
    )


def _entry_plate(entry: dict) -> dict:
    """A weekly entry's recipe dict, whichever shape it was stored in."""
    recipe = entry.get("recipe")
    return recipe if isinstance(recipe, dict) else entry


class _Plate:
    """The attribute access `_grounded` reads, over a weekly entry's dict."""

    def __init__(self, data: dict):
        self.title = data.get("title") or data.get("recipe_title") or ""
        self.recipe_id = data.get("recipe_id") or ""
        self.ingredients = data.get("ingredients") or ""
        self.nutrition = data.get("nutrition")
        self.image_url = data.get("image_url")


def from_canvas(plan: Any, plan_type: str) -> tuple[PastedPlan, list[GroundedMeal]]:
    """`(PastedPlan, grounded)` for a plan on a canvas. `plan_type` is its canvas.

    The `PastedPlan` carries no dish text of its own beyond the titles: it
    exists so `score_payload` can report how many days and meals were read,
    and so the judges are given the same plan listing they are given for a
    pasted one. The grounded list is what every metric actually reads.
    """
    kind = "weekly" if str(plan_type).lower() == "weekly" else "daily"
    grounded: list[GroundedMeal] = []
    by_day: dict[int, list[PastedMeal]] = {}

    def add(day: int, slot: str, plate: Any) -> None:
        dish = _grounded(day, slot, plate)
        if dish is None:
            return
        grounded.append(dish)
        by_day.setdefault(day, []).append(
            PastedMeal(slot=dish.slot, title=dish.title_given, ingredients=dish.ingredients)
        )

    entries = list(getattr(plan, "entries", None) or [])
    if entries:
        for entry in entries:
            try:
                day = int(entry.get("day") or 1)
            except (TypeError, ValueError):
                day = 1
            add(day, str(entry.get("meal_type") or "other"), _Plate(_entry_plate(entry)))
    else:
        days = list(getattr(plan, "days", None) or [])
        if days:
            for day_plan in days:
                try:
                    number = int(getattr(day_plan, "day", 1) or 1)
                except (TypeError, ValueError):
                    number = 1
                for meal in getattr(day_plan, "meals", None) or []:
                    for plate in getattr(meal, "plates", None) or []:
                        add(number, str(getattr(meal, "meal_type", "") or "other"), plate)
        else:
            # A plan stored before `days` existed: three named fields.
            for slot in ("breakfast", "lunch", "dinner"):
                add(1, slot, getattr(plan, slot, None))

    pasted = PastedPlan(
        plan_type=kind,
        days=[
            PastedDay(day=number, meals=by_day[number])
            for number in sorted(by_day)
        ],
    )
    return pasted, grounded
