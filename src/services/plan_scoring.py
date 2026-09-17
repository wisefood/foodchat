"""
The plan scorer's shared measures: guideline text and the variety count.

    guidelines_text(plan_type, profile)  the guideline text the judges are given
    ingredient_names(text)               free-text ingredients → comparable item names
    food_variety_score(courses)          FVS: unique ingredient items across dishes

The normaliser IS the planner's (``plan_quality.extract_ingredient_names``),
so a pasted plan and a generated one count the same ingredients the same way.
``food_variety_score`` differs from ``plan_quality.food_variety`` only in
taking any iterable of dishes rather than a ``ScoredPlan``: a pasted plan has
dishes without slots of their own (two dinners, snacks) and guessed servings
that are not courses at all.

Guidelines: ``guidelines_text(plan_type, profile)`` IS
``guidelines_service.guidelines_text`` — the member's rules from the WiseFood
data catalogue, numbered for the judge, ``""`` when the catalogue cannot
answer. ``plan_type`` is ``"daily"`` or ``"weekly"`` because a weekly rule
(fish twice a week) cannot be kept or broken by a single day, so the daily
judge is not given it.
"""

from __future__ import annotations

from typing import Iterable

from services.guidelines_service import guidelines_text  # noqa: F401 — re-exported
from services.plan_quality import extract_ingredient_names

GUIDELINE_SCOPES = ("daily", "weekly")

ingredient_names = extract_ingredient_names


def food_variety_score(courses: Iterable) -> tuple[int, str]:
    """Count unique food items across dishes (FVS metric).

    ``courses`` is anything with an ``ingredients`` string. Same items and same
    sentence as ``plan_quality.food_variety`` over the same dishes.
    """
    items: list[str] = []
    for course in courses:
        items.extend(ingredient_names(course.ingredients))
    unique_items = sorted(set(items))
    reasoning = (
        f"Unique food items across meals: {len(unique_items)} "
        f"(e.g., {', '.join(unique_items[:8])}{'...' if len(unique_items) > 8 else ''})"
    )
    return len(unique_items), reasoning
