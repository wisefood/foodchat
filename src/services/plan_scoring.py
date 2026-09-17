"""
The plan scorer's shared measures: guideline text and the variety count.

    guidelines_text(scope)       the guideline text the scorer's judge is given
    ingredient_names(text)       free-text ingredients → comparable item names
    food_variety_score(courses)  FVS: unique ingredient items across dishes

The normaliser IS the planner's (``plan_quality.extract_ingredient_names``),
so a pasted plan and a generated one count the same ingredients the same way.
``food_variety_score`` differs from ``plan_quality.food_variety`` only in
taking any iterable of dishes rather than a ``ScoredPlan``: a pasted plan has
dishes without slots of their own (two dinners, snacks) and guessed servings
that are not courses at all.

Guidelines: ``GUIDELINES_PATH`` names a file that is not in this repository,
so today every scope returns ``""`` and logs a warning. ``scope`` is
``"daily"`` or ``"weekly"`` from the start, because the weekly judge is meant
to receive frequency rules (fish twice a week) that make no sense for a single
day. The planner now reads guidance from the data catalogue
(``weekly_planner.explainability.guideline_checklist``); moving the scorer onto
that source is the change this function exists to contain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from services.plan_quality import extract_ingredient_names

logger = logging.getLogger(__name__)

GUIDELINES_PATH = (
    Path(__file__).resolve().parents[2] / "belgium_dietary_guidelines_augmentation.cypher"
)
GUIDELINE_SCOPES = ("daily", "weekly")

ingredient_names = extract_ingredient_names


def guidelines_text(scope: str = "daily") -> str:
    """The guideline text for judging a ``scope`` plan; ``""`` when unavailable."""
    try:
        return GUIDELINES_PATH.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Guidelines file unavailable (%s) — scoring %s plans without it.", e, scope)
        return ""


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
