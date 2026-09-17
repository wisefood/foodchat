"""
Plan-quality metrics shared by the daily planner and the plan scorer.

Hoisted out of ``ChatService`` so a plan FoodChat generated and a plan a
member pasted (``services.plan_scorer``) are graded by the same code:

    guidelines_text(scope)       the guideline text a judge is given
    ingredient_names(text)       free-text ingredients → comparable item names
    food_variety_score(courses)  FVS: unique ingredient items across courses
    plan_as_text(plan)           the daily plan text the planner's judges read
    compute_daily_metrics(...)   the four metrics stored on a daily MealPlan

Behaviour is exactly that of the functions this replaced — the same regular
expressions, the same text, the same dict — and ``tests/test_plan_scoring.py``
pins it against a verbatim copy of the originals. ``chat_service`` and
``weekly_planner.explainability`` import from here; neither keeps its own copy.

Guidelines: ``GUIDELINES_PATH`` names a file that is not in this repository,
so today every scope returns ``""`` and logs a warning, as the daily planner
always has. The text is expected to come from an external endpoint later; this
function is the one place that changes then. ``scope`` is ``"daily"`` or
``"weekly"`` from the start, because the weekly judge is meant to receive
frequency rules (fish twice a week) that make no sense for a single day.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

GUIDELINES_PATH = (
    Path(__file__).resolve().parents[2] / "belgium_dietary_guidelines_augmentation.cypher"
)
GUIDELINE_SCOPES = ("daily", "weekly")


def guidelines_text(scope: str = "daily") -> str:
    """The guideline text for judging a ``scope`` plan; ``""`` when unavailable."""
    try:
        return GUIDELINES_PATH.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Guidelines file unavailable (%s) — scoring %s plans without it.", e, scope)
        return ""


def ingredient_names(ingredients_text: str) -> list[str]:
    """Normalize a free-text ingredients blob into comparable item names."""
    if not isinstance(ingredients_text, str):
        return []
    cleaned = []
    for part in re.split(r"[\n,;•\-]+", ingredients_text):
        t = part.strip().lower()
        t = re.sub(r"\([^\)]*\)", "", t)
        t = re.sub(r"[^a-zA-Z\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            cleaned.append(t)
    return cleaned


def food_variety_score(courses: Iterable) -> tuple[int, str]:
    """Count unique food items across courses (FVS metric).

    ``courses`` is anything with an ``ingredients`` string — a daily plan's
    three ``CandidateRecipe`` courses, or every dish of a pasted plan.
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


def plan_as_text(plan) -> str:
    """A daily ``ScoredPlan`` as the text the diversity and guideline judges read."""
    return "\n".join(
        f"{name}: {course.title}\nIngredients: {course.ingredients}\nDirections: {course.directions}\n"
        for name, course in (
            ("Breakfast", plan.breakfast), ("Lunch", plan.lunch), ("Dinner", plan.dinner),
        )
    )


def compute_daily_metrics(plan, diversity_grader, guideline_grader, guidelines: str) -> dict:
    """The four plan-quality metrics stored on a generated daily ``MealPlan``."""
    plan_text = plan_as_text(plan)
    fvs_count, fvs_reasoning = food_variety_score(plan.courses)
    diversity = diversity_grader.score(plan_text)
    adherence = guideline_grader.score(plan_text, guidelines)
    return {
        "llm_score": plan.score,
        "llm_reasoning": plan.reasoning,
        "fvs_count": fvs_count,
        "fvs_reasoning": fvs_reasoning,
        "diversity_llm_score": int(diversity.get("score", 0)),
        "diversity_llm_reasoning": str(diversity.get("reasoning", "")),
        "guideline_adherence_score": int(adherence.get("score", 0)),
        "guideline_adherence_reasoning": str(adherence.get("reasoning", "")),
    }
