"""
How good is this plan, measured the same way whatever shape it is.

`_compute_metrics` lived on `ChatService` with the two graders it needs as
instance attributes, so the weekly service — a sibling that produces 21 meals
rather than three — could not reach it. Weekly therefore had no variety score,
no diversity judgement and no guideline adherence, and the UI's quality panel
was wired to the daily canvas only. A member planning a week got the deepest
plan the product makes and the least said about it.

The graders are constructed once here rather than per service. They are pooled
Groq clients behind `GROQ_CHAT`, so a second copy would share the same
connection anyway — but two owners means two places to remember when the
scoring changes.

    metrics(plan_or_scored, guidelines="") -> dict

One entry point, taking either a `ScoredPlan` or a produced `MealPlan`, because
the daily path has the former in hand and the weekly path has the latter.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class _Graders:
    """Lazily built, so importing this module costs no client."""

    _diversity = None
    _guideline = None

    @classmethod
    def diversity(cls):
        if cls._diversity is None:
            from agents import MealDiversityGrader

            cls._diversity = MealDiversityGrader()
        return cls._diversity

    @classmethod
    def guideline(cls):
        if cls._guideline is None:
            from agents import GuidelineAdherenceGrader

            cls._guideline = GuidelineAdherenceGrader()
        return cls._guideline


def extract_ingredient_names(ingredients_text: str) -> list[str]:
    """Normalize a free-text ingredients blob into comparable item names.

    Moved here verbatim from `chat_service`, deliberately unchanged. Rewriting
    it while extracting it would silently move every FVS number the product has
    ever reported, and a metric that shifts because someone tidied a regex is a
    metric nobody can compare across two plans.
    """
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


def food_variety(scored) -> tuple[int, str]:
    """Unique foods across every course of the plan (the FVS metric)."""
    items: list[str] = []
    for course in scored.courses:
        items.extend(extract_ingredient_names(course.ingredients))
    unique = sorted(set(items))
    reasoning = (
        f"Unique food items across meals: {len(unique)} "
        f"(e.g., {', '.join(unique[:8])}{'...' if len(unique) > 8 else ''})"
    )
    return len(unique), reasoning


def as_text(scored) -> str:
    """Every course, labelled by its own slot, for the judges' prompts."""
    return "\n".join(
        f"{name.replace('_', ' ').title()}: {course.title}\n"
        f"Ingredients: {course.ingredients}\nDirections: {course.directions}\n"
        for name, course in ((n, scored.slots[n]) for n in scored.slot_names)
    )


def metrics(scored, guidelines: str = "", *, llm_score: Optional[int] = None,
            llm_reasoning: str = "") -> dict:
    """The four quality metrics the API surfaces.

    `llm_score` is the grader's ranking and is passed in rather than derived:
    the daily path has one because it ranked a batch of candidate days, and the
    weekly and structured paths do not because nothing ranked anything. A zero
    there means "not ranked", and inventing a number would make an unranked
    plan indistinguishable from a well-rated one everywhere downstream.
    """
    plan_text = as_text(scored)
    fvs_count, fvs_reasoning = food_variety(scored)

    diversity = _judged("diversity", lambda: _Graders.diversity().score(plan_text))
    # `guidelines` is the member's numbered rules from the data catalog
    # (`guidelines_service.guidelines_text`); empty when the catalog cannot
    # answer, and the judge then grades on its own rubric.
    adherence = _judged(
        "guideline adherence", lambda: _Graders.guideline().score(plan_text, guidelines),
    )

    return {
        "llm_score": int(llm_score if llm_score is not None else getattr(scored, "score", 0)),
        "llm_reasoning": str(llm_reasoning or getattr(scored, "reasoning", "")),
        "fvs_count": fvs_count,
        "fvs_reasoning": fvs_reasoning,
        "diversity_llm_score": int(diversity.get("score", 0)),
        "diversity_llm_reasoning": str(diversity.get("reasoning", "")),
        "guideline_adherence_score": int(adherence.get("score", 0)),
        "guideline_adherence_reasoning": str(adherence.get("reasoning", "")),
    }


def _judged(name: str, judge) -> dict:
    """One judge's `{score, reasoning}`, or an unscored `0` if the call fails.

    These scores describe a plan that has already been chosen. A judge that is
    rate-limited (the on-demand tier allows 8,000 tokens a minute, and grading
    the candidate days spends most of it) used to raise through the whole turn,
    and the member lost a finished plan over a score. `0` is what the judges
    already return for a reply they cannot parse: not scored.
    """
    try:
        return judge()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quality metric '%s' unavailable: %s: %s", name, type(exc).__name__, exc)
        return {"score": 0, "reasoning": ""}


def scored_from_plan(meal_plan, score: int = 0, reasoning: str = ""):
    """A produced plan, in the shape the metrics read.

    Every plate on every day, not day one: variety, diversity and guideline
    adherence all mean "across what the member will actually eat". Judging a
    seven-day plan on its first day reports the variety of a Monday.

    Slots are labelled `day 2 dinner (side)` so the judges' prompt reads as a
    plan rather than a list of dishes, and so two dinners in a week do not
    collapse onto one key.
    """
    from models.recipe import ScoredPlan

    multi_day = len(meal_plan.day_plans) > 1
    slots: dict = {}
    for day in meal_plan.day_plans:
        for meal in day.meals:
            for plate in meal.plates:
                if not getattr(plate, "recipe_id", ""):
                    continue
                name = meal.meal_type
                if multi_day:
                    name = f"day {day.day} {name}"
                if len(meal.plates) > 1:
                    name = f"{name} ({getattr(plate, 'role', 'main')})"
                slots[name] = plate.to_candidate()
    return ScoredPlan(score=score, reasoning=reasoning, slots=slots)
