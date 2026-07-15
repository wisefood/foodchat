"""Lazy nutrition refresh for stored plans.

Plans freeze course nutrition at creation time. Recipes profiled AFTERWARDS
(the RecipeWrangler backfill, a recipe-page visit triggering the background
job) left "(no nutrition data)" in plan summaries forever — starving the
PlanAnalyst and the FoodScholar plan context even though the data now exists.

``refresh_plan_nutrition`` fills any course/entry whose nutrition is missing
from RecipeWrangler's batch details endpoint and persists the plan. Courses
that already carry nutrition — including member-adapted overlays — are never
touched.
"""

from __future__ import annotations

import logging

from .candidates_client import CANDIDATES

logger = logging.getLogger(__name__)


def _nutrition_dict(enrichment) -> dict | None:
    values = {
        "kcal": enrichment.kcal,
        "protein_g": enrichment.protein_g,
        "carbs_g": enrichment.carbs_g,
        "fat_g": enrichment.fat_g,
    }
    if all(v is None for v in values.values()):
        return None
    values["nutri_score_label"] = enrichment.nutri_score_label
    return values


def refresh_plan_nutrition(session, session_service=None) -> int:
    """Fill missing nutrition on the active plan in place; returns fill count."""
    canvas = session.active_canvas
    if canvas is None:
        return 0

    updated = 0
    if canvas.plan_type == "weekly":
        plan = session.get_current_weekly_plan()
        if plan is None:
            return 0
        recipes = [
            entry.get("recipe") for entry in plan.entries
            if isinstance(entry.get("recipe"), dict)
        ]
        missing_ids = list(dict.fromkeys(
            str(r.get("recipe_id")) for r in recipes
            if r.get("recipe_id") and not r.get("nutrition")
        ))
        if not missing_ids:
            return 0
        enriched = CANDIDATES.fetch_details(missing_ids)
        for recipe in recipes:
            rid = str(recipe.get("recipe_id") or "")
            if rid and not recipe.get("nutrition") and rid in enriched:
                nutrition = _nutrition_dict(enriched[rid])
                if nutrition:
                    recipe["nutrition"] = nutrition
                    updated += 1
    else:
        plan = session.get_current_daily_plan()
        if plan is None:
            return 0
        courses = [getattr(plan, slot) for slot in ("breakfast", "lunch", "dinner")]
        missing_ids = list(dict.fromkeys(
            course.recipe_id for course in courses
            if course.recipe_id and not course.nutrition
        ))
        if not missing_ids:
            return 0
        enriched = CANDIDATES.fetch_details(missing_ids)
        for course in courses:
            if not course.nutrition and course.recipe_id in enriched:
                nutrition = _nutrition_dict(enriched[course.recipe_id])
                if nutrition:
                    course.nutrition = nutrition
                    updated += 1

    if updated and session_service is not None:
        try:
            # Same-package internal: canvases are the plan store of record.
            session_service._persist_canvases(session.session_id, session)
        except Exception:  # noqa: BLE001
            logger.warning("Plan nutrition refresh could not persist", exc_info=True)
    if updated:
        logger.info(
            "[%s] refreshed nutrition for %d plan course(s)", session.session_id, updated
        )
    return updated
