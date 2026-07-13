"""Overlay a member's saved adapted recipes onto plan/anchor payloads.

Members can save a personal adapted version of a RecipeWrangler recipe
(an ingredient swap or reduced quantity, produced by the recipe adaptation
endpoints and stored owner-scoped at the gateway). Wherever FoodChat would
surface the ORIGINAL recipe, these helpers swap in the member's adaptation
as the starting point instead — recipe ids stay the original RecipeWrangler
ids so favorites boosting, pinning and recipe links keep working.

The adapted map rides in the session profile (profile["adapted_recipes"],
loaded at session create / diner change) and only ever contains the
member's own adaptations.
"""

import dataclasses
import logging

from models.recipe import ResolvedRecipe

logger = logging.getLogger(__name__)

# Transparency chip attached to adapted courses ([{kind, label}]).
ADAPTED_REASON = {"kind": "adapted", "label": "Your adapted version"}


def adapted_map(profile: dict) -> dict[str, dict]:
    """The member's adaptations keyed by original recipe id (possibly empty)."""
    raw = (profile or {}).get("adapted_recipes") or {}
    return raw if isinstance(raw, dict) else {}


def _display_title(record: dict) -> str | None:
    title = str(record.get("title") or "").strip()
    return title or None


def _ingredients_text(payload: dict) -> str | None:
    """Adapted ingredient rows -> the newline text format candidates use."""
    rows = payload.get("ingredients")
    if not isinstance(rows, list) or not rows:
        return None
    lines = []
    for row in rows:
        if isinstance(row, dict):
            name = str(row.get("name") or "").strip()
            measurement = str(row.get("measurement") or "").strip()
            if name:
                lines.append(f"{measurement} {name}".strip())
        elif isinstance(row, str) and row.strip():
            lines.append(row.strip())
    return "\n".join(lines) or None


def _overlay_nutrition(existing: dict | None, payload: dict) -> dict | None:
    nutrition = payload.get("nutrition")
    if isinstance(nutrition, dict) and nutrition:
        return {**(existing or {}), **nutrition}
    return existing


def overlay_plan(meal_plan, profile: dict) -> int:
    """Overlay all courses of a daily MealPlan in place; returns adapted count."""
    adapted = adapted_map(profile)
    if not adapted:
        return 0
    count = 0
    for course in (meal_plan.breakfast, meal_plan.lunch, meal_plan.dinner):
        record = adapted.get(course.recipe_id)
        if not record:
            continue
        payload = record.get("payload") or {}
        title = _display_title(record)
        if title:
            course.title = title
        text = _ingredients_text(payload)
        if text:
            course.ingredients = text
        course.nutrition = _overlay_nutrition(course.nutrition, payload)
        course.match_reasons = list(course.match_reasons or []) + [dict(ADAPTED_REASON)]
        count += 1
    return count


def overlay_weekly_entries(plan_entries: list[dict], profile: dict) -> int:
    """Overlay weekly-plan entry["recipe"] dicts in place; returns adapted count."""
    adapted = adapted_map(profile)
    if not adapted:
        return 0
    count = 0
    for entry in plan_entries:
        recipe = entry.get("recipe")
        if not isinstance(recipe, dict):
            continue
        record = adapted.get(str(recipe.get("recipe_id") or ""))
        if not record:
            continue
        payload = record.get("payload") or {}
        title = _display_title(record)
        if title:
            recipe["title"] = title
        text = _ingredients_text(payload)
        if text:
            recipe["ingredients"] = text
        nutrition = _overlay_nutrition(recipe.get("nutrition"), payload)
        if nutrition is not None:
            recipe["nutrition"] = nutrition
        recipe["adapted"] = True
        count += 1
    return count


def overlay_resolved(resolved: ResolvedRecipe, profile: dict) -> ResolvedRecipe:
    """Return the member's adapted version of a seed anchor, if one is saved."""
    record = adapted_map(profile).get(resolved.recipe.recipe_id)
    if not record:
        return resolved
    payload = record.get("payload") or {}
    recipe = dataclasses.replace(
        resolved.recipe,
        title=_display_title(record) or resolved.recipe.title,
        ingredients=_ingredients_text(payload) or resolved.recipe.ingredients,
    )
    logger.info("Seed anchor %s uses the member's adapted version.", recipe.recipe_id)
    return dataclasses.replace(resolved, recipe=recipe)
