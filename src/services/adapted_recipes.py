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


def _overlay_course(course, adapted: dict) -> bool:
    """Replace one plate with the member's saved version. True if it changed."""
    record = adapted.get(course.recipe_id)
    if not record:
        return False
    payload = record.get("payload") or {}
    title = _display_title(record)
    if title:
        course.title = title
    text = _ingredients_text(payload)
    if text:
        course.ingredients = text
    course.nutrition = _overlay_nutrition(course.nutrition, payload)
    course.match_reasons = list(course.match_reasons or []) + [dict(ADAPTED_REASON)]
    return True


def overlay_plan(meal_plan, profile: dict) -> int:
    """Overlay every plate of a daily MealPlan in place; returns adapted count.

    Walks `day_plans`, not the three scalar courses.

    Those cover one plate of one day. A plan with a two-course dinner, or more
    than one day, kept the member's adapted version for the main and served the
    original for every other plate — someone who had swapped an ingredient out
    for an allergy or a dislike got their swap on one dish and the thing they
    had rejected on the next.

    `day_plans` presents a legacy plan as one day of single-plate meals, so the
    classic case walks exactly the same three courses it always did.

    The scalar fields are mirrors of day 1's mains, so they are refreshed after
    the walk — the objects are shared for a legacy plan but rebuilt for a
    flexible one, and a reader addressing `plan.dinner` by name must not see the
    unadapted version.
    """
    adapted = adapted_map(profile)
    if not adapted:
        return 0

    count = 0
    seen: set[int] = set()
    for day in meal_plan.day_plans:
        for meal in day.meals:
            for plate in meal.plates:
                # A plan's scalar fields can be the *same objects* as day 1's
                # plates. Overlaying twice would append a second "adapted"
                # reason chip and count one swap as two.
                if id(plate) in seen:
                    continue
                seen.add(id(plate))
                if _overlay_course(plate, adapted):
                    count += 1

    if getattr(meal_plan, "days", None):
        by_slot = {m.meal_type: m.main for m in meal_plan.days[0].meals}
        for slot in ("breakfast", "lunch", "dinner"):
            plate = by_slot.get(slot)
            if plate is not None:
                setattr(meal_plan, slot, plate)

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
