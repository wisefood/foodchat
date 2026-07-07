"""
Transparency (M4a) — the *reasons* behind a plan as structured data.

Pure functions (no LLM, no I/O) that compute what the UI renders:

    match_reasons(course, ...)      → per-course chips ("you liked chickpeas",
                                      "no peanuts — Tom's allergy", "requested by you")
    constraints_ledger(profile, …)  → plan-level hard/soft constraint rows
    personalization_summary(...)    → counts linking to the memory panel

Reason kinds (shared contract with the UI):
    pinned | favorite | memory | profile | feedback | diner | guideline
"""

from models.session import MealCourse


def match_reasons(
    course_recipe_id: str,
    ingredients_text: str,
    profile: dict,
    pinned_recipe_ids: set[str],
) -> list[dict]:
    """Why this recipe is in the plan (order: strongest signal first)."""
    reasons: list[dict] = []

    if course_recipe_id in pinned_recipe_ids:
        reasons.append({"kind": "pinned", "label": "requested by you"})

    if course_recipe_id in (profile.get("favorite_recipe_ids") or []):
        reasons.append({"kind": "favorite", "label": "one of your favorites"})

    text = (ingredients_text or "").lower()
    memory_values = {
        str(e.get("value", "")).lower()
        for e in (profile.get("memory_log") or [])
        if e.get("kind") in ("like", "cuisine")
    }
    for like in profile.get("food_likes") or []:
        like_l = str(like).lower()
        if like_l and like_l in text:
            kind = "memory" if like_l in memory_values else "profile"
            label = f"you like {like_l}"
            reasons.append({"kind": kind, "label": label})
            break  # one like-chip per course keeps cards readable

    return reasons


def constraints_ledger(profile: dict, downvoted_count: int = 0) -> list[dict]:
    """Hard/soft constraint rows for the plan header.

    Sources: the merged session profile (which already unions diner
    constraints — cooking_for_names attributes them) and feedback exclusions.
    """
    ledger: list[dict] = []
    diners = profile.get("cooking_for_names") or []
    source_suffix = f" ({', '.join(diners)})" if len(diners) > 1 else ""

    for allergen in profile.get("allergies") or []:
        ledger.append({
            "constraint": f"no {allergen}",
            "type": "hard", "status": "satisfied",
            "source": f"allergy{source_suffix}",
        })
    for diet in profile.get("diet") or []:
        ledger.append({
            "constraint": str(diet),
            "type": "hard", "status": "satisfied",
            "source": f"dietary group{source_suffix}",
        })
    for dislike in (profile.get("food_dislikes") or [])[:5]:
        ledger.append({
            "constraint": f"avoiding {dislike}",
            "type": "soft", "status": "satisfied",
            "source": "preferences",
        })
    if downvoted_count:
        ledger.append({
            "constraint": f"excluding {downvoted_count} recipe(s) you disliked",
            "type": "soft", "status": "satisfied",
            "source": "your feedback",
        })
    return ledger


def personalization_summary(profile: dict, feedback_lines: int = 0) -> dict:
    """Counts for the "Personalized with …" line (links to the memory panel)."""
    return {
        "memories_used": len(profile.get("memory_log") or []),
        "favorites_used": len(profile.get("favorite_recipe_ids") or []),
        "feedback_signals": feedback_lines,
        "diners": len(profile.get("cooking_for_names") or []) or 1,
    }


def apply_transparency(
    meal_plan,
    profile: dict,
    pinned_recipe_ids: set[str],
    enrichment: dict,
    downvoted_count: int = 0,
    feedback_lines: int = 0,
) -> None:
    """Attach enrichment + transparency to a freshly built MealPlan in place."""
    for course in (meal_plan.breakfast, meal_plan.lunch, meal_plan.dinner):
        rich = enrichment.get(course.recipe_id)
        if rich:
            course.nutrition = rich.nutrition_dict()
            course.image_url = rich.image_url
        course.match_reasons = match_reasons(
            course.recipe_id, course.ingredients, profile, pinned_recipe_ids,
        )
    meal_plan.constraints_applied = constraints_ledger(profile, downvoted_count)
    meal_plan.personalization_summary = personalization_summary(profile, feedback_lines)
