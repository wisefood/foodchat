"""
Weekly plan explainability (M7) — why the week looks the way it does.

Brings the daily plan's transparency story to the weekly plan and goes
further: since the M6 constraints rework the planner is fully
deterministic, so the ledger REPORTS measured numbers (meat meals used,
kcal planned vs budget) instead of declaring "satisfied", and relaxations
recorded at decision time (``reward_logic.apply_hard_constraints``) are
surfaced honestly instead of only warned about in logs.

Everything here is LLM-free and network-free — pure functions over the
final entry list, the member profile, and the selection events the planner
recorded while picking. ``build_weekly_explainability`` is the single
entry point: it attaches per-entry ``match_reasons`` in place (mirroring
``transparency.apply_transparency`` for daily plans) and returns the
``constraints_applied`` / ``personalization_summary`` / ``metrics`` /
``reasoning`` payload stored on ``WeeklyMealPlan``.

Ledger ``status`` values: ``satisfied`` | ``relaxed`` | ``violated``.
Daily rows only ever say "satisfied" — the two new values are additive;
UI consumers should treat unknown statuses as informational.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from services.adapted_recipes import ADAPTED_REASON
from services.transparency import constraints_ledger, match_reasons, personalization_summary

from .day_summary import classify_meal, is_meat_meal
from .reward_logic import candidate_kcal
from .state_tracking import WeeklyNutritionalTracker

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Soft calorie budget counts as satisfied up to this share of the target.
CALORIE_TOLERANCE = 1.05


# --------------------------------------------------------------------- #
# Entry helpers (weekly entries carry recipe_title/recipe_ingredients;   #
# the adapted-recipe overlay adds display title/ingredients on top)      #
# --------------------------------------------------------------------- #

def _recipe(entry: dict) -> dict:
    recipe = entry.get("recipe")
    return recipe if isinstance(recipe, dict) else {}


def _title(recipe: dict) -> str:
    return str(recipe.get("title") or recipe.get("recipe_title") or "")


def _ingredients(recipe: dict) -> str:
    return str(recipe.get("ingredients") or recipe.get("recipe_ingredients") or "")


def _day_name(day: Any) -> str:
    return DAY_NAMES[day - 1] if isinstance(day, int) and 1 <= day <= 7 else f"day {day}"


def _slot_name(event: dict) -> str:
    return f"{_day_name(event.get('day'))} {event.get('meal_type', '')}".strip()


def _ingredient_names(ingredients_text: str) -> List[str]:
    """Normalize an ingredients blob into comparable item names (same
    normalization as the daily FVS metric in ``chat_service``)."""
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


# --------------------------------------------------------------------- #
# Per-entry reasons                                                       #
# --------------------------------------------------------------------- #

def attach_match_reasons(plan_entries: List[dict], profile: dict) -> None:
    """Attach ``recipe["match_reasons"]`` chips to every entry in place.

    Reuses the daily chip logic (pinned/favorite/memory/profile) and maps
    the weekly-only markers onto existing chip kinds: ``recipe["pinned"]``
    → the "requested by you" chip, ``recipe["adapted"]`` → the
    ``ADAPTED_REASON`` chip (the daily overlay already adds it; the weekly
    overlay only sets the flag, unified here).
    """
    pinned_ids = {
        str(_recipe(e).get("recipe_id") or "")
        for e in plan_entries if _recipe(e).get("pinned")
    }
    for entry in plan_entries:
        recipe = _recipe(entry)
        if not recipe:
            continue
        reasons = match_reasons(
            str(recipe.get("recipe_id") or ""),
            _ingredients(recipe),
            profile,
            pinned_ids,
        )
        if recipe.get("adapted"):
            reasons.append(dict(ADAPTED_REASON))
        recipe["match_reasons"] = reasons


# --------------------------------------------------------------------- #
# Weekly metrics (deterministic — no LLM)                                 #
# --------------------------------------------------------------------- #

def variety_metrics(plan_entries: List[dict]) -> dict:
    """Distinct recipes, unique ingredients, and category distribution."""
    recipes = [_recipe(e) for e in plan_entries]
    distinct = len({
        str(r.get("recipe_id") or "") for r in recipes if r.get("recipe_id")
    })
    items: set = set()
    for r in recipes:
        items.update(_ingredient_names(_ingredients(r)))
    categories = Counter(classify_meal(r) for r in recipes if r)
    total = len(plan_entries)
    if distinct == total:
        headline = f"All {total} meals are distinct recipes"
    else:
        headline = f"{distinct} distinct recipes across {total} meals"
    return {
        "distinct_recipes": distinct,
        "total_meals": total,
        "unique_ingredients": len(items),
        "category_distribution": dict(categories),
        "reasoning": (
            f"{headline}; {len(items)} unique ingredients; "
            f"{_category_line(categories)}."
        ),
    }


def _category_line(categories: Dict[str, int]) -> str:
    ordered = sorted(categories.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{n} {cat}" for cat, n in ordered) or "no meals"


def guideline_checklist(category_counts: Dict[str, int], total_meals: int) -> List[dict]:
    """Weekly frequency rules from food-based dietary guidelines (the kind a
    single day can't be graded against), checked from category counts."""
    fish = int(category_counts.get("fish", 0))
    red_meat = int(category_counts.get("red meat", 0))
    plant = int(category_counts.get("vegetarian", 0)) + int(category_counts.get("vegan", 0))
    plant_target = (total_meals + 1) // 2
    return [
        {
            "rule": "eat fish 1–2 times a week",
            "target": "1–2 meals",
            "actual": fish,
            "met": 1 <= fish <= 2,
        },
        {
            "rule": "limit red meat",
            "target": "at most 3 meals",
            "actual": red_meat,
            "met": red_meat <= 3,
        },
        {
            "rule": "make most meals plant-based",
            "target": f"at least {plant_target} of {total_meals} meals",
            "actual": plant,
            "met": plant >= plant_target,
        },
    ]


def nutrition_metrics(plan_entries: List[dict], targets: Dict[str, float]) -> dict:
    """Weekly totals + daily average vs target, with honest coverage.

    Totals are computed from the FINAL entries (post-enrichment,
    post-adapted-overlay) — what the member will actually see — not from
    the selection-time tracker, whose view can miss pinned dishes.
    """
    keys = (
        ("kcal", ("kcal", "calories")),
        ("protein_g", ("protein_g", "protein")),
        ("carbs_g", ("carbs_g", "carbs")),
        ("fat_g", ("fat_g", "fat")),
    )
    totals = {name: 0.0 for name, _ in keys}
    covered = 0
    for entry in plan_entries:
        nutrition = _recipe(entry).get("nutrition") or {}
        if candidate_kcal(_recipe(entry)) is not None:
            covered += 1
        for name, aliases in keys:
            for alias in aliases:
                value = nutrition.get(alias)
                if isinstance(value, (int, float)):
                    totals[name] += float(value)
                    break

    total_meals = len(plan_entries)
    target_kcal = float(targets.get("calories") or 0.0)
    weekly_targets: Dict[str, float] = {}
    if target_kcal > 0:
        weekly_targets["kcal"] = target_kcal
    if float(targets.get("protein") or 0.0) > 0:
        weekly_targets["protein_g"] = float(targets["protein"])

    budget_used_pct = (
        round(100.0 * totals["kcal"] / target_kcal)
        if target_kcal > 0 and covered else None
    )
    note = (
        f"based on {covered} of {total_meals} meals with nutrition data"
        if 0 < covered < total_meals else ""
    )
    return {
        "weekly_totals": {name: round(value, 1) for name, value in totals.items()},
        "daily_average_kcal": round(totals["kcal"] / 7.0, 1) if covered else None,
        "weekly_targets": weekly_targets,
        "budget_used_pct": budget_used_pct,
        "coverage": {"meals_with_data": covered, "total_meals": total_meals},
        "note": note,
    }


def day_breakdown(plan_entries: List[dict], day_summaries: Dict[int, str]) -> List[dict]:
    """Per-day justification rows: headline, kcal, and reason highlights.

    Expects ``attach_match_reasons`` to have run — highlights quote the
    strongest chip per meal ("Grilled Salmon: one of your favorites").
    """
    by_day: Dict[int, List[dict]] = {}
    for entry in plan_entries:
        by_day.setdefault(int(entry.get("day", 0)), []).append(entry)

    days = []
    for day in sorted(by_day):
        entries = sorted(by_day[day], key=lambda e: e.get("meal_idx", 0))
        kcal_values = [candidate_kcal(_recipe(e)) for e in entries]
        known = [v for v in kcal_values if v is not None]
        highlights = []
        for e in entries:
            recipe = _recipe(e)
            reasons = recipe.get("match_reasons") or []
            if reasons:
                highlights.append(f"{_title(recipe)}: {reasons[0].get('label', '')}")
        days.append({
            "day": day,
            "name": _day_name(day),
            "summary": day_summaries.get(day, ""),
            "kcal": round(sum(known), 1) if known else None,
            "meals_with_data": len(known),
            "highlights": highlights[:3],
        })
    return days


# --------------------------------------------------------------------- #
# Measured constraint ledger                                              #
# --------------------------------------------------------------------- #

def weekly_constraints_ledger(
    profile: dict,
    meat_count: int,
    targets: Dict[str, float],
    selection_events: List[dict],
    downvoted_count: int,
    nutrition: dict,
) -> List[dict]:
    """Profile rows (same as daily) + measured weekly rows.

    The weekly rows carry a ``detail`` string with the actual numbers, and
    a status that reports what happened: ``relaxed`` when the planner had
    to relax the meat limit for a slot whose every candidate contained
    meat (recorded at decision time, not reconstructed), ``violated`` when
    the final week breaks the limit anyway (e.g. pinned dishes bypass
    constraints).
    """
    ledger = constraints_ledger(profile, downvoted_count)

    meat_limit = int(targets.get("meat_limit") or 0)
    if meat_limit > 0 or meat_count > 0:
        relaxed = [e for e in selection_events if e.get("type") == "meat_limit_relaxed"]
        pruned = [e for e in selection_events if e.get("type") == "meat_pool_pruned"]
        detail = f"{meat_count} of {meat_limit} meat meal(s) planned"
        if pruned:
            detail += f"; meat dishes left the candidate pool from {_slot_name(pruned[0])} on"
        if relaxed:
            status = "relaxed"
            detail += (
                f"; every available candidate for {_slot_name(relaxed[0])} "
                "contained meat, so the limit was relaxed there"
            )
        elif meat_count > meat_limit:
            status = "violated"
        else:
            status = "satisfied"
        ledger.append({
            "constraint": f"at most {meat_limit} meat meal(s) this week",
            "type": "hard",
            "status": status,
            "source": "dietary preference",
            "detail": detail,
        })

    target_kcal = float(targets.get("calories") or 0.0)
    planned = float((nutrition.get("weekly_totals") or {}).get("kcal") or 0.0)
    covered = int((nutrition.get("coverage") or {}).get("meals_with_data") or 0)
    if target_kcal > 0 and covered:
        pct = round(100.0 * planned / target_kcal)
        detail = f"{planned:,.0f} of {target_kcal:,.0f} kcal planned ({pct}%)"
        if nutrition.get("note"):
            detail += f", {nutrition['note']}"
        ledger.append({
            "constraint": "weekly calorie budget",
            "type": "soft",
            "status": "satisfied" if planned <= target_kcal * CALORIE_TOLERANCE else "violated",
            "source": "calorie target",
            "detail": detail,
        })

    return ledger


# --------------------------------------------------------------------- #
# Whole-week justification (deterministic prose)                          #
# --------------------------------------------------------------------- #

def _compose_reasoning(
    variety: dict,
    meat_count: int,
    meat_limit: int,
    relaxed_slot: str,
    nutrition: dict,
    pinned_count: int,
    adapted_count: int,
) -> str:
    parts = []
    total = variety["total_meals"]
    distinct = variety["distinct_recipes"]
    cat_line = _category_line(variety["category_distribution"])
    if distinct == total:
        parts.append(f"All {total} meals this week are distinct recipes — {cat_line}.")
    else:
        parts.append(f"{distinct} distinct recipes across {total} meals — {cat_line}.")

    if meat_limit > 0:
        if relaxed_slot:
            parts.append(
                f"Your weekly meat limit ({meat_limit}) couldn't be fully honored — "
                f"every available candidate for {relaxed_slot} contained meat."
            )
        elif meat_count > meat_limit:
            parts.append(
                f"The week exceeds your meat limit ({meat_count} of {meat_limit} meat meals)."
            )
        else:
            parts.append(
                f"Stayed within your weekly meat limit ({meat_count} of {meat_limit} meat meals)."
            )

    pct = nutrition.get("budget_used_pct")
    if pct is not None:
        planned = (nutrition.get("weekly_totals") or {}).get("kcal", 0.0)
        target_kcal = (nutrition.get("weekly_targets") or {}).get("kcal", 0.0)
        sentence = (
            f"Planned calories total {planned:,.0f} of your "
            f"{target_kcal:,.0f} kcal weekly budget ({pct}%)"
        )
        if nutrition.get("note"):
            sentence += f" ({nutrition['note']})"
        parts.append(sentence + ".")

    if pinned_count:
        parts.append(f"{pinned_count} dish(es) anchored at your request.")
    if adapted_count:
        parts.append(f"{adapted_count} meal(s) use your saved adapted version.")
    return " ".join(parts)


# --------------------------------------------------------------------- #
# Entry point                                                             #
# --------------------------------------------------------------------- #

def build_weekly_explainability(
    plan_entries: List[dict],
    profile: dict,
    *,
    selection_events: Optional[List[dict]] = None,
    day_summaries: Optional[Dict[int, str]] = None,
    downvoted_count: int = 0,
    feedback_lines: int = 0,
) -> dict:
    """Attach per-entry match reasons in place and build the explainability
    payload stored on ``WeeklyMealPlan``.

    Targets are re-derived from the profile via ``WeeklyNutritionalTracker``
    — the same deterministic derivation the planning env uses — so this
    also works for patched plans (slot edits) where no env exists;
    ``selection_events`` is empty there and statuses come from the final
    counts alone.
    """
    events = list(selection_events or [])
    tracker = WeeklyNutritionalTracker(profile)
    targets = tracker.targets

    attach_match_reasons(plan_entries, profile)

    variety = variety_metrics(plan_entries)
    checklist = guideline_checklist(variety["category_distribution"], variety["total_meals"])
    nutrition = nutrition_metrics(plan_entries, targets)
    days = day_breakdown(plan_entries, day_summaries or {})

    meat_count = sum(
        1 for e in plan_entries
        if is_meat_meal(
            _title(_recipe(e)), _ingredients(_recipe(e)),
            tags=_recipe(e).get("tags"),
            count_fish=tracker.counts_fish_as_meat,
        )
    )
    ledger = weekly_constraints_ledger(
        profile, meat_count, targets, events, downvoted_count, nutrition,
    )

    relaxed = [e for e in events if e.get("type") == "meat_limit_relaxed"]
    pinned_count = sum(1 for e in plan_entries if _recipe(e).get("pinned"))
    adapted_count = sum(1 for e in plan_entries if _recipe(e).get("adapted"))
    reasoning = _compose_reasoning(
        variety,
        meat_count,
        int(targets.get("meat_limit") or 0),
        _slot_name(relaxed[0]) if relaxed else "",
        nutrition,
        pinned_count,
        adapted_count,
    )

    return {
        "constraints_applied": ledger,
        "personalization_summary": personalization_summary(profile, feedback_lines),
        "metrics": {
            "variety": variety,
            "guideline_checklist": checklist,
            "nutrition": nutrition,
            "days": days,
            "selection_events": events,
        },
        "reasoning": reasoning,
    }
