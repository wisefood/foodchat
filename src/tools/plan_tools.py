"""
Plan tools — read a plan, total it, summarise it, replace one day of it.

Every reader here is LLM-free. The numbers are summed from the stored plan, not
estimated by a model, which matters because the daily path had no per-plan
calorie total at all: the prompt asked the model to notice when a day "sums far
outside a sensible intake". `plan_totals` is that arithmetic, done properly and
honest about how many meals it could actually see.

`replace_day` exists because the only way to change a weekly plan was to
refine the whole thing, which regenerates all 21 slots — so a verified slot
edit the member had approved was silently thrown away. Pinning is the
mechanism: the planner bypasses selection for any pinned slot, so pinning the
other 18 and freeing one day's three replaces exactly that day.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from tools import ToolError, tool

logger = logging.getLogger(__name__)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
MEAL_ORDER = ["breakfast", "lunch", "dinner"]

_NUTRIENTS = ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _services():
    """Late import — services import the tools package, so this closes a cycle."""
    import services

    return services


def _session(session_id: str):
    svc = _services()
    session = svc.session_service.get_session(session_id)
    if session is None:
        raise ToolError("That conversation no longer exists.")
    return session


def _day_label(day: Any) -> str:
    try:
        d = int(day)
    except (TypeError, ValueError):
        return str(day)
    return DAY_NAMES[d - 1] if 1 <= d <= 7 else f"Day {d}"


def _weekly_plan(session_id: str):
    session = _session(session_id)
    plan = session.get_current_weekly_plan()
    if plan is None or not plan.entries:
        raise ToolError(
            "There's no weekly plan in this conversation yet — ask for a "
            "weekly plan first and I'll have something to work with."
        )
    return session, plan


def _daily_plan(session_id: str):
    session = _session(session_id)
    plan = session.get_current_daily_plan()
    if plan is None:
        raise ToolError("There's no daily plan in this conversation yet.")
    return session, plan


def _sum_nutrition(pairs: list[tuple[str, dict]]) -> dict:
    """Total a list of (title, nutrition) pairs, saying what it could not see.

    A missing nutrition block is common in the corpus, so a bare total would
    quietly understate the plan. `counted` / `of` is the honesty the weekly
    metrics already use, applied everywhere numbers are reported.
    """
    totals = {k: 0.0 for k in _NUTRIENTS}
    counted = 0
    for _title, nutrition in pairs:
        if not isinstance(nutrition, dict) or not nutrition:
            continue
        got = False
        for key in _NUTRIENTS:
            value = nutrition.get(key)
            if value is None and key == "calories":
                value = nutrition.get("kcal")
            if isinstance(value, (int, float)):
                totals[key] += float(value)
                got = True
        if got:
            counted += 1
    return {
        **{k: round(v, 1) for k, v in totals.items()},
        "meals_counted": counted,
        "meals_total": len(pairs),
        "complete": counted == len(pairs) and len(pairs) > 0,
    }


def _entry_nutrition(entry: dict) -> dict:
    return (entry.get("recipe") or {}).get("nutrition") or {}


def _entry_title(entry: dict) -> str:
    recipe = entry.get("recipe") or {}
    return str(recipe.get("recipe_title") or recipe.get("title") or "")


# --------------------------------------------------------------------------- #
# summarize_week
# --------------------------------------------------------------------------- #

@tool(
    "summarize_week",
    summary="Digest the whole weekly plan: every day, the totals, and what held or slipped.",
    description=(
        "Read-only digest of the current weekly plan. Returns one line per day "
        "with its meals and calories, the week's nutrition totals against the "
        "member's budget, the guideline checklist, and the constraints ledger "
        "split into what held and what was relaxed. Answers 'summarise my week' "
        "without regenerating anything. No model call — every number is summed "
        "from the stored plan."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "The conversation to read."},
        },
        "required": ["session_id"],
    },
    examples=("summarise my week", "how does the week look overall?"),
)
def summarize_week(session_id: str) -> dict:
    from services.transparency import split_ledger

    _session_obj, plan = _weekly_plan(session_id)

    by_day: dict[int, list[dict]] = {}
    for entry in plan.entries:
        by_day.setdefault(int(entry.get("day", 0)), []).append(entry)

    days = []
    for day in sorted(by_day):
        entries = sorted(by_day[day], key=lambda e: e.get("meal_idx", 0))
        pairs = [(_entry_title(e), _entry_nutrition(e)) for e in entries]
        totals = _sum_nutrition(pairs)
        days.append({
            "day": day,
            "name": _day_label(day),
            "summary": (plan.day_summaries or {}).get(str(day))
                       or (plan.day_summaries or {}).get(day) or "",
            "meals": [
                {
                    "meal_type": str(e.get("meal_type", "")),
                    "title": _entry_title(e),
                    "kcal": _entry_nutrition(e).get("calories")
                            or _entry_nutrition(e).get("kcal"),
                }
                for e in entries
            ],
            "kcal": totals["calories"],
            "nutrition_coverage": f"{totals['meals_counted']} of {totals['meals_total']} meals",
        })

    week_pairs = [(_entry_title(e), _entry_nutrition(e)) for e in plan.entries]
    week_totals = _sum_nutrition(week_pairs)
    honored, not_honored = split_ledger(plan.constraints_applied, limit=12)

    # The deterministic weekly metrics the planner already computed —
    # guideline checklist, variety, nutrition trackers. Read back rather than
    # recomputed so the digest cannot disagree with the canvas beside it.
    metrics = plan.metrics or {}
    return {
        "plan_version": plan.version,
        "days": days,
        "week_totals": week_totals,
        "daily_average_kcal": round(week_totals["calories"] / max(len(days), 1), 1),
        "guideline_checklist": metrics.get("guideline_checklist") or [],
        "variety": metrics.get("variety") or {},
        "nutrition": metrics.get("nutrition") or {},
        "constraints_honored": honored,
        "constraints_not_honored": not_honored,
        "personalization": plan.personalization_summary or {},
        "week_summary": plan.reasoning or "",
    }


# --------------------------------------------------------------------------- #
# summarize_day
# --------------------------------------------------------------------------- #

@tool(
    "summarize_day",
    summary="Describe one day of the weekly plan in detail.",
    description=(
        "Read-only. Returns a single day's meals with ingredients, per-meal and "
        "whole-day nutrition, and the reason chips explaining why each dish is "
        "there. Use when the member asks about one day rather than the week. No "
        "model call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "The conversation to read."},
            "day": {
                "type": "integer", "minimum": 1, "maximum": 7,
                "description": "1 = Monday through 7 = Sunday.",
            },
        },
        "required": ["session_id", "day"],
    },
    examples=("what's on Thursday?", "tell me about day 3"),
)
def summarize_day(session_id: str, day: int) -> dict:
    _session_obj, plan = _weekly_plan(session_id)

    entries = sorted(
        (e for e in plan.entries if int(e.get("day", 0)) == day),
        key=lambda e: e.get("meal_idx", 0),
    )
    if not entries:
        raise ToolError(
            f"{_day_label(day)} isn't in this plan — it covers "
            f"{len({int(e.get('day', 0)) for e in plan.entries})} day(s)."
        )

    pairs = [(_entry_title(e), _entry_nutrition(e)) for e in entries]
    meals = []
    for entry in entries:
        recipe = entry.get("recipe") or {}
        meals.append({
            "meal_type": str(entry.get("meal_type", "")),
            "title": _entry_title(entry),
            "recipe_id": str(recipe.get("recipe_id", "")),
            "ingredients": str(recipe.get("recipe_ingredients")
                               or recipe.get("ingredients") or ""),
            "nutrition": _entry_nutrition(entry),
            "why": [
                r.get("label", "") for r in (recipe.get("match_reasons") or [])
            ],
            "pinned": bool(entry.get("pinned")),
        })

    return {
        "day": day,
        "name": _day_label(day),
        "summary": (plan.day_summaries or {}).get(str(day))
                   or (plan.day_summaries or {}).get(day) or "",
        "meals": meals,
        "totals": _sum_nutrition(pairs),
    }


# --------------------------------------------------------------------------- #
# plan_totals
# --------------------------------------------------------------------------- #

@tool(
    "plan_totals",
    summary="Add up a plan's calories and macros, per meal and per day.",
    description=(
        "Read-only arithmetic over the stored plan — works on the daily canvas "
        "or the weekly one. This is the total nobody could get before: the "
        "daily path had no per-plan summation at all, and the prose model was "
        "asked to judge whether a day 'sums far outside a sensible intake'. "
        "Reports how many meals carried nutrition data rather than implying a "
        "complete figure. No model call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "The conversation to read."},
            "plan_type": {
                "type": "string", "enum": ["daily", "weekly"],
                "description": "Which canvas to total. Defaults to daily.",
            },
        },
        "required": ["session_id"],
    },
    examples=("how many calories is this plan?", "what are the macros for the week?"),
)
def plan_totals(session_id: str, plan_type: str = "daily") -> dict:
    if plan_type == "weekly":
        _session_obj, plan = _weekly_plan(session_id)
        per_day = {}
        for entry in plan.entries:
            per_day.setdefault(int(entry.get("day", 0)), []).append(
                (_entry_title(entry), _entry_nutrition(entry))
            )
        days = [
            {"day": d, "name": _day_label(d), **_sum_nutrition(per_day[d])}
            for d in sorted(per_day)
        ]
        overall = _sum_nutrition(
            [p for pairs in per_day.values() for p in pairs]
        )
        return {
            "plan_type": "weekly",
            "plan_version": plan.version,
            "per_day": days,
            "total": overall,
            "daily_average_kcal": round(overall["calories"] / max(len(days), 1), 1),
        }

    _session_obj, plan = _daily_plan(session_id)
    meals = []
    pairs: list[tuple[str, dict]] = []
    for day in plan.day_plans:
        for meal in day.meals:
            for plate in meal.plates:
                nutrition = plate.nutrition or {}
                pairs.append((plate.title, nutrition))
                meals.append({
                    "meal_type": getattr(meal, "meal_type", ""),
                    "title": plate.title,
                    "role": getattr(plate, "role", "") or "",
                    "nutrition": nutrition,
                })
    return {
        "plan_type": "daily",
        "plan_version": plan.version,
        "plates": meals,
        "total": _sum_nutrition(pairs),
    }


# --------------------------------------------------------------------------- #
# replace_day
# --------------------------------------------------------------------------- #

@tool(
    "replace_day",
    summary="Regenerate one day of the weekly plan, leaving every other day untouched.",
    description=(
        "Replaces the three meals of a single day. Every other slot is pinned, "
        "so the rest of the week — including any slot the member already "
        "approved through a verified swap — survives byte for byte. This is the "
        "surgical alternative to refining the week, which regenerates all 21 "
        "slots and silently discards approved edits. Recipes already in the "
        "week are excluded, so the new day does not repeat them. Spends the "
        "candidate fetches for one day; no grading model call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "The conversation to change."},
            "day": {
                "type": "integer", "minimum": 1, "maximum": 7,
                "description": "1 = Monday through 7 = Sunday.",
            },
            "note": {
                "type": "string",
                "description": (
                    "Optional steer for the new day, e.g. 'lighter' or "
                    "'something with the leftover rice'."
                ),
            },
        },
        "required": ["session_id", "day"],
    },
    uses_model=False,
    mutates=True,
    examples=("redo Thursday", "replace day 3 with something lighter"),
)
def replace_day(session_id: str, day: int, note: str = "") -> dict:
    from services.adapted_recipes import overlay_weekly_entries
    from services.candidates_client import CANDIDATES
    from services.weekly_planner.action_adapter import RecipeActionSpace
    from services.weekly_planner.day_summary import build_day_summaries
    from services.weekly_planner.environment import WeeklyMealPlanEnv
    from services.weekly_planner.explainability import build_weekly_explainability
    from services.weekly_planner.planner import (
        PlanGenerationError,
        WeeklyPlanner,
        build_preference_scorer,
    )
    from services.weekly_planner.reward_logic import RewardCalculator

    svc = _services()
    session, plan = _weekly_plan(session_id)

    days_present = {int(e.get("day", 0)) for e in plan.entries}
    if day not in days_present:
        raise ToolError(
            f"{_day_label(day)} isn't in this plan — it covers "
            f"{', '.join(_day_label(d) for d in sorted(days_present))}."
        )

    replaced = [e for e in plan.entries if int(e.get("day", 0)) == day]
    kept = [e for e in plan.entries if int(e.get("day", 0)) != day]

    # Pin every slot except the target day's. The planner bypasses candidate
    # selection entirely for a pinned slot, so this is what makes the operation
    # surgical rather than a regeneration that happens to look similar.
    pinned = {
        (int(e.get("day", 0)), int(e.get("meal_idx", 0))): {
            k: v for k, v in (e.get("recipe") or {}).items()
        }
        for e in kept
    }

    state = svc.session_service.get_planning_state(session_id)
    profile = session.user_profile
    action_space = RecipeActionSpace(
        profile,
        additional_diet=list(state.diet_tags),
        pantry=state.pantry,
    )
    # Nothing already in the week may come back, so the new day is genuinely
    # new rather than a reshuffle of what the member has just seen.
    for entry in kept:
        recipe_id = str((entry.get("recipe") or {}).get("recipe_id", ""))
        if recipe_id:
            action_space.mark_selected(recipe_id)
    # And nothing they have rejected, on this day or any earlier one.
    from services.feedback_service import FeedbackService

    signals = FeedbackService().get_signals(session.member_id)
    for recipe_id in (signals.downvoted_recipe_ids or []):
        action_space.mark_selected(recipe_id)
    for recipe_id in state.excluded_recipe_ids:
        action_space.mark_selected(recipe_id)
    # The day being replaced is being replaced — do not offer it back.
    for entry in replaced:
        recipe_id = str((entry.get("recipe") or {}).get("recipe_id", ""))
        if recipe_id:
            action_space.mark_selected(recipe_id)

    query = f"Replace {_day_label(day)}."
    if note:
        query += f" {note}"

    env = WeeklyMealPlanEnv(
        user_profile=profile,
        action_space=action_space,
        reward_calculator=RewardCalculator(),
        user_query=query,
        stated_diet=list(state.diet_tags),
    )
    try:
        entries = WeeklyPlanner(env).generate_full_plan(
            user_query=query, pinned=pinned,
            scorer=build_preference_scorer(profile, pantry=state.pantry),
        )
    except PlanGenerationError as exc:
        raise ToolError(
            f"I couldn't fill {exc.meal_type} on {_day_label(exc.day)} without "
            "repeating something already in your week. Relax a constraint, or "
            "name a dish you'd like there and I'll plan around it."
        ) from None

    # Enrich only what changed — the other 18 slots already carry their chips.
    new_day = [e for e in entries if int(e.get("day", 0)) == day]
    fresh_ids = [str((e.get("recipe") or {}).get("recipe_id", "")) for e in new_day]
    enrichment = CANDIDATES.fetch_details(fresh_ids)
    for entry in new_day:
        rich = enrichment.get(str((entry.get("recipe") or {}).get("recipe_id", "")))
        if rich:
            entry["recipe"]["nutrition"] = rich.nutrition_dict()
            entry["recipe"]["image_url"] = rich.image_url
            entry["recipe"]["tags"] = rich.tags or []
            entry["recipe"]["dish_types"] = rich.dish_types or []
    overlay_weekly_entries(new_day, profile)

    day_summaries = build_day_summaries(entries)
    explainability = build_weekly_explainability(
        entries, profile,
        selection_events=env.selection_events,
        day_summaries=day_summaries,
    )
    stored = svc.session_service.refine_weekly_meal_plan(
        session_id, entries,
        day_summaries=day_summaries,
        explainability=explainability,
    )

    return {
        "day": day,
        "name": _day_label(day),
        "plan_version": stored.version,
        "replaced": [
            {"meal_type": str(e.get("meal_type", "")), "title": _entry_title(e)}
            for e in sorted(replaced, key=lambda e: e.get("meal_idx", 0))
        ],
        "new": [
            {"meal_type": str(e.get("meal_type", "")), "title": _entry_title(e)}
            for e in sorted(new_day, key=lambda e: e.get("meal_idx", 0))
        ],
        "other_days_untouched": len(kept),
        "totals": _sum_nutrition(
            [(_entry_title(e), _entry_nutrition(e)) for e in new_day]
        ),
    }


# --------------------------------------------------------------------------- #
# swap_meal
# --------------------------------------------------------------------------- #

@tool(
    "swap_meal",
    summary="Swap one meal for another that provably satisfies a directive.",
    description=(
        "Replaces a single slot and proves the change: 'lighter' is checked "
        "against real calories and answered with the delta, 'something with "
        "zucchini' is verified against the candidate's own ingredients. Works "
        "on the daily canvas (omit day) or the weekly one (pass day). A "
        "directive that cannot be verified is reported as unverified rather "
        "than claimed. Spends a candidate fetch and an enrichment call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "The conversation to change."},
            "meal_type": {
                "type": "string", "enum": ["breakfast", "lunch", "dinner"],
                "description": "Which meal to replace.",
            },
            "directive": {
                "type": "string",
                "description": "What the new meal must be: 'lighter', 'quicker', "
                               "'vegetarian', 'something with zucchini'.",
            },
            "day": {
                "type": "integer", "minimum": 1, "maximum": 7,
                "description": "Weekly plans only. Omit for the daily canvas.",
            },
        },
        "required": ["session_id", "meal_type", "directive"],
    },
    mutates=True,
    examples=("swap Tuesday's dinner for something lighter",
              "make lunch vegetarian"),
)
def swap_meal(
    session_id: str, meal_type: str, directive: str, day: Optional[int] = None
) -> dict:
    from services.edit_service import EditService

    edit_service = EditService(_services().session_service)

    where = f"{_day_label(day)} " if day else ""
    outcome = edit_service.process(
        session_id, f"swap {where}{meal_type} for {directive}".strip()
    )
    if getattr(outcome, "unresolved", False):
        raise ToolError(
            "I couldn't read that as a single-slot change. Say which meal and "
            "what you want instead."
        )
    return {
        "text": getattr(outcome, "text", "") or "",
        "changed": bool(getattr(outcome, "changed_slots", None)),
        "changed_slots": getattr(outcome, "changed_slots", None) or [],
    }
