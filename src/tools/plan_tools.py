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


def _day_label(day: Any, *, weekdays: bool = True) -> str:
    """Name a day the way its plan is entitled to name it.

    A weekly plan is calendar-anchored: its day 1 IS Monday. A multi-day plan
    on the daily canvas is not — it starts whenever the member cooks it — so
    calling its day 2 "Tuesday" states a fact the plan never carried.
    `services.edit_service._named_day` refuses to READ weekdays on such a plan
    for the same reason; a reader that WRITES them puts the two out of step.
    """
    try:
        d = int(day)
    except (TypeError, ValueError):
        return str(day)
    if weekdays and 1 <= d <= 7:
        return DAY_NAMES[d - 1]
    return f"Day {d}"


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


def _canvas_kind(session) -> str:
    """Which canvas the member is looking at. Daily when there is nothing."""
    canvas = session.active_canvas
    return canvas.plan_type if canvas else "daily"


def _day_plan(session_id: str, plan_type: Optional[str] = None):
    """The plan a day-aware reader should read: (session, plan, kind).

    `summarize_week` and `summarize_day` read the weekly canvas and nothing
    else, so they answered "there's no weekly plan in this conversation yet" to
    a member sitting in front of a three-day plan. Multi-day plans live on the
    DAILY canvas — that is the shape `plan_structured` produces — and a tool
    whose whole job is to summarise days had no business refusing to look at
    them over which canvas they were stored on.

    An explicit `plan_type` wins, because the UI passes the canvas the menu was
    opened on and the member's own screen is the least ambiguous answer. With
    nothing passed, whichever canvas actually holds a plan is used, weekly
    first — a session that has both was most recently planning a week.
    """
    session = _session(session_id)
    wanted = str(plan_type or "").strip().lower()

    weekly = session.get_current_weekly_plan()
    has_weekly = weekly is not None and bool(weekly.entries)
    daily = session.get_current_daily_plan()

    if wanted == "weekly" or (not wanted and has_weekly):
        if not has_weekly:
            raise ToolError(
                "There's no weekly plan in this conversation yet — ask for a "
                "weekly plan first and I'll have something to work with."
            )
        return session, weekly, "weekly"

    if wanted == "daily" or (not wanted and daily is not None):
        if daily is None:
            raise ToolError("There's no daily plan in this conversation yet.")
        return session, daily, "daily"

    raise ToolError(
        "There's no plan in this conversation yet — ask for one and I'll have "
        "something to work with."
    )


def _spec_of(plan) -> "object":
    """The shape a stored weekly plan actually has, read from the plan itself.

    `replace_day` built its planner with no spec at all, so the walk fell back
    to its default — seven days of breakfast/lunch/dinner — whatever the plan
    in front of it was:

    * replacing a day of a THREE-day week generated the four days it did not
      have, because every unpinned slot gets filled;
    * a week with a snack lost the snack, since the rebuilt day had no such
      slot and the pool was never asked for one;
    * a multi-plate dinner flattened back to one dish, on the one operation
      whose entire promise is that everything else survives byte for byte.

    The plan is the source of truth for its own shape — the same rule
    `_edit_daily` follows for a daily plan with days — so the shape is read
    from the entries rather than from the member's standing spec, which may
    have moved on since this plan was made.
    """
    from models.plan_spec import DEFAULT_MEALS, PlanSpec
    from models.recipe import slot_sort_key

    days = sorted({int(e.get("day") or 0) for e in plan.entries if e.get("day")})

    # slot -> the roles it is served with, in the order they appear. Read from
    # day 1 alone would miss a side that only Thursday has, so every day
    # contributes and the widest reading of each slot wins.
    roles_by_slot: dict[str, list[str]] = {}
    for entry in plan.entries:
        slot = str(entry.get("meal_type") or "").strip().lower()
        if not slot:
            continue
        role = str(entry.get("role") or "main").strip().lower() or "main"
        roles = roles_by_slot.setdefault(slot, [])
        if role not in roles:
            roles.append(role)

    meals = tuple(sorted(roles_by_slot, key=slot_sort_key))
    plates = {
        slot: tuple(roles)
        for slot, roles in roles_by_slot.items()
        if len(roles) > 1
    }
    return PlanSpec(
        num_days=max(len(days), 1),
        # An empty plan cannot describe itself; the default shape is the only
        # honest fallback, and it is the shape this code always assumed anyway.
        meals=meals or DEFAULT_MEALS,
        plates=plates,
    )


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


def _plate_pairs(day) -> list[tuple[str, dict]]:
    """Every plate of a day, for totalling. Sides count — a meal is its plates."""
    return [
        (plate.title, plate.nutrition or {})
        for meal in day.meals
        for plate in meal.plates
    ]


def _meal_rows(day) -> list[dict]:
    """One row per meal of a daily-canvas day, with every plate named.

    A multi-plate meal has no single title: reporting a two-side dinner as
    "dinner: roast chicken" is the same silent drop the weekly grid used to do
    with `.find()`. `title` joins the plates so a reader that only knows about
    titles still sees the whole meal, and `plates` carries the structure for
    one that does.
    """
    rows = []
    for meal in day.meals:
        totals = _sum_nutrition([(p.title, p.nutrition or {}) for p in meal.plates])
        rows.append({
            "meal_type": str(getattr(meal, "meal_type", "")),
            "title": " + ".join(p.title for p in meal.plates if p.title),
            "plates": [
                {
                    "title": p.title,
                    "role": str(getattr(p, "role", "") or ""),
                    "recipe_id": str(getattr(p, "recipe_id", "") or ""),
                }
                for p in meal.plates
            ],
            "kcal": totals["calories"] if totals["meals_counted"] else None,
        })
    return rows


# --------------------------------------------------------------------------- #
# The daily canvas, read day by day
# --------------------------------------------------------------------------- #
#
# A multi-day plan is stored on the daily canvas as `MealPlan.days`, so these
# two read `day_plans` where the weekly digests read `plan.entries`. What they
# deliberately do NOT do is invent the parts a daily plan has no equivalent of:
# no weekday names, no per-day headline (the weekly planner writes those, this
# path has none), and no guideline checklist (a weekly metric). An empty field
# says "this plan does not carry that"; a zero would say "it scored nothing".


def _summarize_daily_plan(plan) -> dict:
    """Digest every day of a plan on the daily canvas."""
    from services.transparency import split_ledger

    days = []
    for day in plan.day_plans:
        totals = _sum_nutrition(_plate_pairs(day))
        days.append({
            "day": day.day,
            "name": _day_label(day.day, weekdays=False),
            "meals": _meal_rows(day),
            "kcal": totals["calories"],
            "nutrition_coverage":
                f"{totals['meals_counted']} of {totals['meals_total']} plates",
        })

    overall = _sum_nutrition(
        [pair for day in plan.day_plans for pair in _plate_pairs(day)]
    )
    honored, not_honored = split_ledger(plan.constraints_applied, limit=12)
    return {
        "plan_type": "daily",
        "plan_version": plan.version,
        "days": days,
        "total": overall,
        "daily_average_kcal": round(overall["calories"] / max(len(days), 1), 1),
        "constraints_honored": honored,
        "constraints_not_honored": not_honored,
        "personalization": plan.personalization_summary or {},
        "plan_summary": plan.reasoning or "",
    }


def _summarize_daily_day(plan, day: int) -> dict:
    """One day of a plan on the daily canvas, plate by plate."""
    match = next((d for d in plan.day_plans if int(d.day) == int(day)), None)
    if match is None:
        covers = ", ".join(
            _day_label(d.day, weekdays=False) for d in plan.day_plans
        )
        raise ToolError(
            f"{_day_label(day, weekdays=False)} isn't in this plan — it covers "
            f"{covers or 'nothing yet'}."
        )

    meals = []
    for meal in match.meals:
        for plate in meal.plates:
            meals.append({
                "meal_type": str(getattr(meal, "meal_type", "")),
                "title": plate.title,
                "recipe_id": str(getattr(plate, "recipe_id", "") or ""),
                "role": str(getattr(plate, "role", "") or ""),
                "ingredients": plate.ingredients or "",
                "nutrition": plate.nutrition or {},
                "why": [
                    r.get("label", "") for r in (plate.match_reasons or [])
                ],
            })

    return {
        "plan_type": "daily",
        "day": int(day),
        "name": _day_label(day, weekdays=False),
        "meals": meals,
        "totals": _sum_nutrition(_plate_pairs(match)),
    }


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
            "plan_type": {
                "type": "string", "enum": ["daily", "weekly"],
                "description": "Which canvas to digest. Defaults to whichever "
                               "holds a plan. A multi-day plan on the daily "
                               "canvas is summarised day by day like a week.",
            },
        },
        "required": ["session_id"],
    },
    examples=("summarise my week", "how does the week look overall?",
              "summarise the whole plan"),
)
def summarize_week(session_id: str, plan_type: str = "") -> dict:
    from services.transparency import split_ledger

    _session_obj, plan, kind = _day_plan(session_id, plan_type)
    if kind == "daily":
        return _summarize_daily_plan(plan)

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
                "description": "On a weekly plan, 1 = Monday through 7 = "
                               "Sunday. On a multi-day daily plan, 1 is its "
                               "first day — such a plan carries no weekdays.",
            },
            "plan_type": {
                "type": "string", "enum": ["daily", "weekly"],
                "description": "Which canvas the day belongs to. Defaults to "
                               "whichever holds a plan.",
            },
        },
        "required": ["session_id", "day"],
    },
    examples=("what's on Thursday?", "tell me about day 3"),
)
def summarize_day(session_id: str, day: int, plan_type: str = "") -> dict:
    _session_obj, plan, kind = _day_plan(session_id, plan_type)
    if kind == "daily":
        return _summarize_daily_day(plan, day)

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
                    "day": day.day,
                    "meal_type": getattr(meal, "meal_type", ""),
                    "title": plate.title,
                    "role": getattr(plate, "role", "") or "",
                    "nutrition": nutrition,
                })
    result = {
        "plan_type": "daily",
        "plan_version": plan.version,
        "plates": meals,
        "total": _sum_nutrition(pairs),
    }
    # A three-day plan totalled as one number reads as one enormous day. The
    # weekly branch has always split per day; the daily branch summed straight
    # through `day_plans` as though there could only ever be one.
    if len(plan.day_plans) > 1:
        result["per_day"] = [
            {
                "day": day.day,
                "name": _day_label(day.day, weekdays=False),
                **_sum_nutrition(_plate_pairs(day)),
            }
            for day in plan.day_plans
        ]
        result["daily_average_kcal"] = round(
            result["total"]["calories"] / len(plan.day_plans), 1
        )
    return result


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
    canvases=("weekly",),
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
    spec = _spec_of(plan)
    action_space = RecipeActionSpace(
        profile,
        additional_diet=list(state.diet_tags),
        pantry=state.pantry,
        # The shape of the plan being edited, not the default seven-by-three.
        spec=spec,
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
        spec=spec,
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
                "description": "Which day, on any plan that has more than one "
                               "— a weekly plan, or a multi-day plan on the "
                               "daily canvas. Omit for a single-day plan.",
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

    # The phrasing has to match what the edit reader will accept. On a weekly
    # plan that is the weekday; on a multi-day daily plan it is "day 2", and
    # `edit_service._named_day(weekdays=False)` reads NOTHING ELSE there — so
    # passing "Tuesday" asked a question the reader could not answer and the
    # edit came back asking which day it was.
    weekdays = _canvas_kind(_session(session_id)) == "weekly"
    where = f"{_day_label(day, weekdays=weekdays)} " if day else ""
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


@tool(
    "save_plan",
    summary="Keep the plan on screen so it outlives this conversation.",
    description=(
        "Saves the plan currently on the canvas — daily or weekly — to the "
        "member's saved plans, optionally under a name they choose. Saved "
        "plans appear in their library and survive the session being closed. "
        "Pass saved=false to take one back off the list. Deterministic: no "
        "model call, and it changes nothing about the plan itself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "The conversation whose plan to save."},
            "title": {
                "type": "string",
                "description": "What to call it — 'Meatless Monday', 'the "
                               "week I liked'. Optional; omit to save it "
                               "unnamed.",
            },
            "saved": {
                "type": "string", "enum": ["true", "false"],
                "description": "'false' un-saves a plan that was saved before.",
            },
        },
        "required": ["session_id"],
    },
    examples=("save this plan", "save my week as Meatless Monday",
              "actually don't keep that one"),
)
def save_plan(session_id: str, title: str = "", saved: str = "true") -> dict:
    """Save the plan the member is looking at.

    The endpoint for this has existed since saved plans were built, and chat
    could not reach it: "save this" was small talk. The tool is a reader in the
    sense that matters — it does not touch the plan, only whether the plan
    outlives the conversation — so `mutates` stays false and the canvas is not
    reloaded afterwards.
    """
    session = _session(session_id)
    canvas = session.active_canvas
    if canvas is None or not canvas.current_id:
        raise ToolError(
            "There's no plan in this conversation yet — ask for one and I'll "
            "save it for you."
        )

    keep = str(saved).strip().lower() != "false"
    name = str(title or "").strip()[:120]
    ok = _services().session_service.set_plan_saved(
        session_id, session.member_id, canvas.current_id, keep, name or None,
    )
    if not ok:
        # The session check inside `set_plan_saved` is the same one the router
        # already ran, so a false here means the plan id has gone — worth
        # saying rather than reporting a save that did not happen.
        raise ToolError("I couldn't find that plan to save.")

    return {
        "saved": keep,
        "title": name or None,
        "plan_type": canvas.plan_type,
        "plan_id": canvas.current_id,
    }


@tool(
    "shopping_list",
    summary="Everything the plan needs to buy, gathered from its own recipes.",
    description=(
        "Collects the ingredient lines from every dish on the plan and groups "
        "the ones that repeat, so a week's plan becomes one list rather than "
        "21. Works on the daily canvas or the weekly one. Deterministic — the "
        "lines come from the stored recipes, nothing is estimated, and no "
        "quantities are invented: the corpus stores ingredients as text, so "
        "the list says which meals need each item rather than how much."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "The conversation to read."},
            "plan_type": {
                "type": "string", "enum": ["daily", "weekly"],
                "description": "Which canvas. Defaults to whichever is active.",
            },
        },
        "required": ["session_id"],
    },
    examples=("what do I need to buy?", "shopping list for the week"),
)
def shopping_list(session_id: str, plan_type: str = "") -> dict:
    """One list for the whole plan, with what each item is for.

    Quantities are deliberately absent. The recipe corpus stores ingredients as
    free text ("2 tbsp olive oil", "olive oil", "olive oil, to serve"), and
    adding those up would mean parsing units the data does not reliably carry.
    A list that says "olive oil — for 4 meals" is true; one that says "6 tbsp"
    would be a number with nothing behind it.
    """
    session = _session(session_id)
    wanted = str(plan_type or "").strip().lower()
    if not wanted:
        canvas = session.active_canvas
        wanted = canvas.plan_type if canvas else "daily"

    if wanted == "weekly":
        _, plan = _weekly_plan(session_id)
        dishes = [
            (
                f"{_day_label(e.get('day'))} {e.get('meal_type') or 'meal'}",
                _entry_title(e),
                ((e.get("recipe") or {}).get("recipe_ingredients") or ""),
            )
            for e in plan.entries
        ]
    else:
        _, plan = _daily_plan(session_id)
        dishes = [
            (
                (f"day {day.day} {meal.meal_type}"
                 if len(plan.day_plans) > 1 else meal.meal_type),
                plate.title,
                plate.ingredients or "",
            )
            for day in plan.day_plans
            for meal in day.meals
            for plate in meal.plates
            if plate.recipe_id
        ]

    if not dishes:
        raise ToolError("That plan has no dishes to shop for yet.")

    # item -> the meals that need it, in the order the plan runs.
    items: dict[str, dict] = {}
    from services.plan_quality import extract_ingredient_names

    for where, title, text in dishes:
        # The plan's own normalizer, not a second one. A shopping list that
        # groups items differently from the way the variety score counts them
        # is two answers to "what is in this plan".
        for name in extract_ingredient_names(text):
            row = items.setdefault(name, {"item": name, "for": []})
            if where not in row["for"]:
                row["for"].append(where)

    ordered = sorted(
        items.values(), key=lambda r: (-len(r["for"]), r["item"]),
    )
    return {
        "plan_type": wanted,
        "dishes": len(dishes),
        "items": ordered,
        "item_count": len(ordered),
        # Said explicitly so a reply cannot imply the list carries amounts.
        "quantities": "not available — the recipes store ingredients as text",
    }
