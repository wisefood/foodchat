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

M9 adds two more things a week can now do, and the labels that keep them
apart from each other and from the member's own requests:

- a recipe may **repeat** (breakfast only — see ``action_adapter``). Every
  repeat carries the day it repeats and whether it was the member's doing
  (a starred recipe) or the plan's own, through to the chip, the ledger row,
  the variety metric and the prose. A duplicate that carries neither reason
  is reported as ``unexplained``, never folded into the sanctioned count.
- the plan may **search later days for ingredients it has already bought**.
  What it searched for is a ledger row; what it actually reused is measured
  separately over the finished week by ``shared_ingredient_facts``, because a
  search that found nothing usable is not a saving.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from services.adapted_recipes import ADAPTED_REASON
from services.transparency import constraints_ledger, match_reasons, personalization_summary

from .day_summary import classify_meal, is_meat_meal
from .planner import nameable_phrases
from .reward_logic import candidate_kcal
from .state_tracking import WeeklyNutritionalTracker

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Soft calorie budget counts as satisfied up to this share of the target.
CALORIE_TOLERANCE = 1.05
# ...and down to this share of it. A ceiling on its own is not a budget check:
# a week planned at 47% of target — 933 kcal a day — passed as "satisfied",
# because nothing ever asked whether the member would be fed. Looser than the
# ceiling on purpose: per-serving figures from a recipe database are
# approximate and real weeks vary, so this is set to catch a week that is
# wrong rather than one that is merely light.
CALORIE_FLOOR = 0.85


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
# Sanctioned repeats (M9)                                                 #
#                                                                         #
# A recipe may now appear twice in a week — breakfast only, spaced, and   #
# capped (`action_adapter`'s repeat policy). Two very different things    #
# can produce that, and the week must never present one as the other:     #
# a breakfast the member starred coming back on Thursday is the plan      #
# doing what they asked, while the planner choosing a second serving on   #
# its own is a mechanism they never opted into. Every repeat therefore    #
# carries the day it repeats and the authority it repeated under, from    #
# the candidate that was picked all the way to the chip on the card.      #
# --------------------------------------------------------------------- #

REPEAT_KIND = "repeat"

# Mirrors `action_adapter.REPEAT_MEMBER_REQUEST` / `REPEAT_PLAN`. Duplicated
# as a presentation constant rather than imported, so the label vocabulary the
# UI depends on cannot change as a side effect of a planning-side edit.
REPEAT_BY_MEMBER = "member_request"
REPEAT_BY_PLAN = "plan"


def repeat_facts(plan_entries: List[dict]) -> dict:
    """Which meals repeat an earlier one, and on whose authority.

    Measured over the finished week, from the flag the action space set on
    the candidate it allowed back — not from the policy constants. If a
    repeat reached the plate some other way (a pinned dish, a slot edit) it
    shows up here as ``unexplained``, which is the honest reading: nobody
    recorded a reason for it, so nobody may claim one.

    Returns ``{"entries": {index: {"of_day", "source"}}, "count": n,
    "by_source": {...}, "min_gap_days": g|None, "max_appearances": n,
    "unexplained": n}``.
    """
    per_entry: Dict[int, dict] = {}
    by_source: Dict[str, int] = {}
    gaps: List[int] = []
    appearances: Counter = Counter()
    # recipe_id -> the days it is actually on, so a flag can be checked
    # against the plan rather than believed. `edit_service` replaces a slot
    # with a freshly built recipe dict, which clears the flag on the slot it
    # edits and leaves the OTHER serving still claiming to repeat a day that
    # no longer has it. A stale flag must not become a ledger row.
    served_on: Dict[str, set] = {}
    for entry in plan_entries:
        recipe_id = str(_recipe(entry).get("recipe_id") or "")
        if recipe_id:
            appearances[recipe_id] += 1
            if isinstance(entry.get("day"), int):
                served_on.setdefault(recipe_id, set()).add(entry["day"])

    for index, entry in enumerate(plan_entries):
        recipe = _recipe(entry)
        of_day = recipe.get("repeat_of_day")
        if not of_day:
            continue
        recipe_id = str(recipe.get("recipe_id") or "")
        day = entry.get("day")
        if int(of_day) not in served_on.get(recipe_id, set()):
            continue
        if isinstance(day, int) and int(of_day) >= day:
            continue
        source = str(recipe.get("repeat_source") or REPEAT_BY_PLAN)
        per_entry[index] = {"of_day": int(of_day), "source": source}
        by_source[source] = by_source.get(source, 0) + 1
        if isinstance(day, int):
            gaps.append(day - int(of_day))

    duplicates = max(sum(appearances.values()) - len(appearances), 0)
    return {
        "entries": per_entry,
        "count": len(per_entry),
        "by_source": by_source,
        "min_gap_days": min(gaps) if gaps else None,
        "max_appearances": max(appearances.values()) if appearances else 0,
        "unexplained": max(duplicates - len(per_entry), 0),
    }


def _repeat_reason(entry: dict, fact: dict) -> dict:
    """One chip saying what came back, from when, and who wanted it."""
    meal = str(entry.get("meal_type") or "meal").lower()
    origin = _day_name(fact["of_day"])
    if fact["source"] == REPEAT_BY_MEMBER:
        label = f"back from {origin}, a favorite of yours"
    else:
        label = f"the same {meal} as {origin}"
    return {"kind": REPEAT_KIND, "label": label, "source": fact["source"]}


def attach_repeat_reasons(plan_entries: List[dict], repeats: dict) -> None:
    """Append the repeat chip to every entry the planner allowed back.

    Appends rather than replaces: a repeated favourite still carries its
    "one of your favourites" chip from `attach_match_reasons`, and dropping
    that to make room would lose the reason it was eligible at all.
    """
    for index, fact in repeats["entries"].items():
        recipe = _recipe(plan_entries[index])
        if not recipe:
            continue
        reasons = list(recipe.get("match_reasons") or [])
        reasons.append(_repeat_reason(plan_entries[index], fact))
        recipe["match_reasons"] = reasons


# --------------------------------------------------------------------- #
# Cross-day ingredient reuse (M8)                                         #
#                                                                         #
# Distinct from the pantry chip, and deliberately worded so a member can  #
# tell which is which. "uses your tomatoes" means the member said they    #
# had tomatoes; "also uses Monday's cabbage" means nothing was said about #
# cabbage and the plan bought it once for two meals. Conflating them      #
# would credit the member for an inference, or credit the planner for the #
# member's own statement.                                                 #
# --------------------------------------------------------------------- #

SHARED_INGREDIENT_KIND = "shared_ingredient"
SHARED_BADGE_FOODWASTE = "reducing food waste"

# Items named in one chip before it stops being readable.
_SHARED_ITEMS_PER_CHIP = 2

# A share is *found* by token and *named* by the phrase the token came from:
# tokenising "self raising flour" once told a member she was reusing
# "Thursday's self" and "Thursday's raising", from one bag of flour. That
# rule now lives in `planner.nameable_phrases`, because the sourcing half of
# this axis needs the same names — an ingredient worth naming to a member is
# exactly an ingredient worth searching RecipeWrangler for, and two
# definitions of "an ingredient" would drift apart within a release. A share
# the phrase rules cannot name is not counted at all: under-reporting is the
# safe direction, the same posture the pantry matcher takes.


def _pantry_stems(pantry: Any) -> set:
    """Stems of every word the member named, so their items are not re-labelled.

    An ingredient the member told us about already carries the pantry chip.
    Adding "also uses Monday's tomatoes" beside "uses your tomatoes" would
    blur exactly the distinction these two chips exist to draw.
    """
    from services.pantry_service import normalize_items, singular

    return {
        singular(word)
        for item in normalize_items(pantry or ())
        for word in str(item).split()
        if word
    }


def shared_ingredient_facts(plan_entries: List[dict], pantry: Any = ()) -> dict:
    """Which meals reuse an ingredient the plan itself introduced.

    Measured over the finished week, not over what the scorer intended: the
    claim is "these two meals share cabbage", which is either true of the
    plan on the page or it is not. A meal is credited only against an
    *earlier* day — the first appearance introduced the ingredient and has
    nothing to point back to — so nothing is double-counted.

    Returns ``{"entries": {index: [(name, day)]}, "meals": n, "items": [...]}``
    where ``name`` is the whole ingredient phrase as the earlier day wrote it.
    """
    pantry_stems = _pantry_stems(pantry)
    phrases = [
        nameable_phrases(_ingredients(_recipe(e)), pantry_stems)
        if isinstance(e.get("day"), int) else []
        for e in plan_entries
    ]
    # A meal that IS an earlier meal is not news about ingredients. Telling a
    # member "the same breakfast as Monday" and "also uses Monday's tomatoes"
    # on one card says the same thing twice, and counting it would inflate the
    # reuse figure with the repeat count. The repeat chip covers it.
    repeated = {
        index for index, entry in enumerate(plan_entries)
        if _recipe(entry).get("repeat_of_day")
    }
    # stem -> (earliest day it appears on, how that day named it)
    origin: Dict[str, tuple] = {}
    for entry, entry_phrases in zip(plan_entries, phrases):
        day = entry["day"]
        for display, stems in entry_phrases:
            for stem in stems:
                if stem not in origin or day < origin[stem][0]:
                    origin[stem] = (day, display)

    per_entry: Dict[int, List[tuple]] = {}
    items: List[str] = []
    for index, (entry, entry_phrases) in enumerate(zip(plan_entries, phrases)):
        if not entry_phrases or index in repeated:
            continue
        day = entry["day"]
        shared: List[tuple] = []
        for _display, stems in entry_phrases:
            for stem in sorted(stems):
                origin_day, origin_name = origin.get(stem, (day, ""))
                if origin_day >= day:
                    continue
                # Keyed by the earlier day's phrase, so several tokens out of
                # one ingredient ("self", "raising") collapse into one item.
                if (origin_name, origin_day) not in shared:
                    shared.append((origin_name, origin_day))
                if origin_name not in items:
                    items.append(origin_name)
        if shared:
            per_entry[index] = shared
    return {"entries": per_entry, "meals": len(per_entry), "items": items}


def _shared_reason(shared: List[tuple]) -> dict:
    """One chip naming what the matcher found, and which day it came from."""
    named = shared[:_SHARED_ITEMS_PER_CHIP]
    parts = [f"{_day_name(day)}'s {item}" for item, day in named]
    return {
        "kind": SHARED_INGREDIENT_KIND,
        "label": "also uses " + " and ".join(parts) + f" — {SHARED_BADGE_FOODWASTE}",
    }


def annotate_shared_ingredients(
    plan_entries: List[dict],
    pantry: Any = (),
    explainability: Optional[dict] = None,
) -> dict:
    """Attach the cross-day reuse chip in place, and a ledger row.

    Runs AFTER ``build_weekly_explainability`` and after the pantry
    annotation, so it appends to the chips those built rather than
    replacing them. Unlike the pantry annotation it runs even when the
    member stated no pantry — this reuse is the plan's own doing, so there
    is nothing for the member to have said first.

    Also appends a sentence to ``explainability["reasoning"]``. Running last
    is what makes the chips additive, but it also meant the whole-week
    justification was composed before anyone had measured the reuse, so the
    one axis a member is most likely to ask about ("why is Wednesday's slaw
    here?") was the one the prose never mentioned.
    """
    facts = shared_ingredient_facts(plan_entries, pantry)
    for index, shared in facts["entries"].items():
        recipe = _recipe(plan_entries[index])
        if not recipe:
            continue
        reasons = list(recipe.get("match_reasons") or [])
        reasons.append(_shared_reason(shared))
        recipe["match_reasons"] = reasons

    if explainability is not None and facts["meals"]:
        ledger = list(explainability.get("constraints_applied") or [])
        ledger.append({
            "constraint": (
                f"{facts['meals']} meal(s) reuse an ingredient from an earlier "
                f"day — {SHARED_BADGE_FOODWASTE}"
            ),
            "type": "soft",
            "status": "satisfied",
            "source": "the plan",
            "detail": "shares " + ", ".join(facts["items"][:6]),
        })
        explainability["constraints_applied"] = ledger

        named = ", ".join(facts["items"][:3])
        sentence = (
            f"{facts['meals']} meal(s) reuse an ingredient introduced earlier "
            f"in the week ({named}), so the week buys fewer separate things "
            "than 21 unrelated recipes would."
        )
        # Appended, never assigned: `_compose_reasoning` has already written
        # the meat, calorie and anchor sentences, and replacing that string
        # would drop them.
        existing = str(explainability.get("reasoning") or "").strip()
        explainability["reasoning"] = f"{existing} {sentence}".strip()
    return facts


# --------------------------------------------------------------------- #
# Weekly metrics (deterministic — no LLM)                                 #
# --------------------------------------------------------------------- #

def variety_metrics(plan_entries: List[dict], repeats: Optional[dict] = None) -> dict:
    """Distinct recipes, planned repeats, unique ingredients, categories.

    A repeat is not automatically a failure of variety. Nobody eats seven
    different breakfasts, and one the member starred coming back on Thursday
    is the week working as intended — so it is counted separately and the
    prose does not treat it as a shortfall.

    A duplicate that nobody sanctioned is a different matter, and stays
    visible as ``unexplained_repeats``: reporting both as one "21 minus
    distinct" number is exactly how a thin candidate pool ends up presented
    as if the member had asked for it.
    """
    facts = repeats if repeats is not None else repeat_facts(plan_entries)
    recipes = [_recipe(e) for e in plan_entries]
    distinct = len({
        str(r.get("recipe_id") or "") for r in recipes if r.get("recipe_id")
    })
    items: set = set()
    for r in recipes:
        items.update(_ingredient_names(_ingredients(r)))
    categories = Counter(classify_meal(r) for r in recipes if r)
    total = len(plan_entries)
    planned_repeats = facts["count"]
    unexplained = facts["unexplained"]
    if distinct == total:
        headline = f"All {total} meals are distinct recipes"
    elif planned_repeats and not unexplained:
        headline = (
            f"{distinct} recipes across {total} meals, "
            f"with {planned_repeats} planned repeat(s)"
        )
    else:
        headline = f"{distinct} distinct recipes across {total} meals"
    return {
        "distinct_recipes": distinct,
        "total_meals": total,
        # Repeats the planner allowed and recorded a reason for, split by that
        # reason. `unexplained_repeats` is the honest remainder: a duplicate
        # with nothing behind it, which is monotony and reads as one.
        "planned_repeats": planned_repeats,
        "repeats_by_source": dict(facts["by_source"]),
        "unexplained_repeats": unexplained,
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


def calorie_budget_status(
    planned: float, target_kcal: float, covered: int, total_meals: int
) -> Optional[str]:
    """``"over"`` | ``"under"`` | ``"on_track"``, or None when nothing is known.

    Both directions, decided in one place so the metric, the ledger row and
    the prose cannot drift apart. The ledger used to check only the ceiling,
    which is not a budget check: a week planned at 47% of target — 933 kcal a
    day — was reported as satisfied and handed to the reply as an honoured
    request, because nothing ever asked whether the member would be fed.

    The floor is measured against the meals we actually have data for, not
    against the whole week. Judging a 19-of-21 week by the full target would
    report "short of target" for two missing data points — a claim about our
    coverage, dressed up as a claim about the member's food.
    """
    if target_kcal <= 0 or not covered:
        return None
    expected = target_kcal * (covered / total_meals if total_meals else 1.0)
    if planned > target_kcal * CALORIE_TOLERANCE:
        return "over"
    if expected > 0 and planned < expected * CALORIE_FLOOR:
        return "under"
    return "on_track"


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

    budget_status = calorie_budget_status(
        totals["kcal"], target_kcal, covered, total_meals,
    )

    return {
        "weekly_totals": {name: round(value, 1) for name, value in totals.items()},
        "daily_average_kcal": round(totals["kcal"] / 7.0, 1) if covered else None,
        "weekly_targets": weekly_targets,
        "budget_used_pct": budget_used_pct,
        # "over" | "under" | "on_track" | None (no target, or nothing measured)
        "budget_status": budget_status,
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
    repeats: Optional[dict] = None,
) -> List[dict]:
    """Profile rows (same as daily) + measured weekly rows.

    The weekly rows carry a ``detail`` string with the actual numbers, and
    a status that reports what happened: ``relaxed`` when the planner had
    to relax the meat limit for a slot whose every candidate contained
    meat (recorded at decision time, not reconstructed), ``violated`` when
    the final week breaks the limit anyway (e.g. pinned dishes bypass
    constraints).
    """
    from .action_adapter import (  # local import; keeps the policy single-sourced
        MAX_APPEARANCES,
        REPEAT_MIN_GAP_DAYS,
    )

    ledger = constraints_ledger(profile, downvoted_count)

    repeats = repeats or {}
    if repeats.get("count"):
        by_source = repeats.get("by_source") or {}
        who = []
        if by_source.get(REPEAT_BY_MEMBER):
            who.append(f"{by_source[REPEAT_BY_MEMBER]} you starred")
        if by_source.get(REPEAT_BY_PLAN):
            who.append(f"{by_source[REPEAT_BY_PLAN]} the plan's own")
        gap = repeats.get("min_gap_days")
        served = int(repeats.get("max_appearances") or 0)
        detail = f"{repeats['count']} meal(s) repeat an earlier day"
        if who:
            detail += " (" + ", ".join(who) + ")"
        if gap is not None:
            detail += f"; closest repeat {gap} day(s) apart"
        detail += f"; no recipe served more than {served} time(s)"
        # Measured against the policy, not asserted from it: a pinned dish or
        # a slot edit can put a duplicate on the plate without ever passing
        # through the cooldown, and the row has to be able to say so.
        within_policy = (
            (gap is None or gap >= REPEAT_MIN_GAP_DAYS)
            and served <= MAX_APPEARANCES
        )
        if by_source.get(REPEAT_BY_MEMBER) and by_source.get(REPEAT_BY_PLAN):
            source = "your favourites and the plan"
        elif by_source.get(REPEAT_BY_MEMBER):
            source = "your favourites"
        else:
            source = "the plan"
        ledger.append({
            "constraint": "repeat meals stay spaced and capped",
            "type": "soft",
            "status": "satisfied" if within_policy else "violated",
            "source": source,
            "detail": detail,
        })
    if repeats.get("unexplained"):
        # A duplicate nobody recorded a reason for. Reported rather than
        # quietly folded into the repeat count above, because the whole point
        # of labelling repeats is that an unexplained one is not a feature.
        ledger.append({
            "constraint": "every repeat has a recorded reason",
            "type": "soft",
            "status": "violated",
            "source": "the plan",
            "detail": (
                f"{repeats['unexplained']} duplicate meal(s) carry no reason — "
                "not the repeat policy's doing"
            ),
        })

    # Cross-day reuse, sourcing half: the days that spent extra RecipeWrangler
    # calls looking for the week's own ingredients. Only the days that actually
    # searched get a row — when the food-waste setting skips it, nothing was
    # applied, so there is no constraint to report; the `derived_pantry_skipped`
    # event in `metrics.selection_events` is where that trace lives instead.
    sourced = [e for e in selection_events if e.get("type") == "derived_pantry_sourced"]
    if sourced:
        searched: List[str] = []
        for event in sourced:
            for item in event.get("items") or []:
                if item not in searched:
                    searched.append(item)
        ledger.append({
            "constraint": "look for recipes using ingredients the week already buys",
            "type": "soft",
            "status": "satisfied",
            "source": "food-waste setting",
            "detail": (
                f"{len(sourced)} day(s) also searched for "
                + ", ".join(searched[:6])
                + " — what the plan then reused is measured separately"
            ),
        })

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
        # A budget has two sides. The row used to check only the ceiling, so a
        # week that fed the member half of what they need was reported as
        # honoured — and `split_ledger` then handed it to the reply as an
        # honoured request.
        #
        # Derived here when the caller's payload predates the field, rather
        # than read as absent: a missing measurement must not turn into a
        # violation, which is the same mistake in the other direction.
        budget_status = nutrition.get("budget_status") or calorie_budget_status(
            planned, target_kcal, covered,
            int((nutrition.get("coverage") or {}).get("total_meals") or 0),
        )
        if budget_status == "under":
            detail += "; short of your target for the meals we have data for"
        elif budget_status == "over":
            detail += "; over your target"
        ledger.append({
            "constraint": "weekly calorie target",
            "type": "soft",
            "status": "satisfied" if budget_status == "on_track" else "violated",
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
    repeats: Optional[dict] = None,
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
        if nutrition.get("budget_status") == "under":
            sentence += " — noticeably short of what you asked for"
        elif nutrition.get("budget_status") == "over":
            sentence += " — over what you asked for"
        parts.append(sentence + ".")

    repeats = repeats or {}
    if repeats.get("count"):
        by_source = repeats.get("by_source") or {}
        who = []
        if by_source.get(REPEAT_BY_MEMBER):
            who.append(f"{by_source[REPEAT_BY_MEMBER]} you'd starred")
        if by_source.get(REPEAT_BY_PLAN):
            who.append(f"{by_source[REPEAT_BY_PLAN]} the plan's own choice")
        sentence = (
            f"{repeats['count']} meal(s) repeat earlier in the week rather than "
            "filling every slot with something new"
        )
        if who:
            sentence += " — " + " and ".join(who)
        gap = repeats.get("min_gap_days")
        if gap is not None:
            sentence += f", never closer together than {gap} day(s)"
        parts.append(sentence + ".")
    if repeats.get("unexplained"):
        parts.append(
            f"{repeats['unexplained']} further duplicate meal(s) appear with no "
            "reason recorded for them."
        )

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
    # Repeats are read off the entries themselves (the flag the action space
    # set on the candidate it allowed back), so this works for patched plans
    # and restored plans exactly as it does for freshly generated ones.
    repeats = repeat_facts(plan_entries)
    attach_repeat_reasons(plan_entries, repeats)

    variety = variety_metrics(plan_entries, repeats)
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
        repeats=repeats,
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
        repeats=repeats,
    )

    return {
        "constraints_applied": ledger,
        "personalization_summary": personalization_summary(profile, feedback_lines),
        "metrics": {
            "variety": variety,
            "guideline_checklist": checklist,
            "nutrition": nutrition,
            "days": days,
            # Counts only. `entries` is keyed by list index, and this payload
            # is stored as JSON — the keys would come back as strings on the
            # next read. Nothing needs it there anyway: which meal is a repeat
            # is already on the entry itself, as `recipe.repeat_of_day`.
            "repeats": {k: v for k, v in repeats.items() if k != "entries"},
            "selection_events": events,
        },
        "reasoning": reasoning,
    }
