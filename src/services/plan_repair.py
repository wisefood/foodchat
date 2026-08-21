"""
Fix the plates the verifier named, once.

The verifier reports which plates failed and why. Until now nothing read that:
`report.offenders` had no consumer anywhere in the codebase. A plan that failed
its own vegetarian check was rendered with a red chip and handed over. The
assistant checked its work and then filed a complaint about it.

This is the missing step, and it is deliberately small:

    one pass          not a loop — a second failure is reported, not chased
    failing plates    the rest of the plan is untouched, byte for byte
    hard checks only  allergens and diet; a calorie miss is worth SAYING, not
                      worth swapping a dish the member may already like
    re-verified       the repaired plan is measured again, so a repair that
                      did not work cannot be announced as one

**One pass is a design decision, not a shortcut.** Each pass costs a fetch per
failing slot and a re-verify, inside a turn budget that also has to buy the
plan itself. And a second pass that fails usually means the corpus cannot
satisfy the constraint at all — a member who wants a vegan plan from a corpus
with four vegan dinners is not helped by three more rounds of searching; they
are helped by being told. Looping there would spend the budget to arrive at the
same sentence.

    repair(plan, brief, report, profile) -> RepairOutcome

Never raises. A repair that cannot run leaves the plan exactly as it was, which
is the behaviour that existed before this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Checks worth swapping a dish over.
#
# Both are hard constraints the member did not ask to have relaxed. Everything
# else the verifier reports — calories, cooking time, Nutri-Score, pantry
# coverage — is a soft signal, and swapping a dish someone may already be
# looking forward to because the day came in 12% over target is a worse outcome
# than the honest sentence "this day runs a bit high".
REPAIRABLE = frozenset({"allergens", "diet"})

# How many plates one pass will replace. A plan where most plates fail a hard
# check is not a repair job — it is a pool that never satisfied the constraint,
# and replacing eleven dishes one at a time would spend the whole turn budget
# discovering that.
MAX_REPAIRS = 4


@dataclass
class RepairOutcome:
    """What the pass did, in terms the reply can use."""

    plan: object
    repaired: list[dict] = field(default_factory=list)
    # Plates that failed and could not be replaced. Named, because "we could
    # not fix this" is the part the member most needs.
    unresolved: list[dict] = field(default_factory=list)
    report: object = None

    @property
    def changed(self) -> bool:
        return bool(self.repaired)


def repairable_offenders(report) -> list[tuple[str, str]]:
    """(recipe_id, check name) for the plates worth replacing, in check order."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for check in getattr(report, "checks", []) or []:
        if check.name not in REPAIRABLE or check.status != "failed":
            continue
        for recipe_id in check.offenders:
            if recipe_id and recipe_id not in seen:
                seen.add(recipe_id)
                out.append((recipe_id, check.name))
    return out


def repair(plan, brief, report, profile: dict, *, client=None) -> RepairOutcome:
    """Replace the plates that failed a hard check, then measure again.

    The plan is mutated in place: it is already assembled, already enriched,
    and already carries the member's own adapted recipes and reason chips on
    every OTHER plate. Rebuilding it to change two dishes would throw all of
    that away to avoid one mutation.
    """
    outcome = RepairOutcome(plan=plan, report=report)
    offenders = repairable_offenders(report)
    if not offenders:
        return outcome

    if len(offenders) > MAX_REPAIRS:
        # Say so rather than silently repairing the first four: a plan where
        # most plates fail a hard check is a pool that never satisfied it, and
        # a partial fix reads as a whole one.
        logger.warning(
            "%d plates fail a hard check — too many to repair; reporting instead",
            len(offenders),
        )
        outcome.unresolved = [
            {"recipe_id": rid, "check": name, "reason": "too many to repair"}
            for rid, name in offenders
        ]
        return outcome

    if client is None:
        from services.candidates_client import CANDIDATES as client

    # Every recipe already on the plan, so a repair cannot introduce a
    # duplicate — or hand back the very dish it is replacing.
    in_plan = [
        plate.recipe_id
        for day in plan.day_plans for meal in day.meals for plate in meal.plates
        if plate.recipe_id
    ]

    for recipe_id, check_name in offenders:
        located = _locate(plan, recipe_id)
        if located is None:
            continue
        meal, index, plate = located

        try:
            candidates = client.slot_candidates(
                profile, meal.meal_type,
                exclude_ids=in_plan + list(brief.exclude_recipe_ids),
                limit=8,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Repair fetch failed for %s: %s", meal.meal_type, exc)
            candidates = []

        if not candidates:
            outcome.unresolved.append({
                "recipe_id": recipe_id, "title": plate.title, "check": check_name,
                "slot": meal.meal_type,
                "reason": "no other dish in the collection fits",
            })
            continue

        replacement = candidates[0]
        from models.session import MealCourse

        meal.plates[index] = MealCourse(
            recipe_id=replacement.recipe_id,
            title=replacement.title,
            ingredients=replacement.ingredients,
            directions=replacement.directions,
            # The plate keeps its place in the meal — a side stays a side.
            role=getattr(plate, "role", "main"),
            match_reasons=[{
                "kind": "profile",
                "label": f"swapped — the first pick did not meet your {check_name}",
            }],
        )
        in_plan.append(replacement.recipe_id)
        outcome.repaired.append({
            "slot": meal.meal_type, "check": check_name,
            "was": plate.title, "now": replacement.title,
        })
        logger.info(
            "Repaired %s: %r -> %r (%s)",
            meal.meal_type, plate.title, replacement.title, check_name,
        )

    if not outcome.repaired:
        return outcome

    # Measure again. A repair that did not work must not be announced as one —
    # and the ledger the member reads has to describe the plan they were given,
    # not the one that existed before the swap.
    from services import plan_verifier

    fresh_ids = [row["now"] for row in outcome.repaired]
    try:
        enrichment = client.fetch_details([
            plate.recipe_id
            for day in plan.day_plans for meal in day.meals for plate in meal.plates
            if plate.recipe_id
        ])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Re-enrichment after repair failed: %s", exc)
        enrichment = {}

    outcome.report = plan_verifier.verify(plan, brief.to_requested(), enrichment)
    logger.info(
        "After repair (%d plate(s), %s): %s",
        len(outcome.repaired), ", ".join(fresh_ids),
        plan_verifier.describe(outcome.report),
    )

    # Anything still failing after one pass is reported, not chased.
    still_failing = repairable_offenders(outcome.report)
    for rid, name in still_failing:
        outcome.unresolved.append({
            "recipe_id": rid, "check": name,
            "reason": "still not right after one attempt",
        })
    return outcome


def _locate(plan, recipe_id: str):
    """(meal, plate index, plate) for a recipe id, or None."""
    for day in plan.day_plans:
        for meal in day.meals:
            for index, plate in enumerate(meal.plates):
                if plate.recipe_id == recipe_id:
                    return meal, index, plate
    return None


def describe(outcome: RepairOutcome) -> Optional[str]:
    """One sentence for the reply, or None when nothing happened.

    Both halves, always. A repair announced without its failures lets the
    member believe a plan is clean when one dish on it is not.
    """
    if not outcome.repaired and not outcome.unresolved:
        return None
    parts = []
    if outcome.repaired:
        swaps = ", ".join(f"{r['was']} → {r['now']}" for r in outcome.repaired[:3])
        parts.append(f"Swapped {swaps} to meet your requirements")
    if outcome.unresolved:
        parts.append(
            f"{len(outcome.unresolved)} dish(es) still fall short and the "
            "collection had nothing better"
        )
    return "; ".join(parts) + "."
