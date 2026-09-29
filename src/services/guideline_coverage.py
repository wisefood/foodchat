"""
Food-based guidelines that shape the plan, not only grade it.

    targets(profile, plan_type)  -> [CoverageTarget]   rules the corpus can act on
    shape(spec, targets)         -> PlanSpec           the fetch, steered
    measure(plan, targets)       -> [row]              counted on the plates
    unverifiable(rules)          -> [row]              what cannot be checked, and why

A member's plan came back and the guideline judge said this:

    The plan lacks wholemeal cereals/breads at each meal (violates G8) and
    provides far fewer than the 3-5 daily servings of whole grains required
    (G57). No dairy servings are included, breaching G5/G11. …

Every word of it true, and none of it reachable. **Guidelines only ever
graded.** Every call site — `chat_service`, `weekly_plan_service`,
`explainability.guideline_checklist`, `plan_scorer` — runs after a plan exists.
Nothing put a guideline in front of the search, so the planner was marked down
for rules it was never told.

The pieces to fix that were already on both sides and nothing joined them:

* RecipeWrangler filters per slot on 13 FoodOn-derived food groups, and
  `PlanSpec.slot_food_groups` has carried that request since the shaped-plan
  work;
* every recipe card comes back carrying its own `food_groups`, so a plan can be
  counted exactly rather than guessed at (`CandidateRecipe.food_groups`);
* a catalogue rule carries `food_groups` of its own.

**The two vocabularies are different, and where they do not meet is the
honest limit of this module.** The catalogue's enum has twelve values; the
corpus has thirteen groups. Five of the catalogue's map onto something that can
be both filtered and counted — and they happen to be exactly the families the
judge cited. The rest cannot, and are reported as unverifiable rather than
searched for, scored, or quietly dropped.

**Soft, never hard.** A guideline joins the fetch as a preference
RecipeWrangler may relax, not a filter it must satisfy. Regional scope, diet,
allergens, dislikes and the recently-served exclusion are already applied; a
sixth mandatory term empties slots, and an empty slot helps nobody. So the plan
is shaped, then measured on what actually arrived, and the gap is reported.

**The member outranks the guideline.** Someone who asks for a low-carb week
gets one; the ledger then says which rules that cost. Same precedence the diet
and allergen paths already have, and the same reason: this is an assistant,
not a clinician.

**Meals are not servings.** The corpus has recipes, not portions. "Three
servings of whole grains a day" is measured here as *meals containing grains*,
and every row says so. Counting one as the other would be a measurement
wearing the wrong name, which is the failure this codebase keeps finding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# The catalogue's `food_groups` enum → the corpus groups that satisfy it.
#
# `protein_foods` is a SET, and deliberately: "two servings of meat, poultry,
# fish, eggs or beans" is met by any of them, so a plate carrying one counts.
# Reading it as a single group is what made `guidelines_service` dismiss this
# field as "a coarse enum [that] cannot name a category" — true for its
# fish-versus-red-meat counter, and not true here.
GUIDELINE_TO_CORPUS: dict[str, tuple[str, ...]] = {
    "fruits": ("fruit",),
    "vegetables": ("vegetables",),
    "grains": ("grains",),
    "dairy": ("dairy",),
    "protein_foods": (
        "meat", "poultry", "fish", "seafood", "eggs", "legumes", "nuts_and_seeds",
    ),
    "sugars_and_sweets": ("sugars",),
}

# The catalogue values with no corpus counterpart, and the reason — which is
# what the ledger says instead of a verdict. Reporting these as failures marks
# a plan down for something nothing could have planned for or measured; leaving
# them out silently is the same fault facing the other way.
NO_CORPUS_EQUIVALENT: dict[str, str] = {
    "fats_and_oils": (
        "recipes carry ingredient text, not portions, so a per-person oil "
        "allowance cannot be measured from a plan"
    ),
    "beverages": "a meal plan carries no drinks, so fluid intake is not in it",
    "salt": "salt content is not annotated per recipe",
}

# Groups whose corpus name is broader than the rule's own words. The filter and
# the count are still real; the row has to say what it actually measured.
BROADER_THAN_ASKED: dict[str, str] = {
    "grains": (
        "measured as grains — the corpus does not distinguish wholemeal from "
        "refined, so a white loaf counts here too"
    ),
}

# "at each meal", "with every meal", "at all main meals".
_PER_MEAL = re.compile(
    r"\b(?:at|with|in|for)\s+(?:each|every|all)\s+(?:main\s+)?meals?\b"
    r"|\b(?:each|every)\s+(?:main\s+)?meal\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CoverageTarget:
    """One food-based rule, in terms the planner and the counter both speak."""

    rule: str
    # Corpus food groups that satisfy it. Any one of them counts.
    groups: tuple[str, ...]
    # "at least" | "at most" | "about"
    direction: str
    low: Optional[float]
    high: Optional[float]
    period: str                       # "day" | "week"
    # The rule asks for this at every meal, not N times over the period.
    per_meal: bool
    source: str
    urn: str
    # The catalogue's own group name, kept for the caveat the row may need.
    guideline_group: str

    @property
    def caveat(self) -> str:
        return BROADER_THAN_ASKED.get(self.guideline_group, "")


def _corpus_groups(rule: dict) -> tuple[str, str, ...]:
    """`(catalogue group, corpus groups)` for a rule, or `("", ())`.

    Exactly one mappable group, or none. "Two servings a day of meat, poultry,
    fish or eggs" is one rule about protein; a rule tagged both `dairy` and
    `grains` is describing a meal, and splitting it into two targets would
    invent a demand it never made.
    """
    named = {
        str(group).strip().lower()
        for group in (rule.get("food_groups") or [])
        if str(group).strip().lower() not in ("", "none", "mixed", "other")
    }
    mappable = {g for g in named if g in GUIDELINE_TO_CORPUS}
    if len(mappable) != 1:
        return "", ()
    group = mappable.pop()
    return group, GUIDELINE_TO_CORPUS[group]


def targets(profile: Optional[dict] = None, plan_type: str = "daily") -> list[CoverageTarget]:
    """The rules this member's scope carries that the corpus can act on.

    `[]` whenever the catalogue cannot answer — a plan's guidelines are not
    worth a failed turn, the same rule `guidelines_text` follows.
    """
    from services import guidelines_service

    try:
        rules = guidelines_service.fetch(profile, plan_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Guideline coverage unavailable: %s", exc)
        return []
    return from_rules(rules)


def from_rules(rules: list[dict]) -> list[CoverageTarget]:
    """The mappable food-based targets in these rules, in catalogue order."""
    from services.guidelines_service import (
        CHECKABLE_TYPES,
        _parse_frequency,
        _quantity_bound,
        _rule_text,
        _source,
    )

    out: list[CoverageTarget] = []
    for rule in rules or []:
        if str(rule.get("guideline_type") or "").strip().lower() not in CHECKABLE_TYPES:
            continue
        guideline_group, groups = _corpus_groups(rule)
        if not groups:
            continue

        text = _rule_text(rule)
        per_meal = bool(_PER_MEAL.search(text))
        bound = _quantity_bound(rule) or _parse_frequency(text)
        if not bound and not per_meal:
            # Neither an amount nor a placement: prose, and the judge's to weigh.
            continue

        out.append(CoverageTarget(
            rule=text[:160],
            groups=groups,
            direction=(bound or {}).get("direction") or "at least",
            low=(bound or {}).get("low"),
            high=(bound or {}).get("high"),
            # A per-meal rule is about every meal of the day whatever period
            # the sentence mentions.
            period="day" if per_meal else ((bound or {}).get("period") or "day"),
            per_meal=per_meal,
            source=_source(rule),
            urn=str(rule.get("id") or rule.get("guide_urn") or ""),
            guideline_group=guideline_group,
        ))

    if out:
        logger.info(
            "Guideline coverage targets: %s",
            "; ".join(f"{t.guideline_group}{'/meal' if t.per_meal else ''}" for t in out),
        )
    return out


# ── shaping the fetch ────────────────────────────────────────────────────

def shape(spec, targets: list[CoverageTarget]):
    """`(spec, notes)` — the plan's shape with guideline groups asked for.

    Only `per_meal` rules reach the spec, and only onto meals the plan already
    serves. A rule that says "at each meal" maps exactly onto
    `slot_food_groups`, which is a per-slot request RecipeWrangler already
    takes and already relaxes rather than emptying a slot.

    A counted rule ("three servings a day") deliberately does NOT become a
    filter. Asking every slot for grains to satisfy a three-a-day floor would
    demand grains at five meals to meet a rule about three, and the member's
    own requests — a taste for one meal, a food group they asked for — would
    lose to it. Those are measured after the fact and reported instead.

    The member's own `slot_food_groups` win: `with_food_groups` merges, so a
    guideline adds to what they asked for and never replaces it.
    """
    result = spec
    notes: list[str] = []
    for target in targets:
        if not target.per_meal:
            continue
        # The FIRST corpus group, not all seven of a protein set: a slot asking
        # for meat OR poultry OR fish is a different query from one asking for
        # the whole family, and `terms` would widen the slot to anything
        # protein-ish. Per-meal rules in practice name one family.
        group = target.groups[0]
        for slot in result.meals:
            before = result
            result = result.with_food_groups(slot, [group])
            if result is not before:
                notes.append(f"{group} at {slot}")
    if notes:
        logger.info("Guidelines shaping the fetch: %s", "; ".join(notes[:6]))
    return result, notes


# ── measuring what came back ─────────────────────────────────────────────

def _plates(plan) -> list:
    days = getattr(plan, "day_plans", None) or getattr(plan, "days", None) or []
    return [
        (int(getattr(day, "day", 1) or 1), plate)
        for day in days
        for meal in (getattr(day, "meals", None) or [])
        for plate in (getattr(meal, "plates", None) or [])
        if getattr(plate, "recipe_id", "")
    ]


def measure(plan, targets: list[CoverageTarget]) -> list[dict]:
    """`{rule, target, actual, met, ...}` rows counted from the plates.

    Counted on each plate's own `food_groups`, which RecipeWrangler derived
    from its ingredients' FoodOn classes — so this is a measurement, not a
    model's reading of a title.

    `met` is `None` when nothing could be counted, and that is the important
    case: a corpus that returned no food groups at all produces "not measured",
    never "0 of 3, failed". A plan marked down for missing data is the same
    lie as one marked satisfied for it.
    """
    plates = _plates(plan)
    if not plates:
        return []
    days = len({day for day, _plate in plates}) or 1
    known = [p for _d, p in plates if getattr(p, "food_groups", ())]

    rows: list[dict] = []
    for target in targets:
        wanted = set(target.groups)
        if not known:
            rows.append(_row(target, None, days, measured=False))
            continue
        if target.per_meal:
            # Meals, not plates: a side salad does not make lunch a grain meal,
            # but it does not have to — the rule asks the MEAL to carry it.
            hits = _meals_carrying(plan, wanted)
            total = _meal_count(plan)
            rows.append(_row(target, hits, days, of=total))
            continue
        hits = sum(1 for plate in known if wanted & set(plate.food_groups))
        rows.append(_row(target, hits, days))
    return rows


def _meals_carrying(plan, wanted: set) -> int:
    days = getattr(plan, "day_plans", None) or getattr(plan, "days", None) or []
    return sum(
        1
        for day in days
        for meal in (getattr(day, "meals", None) or [])
        if any(
            wanted & set(getattr(plate, "food_groups", ()) or ())
            for plate in (getattr(meal, "plates", None) or [])
        )
    )


def _meal_count(plan) -> int:
    days = getattr(plan, "day_plans", None) or getattr(plan, "days", None) or []
    return sum(len(getattr(day, "meals", None) or []) for day in days) or 1


def _row(
    target: CoverageTarget,
    actual: Optional[int],
    days: int,
    *,
    of: Optional[int] = None,
    measured: bool = True,
) -> dict:
    """One checklist row, in the `{rule, target, actual, met}` shape the UI
    already renders and the weekly metrics already carry."""
    if target.per_meal:
        wanted = of or 0
        met = None if not measured else (actual is not None and actual >= wanted)
        target_text = "every meal"
    else:
        low, high = target.low, target.high
        scale = days if target.period == "day" else 1
        bare_count = target.direction == "about" and low == high and low is not None
        if target.direction == "at most":
            limit = (high if high is not None else low) or 0
            met = None if not measured else actual <= limit * scale
            target_text = f"at most {_n(limit)} {_per(target.period)}"
        elif bare_count:
            # A bare count is NOT a bound, and `guidelines_service.split` takes
            # the same line: "choose 2 servings a day from meat, poultry, fish"
            # does not say that three is wrong. So the count is reported and no
            # verdict is claimed — the number still shows a day with none, which
            # is what a member needs to see, without a pass/fail the sentence
            # cannot support.
            met = None
            target_text = f"{_n(low)} {_per(target.period)}"
        elif target.direction == "at least" or high is None:
            floor = low or 0
            met = None if not measured else actual >= floor * scale
            target_text = f"at least {_n(floor)} {_per(target.period)}"
        else:
            met = None if not measured else (low or 0) * scale <= actual <= (high or 0) * scale
            target_text = f"{_n(low)}–{_n(high)} {_per(target.period)}"

    detail = ""
    if not measured:
        detail = (
            "not measured — no dish on this plan carries food-group data, so "
            "nothing was counted"
        )
    elif not target.per_meal and target.direction == "about" \
            and target.low == target.high and target.low is not None:
        detail = "the rule states a count, not a bound, so this is reported and not scored"
        if target.caveat:
            detail += f"; {target.caveat}"
    elif target.caveat:
        detail = target.caveat
    return {
        "rule": target.rule,
        "target": target_text,
        # Meals, and the row says so. The corpus has recipes, not portions, so
        # a "serving" is not a thing this can count.
        "unit": "meals" if target.per_meal else "meals containing it",
        "actual": actual,
        "met": met,
        "source": target.source,
        "urn": target.urn,
        "detail": detail,
    }


def _counted(row: dict) -> str:
    """"1 meal", "2 meals containing it" — the count with its unit, pluralised."""
    actual = row["actual"]
    unit = row["unit"]
    if actual == 1:
        unit = unit.replace("meals", "meal", 1)
    return f"{actual} {unit}"


def _per(period: str) -> str:
    return "a day" if period == "day" else "a week"


def _n(value) -> str:
    if value is None:
        return "0"
    return str(int(value)) if float(value).is_integer() else str(value)


# ── what cannot be checked at all ────────────────────────────────────────

def unverifiable(rules: list[dict]) -> list[dict]:
    """Rows for food-based rules nothing here can act on, naming the reason.

    These are not failures and they are not silence. A member whose guidance
    includes a per-person oil allowance should be told it was not checked and
    why, rather than shown a plan scored against it by a judge reading
    ingredient text, or shown nothing and left to assume it passed.
    """
    from services.guidelines_service import CHECKABLE_TYPES, _rule_text, _source

    rows: list[dict] = []
    seen: set[str] = set()
    for rule in rules or []:
        if str(rule.get("guideline_type") or "").strip().lower() not in CHECKABLE_TYPES:
            continue
        named = {
            str(g).strip().lower() for g in (rule.get("food_groups") or [])
        }
        blocked = sorted(named & set(NO_CORPUS_EQUIVALENT))
        if not blocked or named & set(GUIDELINE_TO_CORPUS):
            continue
        key = blocked[0]
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "rule": _rule_text(rule)[:160],
            "target": "not checkable",
            "unit": "",
            "actual": None,
            "met": None,
            "source": _source(rule),
            "urn": str(rule.get("id") or rule.get("guide_urn") or ""),
            "detail": NO_CORPUS_EQUIVALENT[key],
        })
    return rows


# ── the one seam that reaches every fetch ────────────────────────────────

def per_meal_groups(profile: Optional[dict] = None) -> list[str]:
    """Corpus food groups a guideline asks for at EVERY meal. `[]` normally.

    This is what joins `intent_facets.effective_facets`, and therefore reaches
    all five fetch sites — the classic daily path, the structured path, weekly,
    the pantry search and the pipeline — through the funnel they already share,
    instead of five separate edits that would drift apart the way the facet
    families themselves once did.

    Only per-meal rules. A counted rule ("three servings a day") must not
    become a plan-wide filter: it would demand the group at every slot to
    satisfy a floor of three, which asks for more than the rule does and
    crowds out what the member asked for. Those are measured afterwards.

    Safe to call on a hot path: `backend.catalog.CATALOG.search` caches per
    scope for an hour and caches empty results too, so a member whose region
    has no such rule pays one lookup a session.
    """
    groups: list[str] = []
    for target in targets(profile, "daily"):
        if not target.per_meal:
            continue
        group = target.groups[0]
        if group not in groups:
            groups.append(group)
    return groups


def ledger_rows(plan, profile: Optional[dict] = None) -> list[dict]:
    """The guideline rows for a finished plan's constraint ledger.

    One place, called by both the daily and the weekly path, so the two cannot
    drift into saying different things about the same rule.

    Statuses map onto the four the UI already knows:

        met is True   -> satisfied
        met is False  -> relaxed, never `violated`

    `relaxed` and not `violated` because a guideline is not a rule the member
    set. It is guidance FoodChat searched for and could not fully honour from
    the corpus available under everything else they asked for, and a red chip
    for that is the weekly meat limit's mistake — a house standard reported as
    the member's own broken promise.

        met is None   -> unsupported (the UI's grey, informational chip)

    which covers both "nothing on this plan carries food-group data" and "the
    rule states a count, not a bound". Neither is a pass and neither is a fail,
    and inventing a fifth chip state for a distinction the member has no use
    for is worse than reusing the one that already says "we did not prove
    this".
    """
    from services import guidelines_service

    try:
        rules = guidelines_service.fetch(profile, "daily")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Guideline ledger unavailable: %s", exc)
        return []
    if not rules:
        return []

    rows = measure(plan, from_rules(rules)) + unverifiable(rules)
    out: list[dict] = []
    for row in rows:
        met = row.get("met")
        status = "satisfied" if met is True else ("relaxed" if met is False else "unsupported")
        detail = _counted(row) if row.get("actual") is not None else ""
        if detail and row.get("target"):
            detail += f", against {row['target']}"
        if row.get("detail"):
            detail = f"{detail} — {row['detail']}" if detail else row["detail"]
        out.append({
            "constraint": row["rule"],
            # Never hard. Allergens and diet are the member's; this is guidance.
            "type": "soft",
            "status": status,
            "source": row.get("source") or "dietary guidelines",
            "detail": detail,
        })
    if out:
        logger.info(
            "Guideline coverage: %s",
            "; ".join(f"{r['status']}" for r in out),
        )
    return out
