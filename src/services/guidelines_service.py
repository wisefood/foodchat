"""
Real dietary guidelines, for the member's own region and life stage.

FoodChat grades plans against dietary guidelines and has never read one. The
checklist it reports — eat fish 1–2 times a week, limit red meat, make most
meals plant-based — is three rules hardcoded in the weekly explainability
module. They are real guidance. They are also the same three rules for a
member in Ireland, Slovenia, Hungary or Greece, and the same three for a
pregnant member, a teenager and a 70-year-old.

The catalog holds ~2,700 rules across 31 guides, faceted by exactly the things
that would make them different for those people.

**Most of them cannot be checked, and this module says so.** Measured over a
1,334-rule sample: 6.4% are food-group frequencies a plan can be counted
against, 3% are nutrient thresholds, and 73.4% are prose — "choose wholegrain
varieties where possible" is advice, not an assertion with a truth value. So
the rules are split:

    checkable   → the `{rule, target, actual, met}` checklist that already
                  exists, now built from the member's own guidance
    prose       → context handed to the grader, which can weigh advice a
                  counter cannot

Inventing a target for a prose rule so it can appear in the checklist would be
the same mistake the constraint ledger made: a number with nothing behind it,
rendered as if it were a measurement.

Everything degrades to the hardcoded three. A catalog that is unreachable, or
simply not configured, costs the plan its regional detail and nothing else.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# The catalog's own classifier for what KIND of rule this is. `action_type` is
# a verb classifier ("eat", "drink", "limit") and says nothing about whether a
# plan can be counted against the rule; `guideline_type` is the closest thing
# the schema has to a checkability signal.
CHECKABLE_TYPES = frozenset({"food_based"})

# Food groups a weekly plan can actually count, mapped onto the category names
# the weekly planner tracks. A rule about a group not in here is prose as far
# as this module is concerned, whatever its type says.
COUNTABLE_GROUPS: dict[str, str] = {
    "fish": "fish",
    "seafood": "fish",
    "oily fish": "fish",
    "red meat": "red meat",
    "processed meat": "red meat",
    "meat": "red meat",
    "vegetables": "vegetables",
    "fruit": "fruit",
    "fruit and vegetables": "vegetables",
    "legumes": "legumes",
    "pulses": "legumes",
    "wholegrains": "wholegrains",
    "dairy": "dairy",
}

# "twice a week", "2-3 times per week", "at least 5 a day", and — because
# guideline prose reads like prose — "no more than 1 portion of processed meat
# a week", where the food sits between the count and the period.
_FREQUENCY_RE = re.compile(
    r"(?P<qualifier>at least|at most|no more than|up to|under|over)?\s*"
    r"(?P<low>\d+)"
    r"(?:\s*(?:[-–]|to)\s*(?P<high>\d+))?"
    r"\s*(?:times?|portions?|servings?|meals?)?"
    r"(?:\s+of\s+[a-z][a-z\s-]{0,30}?)?"
    r"\s*(?:a|per|each)\s+(?P<period>week|day)",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "once": 1, "twice": 2, "thrice": 3,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def facets_for(profile: dict, brief=None) -> dict[str, list[str]]:
    """Which guidelines apply to this member.

    Derived from what the profile actually holds. A facet nobody can fill is
    left out rather than guessed: asking the catalog for `life_stage:adult`
    because most members are adults would silently exclude the rules that exist
    precisely for the members who are not.
    """
    profile = profile or {}
    facets: dict[str, list[str]] = {}

    region = (
        profile.get("region")
        or profile.get("country")
        or profile.get("household_country")
    )
    if region:
        facets["region"] = [str(region).strip().lower()]

    life_stage = _life_stage(profile)
    if life_stage:
        facets["life_stage"] = [life_stage]

    conditions = [
        str(c).strip().lower()
        for c in (profile.get("health_conditions") or profile.get("conditions") or [])
        if c
    ]
    if conditions:
        facets["health_conditions"] = conditions

    # Only the food groups the plan is actually shaped around — asking for
    # every group returns the whole corpus and tells the grader nothing.
    if brief is not None and getattr(brief, "food_groups", None):
        facets["food_groups"] = list(brief.food_groups)

    return facets


def _life_stage(profile: dict) -> Optional[str]:
    """The catalog's life-stage bucket, from an age group if one is recorded."""
    raw = str(
        profile.get("life_stage") or profile.get("age_group") or ""
    ).strip().lower()
    if not raw:
        return None
    if raw in {"infant", "toddler", "child", "children", "teen", "adolescent",
               "adult", "older_adult", "elderly", "pregnancy", "lactation"}:
        return raw
    # The picker's own vocabulary, mapped rather than passed through: sending
    # "65+" to a facet that stores "older_adult" matches nothing, and matching
    # nothing here looks identical to "this member has no special guidance".
    if raw in {"65+", "over 65", "senior", "seniors"}:
        return "older_adult"
    if raw in {"13-18", "teenager", "teenagers"}:
        return "teen"
    if raw in {"0-2", "0-3"}:
        return "toddler"
    if raw in {"4-12", "kids"}:
        return "child"
    if raw in {"19-64", "18-64"}:
        return "adult"
    return None


def fetch(profile: dict, brief=None, limit: int = 40) -> list[dict]:
    """The guidelines that apply here. `[]` when the catalog cannot answer."""
    from backend.catalog import CATALOG

    if not CATALOG.available():
        return []
    facets = facets_for(profile, brief)
    rules = CATALOG.search_guidelines(facets, limit=limit)
    # A region with no rules of its own is common — most of the corpus is
    # Ireland. Falling back to the unfiltered set beats reporting that a member
    # in Slovenia has no dietary guidance at all.
    if not rules and "region" in facets:
        narrower = {k: v for k, v in facets.items() if k != "region"}
        rules = CATALOG.search_guidelines(narrower, limit=limit)
        if rules:
            logger.info("No region-specific guidelines; using the general set.")
    return rules


def split(rules: list[dict]) -> tuple[list[dict], list[dict]]:
    """(checkable, prose).

    Checkable means: the catalog calls it food-based, it names a food group
    this plan can count, and it states a frequency. All three, because a rule
    missing any one of them cannot produce an honest `actual` — and a checklist
    row whose `actual` is a guess is worse than no row.
    """
    checkable: list[dict] = []
    prose: list[dict] = []
    for rule in rules or []:
        parsed = _as_frequency(rule)
        (checkable if parsed else prose).append(parsed or rule)
    return checkable, prose


def _as_frequency(rule: dict) -> Optional[dict]:
    """A rule turned into something countable, or None."""
    if str(rule.get("guideline_type") or "").strip().lower() not in CHECKABLE_TYPES:
        return None

    category = _countable_category(rule)
    if not category:
        return None

    bound = _quantity_bound(rule) or _parse_frequency(_rule_text(rule))
    if not bound:
        return None

    return {
        "rule": _rule_text(rule)[:160],
        "category": category,
        "period": bound["period"],
        "low": bound.get("low"),
        "high": bound.get("high"),
        "direction": bound["direction"],
        "source": rule.get("guide_title") or rule.get("guide_urn") or "",
        "urn": rule.get("urn") or rule.get("id") or "",
    }


def _rule_text(rule: dict) -> str:
    return str(rule.get("rule_text") or rule.get("title") or "").strip()


def _countable_category(rule: dict) -> Optional[str]:
    for group in rule.get("food_groups") or []:
        category = COUNTABLE_GROUPS.get(str(group).strip().lower())
        if category:
            return category
    # Some rules name the food only in their text.
    text = _rule_text(rule).lower()
    for group, category in COUNTABLE_GROUPS.items():
        if re.search(rf"\b{re.escape(group)}\b", text):
            return category
    return None


def _quantity_bound(rule: dict) -> Optional[dict]:
    """The catalog's own structured quantity, when it has one.

    `quantity` is a documented `{operator, value, unit, period}` triple with a
    full mapping — and, at the time of writing, nothing populates it: the write
    path exists and the producer does not. Read first anyway, because the day
    an import fills it, this is the accurate source and the regex below stops
    being the only one.
    """
    quantity = rule.get("quantity")
    if not isinstance(quantity, dict):
        return None
    try:
        value = float(quantity.get("value"))
    except (TypeError, ValueError):
        return None
    period = str(quantity.get("period") or "week").strip().lower()
    if period not in {"day", "week"}:
        return None
    operator = str(quantity.get("operator") or "").strip().lower()
    if operator in {"<=", "lt", "lte", "max", "at_most"}:
        return {"direction": "at most", "high": value, "period": period}
    if operator in {">=", "gt", "gte", "min", "at_least"}:
        return {"direction": "at least", "low": value, "period": period}
    return {"direction": "about", "low": value, "high": value, "period": period}


def _parse_frequency(text: str) -> Optional[dict]:
    """A frequency read out of the rule's own sentence."""
    if not text:
        return None
    lowered = text.lower()

    for word, number in _WORD_NUMBERS.items():
        match = re.search(rf"\b{word}\b\s+(?:a|per)\s+(week|day)", lowered)
        if match:
            return {"direction": "about", "low": number, "high": number,
                    "period": match.group(1)}

    match = _FREQUENCY_RE.search(lowered)
    if not match:
        return None
    low = float(match.group("low"))
    high = float(match.group("high")) if match.group("high") else None
    qualifier = (match.group("qualifier") or "").strip()
    period = match.group("period")

    if qualifier in {"at most", "no more than", "up to", "under"}:
        return {"direction": "at most", "high": low, "period": period}
    if qualifier in {"at least", "over"}:
        return {"direction": "at least", "low": low, "period": period}
    return {"direction": "about", "low": low, "high": high or low, "period": period}


def checklist(rules: list[dict], category_counts: dict, total_meals: int) -> list[dict]:
    """The `{rule, target, actual, met}` rows, from real guidance.

    Same shape the UI already renders and the weekly metrics already carry —
    the receiver was right all along, it was just being fed three constants.
    """
    out: list[dict] = []
    for parsed in rules:
        actual = int(category_counts.get(parsed["category"], 0))
        low, high = parsed.get("low"), parsed.get("high")
        direction = parsed["direction"]

        if direction == "at most":
            met = actual <= (high or 0)
            target = f"at most {_n(high)}"
        elif direction == "at least":
            met = actual >= (low or 0)
            target = f"at least {_n(low)}"
        else:
            met = (low or 0) <= actual <= (high if high is not None else low or 0)
            target = _n(low) if low == high else f"{_n(low)}–{_n(high)}"

        per = "a week" if parsed["period"] == "week" else "a day"
        out.append({
            "rule": parsed["rule"],
            "target": f"{target} {per}",
            "actual": actual,
            "met": bool(met),
            "source": parsed.get("source") or "",
        })
    return out


def _n(value) -> str:
    if value is None:
        return "0"
    return str(int(value)) if float(value).is_integer() else str(value)


def prose_context(rules: list[dict], limit: int = 8) -> str:
    """The advice a counter cannot check, for the grader that can weigh it.

    Capped: a grader handed 400 lines of guidance is a grader that reads the
    first few and pads the rest of its answer, which is worse than being handed
    the few that matter.
    """
    lines: list[str] = []
    for rule in rules[:limit]:
        text = _rule_text(rule)
        if not text:
            continue
        source = rule.get("guide_title") or ""
        lines.append(f"- {text}" + (f" ({source})" if source else ""))
    return "\n".join(lines)


def reason_chips(plate_ingredients: str, rules: list[dict]) -> list[dict]:
    """`{kind: "guideline"}` chips for a plate that satisfies a countable rule.

    `guideline` has been a documented reason kind — declared in the shared
    contract, rendered by the UI with its own icon — that nothing ever emitted.
    """
    text = (plate_ingredients or "").lower()
    chips: list[dict] = []
    seen: set[str] = set()
    for parsed in rules:
        category = parsed.get("category")
        if not category or category in seen:
            continue
        # Only the "eat more of this" direction. A chip saying a dish helps
        # you limit red meat, on a dish containing red meat, is nonsense.
        if parsed.get("direction") == "at most":
            continue
        if re.search(rf"\b{re.escape(category)}\b", text):
            seen.add(category)
            chips.append({
                "kind": "guideline",
                "label": f"{category} — {parsed['rule'][:60]}",
            })
    return chips
