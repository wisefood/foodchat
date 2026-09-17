"""
Dietary guidelines for a plan: which rules, the text the judges read, and the
weekly checklist rows a counter can honestly fill.

    resolve_scope(profile, plan_type, override)  → GuidelineScope
    fetch(profile, plan_type, override)          → [rule dict] from the catalog
    guidelines_text(plan_type, profile, override)→ numbered rules for a judge
    split(rules)                                 → (checkable, prose)
    checklist(checkable, category_counts, total) → {rule, target, actual, met}

**Which rules.** The deployment default — Ireland, adults
(`GUIDELINES_DEFAULT_REGION`, `DEFAULT_LIFE_STAGE`) — for every member today:
the profile carries no region or age group yet. `resolve_scope` reads
``region`` / ``age_group`` (or ``life_stage``) from the profile when they
appear, so supplying them later is a profile change, not a change here. A
caller that wants another country, one guide, or a hand-picked subset of rules
passes an ``override`` scope. Rules tagged for nobody in particular are
ordinary rules for everyone, including the untagged ones that were plainly
written for children ("Offer red meat 3 times a week").

**The judges get the rules** (`guidelines_text`): a numbered list the LLM can
cite, rules that state a frequency first, capped (`GUIDELINES_MAX_CHARS`) so the
judge fits in the same minute as the candidate grading on Groq's on-demand
tier.

**The checklist gets very few.** Most rules cannot be counted from a plan —
"choose wholegrain varieties where possible" is advice, not an assertion with a
truth value. A rule becomes a checklist row only when it is food-based, names a
meal category the weekly planner actually counts (fish, red meat, poultry), and
states a weekly floor, ceiling or range. In the default Irish adult set no rule
does, so the checklist keeps its three built-in rules there. Everything else is
prose. Inventing a target for a prose rule — or counting "vegetables" against a category counter that never
counts vegetables — would render a number with nothing behind it as if it were
a measurement.

Everything degrades. A catalog that is unreachable, or simply not configured,
yields ``[]`` / ``""``; the weekly checklist falls back to its hardcoded three
rules and the judges say they had no guideline text.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from models.guidelines import LIFE_STAGES, GuidelineScope

logger = logging.getLogger(__name__)

DEFAULT_REGION = (os.getenv("GUIDELINES_DEFAULT_REGION") or "IE").strip().upper()
DEFAULT_LIFE_STAGE = "adulthood"

REGION_NAMES = {"IE": "Ireland", "HU": "Hungary", "SI": "Slovenia"}
_STAGE_LABELS = {
    "infancy": "infants", "early_childhood": "young children",
    "school_age": "school-age children", "adolescence": "adolescents",
    "adulthood": "adults", "older_adulthood": "older adults",
    "pregnancy": "pregnancy", "lactation": "breastfeeding",
}
_REGION_ALIASES = {
    "ireland": "IE", "irl": "IE", "eire": "IE", "éire": "IE",
    "hungary": "HU", "hun": "HU", "magyarország": "HU",
    "slovenia": "SI", "svn": "SI", "slovenija": "SI",
}

# The gateway's AgeGroupEnum, onto the catalog's life stages.
_AGE_GROUP_STAGE = {
    "baby": "infancy",
    "child": "school_age",
    "teen": "adolescence",
    "young_adult": "adulthood",
    "adult": "adulthood",
    "middle_aged": "adulthood",
    "senior": "older_adulthood",
}

# What the judge is handed, in characters of rule text (~4 per token). The
# adherence judge runs in the same minute as the candidate grading, and on
# Groq's on-demand tier (8,000 tokens a minute) the whole Irish adult set —
# 77 rules, ~6k characters — pushed it past the limit on a live daily plan, so
# the score came back empty. 3,500 keeps 45 of those 77 — every rule that
# states a frequency, portion definitions last; a deployment on a larger tier
# can raise it.
MAX_RULES = 90
MAX_CHARS = int(os.getenv("GUIDELINES_MAX_CHARS", "3500"))

# A rule that states how often ("5-7 servings a day", "at each meal") says
# something a plan can be judged against, whatever its type.
_STATES_AN_AMOUNT = re.compile(
    r"\b(?:a|per|each|every)\s+(?:day|week|meal)\b|\b(?:daily|weekly|once|twice)\b",
    re.IGNORECASE,
)

# The catalog's classifier for what KIND of rule this is. `action_type` is
# "eat" on 2,765 of 2,790 rules and says nothing.
CHECKABLE_TYPES = frozenset({"food_based"})

# The categories `weekly_planner.day_summary.classify_meal` assigns — the ONLY
# things a weekly checklist can count. `topic` (free-form, from enrichment) is
# read first; `food_groups` is a coarse enum ("protein_foods" is fish AND meat)
# and cannot name a category. "processed meat" is not here: every red-meat
# meal would be counted against a processed-meat limit.
TOPIC_CATEGORIES: dict[str, str] = {
    "fish": "fish",
    "oily_fish": "fish",
    "seafood": "fish",
    "red_meat": "red meat",
    "poultry": "poultry",
}
TEXT_CATEGORIES: dict[str, str] = {
    "oily fish": "fish",
    "fish": "fish",
    "seafood": "fish",
    "red meat": "red meat",
    "poultry": "poultry",
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


# ── which rules ───────────────────────────────────────────────────────────

def resolve_scope(
    profile: Optional[dict] = None,
    plan_type: Optional[str] = None,
    override: Optional[GuidelineScope] = None,
) -> GuidelineScope:
    """The scope a plan is judged in.

    ``override`` wins outright (its own plan type is kept if it set one).
    Otherwise the member's region and life stage when the profile carries
    them, and the deployment default for whichever it does not.
    """
    plan_type = plan_type if plan_type in ("daily", "weekly") else None
    if override is not None:
        if override.plan_type is None and plan_type:
            return override.model_copy(update={"plan_type": plan_type})
        return override

    profile = profile or {}
    return GuidelineScope(
        regions=(region_code(profile.get("region")) or DEFAULT_REGION,),
        life_stage=life_stage_of(profile) or DEFAULT_LIFE_STAGE,
        plan_type=plan_type,
    )


def region_code(raw) -> Optional[str]:
    """An ISO alpha-2 code for a region the catalog has, or None.

    ``household.region`` on the gateway is free text. A value that is not a
    catalog country is None (and the caller uses the default) — never a guess.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    if text.upper() in REGION_NAMES:
        return text.upper()
    code = _REGION_ALIASES.get(text.lower())
    if code is None:
        logger.info("No guidelines for region %r; using the default.", text)
    return code


def life_stage_of(profile: dict) -> Optional[str]:
    stage = str(profile.get("life_stage") or "").strip().lower()
    if stage in LIFE_STAGES:
        return stage
    return _AGE_GROUP_STAGE.get(str(profile.get("age_group") or "").strip().lower())


def fetch(
    profile: Optional[dict] = None,
    plan_type: Optional[str] = None,
    override: Optional[GuidelineScope] = None,
) -> list[dict]:
    """The rules that apply here, as dicts. `[]` when the catalog cannot answer."""
    from backend.catalog import CATALOG

    if not CATALOG.available():
        return []
    scope = resolve_scope(profile, plan_type, override)
    return [rule.model_dump() for rule in CATALOG.search(scope)]


# ── the judges' text ──────────────────────────────────────────────────────

def guidelines_text(
    plan_type: str = "daily",
    profile: Optional[dict] = None,
    override: Optional[GuidelineScope] = None,
) -> str:
    """Numbered guideline rules for an LLM judge; ``""`` when there are none.

    Never raises: a plan's scores are not worth a failed turn.
    """
    try:
        scope = resolve_scope(profile, plan_type, override)
        return render(fetch(profile, plan_type, override), scope)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Guideline text unavailable: %s", exc)
        return ""


def render(rules: list[dict], scope: Optional[GuidelineScope] = None) -> str:
    """The rules as a judge reads them.

    ::

        Dietary guidelines: Ireland, adults — 81 of 81 rules (WiseFood catalogue).
        Sources: Healthy food for life full guide; Fsai healthy eating guidelines.
        Cite rules by id. ...
        [G1] (weekly) Offer oily fish such as mackerel, ... once a week.

    Stable ids so a reasoning line can point at a rule. Ordered so the cap cuts
    the least checkable rules first: rules that state an amount or frequency
    before those that do not, then food-based before nutrient-based before
    untyped before behavioural. A rule too long for what is left of the budget
    is skipped, not the end of the list.
    """
    rules = [r for r in rules or [] if _rule_text(r)]
    if not rules:
        return ""

    ordered = sorted(rules, key=_judge_priority)
    kept: list[dict] = []
    chars = 0
    for rule in ordered:
        text = _rule_text(rule)
        if len(kept) >= MAX_RULES:
            break
        if chars + len(text) > MAX_CHARS:
            continue
        kept.append(rule)
        chars += len(text)

    lines = [
        f"Dietary guidelines: {_scope_label(scope, rules)} — "
        f"{len(kept)} of {len(rules)} rules (WiseFood catalogue).",
    ]
    sources = list(dict.fromkeys(_source(r) for r in kept if _source(r)))
    if sources:
        lines.append("Sources: " + "; ".join(sources[:6]) + ".")
    lines.append(
        "Cite rules by id. A rule a meal plan cannot show — portion measuring, "
        "mealtime behaviour, advice addressed to parents — is context, not a failure."
    )
    for n, rule in enumerate(kept, start=1):
        frequency = rule.get("frequency")
        tag = f"({frequency}) " if frequency else ""
        lines.append(f"[G{n}] {tag}{_rule_text(rule)}")
    return "\n".join(lines)


_TYPE_RANK = {"food_based": 0, "nutrient_based": 1, None: 2, "behavioral": 3}


def _judge_priority(rule: dict):
    states_amount = bool(rule.get("frequency")) or bool(
        _STATES_AN_AMOUNT.search(_rule_text(rule))
    )
    return (
        0 if states_amount else 1,
        _TYPE_RANK.get(rule.get("guideline_type"), 2),
    )


def _scope_label(scope: Optional[GuidelineScope], rules: list[dict]) -> str:
    if scope is not None and scope.rule_ids:
        return "a selected subset"
    regions = list(scope.regions) if scope and scope.regions else sorted(
        {r.get("guide_region") for r in rules if r.get("guide_region")}
    )
    where = ", ".join(REGION_NAMES.get(code, code) for code in regions) or "all regions"
    stage = scope.life_stage if scope else None
    return f"{where}, {_STAGE_LABELS.get(stage, stage)}" if stage else where


def _source(rule: dict) -> str:
    """A readable guide name from its URN.

    ``urn:guide:healthy-food-for-life-full-guide-20260330134146569`` →
    "Healthy food for life full guide". The row carries no guide title, and a
    lookup per guide to fetch one would be a request per plan for a label.
    """
    urn = str(rule.get("guide_urn") or "")
    slug = urn.rsplit(":", 1)[-1]
    slug = re.sub(r"[-_]?\d{8,}$", "", slug)
    words = re.sub(r"[-_]+", " ", slug).strip()
    return words[:1].upper() + words[1:] if words else ""


def _rule_text(rule: dict) -> str:
    return str(rule.get("rule_text") or rule.get("title") or "").strip()


# ── the checklist ─────────────────────────────────────────────────────────

def split(rules: list[dict]) -> tuple[list[dict], list[dict]]:
    """(checkable, prose).

    Checkable means: the catalog calls it food-based, it names one meal
    category the weekly planner counts, and it states a weekly bound — a floor
    ("at least once"), a ceiling ("limit to 3", "no more than 2") or a range
    ("1-2 times"). All of them, because a rule missing any one cannot produce
    an honest `met` — and a checklist row whose verdict is a guess is worse
    than no row.

    A bare count is not a bound. "Offer oily fish once a week" does not say
    that two fish dinners break it, and "Offer red meat 3 times a week" does
    not say that one red-meat dinner does; checked as exact targets, both
    would report a sound week as failing. Those rules stay prose, where the
    judge can weigh them.
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
    # Category counts are per plan, and a plan is a week. "Fish twice a day"
    # checked against a weekly count would be met by two fish dinners.
    if not bound or bound["period"] != "week":
        return None
    if bound["direction"] == "about" and bound.get("low") == bound.get("high"):
        return None

    return {
        "rule": _rule_text(rule)[:160],
        "category": category,
        "period": bound["period"],
        "low": bound.get("low"),
        "high": bound.get("high"),
        "direction": bound["direction"],
        "source": _source(rule),
        "urn": rule.get("id") or rule.get("guide_urn") or "",
    }


def _countable_category(rule: dict) -> Optional[str]:
    topics = {
        TOPIC_CATEGORIES[str(t).strip().lower()]
        for t in rule.get("topic") or []
        if str(t).strip().lower() in TOPIC_CATEGORIES
    }
    if len(topics) == 1:
        return topics.pop()
    if topics:
        return None

    # Some rules name the food only in their text. Exactly one category, or
    # none: "2 servings a day of meat, poultry, fish, eggs" is not a fish rule.
    text = _rule_text(rule).lower()
    found = set()
    for phrase, category in TEXT_CATEGORIES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            found.add(category)
    return found.pop() if len(found) == 1 else None


def _quantity_bound(rule: dict) -> Optional[dict]:
    """The catalog's own structured quantity, when it has one.

    `quantity` is a documented `{operator, value, unit, period}` object and is
    empty on every rule today. Read first anyway: the day an import fills it,
    it is the accurate source and the regex below stops being the only one.
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
    if operator in {"lt", "lte"}:
        return {"direction": "at most", "high": value, "period": period}
    if operator in {"gt", "gte"}:
        return {"direction": "at least", "low": value, "period": period}
    return {"direction": "about", "low": value, "high": value, "period": period}


def _parse_frequency(text: str) -> Optional[dict]:
    """A frequency read out of the rule's own sentence."""
    if not text:
        return None
    lowered = text.lower()

    for word, number in _WORD_NUMBERS.items():
        match = re.search(rf"\b{word}\b\s+(?:a|per|each)\s+(week|day)", lowered)
        if match:
            return _bounded(_direction_before(lowered[:match.start()]),
                            number, number, match.group(1))

    match = _FREQUENCY_RE.search(lowered)
    if not match:
        return None
    low = float(match.group("low"))
    high = float(match.group("high")) if match.group("high") else None
    qualifier = (match.group("qualifier") or "").strip()
    if qualifier in {"at most", "no more than", "up to", "under"}:
        direction = "at most"
    elif qualifier in {"at least", "over"}:
        direction = "at least"
    else:
        direction = _direction_before(lowered[:match.start()])
    return _bounded(direction, low, high or low, match.group("period"))


def _direction_before(before: str) -> str:
    """The words ahead of the count: "at least once" is a floor, "limit red
    meat to once" a ceiling, anything else a target."""
    if re.search(r"\bat least\s*$", before):
        return "at least"
    if re.search(r"\b(?:at most|no more than|up to|maximum of)\s*$", before):
        return "at most"
    if re.search(r"\blimit\b", before):
        return "at most"
    return "about"


def _bounded(direction: str, low: float, high: float, period: str) -> dict:
    if direction == "at most":
        return {"direction": direction, "high": high, "period": period}
    if direction == "at least":
        return {"direction": direction, "low": low, "period": period}
    return {"direction": direction, "low": low, "high": high, "period": period}


def checklist(rules: list[dict], category_counts: dict, total_meals: int) -> list[dict]:
    """The `{rule, target, actual, met}` rows, from real guidance.

    Same shape the UI already renders and the weekly metrics already carry.
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


def reason_chips(plate_ingredients: str, rules: list[dict]) -> list[dict]:
    """`{kind: "guideline"}` chips for a plate that satisfies a countable rule.

    `guideline` is a documented reason kind — declared in the shared contract,
    rendered by the UI with its own icon.
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

