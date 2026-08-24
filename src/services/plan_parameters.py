"""
Interactive plan parameters — the slider card shown with fresh daily plans.

These are the clarification topics the pipeline used to ask about in text
("cooking time", "difficulty level", "goal") before the interrogation was
tuned out. Instead of questions, the UI renders an optional card with a
predefined scale per parameter; whatever the user applies comes back through
POST /sessions/{id}/plan-parameters as a deterministic refinement — no intent
classification, no clarification round-trip.

Everything here is static and LLM-free: the card definition, value
sanitization, and the canonical refinement text built from applied values.
Applied values live on the session profile under ``plan_parameters`` (so the
card can show current settings) and are appended to the profile history (so
the reconciler treats them as known facts and never asks again).

The one thing here that is not about the card: `extract_time_delta` reads a
cooking-time ceiling out of a sentence. It lives here because this module owns
the `cooking_time` key and `max_duration_minutes`, which is the single place
every fetch site reads the limit from — so the slider and the sentence land on
the same constraint rather than on two that can disagree.
"""

import logging
import re

_logger = logging.getLogger(__name__)

# Kinds: "scale" is a numeric slider; "choice" is a discrete labeled scale
# (still rendered as a draggable knob over fixed stops in the UI).
PARAMETER_DEFS: list[dict] = [
    {
        "key": "cooking_time",
        "label": "Cooking time",
        "kind": "scale",
        "min": 10,
        "max": 90,
        "step": 5,
        "unit": "min",
        "default": 30,
    },
    {
        # Was "Difficulty", with Easy / Medium / Elaborate — and two of those
        # three filtered nothing. `grep -ri difficulty` across RecipeWrangler
        # returns nothing: there is no difficulty field, no tag and no
        # vocabulary, so "Elaborate" was a control that changed the plan in no
        # way whatsoever and said nothing about it.
        #
        # What the corpus CAN express is simplicity: `5_ingredients_or_less`
        # (563 recipes). So the parameter is now that, under a name that says
        # it, with the option that meant nothing removed. Deliberately NOT
        # `30_minutes_or_less` as well — duration is the control right above
        # this one, and two controls setting the same filter is how they end up
        # disagreeing.
        #
        # `ordered` marks a scale the UI can render as a draggable toggle
        # rather than a row of pills: simple → any is a direction, unlike the
        # goals below, which are alternatives.
        "key": "difficulty",
        "label": "Effort",
        "kind": "choice",
        "ordered": True,
        "options": [
            {"value": "easy", "label": "Simple"},
            {"value": "medium", "label": "Any"},
        ],
        "default": "medium",
    },
    {
        "key": "goal",
        "label": "Goal",
        "kind": "choice",
        "options": [
            {"value": "weight_loss", "label": "Lose weight"},
            {"value": "balanced", "label": "Balanced"},
            {"value": "high_protein", "label": "High protein"},
            {"value": "energy", "label": "Energy boost"},
        ],
        "default": "balanced",
    },
    {
        # Food waste is a *dimension of the plan*, not of any one recipe: a
        # week where Monday's leftover half-cabbage reappears in Wednesday's
        # dinner wastes less than one where every meal opens a new set of
        # ingredients. Nothing in a single prompt reliably says whether the
        # member wants that trade — reuse pulls against variety — so it is a
        # control, not an inference.
        #
        # One control, not a toggle plus a scope: "off" is the scope's own
        # zero, and two knobs that can contradict each other ("waste: on,
        # scope: off") is a settings bug shipped as a feature.
        "key": "food_waste",
        "label": "Food waste",
        "kind": "choice",
        "ordered": True,
        "options": [
            {"value": "off", "label": "Off"},
            # Reuse fresh ingredients across the plan; pantry staples (oil,
            # flour, spices) don't count as waste and don't constrain.
            {"value": "reuse", "label": "Reuse ingredients"},
            # Also shrink the total shopping list: fewer distinct
            # ingredients overall, at some cost to variety.
            {"value": "strict", "label": "Minimal shopping"},
        ],
        "default": "off",
    },
]

# How each applied value reads in the canonical refinement query.
_PHRASES = {
    "cooking_time": lambda v: f"keep cooking time under {v} minutes per meal",
    "difficulty": {
        "easy": "keep recipes easy to cook",
        "medium": "medium cooking difficulty is fine",
        "hard": "elaborate recipes are welcome",
    },
    "goal": {
        "weight_loss": "aim for lighter, lower-calorie meals for weight loss",
        "balanced": "aim for balanced, generally healthy meals",
        "high_protein": "aim for high-protein meals",
        "energy": "aim for energizing, sustaining meals",
    },
    "food_waste": {
        "off": "no ingredient-reuse constraint",
        "reuse": "favour meals that reuse each other's fresh ingredients to reduce food waste",
        "strict": "keep the overall shopping list small — strongly favour meals sharing ingredients, even at some cost to variety",
    },
}


def waste_mode(values: dict) -> str:
    """The applied food-waste setting: 'off', 'reuse' or 'strict'.

    Two readers, because the two paths choose differently. The weekly planner
    selects without an LLM, so there it becomes a number in the preference
    scorer — a preference that never becomes a number there does not exist. The
    daily path ranks combinations with a grader, so there it becomes a line in
    the grader's query: whether three meals share a bunch of coriander is a
    property of the COMBINATION, and the grader is the only thing that sees all
    three at once.

    It used to say the daily path heard this "as prose via `describe`". It did
    not: `describe` builds the canonical message for a slider APPLY, so a
    member with reuse standing got it once, on the turn they set it, and every
    later "plan my day" ignored it.
    """
    value = values.get("food_waste")
    return value if value in ("reuse", "strict") else "off"


def build_card(profile: dict, plan_type: str = "daily") -> dict:
    """The card payload for a turn: definitions plus current applied values.

    ``plan_type`` is the card's address — the plan it was rendered with. The
    client sends it back on apply so the values refine THAT plan, not
    whichever canvas happens to be newest by then.
    """
    applied = profile.get("plan_parameters") or {}
    parameters = []
    for definition in PARAMETER_DEFS:
        param = dict(definition)
        param["value"] = applied.get(definition["key"])
        parameters.append(param)
    return {"parameters": parameters, "plan_type": plan_type}


def sanitize(values: dict) -> dict:
    """Whitelist keys, clamp scales to their range/step, validate choices.

    Anything unusable is dropped; an empty dict means the caller sent nothing
    actionable (the router turns that into a 400).
    """
    defs = {d["key"]: d for d in PARAMETER_DEFS}
    clean: dict = {}
    for key, value in (values or {}).items():
        definition = defs.get(key)
        if definition is None:
            continue
        if definition["kind"] == "scale":
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            step = definition["step"]
            snapped = round(number / step) * step
            clean[key] = int(min(definition["max"], max(definition["min"], snapped)))
        else:
            if value in {o["value"] for o in definition["options"]}:
                clean[key] = value
    return clean


def max_duration_minutes(values: dict) -> int | None:
    """The cooking-time slider as a real constraint, not prose.

    `describe` renders this value as "keep cooking time under N minutes per
    meal" and hands it to the LLM grader, which reads it as a hint and ranks
    accordingly. RecipeWrangler can filter on duration directly, so the slider
    can now narrow the candidate set instead of merely nudging how it is scored
    — a member who set 20 minutes stops being shown 90-minute braises at all.

    Returned as-is rather than clamped: the slider's own bounds (10-90) are
    enforced by `sanitize`, and inventing a second, different limit here is how
    two components end up disagreeing about what the user asked for.
    """
    value = values.get("cooking_time")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# The same constraint, said out loud                                            #
# --------------------------------------------------------------------------- #
#
# "Keep it under 20 minutes" reached nothing. The persona has promised for a
# long time that a member can steer by cooking time, and only the slider ever
# could — every one of the seven fetch sites reads the limit from
# `profile["plan_parameters"]["cooking_time"]`, and nothing but the slider ever
# wrote it.
#
# Deliberately a regex, not a model call. A stated duration is one of the few
# things members say in a small number of shapes, the answer is a number that
# becomes a hard filter, and a model that hallucinates 20 when someone said
# 90 is worse than one that says nothing.

_TIME_QUALIFIER = (
    r"(?:under|below|less than|no more than|not? more than|no longer than|"
    r"nothing over|nothing longer than|within|at most|max(?:imum)?(?: of)?|"
    r"keep it to|in|only|up to)"
)
_HOUR_WORDS = {"half an hour": 30, "half hour": 30, "an hour": 60, "one hour": 60}

# "under 30 minutes", "in 20 min", "max 45 mins"
_MINUTES_RE = re.compile(
    rf"\b{_TIME_QUALIFIER}\s+(?P<n>\d{{1,3}})\s*(?:-|\s)?\s*(?:minute|minutes|min|mins)\b",
    re.IGNORECASE,
)
# "30 minutes or less", "20 mins tops"
_MINUTES_TRAILING_RE = re.compile(
    r"\b(?P<n>\d{1,3})\s*(?:-|\s)?\s*(?:minute|minutes|min|mins)\b"
    r"\s*(?:or less|or under|tops|max(?:imum)?)\b",
    re.IGNORECASE,
)
# "30-minute meals", "20 minute dinners"
_MINUTES_COMPOUND_RE = re.compile(
    r"\b(?P<n>\d{1,3})\s*-?\s*(?:minute|min)\s+"
    r"(?:meal|meals|dinner|dinners|lunch|lunches|breakfast|breakfasts|recipe|recipes|dish|dishes)\b",
    re.IGNORECASE,
)
# "under an hour", "within half an hour", "in 2 hours"
_HOURS_RE = re.compile(
    rf"\b{_TIME_QUALIFIER}\s+(?P<h>half an hour|half hour|an hour|one hour|\d{{1,2}}\s*(?:hour|hours|hr|hrs))\b",
    re.IGNORECASE,
)

# Speed asked for without a number. These become the corpus's own
# `30_minutes_or_less` annotation, NOT an invented 30-minute ceiling: a member
# who said "quick" did not say a number, and turning their adjective into a
# hard numeric filter is the invented-constraint bug the ledger exists to stop.
_VAGUE_SPEED_RE = re.compile(
    r"\b(quick|quicker|quickly|fast|speedy|super fast|in a hurry|no time|"
    r"something quick|weeknight)\b",
    re.IGNORECASE,
)
_SPEED_CLAIM_TAG = "30_minutes_or_less"

# "take as long as you like" — the retraction, so a ceiling stated once is not
# permanent when the member changes their mind on a Sunday.
_TIME_CLEAR_RE = re.compile(
    r"\b(take as long as|as long as (it|you) (takes|like|need)|"
    r"time is no|no (time )?(limit|rush)|don'?t (worry|mind) about (the )?time)\b",
    re.IGNORECASE,
)

# Bounds. Below the floor is a typo or a joke; above the ceiling is not a
# constraint a recipe corpus can honour. Deliberately WIDER than the slider's
# 10-90: the slider's range is a UI affordance, and refusing "under two hours"
# because no knob goes there would be the tool telling the member they are
# wrong about their own evening.
_MIN_MINUTES = 5
_MAX_MINUTES = 240


def extract_time_delta(message: str):
    """A cooking-time ceiling stated in words. Never raises.

    Returns an empty delta when the message says nothing about time, which is
    most messages.
    """
    from models.planning_state import PlanningStateDelta

    text = (message or "").strip()
    if not text:
        return PlanningStateDelta()

    if _TIME_CLEAR_RE.search(text):
        return PlanningStateDelta(max_minutes_clear=True)

    minutes = _normalize_minutes(_stated_minutes(text))
    if minutes is not None:
        return PlanningStateDelta(max_minutes=minutes)

    if _VAGUE_SPEED_RE.search(text):
        return PlanningStateDelta(claim_tags=(_SPEED_CLAIM_TAG,))
    return PlanningStateDelta()


def _normalize_minutes(minutes: int | None) -> int | None:
    """A stated duration expressed in the same units the slider uses, or None.

    One constraint, one number. The slider and the sentence must land on the
    same value or the card will show a limit the planner is not using, and
    pressing Apply would then silently overwrite what the member said.

    Two edges, both decided in the safe direction:

    * **Tighter than the slider's floor.** "In five minutes" snaps up to 10.
      Nothing in the corpus is a five-minute meal, so the choice is between the
      tightest limit the product supports and no plan at all.
    * **Looser than its ceiling.** "Under two hours" sets NO limit rather than
      90. Ninety would be a tighter constraint than the member stated — the
      invented-constraint bug — and 120 would put the card's knob off its own
      track. Above 90 the limit filters out almost nothing anyway, so dropping
      it costs the member nothing and asserts nothing untrue.
    """
    if minutes is None:
        return None
    ceiling = next(d["max"] for d in PARAMETER_DEFS if d["key"] == "cooking_time")
    if minutes > ceiling:
        _logger.info(
            "Stated cooking limit of %d min is looser than the %d min the card "
            "can express — no ceiling applied.", minutes, ceiling,
        )
        return None
    return sanitize({"cooking_time": minutes}).get("cooking_time")


def _stated_minutes(text: str) -> int | None:
    """The tightest explicit limit in the message, or None.

    Tightest, not first: "under an hour, ideally 20 minutes" is a member
    telling you both their limit and their preference, and honouring the looser
    of the two answers the wrong one.
    """
    found: list[int] = []
    for pattern in (_MINUTES_RE, _MINUTES_TRAILING_RE, _MINUTES_COMPOUND_RE):
        for match in pattern.finditer(text):
            found.append(int(match.group("n")))
    for match in _HOURS_RE.finditer(text):
        raw = match.group("h").lower().strip()
        if raw in _HOUR_WORDS:
            found.append(_HOUR_WORDS[raw])
        else:
            digits = re.search(r"\d{1,2}", raw)
            if digits:
                found.append(int(digits.group()) * 60)

    usable = [m for m in found if _MIN_MINUTES <= m <= _MAX_MINUTES]
    return min(usable) if usable else None


def apply_state(profile: dict, state) -> dict:
    """The profile a planning path should use, given the standing constraints.

    Only the cooking-time ceiling, and only because it is the one standing
    constraint whose consumers do NOT read the planning state — all seven fetch
    sites read `profile["plan_parameters"]["cooking_time"]` through
    `max_duration_minutes`. Writing it here means a sentence and a slider are
    the same constraint at every one of them, rather than a seventh place to
    remember.

    Mutates and returns `profile`: every caller has already taken its own copy
    of the session profile, and returning a second one would leave the
    underscore keys the paths write next on the wrong dict.
    """
    minutes = getattr(state, "max_minutes", None)
    if minutes:
        params = dict(profile.get("plan_parameters") or {})
        params["cooking_time"] = int(minutes)
        profile["plan_parameters"] = params
    return profile


def describe(values: dict) -> str:
    """Canonical refinement message for sanitized values (deterministic)."""
    phrases = []
    for definition in PARAMETER_DEFS:  # definition order keeps output stable
        key = definition["key"]
        if key not in values:
            continue
        phrase = _PHRASES[key]
        phrases.append(phrase(values[key]) if callable(phrase) else phrase[values[key]])
    return "Adjust my meal plan to these settings: " + "; ".join(phrases) + "."


def history_line(values: dict) -> str:
    """Short known-facts line so the reconciler never re-asks these topics."""
    defs = {d["key"]: d for d in PARAMETER_DEFS}
    parts = []
    for key, value in values.items():
        definition = defs[key]
        if definition["kind"] == "scale":
            parts.append(f"{definition['label']}: {value} {definition['unit']}")
        else:
            label = next(o["label"] for o in definition["options"] if o["value"] == value)
            parts.append(f"{definition['label']}: {label}")
    return "User set plan parameters — " + "; ".join(parts)
