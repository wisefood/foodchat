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
"""

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
        "key": "difficulty",
        "label": "Difficulty",
        "kind": "choice",
        "options": [
            {"value": "easy", "label": "Easy"},
            {"value": "medium", "label": "Medium"},
            {"value": "hard", "label": "Elaborate"},
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
    {
        # How much of the week may be the same food twice. Like `food_waste`
        # this is a *dimension of the plan*, not of any one recipe, and it
        # pulls against variety — so it is a control rather than an inference:
        # nothing in "plan my week" says whether repeating Tuesday's dinner on
        # Wednesday reads as familiar or as lazy, and that answer differs by
        # household rather than by request.
        #
        # One ordered scale, each stop a superset of the one before it, for
        # the reason the food-waste comment gives: two knobs that can
        # contradict each other ("repeats: off, leftovers: on") is a settings
        # bug shipped as a feature.
        #
        # `breakfast` is the default because it is what the planner already
        # did before this control existed (M9). A member who never touches
        # this card gets exactly the week they got yesterday.
        "key": "repeat_meals",
        "label": "Repeat meals",
        # Weekly only. "The same breakfast may come back later in the week" is
        # not a setting a one-day plan can honour or even parse, and a control
        # that visibly does nothing teaches members that none of them work.
        # `sanitize` still accepts the key from any card: the value is the
        # member's standing preference, stored on the profile, and the weekly
        # planner is simply the only thing that reads it.
        "plan_types": ("weekly",),
        "kind": "choice",
        "options": [
            # 21 slots, 21 recipes — the pre-M9 rule, still what some people
            # want from a plan.
            {"value": "off", "label": "All different"},
            # A breakfast may come back after a gap. Nobody eats seven
            # different breakfasts.
            {"value": "breakfast", "label": "Repeat breakfasts"},
            # ...and so may a lunch or a dinner, under the same cooldown and
            # the same cap.
            {"value": "all", "label": "Repeat any meal"},
            # Plus the cross-slot one: yesterday's dinner may be today's
            # lunch. Deliberately the last stop — it is the only setting that
            # removes a cooking session rather than a shopping line.
            {"value": "leftovers", "label": "Cook once, eat twice"},
        ],
        "default": "breakfast",
    },
]

# The ordered stops of `repeat_meals`, loosest last. Index comparisons here
# are what make each stop a superset of the one before it, so a new stop is
# added to this tuple and nowhere else.
REPEAT_MODES: tuple = ("off", "breakfast", "all", "leftovers")
REPEAT_MODE_DEFAULT = "breakfast"

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
    "repeat_meals": {
        "off": "give every meal a different recipe",
        "breakfast": "the same breakfast may come back later in the week",
        "all": "any meal may come back later in the week, well spaced",
        "leftovers": "any meal may come back later in the week, and yesterday's dinner may be today's lunch — cook once, eat twice",
    },
}


def waste_mode(values: dict) -> str:
    """The applied food-waste setting: 'off', 'reuse' or 'strict'.

    Read by the weekly planner's scorer, which selects without an LLM — a
    preference that never becomes a number there is a preference that does
    not exist. The daily path hears the same setting as prose via `describe`.
    """
    value = values.get("food_waste")
    return value if value in ("reuse", "strict") else "off"


def repeat_mode(values: dict) -> str:
    """The applied repeat setting: 'off', 'breakfast', 'all' or 'leftovers'.

    Read by ``weekly_planner.action_adapter``, which decides whether a recipe
    may come back at all, and by the weekly scorer, which decides whether one
    is actually chosen. Both read it through this function rather than off the
    dict, so an unrecognised stored value degrades to the default instead of
    disabling the policy — a profile written by an older release must not
    silently become "all different".

    The default is what the planner did before the control existed, so an
    untouched card changes nothing.
    """
    value = values.get("repeat_meals")
    return value if value in REPEAT_MODES else REPEAT_MODE_DEFAULT


def repeat_mode_is_explicit(values: dict) -> bool:
    """Whether the member actually SET the repeat control.

    ``repeat_mode`` cannot answer this: it returns ``"breakfast"`` both for a
    member who chose "Repeat breakfasts" and for one who has never seen the
    card. Those deserve different planning, so the difference has to survive.

    The distinction is what gates the two things that make a repeat likely
    rather than merely legal — injecting an eligible earlier dish into the
    day's pool (``action_adapter``) and paying it a bonus
    (``planner._REPEAT_BONUS``). Both are a member's request being honoured;
    neither may fire on a default, because a default is not a request. So a
    week planned for someone who never touched this card is byte-for-byte the
    week they got before the control existed.

    An unrecognised stored value is NOT explicit. `repeat_mode` degrades it to
    the default, and acting on a value we could not read would be inventing a
    request rather than honouring one.
    """
    return values.get("repeat_meals") in REPEAT_MODES


def repeats_allowed(values: dict, meal_type: str) -> bool:
    """Whether a recipe already served may return to ``meal_type``.

    The slot question only — the gap and the cap are the action space's, and
    live next to the constants that state them.
    """
    mode = repeat_mode(values)
    if mode == "off":
        return False
    if mode == "breakfast":
        return str(meal_type).lower() == "breakfast"
    return True


def leftovers_allowed(values: dict) -> bool:
    """Whether yesterday's dinner may be served as today's lunch."""
    return repeat_mode(values) == "leftovers"


def build_card(profile: dict, plan_type: str = "daily") -> dict:
    """The card payload for a turn: definitions plus current applied values.

    ``plan_type`` is the card's address — the plan it was rendered with. The
    client sends it back on apply so the values refine THAT plan, not
    whichever canvas happens to be newest by then.
    """
    applied = profile.get("plan_parameters") or {}
    parameters = []
    for definition in PARAMETER_DEFS:
        # A definition may declare which plans it applies to; most apply to
        # both and say nothing.
        applies_to = definition.get("plan_types")
        if applies_to and plan_type not in applies_to:
            continue
        param = {k: v for k, v in definition.items() if k != "plan_types"}
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
