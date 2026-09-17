"""
"Add breakfast." "And a salad on the side."

Adding was the one thing nothing could do.

    add breakfast as well i dont have one
      → "Day 1 doesn't have a breakfast in this plan — it has lunch, dinner.
         Which of those should I change?"
    okay yes but my day doesnt have breakfast
      → the same sentence again

There is no way out of that loop, and it is not the edit path being unhelpful:
an edit REPLACES the dish on a slot, and the member is asking for a slot that
does not exist. The request is a change to the plan's SHAPE, and `PlanSpec` is
the type that holds shape — so it has to reach the spec, not the editor.

The same fault, quieter, on plates:

    also lighter lunch and add a salad as well there for side
      → lunch swapped for something lighter, and no salad

The swap was classified, executed and reported. The addition was heard by
nobody, so the member read a confident answer to half of what they asked.

Deterministic on purpose, like the cooking-time and facet-removal readers. The
targets are closed sets — the slots RecipeWrangler fills and the roles a plate
can have — so there is nothing to invent, and a missed addition is a worse
outcome than a wrong one only if the wrong one is possible. It is not: an
unrecognised word matches nothing and the turn routes exactly as it did.

    additions(message, spec) -> (PlanSpec, list[str])
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# "add", "include", "and", "also", "with", "throw in", "chuck in", "put".
#
# Deliberately broad on the verb and narrow on the target: the target has to be
# a slot or a role the corpus can actually fill, so a loose verb cannot invent
# a meal.
_ADD = (
    r"(?:add|include|throw in|chuck in|put in|put|give me|i want|"
    r"i'd like|i would like|"
    # "lunch should have a soup as well" is an addition phrased as a
    # requirement rather than a command, and it was reaching nothing.
    r"should (?:have|get|come with)|needs?|wants?|have|"
    r"also|as well as|plus|with|and)"
)

# Words that mean "not that" — so "no dessert" and "without a salad" are never
# read as additions.
#
# Matched only in the few words IMMEDIATELY BEFORE the target, not anywhere in
# the clause. A clause-wide check read "add breakfast as well i dont have one"
# as a negation, because "dont" is in it — when that phrase is the member's
# REASON for adding, not a refusal. Same for "my day doesnt have breakfast".
_NEGATED_BEFORE = re.compile(
    r"\b(?:no|not|without|skip|drop|remove|forget|lose|never|don'?t|doesn'?t)\b"
    r"(?:\s+\w+){0,2}\s*$",
    re.IGNORECASE,
)

# "on the side", "for side", "as a side" — the phrasing that means a PLATE
# rather than a meal, when the word itself could be either.
_AS_PLATE = re.compile(
    r"\b(?:on|as|for|to)\s+(?:the\s+|a\s+|its\s+)?side\b|\bon the side\b",
    re.IGNORECASE,
)


def additions(
    message: str,
    spec,
    *,
    focus_slot: Optional[str] = None,
) -> tuple[object, list[str]]:
    """`(spec, what changed)`. The same spec and `[]` when nothing was added.

    `focus_slot` is the meal the rest of the turn is about — "add a salad as
    well THERE" names no meal, and the sentence beside it does ("lighter
    lunch"). Without a focus and without a named slot a plate addition is
    ambiguous, and it is left alone rather than applied to a meal the member did
    not choose.
    """
    from models.plan_spec import KNOWN_SLOTS, ROLES

    text = (message or "").strip()
    if not text:
        return spec, []

    changed: list[str] = []
    result = spec

    # Which meal a plate would go on, worked out first — because it decides how
    # to read the words that are BOTH a slot and a role.
    #
    # `side`, `dessert` and `drink` are each a meal in their own right and a
    # plate on another meal. "Add a side to dinner" names dinner, so it is a
    # plate; "add a dessert" names nothing, so it is a course of its own. Read
    # the wrong way round it produced a meal called "side", which is not a
    # thing anybody asked for.
    target = _named_slot(text, spec) or focus_slot

    for slot in KNOWN_SLOTS:
        if slot in result.meals:
            continue
        if not _mentioned_as_addition(text, slot):
            continue
        if slot in ROLES and (target or _AS_PLATE.search(text)):
            continue                      # a plate on `target`, handled below
        result = result.with_meal(slot)
        if result is not spec:
            changed.append(f"added {slot}")
    # A named course wins over the generic "side".
    #
    # "add a salad as well there for side" says WHERE the salad goes, not that a
    # salad and a side are both wanted — reading both gave lunch three plates
    # for a request that named one.
    wanted = [
        role for role in ROLES
        if role != "main" and _mentioned_as_addition(text, role)
    ]
    specific = [r for r in wanted if r != "side"]
    if specific and "side" in wanted:
        wanted = specific

    for role in wanted:
        if not target:
            logger.info(
                "Heard %r as a plate to add but no meal was named — leaving the "
                "shape alone rather than guessing which meal", role,
            )
            continue
        before = result
        result = result.with_plate(target, role)
        if result is not before:
            changed.append(f"added a {role} to {target}")

    if changed:
        logger.info("Shape additions: %s -> %s", "; ".join(changed), result.describe())
    return result, changed


# "remove the snack", "drop the dessert", "no snack today", "skip breakfast".
#
# Narrow on the verb and narrow on the target, unlike `_ADD`: a removal throws
# work away, so a loose verb that fires by accident costs the member a meal.
_REMOVE = (
    r"(?:remove|drop|delete|cancel|skip|lose|get rid of|take out|take off|"
    r"do not want|don'?t want|no more|without|no)"
)


def removals(message: str, spec, *, focus_slot: Optional[str] = None):
    """Meals and plates the message asks to take OUT. `(spec, notes)`.

    The counterpart to `additions`, and it was missing — so "remove the snack
    from midday" reached the edit path, which can only replace the dish on a
    slot. It swapped the member's lunch for a recipe called "Oat Snack Cakes",
    because the leftover word "snack" was searched as a dish title, and said
    "Done".

    Deliberately conservative in three ways, because this one destroys:

    * the verb must be a removal verb and it must sit just before the target,
      so "I had a snack earlier, plan dinner" is not a removal;
    * a meal is only removed if the plan HAS it — there is nothing to say
      about a breakfast that was never there;
    * the last meal cannot go, and neither can a main. "Cancel the plan" is a
      different request and this is not it.
    """
    from models.plan_spec import KNOWN_SLOTS, ROLES

    text = (message or "").strip()
    if not text:
        return spec, []

    changed: list[str] = []
    result = spec
    target = _named_slot(text, spec) or focus_slot

    for slot in KNOWN_SLOTS:
        if slot not in result.meals:
            continue
        if not _mentioned_as_removal(text, slot):
            continue
        # A word that is both a slot and a role, named alongside another meal,
        # is a plate of that meal — "drop the side from dinner".
        if slot in ROLES and target and target != slot:
            continue
        before = result
        result = result.without_meal(slot)
        if result is not before:
            changed.append(f"removed {slot}")
        else:
            logger.info(
                "Heard %r as a meal to remove but it is the only one left — "
                "leaving the plan alone", slot,
            )

    for role in ROLES:
        if role == "main" or not _mentioned_as_removal(text, role):
            continue
        slot = target if target and target != role else None
        if slot is None:
            logger.info(
                "Heard %r as a plate to remove but no meal was named — leaving "
                "the shape alone rather than guessing", role,
            )
            continue
        before = result
        result = result.without_plate(slot, role)
        if result is not before:
            changed.append(f"removed the {role} from {slot}")

    if changed:
        logger.info("Shape removals: %s -> %s", "; ".join(changed), result.describe())
    return result, changed


def _mentioned_as_removal(text: str, word: str) -> bool:
    """A removal verb, then the target, within the same breath."""
    pattern = (
        _REMOVE + r"(?:\s+\w+){0,2}\s+(?:the\s+|a\s+|an\s+|my\s+|any\s+)?"
        + word + r"(?:e?s)?\b"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


# "switch to daily", "make it one day", "just today", "three days", "a week".
#
# Deterministic, because the LLM shape extractor is allowed to abstain and
# routinely does on a REFINEMENT — "switch to daily from three days plan" is a
# sentence about the plan already on screen, and `plan_horizon` deliberately
# leaves a refinement's days alone. So the member asked for one day, nothing
# read it as a number, and the three-day plan came back three days long.
_DAY_WORDS = {
    "one": 1, "a": 1, "single": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7,
}
# The count has to be ASKED FOR, not merely mentioned. "I have three days of
# leftovers" is a fact about a fridge, and the first version of this read it as
# a three-day plan — the exact false positive the docstring below warns about,
# committed in the same breath as the warning.
_N_DAYS = re.compile(
    r"(?:^(?:for\s+)?|\b(?:plan|planning|make|make it|give me|do|want|need|"
    r"switch to|just|only|create)\s+(?:me\s+|it\s+|for\s+)?(?:a\s+)?)"
    r"(?P<n>\d|" + "|".join(_DAY_WORDS) + r")\s+days?\b",
    re.IGNORECASE,
)
_ONE_DAY = re.compile(
    r"\b(?:switch to|make it|just|only)\s+(?:a\s+)?(?:daily|one day|today|a day)\b"
    r"|\bdaily (?:plan|instead)\b|\bjust (?:today|one day)\b",
    re.IGNORECASE,
)
_A_WEEK = re.compile(
    r"\b(?:switch to|make it|for)\s+(?:a\s+)?(?:weekly|week)\b|\bweekly plan\b",
    re.IGNORECASE,
)


def horizon(message: str) -> Optional[int]:
    """How many days the member just asked for, or None.

    Narrow on purpose. "I have three days of leftovers" is not a plan horizon,
    so the number has to be attached to the word `day` or be one of the few
    phrasings that mean a horizon and nothing else.
    """
    text = (message or "").strip()
    if not text:
        return None
    if _ONE_DAY.search(text):
        return 1
    match = _N_DAYS.search(text)
    if match:
        raw = match.group("n").lower()
        days = int(raw) if raw.isdigit() else _DAY_WORDS.get(raw, 0)
        if 1 <= days <= 7:
            return days
    if _A_WEEK.search(text):
        return 7
    return None


# "my day doesn't have breakfast", "i don't have a lunch".
#
# A statement that the plan LACKS something is a request to add it — it is what
# the member says when "add breakfast" has already failed once. Narrow on
# purpose: it needs the subject and the verb, so it cannot catch "I don't have
# time" or "we don't have nuts in".
_MISSING = (
    r"\b(?:i|we|my (?:day|plan|meal plan)|it|this|there)\s+"
    r"(?:do(?:es)?n'?t|do(?:es)? not|hav(?:e|en'?t)\s+no)\s+"
    r"(?:have\s+)?(?:a\s+|any\s+)?{word}\b"
)


def _stated_missing(text: str, word: str) -> bool:
    return bool(re.search(_MISSING.format(word=re.escape(word)), text, re.IGNORECASE))


def _mentioned_as_addition(text: str, word: str) -> bool:
    """Whether `word` appears as something to ADD, not to remove.

    The add verb may sit before the target ("add a salad") or the target may
    simply be listed ("lunch with a salad"). What disqualifies it is a negation
    anywhere in the same clause — "no dessert" must never grow a dessert.
    """
    # An optional plural, because "add two sides to dinner" is the same request
    # as "add a side" and matched nothing at all.
    pattern = re.compile(rf"\b{re.escape(word)}(?:e?s)?\b", re.IGNORECASE)
    for match in pattern.finditer(text):
        clause = _clause_around(text, match.start())
        # Where the target sits inside its own clause, so "not" is only a
        # refusal when it is attached to THIS word.
        local = clause.lower().rfind(word.lower())
        if local != -1 and _NEGATED_BEFORE.search(clause[:local]):
            continue
        if re.search(
            rf"{_ADD}\b[^.;]*\b{re.escape(word)}(?:e?s)?\b", clause, re.IGNORECASE,
        ):
            return True
    return _stated_missing(text, word)


def _clause_around(text: str, index: int) -> str:
    """The sentence-ish fragment containing `index`.

    Clause-scoped rather than whole-message, so "add a salad, no dessert" adds
    the salad and does not add the dessert — a single negation anywhere in a
    long message must not veto every addition in it.
    """
    start = max(
        (text.rfind(sep, 0, index) for sep in (".", ";", ",", " and ", " but ")),
        default=-1,
    )
    ends = [pos for pos in (text.find(sep, index) for sep in (".", ";", ",")) if pos != -1]
    end = min(ends) if ends else len(text)
    return text[start + 1:end]


def _named_slot(text: str, spec) -> Optional[str]:
    """The meal this message names, if exactly one.

    A slot the plan does not serve yet still counts: "add a salad to lunch" on
    a dinner-only plan is a request for a lunch with a salad, and requiring the
    meal to already exist made that do nothing at all. `with_plate` adds the
    meal when the plate implies it.

    Meals the plan HAS are preferred when the message names several, because
    then the member is most likely pointing at one of them.
    """
    from models.plan_spec import KNOWN_SLOTS

    from models.plan_spec import ROLES

    named = [
        slot for slot in KNOWN_SLOTS
        if re.search(rf"\b{slot}(?:e?s)?\b", text, re.IGNORECASE)
    ]
    # A word that is also a ROLE is not the naming of a meal.
    #
    # "add a salad as well there for side" and "add a salad but no dessert"
    # both mention a slot word — `side`, `dessert` — that is doing a different
    # job in the sentence. Treating it as the target put the salad on a meal
    # called "side", and left "add a dessert" adding a dessert to a dessert.
    # A meal the plan already serves is exempt: then it really is being pointed
    # at.
    named = [slot for slot in named if slot not in ROLES or slot in spec.meals]
    if not named:
        return None
    existing = [slot for slot in named if slot in spec.meals]
    candidates = existing or named
    return candidates[0] if len(candidates) == 1 else None
