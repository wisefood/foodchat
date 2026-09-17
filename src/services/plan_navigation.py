"""Moving around a plan's history, and saying so when we cannot.

Two requests arrive as ordinary sentences and used to reach the edit path,
which can only replace the dish on a slot:

* **"go back to the first version"** — answered with a NEW plan, which is the
  one thing the member was asking not to happen. The canvas has always been a
  versioned lineage; nothing could read it back.
* **"put the snack before lunch"** — answered by swapping a dish, because the
  leftover words were searched as a recipe title. Order is genuinely not
  expressible yet, and the honest answer says so instead of changing food
  nobody asked about.

Deterministic on purpose. A member undoing their work is the worst moment to
spend a model call that might mean something else by it.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# "go back to the first version", "restore v2", "undo", "the previous one".
_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "original": 1, "initial": 1,
}
_RESTORE_VERB = r"(?:go back to|back to|restore|revert to|revert|return to|bring back|reinstate)"
_VERSION_WORD = r"(?:version|v|one|plan)"

_NUMBERED = re.compile(
    _RESTORE_VERB + r"\s+(?:the\s+)?(?:" + _VERSION_WORD + r"\s*)?[#v]?\s*(\d{1,2})\b",
    re.IGNORECASE,
)
_ORDINAL = re.compile(
    _RESTORE_VERB + r"\s+(?:the\s+)?(" + "|".join(_ORDINALS) + r")\b",
    re.IGNORECASE,
)
# "undo", "undo that", "revert" on its own — one step back, not a named version.
_UNDO = re.compile(
    r"^\s*(?:please\s+)?(?:undo|revert)(?:\s+(?:that|this|it|the last( change)?))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_PREVIOUS = re.compile(
    _RESTORE_VERB + r"\s+(?:the\s+)?(?:previous|last|earlier|prior)\b", re.IGNORECASE,
)


def restore_request(message: str) -> Optional[object]:
    """What version the member asked to go back to.

    Returns an int (that version), the string ``"previous"`` (one step back),
    or ``None``. The two are different answers: "go back to the first version"
    names one, "undo" means the step just taken, and guessing either way gets
    somebody the wrong plan.
    """
    text = (message or "").strip()
    if not text:
        return None
    if _UNDO.match(text) or _PREVIOUS.search(text):
        return "previous"
    match = _NUMBERED.search(text)
    if match:
        return int(match.group(1))
    match = _ORDINAL.search(text)
    if match:
        return _ORDINALS[match.group(1).lower()]
    return None


# "before lunch", "after dinner", "move the snack", "swap them round".
#
# Narrow: it has to name a MEAL to sit beside, or say "move"/"reorder" outright.
# "Something lighter before I go out" is not a reorder, and reading it as one
# would decline a request the edit path can serve perfectly well.
_REORDER = re.compile(
    r"\b(?:"
    r"(?:before|after|ahead of|earlier than|later than)\s+(?:the\s+|my\s+)?"
    r"(?:breakfast|brunch|lunch|dinner|snack|dessert|supper)"
    r"|(?:move|reorder|re-?order|rearrange|shift|swap)\s+(?:the\s+|my\s+)?"
    r"(?:breakfast|brunch|lunch|dinner|snack|dessert|order)"
    r"|(?:the\s+)?order\s+of\s+(?:the\s+)?(?:meals|day|plan)"
    r")\b",
    re.IGNORECASE,
)


def asks_to_reorder(message: str) -> bool:
    """Whether the member is asking for the day in a different ORDER.

    Cannot be served today, and that is a fact about the model rather than a
    missing branch: `slot_sort_key` derives eating order from the slot NAME, so
    a snack sits between lunch and dinner whatever the member's day looks like,
    and the UI sorts the same way independently. Both would have to learn an
    explicit order before this can mean anything.

    Detected anyway so the turn can SAY that. Left to the edit path, "put the
    snack before lunch" became a search for a recipe called "lunch" and the
    member's 82 kcal breakfast became a 1,907 kcal one, reported as done.
    """
    return bool(_REORDER.search(message or ""))


# "the snack before lunch", "move dinner after the snack", "lunch first".
_MOVE = re.compile(
    r"\b(?:move|shift|put|have|take|bring|make)?\s*(?:the\s+|my\s+)?"
    r"(?P<slot>breakfast|brunch|lunch|dinner|snack|dessert|supper)\b"
    r"[^.!?]{0,40}?"
    r"\b(?P<where>before|after|ahead of|earlier than|later than)\s+"
    r"(?:the\s+|my\s+)?(?P<anchor>breakfast|brunch|lunch|dinner|snack|dessert|supper)\b",
    re.IGNORECASE,
)
_BEFORE_WORDS = {"before", "ahead of", "earlier than"}


def reorder_request(message: str) -> Optional[tuple]:
    """`(slot, "before"|"after", anchor)`, or None.

    Reads only what it can act on: two named meals and a direction between
    them. "Better before lunch" names one meal and a direction, and the meal it
    means is whatever the turn was already about — that ambiguity is the
    caller's to resolve with a focus slot, not this reader's to guess.
    """
    match = _MOVE.search(message or "")
    if match is None:
        return None
    slot = match.group("slot").lower()
    anchor = match.group("anchor").lower()
    if slot == anchor:
        return None
    where = "before" if match.group("where").lower() in _BEFORE_WORDS else "after"
    return slot, where, anchor


# The direction and the anchor, when the meal to MOVE was left implied —
# "hmmm better before lunch" is the member's own phrasing and it names one meal,
# not two.
_HALF_MOVE = re.compile(
    r"\b(?P<where>before|after|ahead of|earlier than|later than)\s+"
    r"(?:the\s+|my\s+)?(?P<anchor>breakfast|brunch|lunch|dinner|snack|dessert|supper)\b",
    re.IGNORECASE,
)


def reorder_question(message: str, meals) -> Optional[str]:
    """What to ask when the member asked for an order but not a whole one.

    Asking beats guessing and it beats declining. The meal they mean is usually
    the one they were just talking about, and "usually" is how a member's
    breakfast gets moved when they meant their snack.
    """
    match = _HALF_MOVE.search(message or "")
    if match is None:
        return "Which meal should move, and where should it go?"
    anchor = match.group("anchor").lower()
    where = "before" if match.group("where").lower() in _BEFORE_WORDS else "after"
    movable = [m for m in (meals or ()) if m != anchor]
    if not movable:
        return None
    if len(movable) == 1:
        return f"Shall I move the {movable[0]} {where} the {anchor}?"
    listed = ", ".join(movable[:-1]) + f" or {movable[-1]}"
    return f"Which one should go {where} the {anchor} — the {listed}?"
