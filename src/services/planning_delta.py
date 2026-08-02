"""Reading one turn's message into a `PlanningStateDelta`.

Only what this turn *mentioned*. Silence must stay silence: a delta that fills
in unmentioned fields would overwrite standing constraints with defaults every
turn, which is the bug the standing state exists to fix.

Two extractors, deliberately different in kind:

**Reset** is a regex. "Start over" has to work when the model is down, and a
member who wants to abandon three turns of accumulated constraints should not
be arguing with an LLM about whether they meant it.

**Shape** is the LLM `PlanSpecExtractor`, which already abstains when a message
says nothing about structure. Its `mentioned` flag is exactly the distinction
this module is built around, so it is reused rather than reimplemented.
"""

from __future__ import annotations

import logging
import re

from models.planning_state import PlanningStateDelta

logger = logging.getLogger(__name__)

# Phrases that abandon the accumulated plan. Anchored so "forget the salt"
# inside a longer request does not wipe a member's whole session.
_RESET = re.compile(
    r"^\s*(start over|start again|forget (about )?(this|that|it|everything)|"
    r"scrap (this|that|it)|from scratch|reset|never mind( all( of)?)? that)\b",
    re.IGNORECASE,
)


def extract_state_delta(message: str, *, extractor=None) -> PlanningStateDelta:
    """What this turn changes about the standing plan constraints.

    Never raises. A turn whose delta cannot be read is a turn that changes
    nothing, which leaves the member with the constraints they already had —
    the safe direction.
    """
    text = (message or "").strip()
    if not text:
        return PlanningStateDelta()

    if _RESET.match(text):
        logger.info("Plan state reset requested")
        return PlanningStateDelta(reset=True)

    spec = None
    try:
        if extractor is None:
            from agents import PlanSpecExtractor

            extractor = PlanSpecExtractor()
        candidate = extractor.extract(text)
        # `is_default` here means the extractor abstained — it says nothing
        # about shape, so the standing shape must be left alone. Sending the
        # default would reset a member's "salads on the side" every time they
        # said anything else.
        if not candidate.is_default:
            spec = candidate
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plan shape extraction failed: %s", exc)

    return PlanningStateDelta(spec=spec)
