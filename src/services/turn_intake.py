"""
Hearing the member, once per turn, whatever the turn turns out to be.

FoodChat has four extractors — shape, pantry, diet, facets — and until now they
lived inside the handlers rather than in front of them. The daily path ran all
four. The weekly path ran two. **Every other kind of turn ran none.** So this
conversation lost a hard constraint:

    "swap Tuesday's dinner"        → edit turn
    "by the way, I'm coeliac now"  → still an edit turn: extracted nothing
    "plan next week"               → weekly, gluten-free never stated

The member said it. The system routed on it. Nothing recorded it. That is the
same failure the standing planning state was built to fix, one level up: it
fixed *silence is not a retraction*, and left *a statement is only heard on the
path that happens to listen*.

Intake is that missing seam. It runs before routing, so every turn — smalltalk,
a question about the plan, a slot edit, a tool call — updates the standing
constraints, and every path downstream reads the same state it always did.

**Concurrent, because it is four independent HTTP calls.** Run in sequence the
daily path already paid four fast-tier round trips before planning began;
hoisting that onto every turn in sequence would have made the cheap turns
expensive. Fanned out, intake costs roughly one call of latency, and the daily
path gets *faster* than it was.

**Never skipped, whatever the budget says.** `turn_budget` sheds grading,
metrics and the response writer when a turn runs late — everything that makes a
plan better rather than makes it exist. Hearing the member is not in that
category: a dropped "I'm coeliac" does not produce a worse plan, it produces a
confidently wrong one.

**Memoised per turn.** `chat_service` and `weekly_plan_service` still call
intake where they always extracted, so their code reads the same and neither
depends on the orchestrator having gone first. The second call in a turn is
free.

    state = turn_intake.intake(session_id, message)
"""

from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Callable, Optional

from models.planning_state import PlanningState, PlanningStateDelta

logger = logging.getLogger(__name__)

# The result of this turn's intake: (session_id, message, state).
#
# Keyed by the message as well as the session so a second turn in the same
# request context — `apply_plan_parameters` regenerating through a handler —
# re-reads rather than reusing the previous turn's answer.
_TURN: ContextVar[Optional[tuple[str, str, PlanningState]]] = ContextVar(
    "turn_intake", default=None
)


def _run(fns: list[Callable[[], PlanningStateDelta]]) -> list[PlanningStateDelta]:
    """Every extractor at once, each in a copy of this turn's context.

    The context copy is what keeps Langfuse attribution working: `trace_context`
    binds the session and member on a ContextVar, and a bare thread would start
    with none of it, so four extractions per turn would appear as orphan traces
    belonging to nobody.
    """
    with ThreadPoolExecutor(max_workers=len(fns)) as pool:
        # A fresh copy per call: a Context can only be entered once at a
        # time, so handing the same one to four threads raises rather than
        # extracting anything.
        futures = [pool.submit(contextvars.copy_context().run, fn) for fn in fns]
        out = []
        for future in futures:
            try:
                out.append(future.result())
            except Exception as exc:  # noqa: BLE001
                # Each extractor already promises not to raise, so this is the
                # belt to that brace — and it is worth having, because the
                # alternative is one extractor's bug taking down the turn AND
                # the three statements its siblings heard correctly.
                logger.warning("Intake extractor failed: %s", exc)
                out.append(PlanningStateDelta())
        return out


def extract(message: str) -> list[PlanningStateDelta]:
    """This turn's deltas, in merge order. Never raises.

    Always the member's own words, never a refinement context. Pantry, diet and
    facets were already read from the raw message for a documented reason — a
    refinement quotes the plan on screen, and the chicken in a recipe someone is
    *looking at* is not a statement that they eat chicken. The shape extractor
    was the one that still saw the context, and it had the same problem in a
    sharper form: shown a five-meal plan followed by "actually just breakfast
    and dinner", the strongest shape signal in its input is the shape being
    replaced. Reading the raw message is the consistent choice and the cheaper
    one — the context dump was the longest input any extractor received.
    """
    from services import diet_intent, intent_facets, pantry_service
    from services.planning_delta import extract_state_delta

    text = (message or "").strip()
    if not text:
        return []

    deltas = _run([
        lambda: extract_state_delta(text),
        lambda: pantry_service.extract_pantry_delta(text),
        lambda: diet_intent.extract_diet_delta(text),
        lambda: intent_facets.extract_facet_delta(text),
    ])
    # Reset first, so "start over — but I still have the spinach" keeps the
    # spinach. `merge` returns a blank state for a reset delta and discards
    # everything already on it, so a reset arriving last would wipe the three
    # statements this same turn made.
    return sorted(deltas, key=lambda d: 0 if d.reset else 1)


def intake(session_id: str, message: str, *,
           session_service=None) -> PlanningState:
    """Merge this turn's statements into the standing state, and persist them.

    Returns the state the rest of the turn should plan against. Safe to call
    more than once per turn: the second call returns the first one's answer
    rather than paying for the extraction again.
    """
    if session_service is None:
        import services as _services

        session_service = _services.session_service

    cached = _TURN.get()
    if cached is not None and cached[0] == session_id and cached[1] == message:
        return cached[2]

    state = session_service.get_planning_state(session_id)
    before = state
    for delta in extract(message):
        if not delta.is_empty:
            state = state.merge(delta)

    if state != before:
        try:
            session_service.set_planning_state(session_id, state)
        except Exception as exc:  # noqa: BLE001
            # A session that vanished mid-turn, most likely. The turn can still
            # plan against what was heard; it just will not survive to the next.
            logger.warning("[%s] Could not persist planning state: %s", session_id, exc)
        logger.info("[%s] Standing plan state: %s", session_id, state.describe())

    _TURN.set((session_id, message, state))
    return state


def forget() -> None:
    """Drop the memo. For tests, and for a caller replaying one session."""
    _TURN.set(None)
