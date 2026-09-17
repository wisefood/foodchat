"""
Hearing the member, once per turn, whatever the turn turns out to be.

FoodChat's extractors — shape, pantry, diet, facets, cooking time — lived
inside the handlers rather than in front of them. The daily path ran four of
them. The weekly path ran two. **Every other kind of turn ran none.** So this
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

**Concurrent, because they are independent HTTP calls.** Run in sequence the
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
from dataclasses import replace
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

# What this turn added to the plan's SHAPE, if anything — "added breakfast",
# "added a salad to lunch". Read by the router: a turn that grows the shape is a
# re-plan, not a slot swap, whatever the intent classifier called it.
_SHAPE: ContextVar[list] = ContextVar("turn_shape_additions", default=[])
# Whether the shape extractor spoke THIS turn — as opposed to the standing
# spec merely carrying what an earlier turn said. The router needs the
# difference: a standing seven-day horizon is not a request for seven days.
_NAMED: ContextVar[bool] = ContextVar("turn_shape_named", default=False)


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
                # the statements its siblings heard correctly.
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
    from services import diet_intent, intent_facets, pantry_service, plan_parameters
    from services.planning_delta import extract_state_delta

    text = (message or "").strip()
    if not text:
        return []

    deltas = _run([
        lambda: extract_state_delta(text),
        lambda: pantry_service.extract_pantry_delta(text),
        lambda: diet_intent.extract_diet_delta(text),
        lambda: intent_facets.extract_facet_delta(text),
        # A regex, not a model call, and cheap enough that it rides the same
        # fan-out rather than earning its own branch.
        lambda: plan_parameters.extract_time_delta(text),
    ])
    # Reset first, so "start over — but I still have the spinach" keeps the
    # spinach. `merge` returns a blank state for a reset delta and discards
    # everything already on it, so a reset arriving last would wipe every other
    # statement this same turn made.
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
    standing_spec = state.spec
    deltas = extract(message)
    _NAMED.set(any(d.spec is not None for d in deltas))

    # The shape reader runs FIRST, against the shape that is STANDING.
    #
    # It used to run after the merge, and that made it blind exactly when it
    # was needed. The LLM shape extractor answers with the meals the MESSAGE
    # mentions, not with the plan the member wants: for "add a snack after my
    # lunch" it says `lunch, snack`. `merge` takes a delta's spec wholesale, so
    # the standing breakfast/lunch/dinner became lunch+snack — breakfast and
    # dinner deleted — and the reader, running next, then saw a snack already
    # in the shape and reported no addition at all. The member got their plan
    # quietly cut to two meals AND "which meal should I swap?", because the
    # router's re-plan guard reads that empty answer.
    #
    # Invisible offline, which is why the suite never caught it: with no
    # network the extractor fails, the delta is empty, and the reader sees the
    # real standing shape.
    from services import shape_intent

    grown, added = shape_intent.additions(message, standing_spec)
    # Removals read against the GROWN shape, so one message can do both —
    # "drop the snack and add a dessert" is one sentence and two changes.
    shaped, removed = shape_intent.removals(message, grown)
    changed_shape = added + removed

    # "Add a snack BEFORE MY LUNCH" is one request, and the second half used to
    # reach nothing: the router skips its reorder branch whenever the shape
    # grew, on the reasoning that an addition mentioning an order is still an
    # addition. It is — and the order is still part of what was asked for, so
    # it is applied to the shape the addition just produced, where it costs
    # nothing and lands in the same re-plan.
    from services import plan_navigation

    move = plan_navigation.reorder_request(message)
    if move is not None:
        slot, where, anchor = move
        moved = shaped.reorder(slot, **{where: anchor})
        if moved is not shaped:
            shaped = moved
            changed_shape.append(f"moved {slot} {where} {anchor}")

    # How many days, read deterministically. The LLM extractor may abstain, and
    # on a REFINEMENT `plan_horizon` deliberately leaves the days alone — so
    # "switch to daily from three days plan" said one day, nothing read it as a
    # number, and the three-day plan came back three days long.
    days = shape_intent.horizon(message)
    if days is not None and days != shaped.num_days:
        shaped = shaped.with_days(days)
        changed_shape.append(f"{days} day(s)")

    _SHAPE.set(changed_shape)

    for delta in deltas:
        if delta.is_empty:
            continue
        if changed_shape and delta.spec is not None:
            # An addition AMENDS the plan; it never replaces it. The horizon is
            # still the extractor's to state — "three days, and add a snack" is
            # one message — so only the meals and plates are refused.
            shaped = shaped.with_days(delta.spec.num_days)
            delta = replace(delta, spec=None)
        state = state.merge(delta)

    if changed_shape:
        state = state.merge(PlanningStateDelta(spec=shaped))

    # Shape ADDITIONS, before the facet retraction.
    #
    # "Add breakfast" and "a salad on the side" are changes to the plan's
    # SHAPE, and nothing could act on them: an edit replaces the dish on a slot,
    # so a request for a slot that does not exist got "this plan has lunch,
    # dinner — which of those should I change?", and the member said "I don't
    # have a breakfast" again. There was no way out of that loop.
    #
    # Additive against the standing spec rather than replacing it, which is the
    # other half: the shape extractor answers with the meals the MESSAGE
    # mentions, and `merge` takes a delta's spec wholesale — so "add a salad to
    # lunch" would have set the day to lunch alone.
    # Taking a facet back, last and against the merged state.
    #
    # Last because it can only remove something that is standing, and the thing
    # being retracted may have arrived this very turn — a facet extractor that
    # reads "not spicy" as a request for spicy is repaired here rather than
    # shipping the opposite of what was said. The UI has always been able to do
    # this: a chip has an × and it calls DELETE /facets/{value}. Chat could not.
    from services import intent_facets

    removals = intent_facets.extract_facet_removals(message, state)
    if not removals.is_empty:
        state = state.merge(removals)

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
    _SHAPE.set([])
    _NAMED.set(False)


def plan_horizon(state: PlanningState, *, is_refinement: bool) -> PlanningState:
    """The state a FRESH daily request should plan against: one day, unless
    this turn said otherwise.

    The horizon follows the request; the shape is standing. "Plan my week"
    writes `num_days=7` into the standing spec, where it sits for the rest of
    the session. A later "plan for today" says nothing the shape extractor can
    read, so it abstains, the standing spec is left alone — and the day the
    member asked for comes out as seven days on the daily canvas. That is the
    "daily plans are still weekly plans, just with another layout" report: a
    fresh session has no standing week and gets the three recipes it asked
    for; a session that ever planned a week never does again.

    Meals and plates are left exactly as they stand — "salads on the side" is
    a preference and survives. A refinement keeps its days too: "make day 2
    lighter" on a three-day plan is about that plan. And a turn that named a
    horizon itself ("three days, please") already merged it, so it is kept.
    """
    if is_refinement or state.spec.num_days <= 1 or named_shape():
        return state
    return state.merge(PlanningStateDelta(spec=state.spec.with_days(1)))


def named_shape() -> bool:
    """Whether this turn's message itself described the plan's shape.

    False means the extractor abstained and everything on `state.spec` is
    standing — carried from an earlier turn. The distinction matters for the
    horizon: the standing spec keeps `num_days=7` after "plan my week", and a
    later "plan for today" that says nothing about days must not inherit it.
    """
    return bool(_NAMED.get())


def added_shape() -> list:
    """What this turn CHANGED about the plan's shape — `[]` when nothing.

    Additions and removals both, because the router asks one question of this:
    did the shape move? "Add a snack" and "drop the snack" are the same kind of
    turn — a re-plan — and only the edit path cares which.

    The router reads this rather than re-deriving it: intake has already done
    the work, and two places deciding what "add a salad" means is how they come
    to disagree.
    """
    return list(_SHAPE.get() or [])


def current(session_id: str, *, session_service=None) -> PlanningState:
    """The standing state for this turn, without extracting anything.

    For paths that need to read the constraints but are not the turn's entry
    point — an edit fetching a replacement, say. Returns the memo when intake
    has already run this turn, and the stored state otherwise, so a handler
    called outside a turn still sees the member's accumulated constraints
    rather than a blank slate.
    """
    cached = _TURN.get()
    if cached is not None and cached[0] == session_id:
        return cached[2]
    if session_service is None:
        import services as _services

        session_service = _services.session_service
    return session_service.get_planning_state(session_id)
