"""
How long a turn is allowed to take, and what it gives up when it runs late.

The timeout ladder was upside down. The UI waited 180 seconds, the gateway gave
up at 90, FoodChat had no limit of its own, and one Groq call had no limit
either. So a slow planning turn produced the worst possible outcome: the
gateway cut the connection at 90 seconds and the member saw a failure, while
FoodChat carried on, finished the plan, and stored it. The plan existed. The
member had been told it did not. They found it on the next reload.

A budget fixes that by making the innermost layer the one that decides. The
turn knows its own deadline, and when it is running out it **sheds optional
work rather than dying**:

    grading            skipped   → the plan is unranked, not absent
    quality metrics    skipped   → no scores on the card
    the response writer  skipped → a plain sentence instead of a written one
    fetching, storing  never skipped — that IS the plan

That ordering is the whole design. Everything above the line makes a plan
better; nothing above the line makes it exist. A member would rather have an
unranked plan in forty seconds than a beautifully graded one they never see.

    with turn_budget.start():
        ...
        if turn_budget.can_afford(GRADING_COST):
            plans = grade(plans)

The budget is advisory by construction — nothing here interrupts a call in
flight. It is checked BETWEEN stages, which is the only place a Python service
can safely give up without leaving a half-written plan behind.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

# The wall-clock budget for one turn.
#
# Must stay comfortably under the gateway's own timeout for FoodChat (90s), so
# that a slow turn is FoodChat's decision and not a cut connection. 70 leaves
# room for the response to serialise and travel.
TURN_BUDGET_SECONDS = float(os.getenv("FOODCHAT_TURN_BUDGET", "70"))

# Rough costs, used only to answer "is there time for this?". Deliberately
# pessimistic: skipping a stage that would have fit is a cheaper mistake than
# starting one that will not.
COST_GRADING = 25.0
COST_METRICS = 20.0
COST_WRITER = 10.0
COST_FETCH = 20.0

_deadline: ContextVar[Optional[float]] = ContextVar("turn_deadline", default=None)


@contextmanager
def start(seconds: Optional[float] = None):
    """Open a budget for one turn. Nested calls keep the outermost deadline.

    Nesting matters: `apply_plan_parameters` and `regenerate` both route into
    the same handlers a chat turn uses, and a nested budget would silently give
    the inner stage a fresh full allowance — which is exactly the runaway the
    budget exists to prevent.
    """
    existing = _deadline.get()
    if existing is not None:
        yield
        return
    token = _deadline.set(time.monotonic() + (seconds or TURN_BUDGET_SECONDS))
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining() -> Optional[float]:
    """Seconds left, or None when no budget is running.

    None is not zero. Outside a turn — a tool invocation, a test, a background
    job — there is no deadline, and `can_afford` says yes to everything. A
    budget that defaulted to expired would disable grading everywhere it was
    not explicitly started.
    """
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()


def can_afford(cost: float) -> bool:
    """Whether an optional stage of roughly this cost still fits."""
    left = remaining()
    if left is None:
        return True
    return left >= cost


def expired() -> bool:
    left = remaining()
    return left is not None and left <= 0


def skip(stage: str, cost: float) -> bool:
    """`True` when `stage` should be skipped, with one line saying why.

    Logged rather than silent: a plan that came back unranked because the turn
    was running late looks identical to a grader that is broken, and the
    difference is the first thing anyone debugging will want.
    """
    if can_afford(cost):
        return False
    left = remaining()
    logger.warning(
        "Turn budget low (%.1fs left, %s needs ~%.0fs) — skipping it. "
        "The plan is returned without it.",
        left if left is not None else -1.0, stage, cost,
    )
    return True
