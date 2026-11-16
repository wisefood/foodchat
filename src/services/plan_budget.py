"""Which of the ranked days to serve, when the member set a calorie budget.

The grader ranks candidate days against the request and the profile, and it
sees each dish's calories — but it is a model, and a model can rank a day at
half its budget first. When that happens and another day the grader rated
almost as highly lands near the budget, that day is the better one to serve:
the member asked for a number, and a near-tie in the grader's opinion is not
a reason to miss it by a thousand calories.

Only ever a choice between days the grader already ranked. Nothing here
fetches, composes or scores, and with no budget it changes nothing.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# The ranked day is left alone while it is within this share of the budget.
FAR = 0.25
# A replacement must land within this share of it.
NEAR = 0.20
# And may be rated at most this much below the ranked day.
MAX_SCORE_DROP = 1


def day_kcal(plan) -> Optional[float]:
    """The day's calories from its dishes, or None when any dish has no figure.

    A partial sum would make a day with a missing dish look light, and this
    choice is exactly about which day is light.
    """
    total = 0.0
    slots = getattr(plan, "slots", None) or {}
    if not slots:
        return None
    for course in slots.values():
        nutrition = getattr(course, "nutrition", None) or {}
        kcal = nutrition.get("calories", nutrition.get("kcal"))
        if not isinstance(kcal, (int, float)):
            return None
        total += float(kcal)
    return total


def pick(plans: Sequence, kcal_target: Optional[float]):
    """The day to serve: the grader's first, unless it misses the budget badly.

    ``kcal_target`` is the member's own budget (or one their words named) —
    never a population reference, which a plan may be reported against but
    must not be chosen by.
    """
    if not plans:
        return None
    best = plans[0]
    if not kcal_target or kcal_target <= 0:
        return best
    best_kcal = day_kcal(best)
    if best_kcal is None or abs(best_kcal - kcal_target) <= FAR * kcal_target:
        return best
    for alternative in plans[1:]:
        kcal = day_kcal(alternative)
        if kcal is None:
            continue
        if (
            abs(kcal - kcal_target) <= NEAR * kcal_target
            and alternative.score >= best.score - MAX_SCORE_DROP
        ):
            logger.info(
                "Serving the day ranked %s (%d kcal) over the first (%d kcal): "
                "the budget is %d kcal",
                list(plans).index(alternative) + 1, int(kcal), int(best_kcal),
                int(kcal_target),
            )
            return alternative
    return best
