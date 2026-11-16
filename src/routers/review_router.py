"""Review-scoped reads: a chat exchange or a meal plan, for somebody reviewing them.

Separate from `foodchat_router` on purpose. Every route there is member-scoped
and passes through `_require_session`, and `tests/test_route_authorization.py`
enumerates that router to prove it. A reviewer reads conversations they do not
own, so these routes cannot satisfy that check — and rather than add an
exemption to a router whose whole guarantee is that there are none, they live
here, under their own prefix, with the asymmetry stated out loud.

**The authorization is the gateway's.** It requires an admin or expert token,
records the read as expert activity, and is the only route into this service
from outside the cluster. That is the same arrangement FoodScholar's Q&A review
endpoints already use; this router does not invent a posture, it matches one.
Where assertions are configured, the gateway signs these reads as the
reviewer (`auth.REVIEWER`) and every route here requires exactly that.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

import auth
from services.chat_review_service import CHAT_REVIEW_SERVICE, DEFAULT_WINDOW, MAX_WINDOW

logger = logging.getLogger(__name__)


def _reviewer_only() -> None:
    """The gateway signs review reads as the reviewer; nothing else gets in."""
    try:
        auth.check_reviewer()
    except auth.AssertionError_ as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


router = APIRouter(
    prefix="/foodchat/review",
    tags=["foodchat-review"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(_reviewer_only)],
)


@router.get("/messages/{message_id}/context")
def get_message_context(
    message_id: str,
    window: int = Query(DEFAULT_WINDOW, ge=1, le=MAX_WINDOW),
):
    """The rated message and the turns either side of it.

    404 when the message is gone. Chat sessions are pruned and a complaint
    outlives the conversation it was about, so this is an ordinary outcome the
    console renders as "the exchange is no longer stored" — not an error.
    """
    context = CHAT_REVIEW_SERVICE.message_context(message_id, window=window)
    if context is None:
        raise HTTPException(status_code=404, detail="No such chat message")
    return context


@router.get("/plans/{plan_id}")
def get_plan_summary(plan_id: str):
    """One meal plan's dishes, judges' scores and constraint rows.

    404 when the plan is gone: plans die with their conversation unless the
    member saved them, and the console says so rather than erroring.
    """
    summary = CHAT_REVIEW_SERVICE.plan_summary(plan_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="No such meal plan")
    return summary
