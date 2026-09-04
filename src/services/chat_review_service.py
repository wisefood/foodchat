"""Reading a chat exchange back, for someone reviewing feedback about it.

A thumbs-down on a chat message used to arrive in the shared feedback inbox as
a rating, a message id, and nothing else. The reviewer saw that somebody was
unhappy and had no way to see what they were unhappy *about* — the id points at
a row in this service's database, and the console has no route to it.

This is the route. Deliberately narrow: one message, the turns immediately
around it, and nothing else. Enough to judge the complaint, not a general
window onto anyone's chat history.

**Authorization lives at the gateway**, which requires an admin or expert token
and records the read as expert activity. This service verifies the caller is
the gateway the same way every other cross-service read here does, and does not
apply the member-ownership check that guards the member-facing routes: a
reviewer legitimately reads conversations they do not own, which is the whole
point, and applying an ownership check would make the endpoint useless rather
than safe. That asymmetry is why this lives in its own module with its own
name, instead of a flag on the member-facing conversation endpoint where it
would be one wrong default away from leaking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Turns either side of the rated message. Four is enough to see the question
#: that prompted an answer and the correction that followed it; more starts
#: being someone's conversation rather than the context for one complaint.
DEFAULT_WINDOW = 4
MAX_WINDOW = 20

#: Chat messages are free text a person wrote. Truncated so a reviewer sees
#: the exchange without the inbox becoming a transcript store.
MAX_CONTENT_CHARS = 4000


def _to_dict(row: Any, *, rated: bool) -> Dict[str, Any]:
    content = row.content or ""
    return {
        "id": row.id,
        "role": row.role,
        "content": content[:MAX_CONTENT_CHARS],
        "truncated": len(content) > MAX_CONTENT_CHARS,
        "intent": row.intent,
        "plan_id": row.plan_id,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "attribution": _load_json(getattr(row, "attribution", None)),
        # The message the feedback was actually about, so the console does not
        # have to compare ids to highlight it.
        "is_rated": rated,
    }


def _load_json(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


class ChatReviewService:
    """One rated message with the turns around it."""

    def message_context(
        self, message_id: str, *, window: int = DEFAULT_WINDOW
    ) -> Optional[Dict[str, Any]]:
        """The exchange a piece of feedback was about, or None if it is gone.

        None rather than an error for a missing message: chat sessions are
        pruned and a complaint outlives the conversation it was about, so an
        inbox item pointing at nothing is expected rather than exceptional.
        """
        from db import MessageRow, SessionLocal

        try:
            target_id = int(message_id)
        except (TypeError, ValueError):
            # Feedback carries whatever the client sent. A non-numeric id is
            # not a message here, and is not worth a database round trip.
            return None

        span = max(1, min(int(window or DEFAULT_WINDOW), MAX_WINDOW))
        db = SessionLocal()
        try:
            rated = db.query(MessageRow).filter(MessageRow.id == target_id).first()
            if rated is None:
                return None

            # Ordered by id rather than timestamp: two turns in one second are
            # common and their order is what makes the exchange readable.
            before = (
                db.query(MessageRow)
                .filter(
                    MessageRow.session_id == rated.session_id,
                    MessageRow.id < target_id,
                )
                .order_by(MessageRow.id.desc())
                .limit(span)
                .all()
            )
            after = (
                db.query(MessageRow)
                .filter(
                    MessageRow.session_id == rated.session_id,
                    MessageRow.id > target_id,
                )
                .order_by(MessageRow.id.asc())
                .limit(span)
                .all()
            )
            total = (
                db.query(MessageRow)
                .filter(MessageRow.session_id == rated.session_id)
                .count()
            )
        except Exception:
            logger.warning("chat_review.context_failed id=%s", message_id, exc_info=True)
            return None
        finally:
            db.close()

        messages: List[Dict[str, Any]] = [
            _to_dict(row, rated=False) for row in reversed(before)
        ]
        messages.append(_to_dict(rated, rated=True))
        messages.extend(_to_dict(row, rated=False) for row in after)

        return {
            "message_id": str(target_id),
            "session_id": rated.session_id,
            "messages": messages,
            "window": span,
            # So the console can say "5 of 42 turns" rather than implying the
            # exchange shown is the whole conversation.
            "session_messages": total,
        }


CHAT_REVIEW_SERVICE = ChatReviewService()
