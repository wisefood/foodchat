"""Reporting what only FoodChat knows.

The gateway records that a chat turn was requested and by which member. It
cannot see what the turn cost, which intent it was routed to, whether a plan
came out of it, or how many model calls it took to get there. Those are
reported from here.

Identity is the awkward part. FoodChat receives a signed *member* assertion,
not a Keycloak subject — deliberately, since it has no business holding one —
so it reports `member_id` and the correlation id, and the gateway's own record
for that same id supplies the user. Nothing here invents an identity it cannot
prove.

Everything is a no-op unless ``ANALYTICS_ENABLED`` is true and an ingest secret
is configured. Nothing raises.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import wf_telemetry

logger = logging.getLogger(__name__)

APP = "foodchat"


def current_identity() -> Dict[str, Optional[str]]:
    """The member this turn belongs to, resolved at call time.

    Read from the trace context the orchestrator binds per turn, falling back
    to the middleware's asserted member. Resolved lazily because the callbacks
    that use it are attached once to a pooled LLM client and then serve every
    member — an identity captured at construction would label every model call
    with whoever happened to warm the pool.
    """
    member_id = None
    try:
        from backend.observability import current_trace_context

        member_id = (current_trace_context() or {}).get("user_id")
    except Exception:
        member_id = None
    if not member_id:
        try:
            import auth

            member_id = auth.asserted_member()
        except Exception:
            member_id = None
    return {"user_id": None, "member_id": member_id}


def report_turn(
    *,
    session_id: str,
    intent: Optional[str],
    plan_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Report one completed chat turn."""
    try:
        props: Dict[str, Any] = {
            "session_id": session_id,
            "intent": intent,
            "plan_id": plan_id,
            "latency_ms": int(latency_ms) if latency_ms is not None else None,
        }
        props.update(dict(extra or {}))
        wf_telemetry.TELEMETRY.event(
            "chat.turn", props=props, app=APP, **current_identity()
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("activity.report_turn_failed", exc_info=True)


def report_event(event_type: str, *, props: Optional[Mapping[str, Any]] = None) -> None:
    try:
        wf_telemetry.TELEMETRY.event(
            event_type, props=dict(props or {}), app=APP, **current_identity()
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("activity.report_event_failed", exc_info=True)


def report_feedback(
    *,
    message_id: str,
    rating: str,
    comment: Optional[str] = None,
    member_id: Optional[str] = None,
) -> None:
    """Mirror a thumbs rating to the shared feedback inbox.

    FoodChat keeps its own `feedback` table — that one drives personalisation
    and must stay local. This copy is so an expert sees chat feedback next to
    FoodScholar's and the platform widget's, rather than in a third table that
    joins to neither.
    """
    try:
        wf_telemetry.TELEMETRY.feedback(
            target_type="chat_message",
            target_id=message_id,
            rating_kind="thumbs",
            rating_value=rating,
            comment=comment,
            member_id=member_id or current_identity().get("member_id"),
            app=APP,
        )
    except Exception:  # pragma: no cover - defensive
        logger.debug("activity.report_feedback_failed", exc_info=True)


def usage_callback(feature: str, *, provider: str = "groq"):
    """A LangChain callback reporting each model call's token cost.

    Attached once at the client pool, exactly where the Langfuse handler is, so
    a new agent is traced and costed without touching its call site.
    """
    return wf_telemetry.usage_callback(
        feature, provider=provider, app=APP, identity=current_identity
    )
