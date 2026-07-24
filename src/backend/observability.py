"""
LLM observability — Langfuse tracing + trace enrichment for every Groq call.

One module owns ALL Langfuse coupling (the WiseFood convention); nothing else
in the codebase imports the SDK directly. It provides:

  - ``langfuse_enabled()``      — the single enablement guard
  - ``get_callback_handler()``  — the shared LangChain callback handler
  - ``get_langfuse_client()``   — the process-wide client (prompt fetch + flush)
  - ``build_trace_config()``    — the ONLY way to build per-invoke trace config
  - ``trace_context(...)``      — request-scoped session/user, read by the above
  - ``flush_langfuse()``        — drain buffered traces on shutdown

Design principles (see langfuse-integration-guide.md):
  1. Strictly optional. No keys / package missing → every helper no-ops and
     chat behaves exactly as before. Observability must never take down chat.
  2. Never raise from a tracing path. Every SDK call is wrapped and logged at
     WARNING; a tracing failure is never a request failure.
  3. No PII in trace metadata — only opaque IDs (session/member) and feature
     tags. The LLM message payload is the only place user context appears.

Enablement rule: tracing activates only when BOTH ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are present AND ``import langfuse`` succeeds. One key
alone = disabled. When Langfuse runs in-cluster the deployment exports
``LANGFUSE_BASE_URL`` (the platform convention); the SDK reads ``LANGFUSE_HOST``,
so we bridge the two before constructing any client.
"""

import contextlib
import contextvars
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Request-scoped trace identifiers (session/member). Set once per turn by
# ``trace_context`` at the orchestrator entry points, then read by
# ``build_trace_config`` so individual agent call sites don't have to thread
# session_id/user_id through every method signature. Contextvars are
# per-execution-context, so concurrent requests never see each other's ids.
_trace_ctx: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "langfuse_trace_ctx", default=None
)


def langfuse_enabled() -> bool:
    """True only when both keys are set AND the langfuse package imports."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except Exception as exc:  # missing/broken package — treat as disabled
        logger.warning("Langfuse keys set but package not importable: %s", exc)
        return False
    return True


def _bridge_host() -> None:
    """Map the platform's LANGFUSE_BASE_URL onto the SDK's LANGFUSE_HOST.

    The FoodScholar/CityCloud deployments export ``LANGFUSE_BASE_URL`` (the
    in-cluster service URL); the Langfuse SDK reads ``LANGFUSE_HOST``. Bridge
    them so either name works. An explicit ``LANGFUSE_HOST`` always wins.
    """
    if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]


@lru_cache(maxsize=1)
def get_callback_handler() -> Optional[Any]:
    """The shared LangChain ``CallbackHandler`` (``None`` when disabled).

    The handler is stateless and reads credentials via the singleton client,
    so ONE instance can safely be attached to every pooled ChatGroq client and
    shared across threads. Cached so per-request instantiation never leaks.
    """
    if not langfuse_enabled():
        return None
    _bridge_host()
    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        logger.info("Langfuse tracing enabled (host=%s).", os.getenv("LANGFUSE_HOST", "cloud"))
        return handler
    except Exception as exc:  # bad config / package mismatch — never break chat
        logger.warning("Failed to init Langfuse CallbackHandler: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_langfuse_client() -> Optional[Any]:
    """Process-wide Langfuse client for prompt fetching + flushing.

    The SDK client is a singleton — per-request instantiation leaks memory —
    so it is cached here. ``None`` when tracing is disabled.
    """
    if not langfuse_enabled():
        return None
    _bridge_host()
    try:
        from langfuse import Langfuse
        return Langfuse()  # reads keys + host from the environment
    except Exception as exc:
        logger.warning("Failed to init Langfuse client: %s", exc)
        return None


# Kept for backward compatibility with callers that want a plain callbacks
# list; new code should attach ``get_callback_handler()`` directly.
def langchain_callbacks() -> list:
    """The callbacks to attach to LLM clients ([] when tracing is disabled)."""
    handler = get_callback_handler()
    return [handler] if handler is not None else []


@contextlib.contextmanager
def trace_context(session_id: Optional[str] = None, user_id: Optional[str] = None):
    """Bind request-scoped trace ids for the duration of a turn.

    Wrap the orchestrator's per-turn entry points in this so every downstream
    agent invocation groups under the same Langfuse Session (session_id) and is
    attributed to the same User (member_id) without each agent needing the ids.
    Only opaque identifiers belong here — never PII.
    """
    ctx: Dict[str, str] = {}
    if session_id is not None:
        ctx["session_id"] = str(session_id)
    if user_id is not None:
        ctx["user_id"] = str(user_id)
    token = _trace_ctx.set(ctx or None)
    try:
        yield
    finally:
        _trace_ctx.reset(token)


def build_trace_config(
    *,
    run_name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build the LangChain ``config`` for one LLM invocation.

    This is the ONLY sanctioned way to construct trace config — do not
    hand-roll metadata dicts. ``session_id``/``user_id`` default to whatever
    ``trace_context`` bound for the current turn; pass them explicitly only to
    override. Works whether or not Langfuse is enabled: ``run_name`` is a
    standard LangChain config key and the ``langfuse_*`` metadata keys are
    simply ignored when no handler is attached.

    🔒 PII policy: only opaque IDs and feature tags. Never put allergies,
    dietary profiles, member details, or health conditions in ``extra_metadata``
    — the message payload is the only place user context legitimately appears.
    """
    ctx = _trace_ctx.get() or {}
    if session_id is None:
        session_id = ctx.get("session_id")
    if user_id is None:
        user_id = ctx.get("user_id")

    config: Dict[str, Any] = {"run_name": run_name}
    metadata: Dict[str, Any] = {}
    # Langfuse v3 metadata keys — these exact names drive the Sessions/Users
    # views and tag filters. Values must be strings; None is omitted.
    if session_id is not None:
        metadata["langfuse_session_id"] = str(session_id)
    if user_id is not None:
        metadata["langfuse_user_id"] = str(user_id)
    if tags:
        metadata["langfuse_tags"] = list(tags)
    if extra_metadata:
        metadata.update(extra_metadata)
    if metadata:
        config["metadata"] = metadata
    return config


def flush_langfuse() -> None:
    """Drain buffered traces (call on shutdown; traces are batched)."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("Failed to flush Langfuse traces: %s", exc)
