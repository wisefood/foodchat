"""
LLM observability (M5) — Langfuse tracing for every Groq call.

Entirely optional and env-gated: when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY
are set (and the ``langfuse`` package is installed), a LangChain callback
handler is attached to every pooled ChatGroq client, so all agent calls
(orchestrator, graders, extractors, response writer, …) appear as traces in
the platform's Langfuse — the same instance FoodScholar reports to.

Missing keys or a missing package degrade to no-op silently; tracing must
never affect chat behavior.
"""

import logging
import os

logger = logging.getLogger(__name__)

_handler = None
_initialized = False


def langchain_callbacks() -> list:
    """The callbacks to attach to LLM clients ([] when tracing is disabled)."""
    global _handler, _initialized
    if _initialized:
        return [_handler] if _handler else []

    _initialized = True
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse tracing disabled (no keys configured).")
        return []
    # The platform deployment exports LANGFUSE_BASE_URL (FoodScholar's
    # convention); the Langfuse SDK reads LANGFUSE_HOST — bridge them.
    if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
    try:
        from langfuse.langchain import CallbackHandler
        _handler = CallbackHandler()
        logger.info("Langfuse tracing enabled (host=%s).", os.getenv("LANGFUSE_HOST", "default"))
        return [_handler]
    except Exception as e:  # missing package, bad config — never break chat
        logger.warning("Langfuse tracing unavailable: %s", e)
        return []
