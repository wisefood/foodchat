"""
FoodScholar bridge — answers nutrition-science questions inside FoodChat.

Pre-M1, FoodChat refused anything beyond meal planning ("I cannot answer
generic nutrition questions"). This service delegates such questions to
FoodScholar's QA API and returns the cited answer as a normal chat turn with
an ``Attribution`` (source badge + citations + "Learn more in FoodScholar"
deep link) for the UI.

Upstream contract: FoodScholar ``POST {FOODSCHOLAR_API_URL}/api/v1/qa/ask``
(``QARequest``/``QAResponse`` in foodscholar ``src/models/qa.py``). Notable:
the response may ask for clarification (``needs_clarification`` +
``qa_thread_id``); we surface that question conversationally and resume the
thread with the user's next message. The pending thread is persisted in the
session's clarification state (``{"kind": "foodscholar", ...}``) — the same
restart-safe mechanism the plan flow uses (services/clarification.py), and
it is FoodScholar (not us) that decides when it has enough to answer.

FoodScholar is internally unauthenticated; identity is the opaque member_id,
consistent with the platform's gateway trust model (see README).
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import httpx

from models.attribution import Attribution, Citation
from .session_service import SessionService

logger = logging.getLogger(__name__)

FOODSCHOLAR_API_URL = os.getenv("FOODSCHOLAR_API_URL", "http://foodscholar:8001")
FOODSCHOLAR_TIMEOUT = float(os.getenv("FOODSCHOLAR_TIMEOUT", "90"))
# Sources requested per question — 5 balances answer quality vs. latency.
FOODSCHOLAR_TOP_K = int(os.getenv("FOODSCHOLAR_TOP_K", "5"))

UNAVAILABLE_MESSAGE = (
    "I tried to look that up with FoodScholar, our nutrition-science service, "
    "but it isn't reachable right now. Please try again in a moment — "
    "or ask me for a meal plan in the meantime!"
)


@dataclass(frozen=True)
class FoodScholarTurn:
    """Outcome of one delegated QA exchange (answer OR clarification question)."""

    text: str
    needs_clarification: bool = False
    attribution: Optional[Attribution] = None
    # Set while FoodScholar is waiting for a clarification answer
    qa_thread_id: Optional[str] = None
    clarification_id: Optional[str] = None


class FoodScholarClient:
    """Thin HTTP client for FoodScholar's QA endpoint (stateless)."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or FOODSCHOLAR_API_URL).rstrip("/")

    def ask(
        self,
        question: str,
        member_id: str,
        qa_thread_id: Optional[str] = None,
        clarification_answer: Optional[str] = None,
        clarification_id: Optional[str] = None,
    ) -> dict:
        """Call ``/api/v1/qa/ask``; raises httpx.HTTPError on failure."""
        payload: dict = {
            "question": question,
            "mode": "simple",
            "member_id": member_id,
            "top_k": FOODSCHOLAR_TOP_K,
        }
        if qa_thread_id:
            payload["qa_thread_id"] = qa_thread_id
            # Clarification answers are sent as free text — chat has no
            # structured option picker; FoodScholar accepts free_text always.
            payload["clarification_response"] = {
                "question_id": clarification_id,
                "free_text": clarification_answer,
            }

        with httpx.Client(timeout=FOODSCHOLAR_TIMEOUT) as client:
            response = client.post(f"{self.base_url}/api/v1/qa/ask", json=payload)
            response.raise_for_status()
            return response.json()


class FoodScholarService:
    """Conversation-level handling of 'nutrition_question' turns.

    Mirrors ChatService's shape: owns message persistence for its turns and
    the clarification hand-back; called only by OrchestratorService.
    """

    CLARIFICATION_KIND = "foodscholar"

    def __init__(self, session_service: SessionService, client: Optional[FoodScholarClient] = None):
        self.session_service = session_service
        self.client = client or FoodScholarClient()
        logger.info("FoodScholarService initialized (base=%s).", self.client.base_url)

    # ------------------------------------------------------------------ #
    # Entry points (called by OrchestratorService)                         #
    # ------------------------------------------------------------------ #

    def process_question(self, session_id: str, message: str,
                         question: Optional[str] = None) -> FoodScholarTurn:
        """Handle a fresh nutrition-science question.

        ``message`` is what the member typed (persisted to the transcript);
        ``question`` is what FoodScholar is asked — the orchestrator may
        enrich it with plan context the transcript shouldn't display.
        """
        session = self._get_session(session_id)
        self.session_service.add_message(session_id, "user", message)
        return self._ask_and_record(session, question or message)

    def continue_clarification(self, session_id: str, message: str) -> FoodScholarTurn:
        """Feed the user's answer back into the pending FoodScholar thread."""
        session = self._get_session(session_id)
        self.session_service.add_message(session_id, "user", message)

        pending = session.clarification or {}
        self.session_service.clear_clarification_state(session_id)
        return self._ask_and_record(
            session,
            pending.get("original_question", message),
            qa_thread_id=pending.get("qa_thread_id"),
            clarification_answer=message,
            clarification_id=pending.get("clarification_id"),
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _ask_and_record(
        self,
        session,
        question: str,
        qa_thread_id: Optional[str] = None,
        clarification_answer: Optional[str] = None,
        clarification_id: Optional[str] = None,
    ) -> FoodScholarTurn:
        session_id = session.session_id
        try:
            data = self.client.ask(
                question,
                member_id=session.member_id,
                qa_thread_id=qa_thread_id,
                clarification_answer=clarification_answer,
                clarification_id=clarification_id,
            )
        except httpx.HTTPError as e:
            # Degrade gracefully — a broken FoodScholar must never 500 the chat.
            logger.error("[%s] FoodScholar unreachable: %s", session_id, e)
            self.session_service.add_message(session_id, "assistant", UNAVAILABLE_MESSAGE)
            return FoodScholarTurn(text=UNAVAILABLE_MESSAGE)

        if data.get("needs_clarification") and data.get("clarification"):
            text = self._render_clarification(data["clarification"])
            self.session_service.set_clarification_state(session_id, {
                "kind": self.CLARIFICATION_KIND,
                "original_question": question,
                "qa_thread_id": data.get("qa_thread_id"),
                "clarification_id": data["clarification"].get("id"),
            })
            self.session_service.add_message(session_id, "assistant", text)
            logger.info("[%s] FoodScholar requested clarification (thread=%s).",
                        session_id, data.get("qa_thread_id"))
            return FoodScholarTurn(
                text=text,
                needs_clarification=True,
                qa_thread_id=data.get("qa_thread_id"),
                clarification_id=data["clarification"].get("id"),
            )

        primary = data.get("primary_answer") or {}
        text = primary.get("answer") or UNAVAILABLE_MESSAGE
        attribution = self._build_attribution(question, primary)
        self.session_service.add_message(session_id, "assistant", text)
        logger.info(
            "[%s] FoodScholar answered (confidence=%s, %d citations).",
            session_id, attribution.confidence, len(attribution.citations),
        )
        return FoodScholarTurn(text=text, attribution=attribution)

    @staticmethod
    def _render_clarification(clarification: dict) -> str:
        """Flatten FoodScholar's structured clarification into chat text."""
        text = clarification.get("question", "Could you clarify your question?")
        options = clarification.get("options") or []
        if options:
            labels = " · ".join(o.get("label", "") for o in options if o.get("label"))
            if labels:
                text = f"{text}\n\nFor example: {labels}"
        return text

    @staticmethod
    def _build_attribution(question: str, primary_answer: dict) -> Attribution:
        citations = [
            Citation(
                title=c.get("source_title", ""),
                source_type=c.get("source_type", "article"),
                url=c.get("source_url"),
                label=c.get("display_label"),
            )
            for c in (primary_answer.get("citations") or [])
        ]
        return Attribution(
            source="foodscholar",
            confidence=primary_answer.get("confidence"),
            citations=citations,
            # UI-relative deep link; the frontend prefills and auto-asks.
            learn_more_url=f"/foodscholar?q={quote(question)}",
        )

    def _get_session(self, session_id: str):
        session = self.session_service.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session
