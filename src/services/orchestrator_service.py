"""
OrchestratorService — unified conversational entry point.

Receives every user message, classifies intent via OrchestratorAgent,
then routes to the appropriate sub-service:

  daily_plan   → ChatService
  weekly_plan  → WeeklyPlanService
  refine_plan  → ChatService or WeeklyPlanService depending on active_context
  chat         → ChatService (chatbot path)

Returns a unified ChatTurn so the router only needs one response model.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any

from agents import OrchestratorAgent
from models.session import MealPlan, WeeklyMealPlan
from services.session_service import SessionService

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    role: str
    content: str
    intent: str
    needs_clarification: bool = False
    meal_plan: Optional[MealPlan] = None
    weekly_meal_plan: Optional[WeeklyMealPlan] = None
    at_message_limit: bool = False


class OrchestratorService:

    def __init__(self, session_service: SessionService, chat_service: Any, weekly_plan_service: Any):
        self.session_service = session_service
        self.chat_service = chat_service
        self.weekly_plan_service = weekly_plan_service
        self.orchestrator = OrchestratorAgent()

    def process(self, session_id: str, member_id: str, message: str) -> ChatTurn:
        """
        Main entry point. Validates ownership, checks message cap, classifies
        intent, and delegates to the right service.
        """
        session = self.session_service.get_session(session_id, member_id=member_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or access denied")

        if session.is_at_message_limit:
            return ChatTurn(
                role="assistant",
                content=(
                    f"This conversation has reached the {session.max_messages}-message limit. "
                    "Please start a new session to continue."
                ),
                intent="chat",
                at_message_limit=True,
            )

        # Build history snapshot for the orchestrator (role + first 300 chars)
        history = [
            {"role": m.role, "content": m.content}
            for m in session.conversation[-12:]
        ]

        # Classify intent — skip orchestrator if already in clarification flow
        if session.state == "clarifying":
            intent = "daily_plan"
        else:
            intent = self.orchestrator.classify(message, history)

        logger.info(f"[{session_id}] intent={intent}")

        if intent == "weekly_plan":
            return self._handle_weekly(session_id, message, intent)

        if intent == "refine_plan" and session.active_context:
            if session.active_context.plan_type == "weekly":
                return self._handle_weekly(session_id, message, intent)
            # Daily refinement falls through to ChatService (it re-runs RAG with prior context)

        # daily_plan, refine_plan (daily), chat all go through ChatService
        return self._handle_chat(session_id, message, intent)

    # ------------------------------------------------------------------ #

    def _handle_chat(self, session_id: str, message: str, intent: str) -> ChatTurn:
        response_text, needs_clarification, meal_plan = self.chat_service.process_message(
            session_id, message
        )
        # Tag the last assistant message with intent
        self._tag_last_message(session_id, intent, meal_plan.id if meal_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan,
        )

    def _handle_weekly(self, session_id: str, message: str, intent: str) -> ChatTurn:
        response_text, weekly_plan = self.weekly_plan_service.process_message(
            session_id, message
        )
        self._tag_last_message(session_id, intent, weekly_plan.id if weekly_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            weekly_meal_plan=weekly_plan,
        )

    def _tag_last_message(self, session_id: str, intent: str, plan_id: Optional[str]) -> None:
        """Back-fill intent and plan_id onto the last assistant message in-memory."""
        session = self.session_service._sessions.get(session_id)
        if not session or not session.conversation:
            return
        last = session.conversation[-1]
        if last.role == "assistant":
            last.intent = intent
            last.plan_id = plan_id
