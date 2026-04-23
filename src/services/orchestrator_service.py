"""
OrchestratorService — unified conversational entry point.

Receives every user message, classifies intent via OrchestratorAgent,
then routes to the appropriate sub-service:

  daily_plan       → ChatService (fresh plan, new canvas)
  weekly_plan      → WeeklyPlanService (fresh plan, new canvas)
  switch_plan_type → abandons current canvas type, starts fresh canvas of target type
  chat             → ChatService (chatbot path)

Each plan type maintains an independent versioned canvas so history is
always retrievable, even after a switch.

Returns a unified ChatTurn so the router only needs one response model.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any

from agents import OrchestratorAgent
from models.session import MealPlan, WeeklyMealPlan
from .session_service import SessionService

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
    # Version metadata surfaced to the caller
    plan_version: Optional[int] = None
    plan_parent_id: Optional[str] = None


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

        # Skip orchestrator if already mid-clarification
        if session.state == "clarifying":
            classification = {"intent": "daily_plan", "target_plan_type": None}
        else:
            classification = self.orchestrator.classify(message, history)

        intent = classification["intent"]
        target_plan_type = classification.get("target_plan_type")

        logger.info("[%s] intent=%s target=%s", session_id, intent, target_plan_type)

        # ---- routing ------------------------------------------------- #

        if intent == "switch_plan_type":
            return self._handle_switch(session_id, message, target_plan_type)

        if intent == "weekly_plan":
            return self._handle_weekly(session_id, message, intent, is_refinement=False)

        if intent == "daily_plan":
            return self._handle_chat(session_id, message, intent, is_refinement=False)

        # "chat" and any unexpected values
        return self._handle_chat(session_id, message, intent, is_refinement=False)

    # ------------------------------------------------------------------ #
    # Handlers                                                             #
    # ------------------------------------------------------------------ #

    def _handle_chat(
        self,
        session_id: str,
        message: str,
        intent: str,
        is_refinement: bool,
    ) -> ChatTurn:
        response_text, needs_clarification, meal_plan = self.chat_service.process_message(
            session_id, message, is_refinement=is_refinement
        )
        self._tag_last_message(session_id, intent, meal_plan.id if meal_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan,
            plan_version=meal_plan.version if meal_plan else None,
            plan_parent_id=meal_plan.parent_id if meal_plan else None,
        )

    def _handle_weekly(
        self,
        session_id: str,
        message: str,
        intent: str,
        is_refinement: bool,
    ) -> ChatTurn:
        response_text, weekly_plan = self.weekly_plan_service.process_message(
            session_id, message, is_refinement=is_refinement
        )
        self._tag_last_message(session_id, intent, weekly_plan.id if weekly_plan else None)

        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=intent,
            weekly_meal_plan=weekly_plan,
            plan_version=weekly_plan.version if weekly_plan else None,
            plan_parent_id=weekly_plan.parent_id if weekly_plan else None,
        )

    def _handle_switch(
        self,
        session_id: str,
        message: str,
        target_plan_type: Optional[str],
    ) -> ChatTurn:
        """
        User wants to abandon the current plan type and start a fresh one.
        The old canvas is preserved (history still retrievable); we simply
        start generating a new canvas for the target type.
        """
        # Normalise — default to weekly if unspecified
        target = target_plan_type if target_plan_type in ("daily", "weekly") else "weekly"

        logger.info("[%s] switch_plan_type → %s", session_id, target)

        if target == "weekly":
            # Acknowledge the switch in the conversation, then generate
            ack = (
                "Sure! I'm setting aside the daily plan and starting a fresh weekly plan for you."
            )
            self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
            return self._handle_weekly(session_id, message, "weekly_plan", is_refinement=False)
        else:
            ack = (
                "Sure! I'm setting aside the weekly plan and starting a fresh daily plan for you."
            )
            self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
            return self._handle_chat(session_id, message, "daily_plan", is_refinement=False)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _tag_last_message(self, session_id: str, intent: str, plan_id: Optional[str]) -> None:
        """Back-fill intent and plan_id onto the last assistant message in-memory."""
        session = self.session_service._sessions.get(session_id)
        if not session or not session.conversation:
            return
        last = session.conversation[-1]
        if last.role == "assistant":
            last.intent = intent
            last.plan_id = plan_id
