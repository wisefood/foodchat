"""
OrchestratorService — unified conversational entry point.

The ONLY intent classification in the pipeline happens here (one
``OrchestratorAgent.classify`` call per turn), then the turn is routed:

  daily_plan       → ChatService.process_plan_request (fresh canvas)
  refine_plan      → ChatService / WeeklyPlanService with is_refinement=True,
                     targeting whichever canvas was most recently updated
  weekly_plan      → WeeklyPlanService (fresh canvas)
  switch_plan_type → acknowledge, then fresh canvas of the target type
  chat             → ChatService.process_smalltalk

While a session is mid-clarification (session.state == "clarifying"), the
classifier is skipped entirely and the message is fed to
``ChatService.continue_clarification`` — the original intent is restored from
the persisted clarification state (restart-safe; see services/clarification.py).

Returns a unified ChatTurn so the router needs one response model.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

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
    # Version metadata surfaced to the caller (canvas tracking in the UI)
    plan_version: Optional[int] = None
    plan_parent_id: Optional[str] = None


class OrchestratorService:

    def __init__(self, session_service: SessionService, chat_service: Any, weekly_plan_service: Any):
        self.session_service = session_service
        self.chat_service = chat_service
        self.weekly_plan_service = weekly_plan_service
        self.orchestrator = OrchestratorAgent()

    def process(self, session_id: str, member_id: str, message: str) -> ChatTurn:
        """Validate ownership, check the message cap, classify, and route."""
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

        # Mid-clarification turns bypass classification — the user is answering
        # our question, not expressing a new intent.
        if session.state == "clarifying":
            return self._handle_clarification_turn(session_id, message)

        history = [
            {"role": m.role, "content": m.content}
            for m in session.conversation[-12:]
        ]
        classification = self.orchestrator.classify(message, history)
        intent = classification["intent"]
        target_plan_type = classification.get("target_plan_type")
        logger.info("[%s] intent=%s target=%s", session_id, intent, target_plan_type)

        if intent == "switch_plan_type":
            return self._handle_switch(session_id, message, target_plan_type)

        if intent == "weekly_plan":
            return self._handle_weekly(session_id, message, intent, is_refinement=False)

        if intent == "daily_plan":
            return self._handle_plan(session_id, message, intent, is_refinement=False)

        if intent == "refine_plan":
            canvas = session.active_canvas
            if canvas is None:
                # Nothing to refine yet — treat as a fresh daily plan.
                logger.info("[%s] refine_plan with no active canvas — fresh daily plan", session_id)
                return self._handle_plan(session_id, message, "daily_plan", is_refinement=False)
            if canvas.plan_type == "weekly":
                return self._handle_weekly(session_id, message, intent, is_refinement=True)
            return self._handle_plan(session_id, message, intent, is_refinement=True)

        # "chat" and any unexpected values
        return self._handle_smalltalk(session_id, message)

    # ------------------------------------------------------------------ #
    # Handlers                                                             #
    # ------------------------------------------------------------------ #

    def _handle_clarification_turn(self, session_id: str, message: str) -> ChatTurn:
        response_text, needs_clarification, meal_plan, origin_intent = (
            self.chat_service.continue_clarification(session_id, message)
        )
        self._tag_last_message(session_id, origin_intent, meal_plan.id if meal_plan else None)
        return ChatTurn(
            role="assistant",
            content=response_text,
            intent=origin_intent,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan,
            plan_version=meal_plan.version if meal_plan else None,
            plan_parent_id=meal_plan.parent_id if meal_plan else None,
        )

    def _handle_smalltalk(self, session_id: str, message: str) -> ChatTurn:
        response_text, _, _ = self.chat_service.process_smalltalk(session_id, message)
        self._tag_last_message(session_id, "chat", None)
        return ChatTurn(role="assistant", content=response_text, intent="chat")

    def _handle_plan(self, session_id: str, message: str, intent: str, is_refinement: bool) -> ChatTurn:
        response_text, needs_clarification, meal_plan = self.chat_service.process_plan_request(
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

    def _handle_weekly(self, session_id: str, message: str, intent: str, is_refinement: bool) -> ChatTurn:
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

    def _handle_switch(self, session_id: str, message: str, target_plan_type: Optional[str]) -> ChatTurn:
        """Freeze the current canvas type and start a fresh one of the target type.

        Both canvases stay retrievable — switching never destroys history.
        """
        target = target_plan_type if target_plan_type in ("daily", "weekly") else "weekly"
        logger.info("[%s] switch_plan_type → %s", session_id, target)

        if target == "weekly":
            ack = "Sure! I'm setting aside the daily plan and starting a fresh weekly plan for you."
            self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
            return self._handle_weekly(session_id, message, "weekly_plan", is_refinement=False)

        ack = "Sure! I'm setting aside the weekly plan and starting a fresh daily plan for you."
        self.session_service.add_message(session_id, "assistant", ack, intent="switch_plan_type")
        return self._handle_plan(session_id, message, "daily_plan", is_refinement=False)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _tag_last_message(self, session_id: str, intent: str, plan_id: Optional[str]) -> None:
        """Back-fill intent and plan_id onto the last assistant message.

        The UI uses these tags to associate chat bubbles with canvas versions.
        """
        session = self.session_service.get_session(session_id)
        if not session or not session.conversation:
            return
        last = session.conversation[-1]
        if last.role == "assistant":
            last.intent = intent
            last.plan_id = plan_id
