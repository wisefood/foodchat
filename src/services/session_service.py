from typing import Dict, Optional
from models.session import Session, Message, MealPlan


class SessionService:
    """In-memory session management service."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self, member_id: str, user_profile: dict) -> Session:
        """Create a new session for a member."""
        session = Session.create(member_id=member_id, user_profile=user_profile)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        """Add a message to a session."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        message = Message(role=role, content=content)
        session.messages.append(message)
        return message

    def add_meal_plan(
        self, session_id: str, meal_plan_tuple: tuple, reasoning: str
    ) -> MealPlan:
        """Add a meal plan to a session."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        meal_plan = MealPlan.from_response(meal_plan_tuple, reasoning)
        session.meal_plans.append(meal_plan)
        return meal_plan

    def set_clarification_state(
        self, session_id: str, generator, pending_data: dict
    ) -> None:
        """Store clarification generator for multi-turn flow."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        session.state = "clarifying"
        session.clarification_generator = generator
        session.pending_rag_data = pending_data

    def clear_clarification_state(self, session_id: str) -> None:
        """Clear clarification state after completing the flow."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        session.state = "ready"
        session.clarification_generator = None
        session.pending_rag_data = None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[Session]:
        """List all active sessions."""
        return list(self._sessions.values())
