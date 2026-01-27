from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Any
import uuid


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MealPlan:
    id: str
    created_at: datetime
    breakfast: str
    lunch: str
    dinner: str
    reasoning: str
    raw_data: tuple

    @classmethod
    def from_response(cls, meal_plan_tuple: tuple, reasoning: str) -> "MealPlan":
        """Create a MealPlan from RAG chain response."""
        return cls(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            breakfast=meal_plan_tuple[0] if len(meal_plan_tuple) > 0 else "",
            lunch=meal_plan_tuple[1] if len(meal_plan_tuple) > 1 else "",
            dinner=meal_plan_tuple[2] if len(meal_plan_tuple) > 2 else "",
            reasoning=reasoning,
            raw_data=meal_plan_tuple,
        )


@dataclass
class Session:
    session_id: str
    member_id: str
    user_profile: dict
    messages: list[Message] = field(default_factory=list)
    meal_plans: list[MealPlan] = field(default_factory=list)
    state: Literal["ready", "clarifying"] = "ready"
    clarification_generator: Optional[Any] = field(default=None, repr=False)
    pending_rag_data: Optional[dict] = field(default=None, repr=False)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(cls, member_id: str, user_profile: dict) -> "Session":
        """Create a new session for a member."""
        return cls(
            session_id=str(uuid.uuid4()),
            member_id=member_id,
            user_profile=user_profile,
        )
