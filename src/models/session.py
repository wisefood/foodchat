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
class MealCourse:
    recipe_id: str
    title: str
    ingredients: str
    directions: str

    @classmethod
    def from_list(cls, data) -> "MealCourse":
        """Create a MealCourse from a [recipe_id, title, ingredients, directions] list."""
        if isinstance(data, (list, tuple)) and len(data) >= 4:
            return cls(
                recipe_id=str(data[0]),
                title=str(data[1]),
                ingredients=str(data[2]),
                directions=str(data[3]),
            )
        return cls(recipe_id="", title="", ingredients="", directions="")


@dataclass
class MealPlan:
    id: str
    created_at: datetime
    breakfast: MealCourse
    lunch: MealCourse
    dinner: MealCourse
    reasoning: str

    @classmethod
    def from_response(cls, meal_plan_tuple: tuple, reasoning: str) -> "MealPlan":
        """Create a MealPlan from RAG chain response.

        Each element of meal_plan_tuple is [recipe_id, title, ingredients, directions].
        """
        return cls(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            breakfast=MealCourse.from_list(meal_plan_tuple[0] if len(meal_plan_tuple) > 0 else []),
            lunch=MealCourse.from_list(meal_plan_tuple[1] if len(meal_plan_tuple) > 1 else []),
            dinner=MealCourse.from_list(meal_plan_tuple[2] if len(meal_plan_tuple) > 2 else []),
            reasoning=reasoning,
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
