from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Any
import uuid

MAX_MESSAGES_PER_SESSION = int(__import__("os").getenv("SESSION_MAX_MESSAGES", "200"))


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # Conversation turn metadata
    intent: Optional[Literal["daily_plan", "weekly_plan", "refine_plan", "chat"]] = None
    plan_id: Optional[str] = None  # references MealPlan.id or WeeklyMealPlan.id


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
    llm_score: int = 0
    llm_reasoning: str = ""
    fvs_count: int = 0
    fvs_reasoning: str = ""
    diversity_llm_score: int = 0
    diversity_llm_reasoning: str = ""
    guideline_adherence_score: int = 0
    guideline_adherence_reasoning: str = ""

    @classmethod
    def from_response(cls, meal_plan_tuple: tuple, reasoning: str, metrics: Optional[dict] = None) -> "MealPlan":
        metrics = metrics or {}
        return cls(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            breakfast=MealCourse.from_list(meal_plan_tuple[0] if len(meal_plan_tuple) > 0 else []),
            lunch=MealCourse.from_list(meal_plan_tuple[1] if len(meal_plan_tuple) > 1 else []),
            dinner=MealCourse.from_list(meal_plan_tuple[2] if len(meal_plan_tuple) > 2 else []),
            reasoning=reasoning,
            llm_score=int(metrics.get("llm_score", 0)),
            llm_reasoning=str(metrics.get("llm_reasoning", "")),
            fvs_count=int(metrics.get("fvs_count", 0)),
            fvs_reasoning=str(metrics.get("fvs_reasoning", "")),
            diversity_llm_score=int(metrics.get("diversity_llm_score", 0)),
            diversity_llm_reasoning=str(metrics.get("diversity_llm_reasoning", "")),
            guideline_adherence_score=int(metrics.get("guideline_adherence_score", 0)),
            guideline_adherence_reasoning=str(metrics.get("guideline_adherence_reasoning", "")),
        )


@dataclass
class WeeklyMealPlan:
    id: str
    created_at: datetime
    entries: list[dict]


@dataclass
class ActiveContext:
    """Tracks the most recently generated plan so refinements have context."""
    plan_type: Literal["daily", "weekly"]
    plan_id: str


@dataclass
class Session:
    session_id: str
    member_id: str          # WiseFood member ID — used for user_id scoping
    user_profile: dict

    # Unified conversation thread (replaces separate messages + weekly_messages)
    conversation: list[Message] = field(default_factory=list)

    # Plan stores (kept separate for typed access)
    meal_plans: list[MealPlan] = field(default_factory=list)
    weekly_meal_plans: list[WeeklyMealPlan] = field(default_factory=list)

    # Tracks what plan was last shown — drives refine_plan routing
    active_context: Optional[ActiveContext] = None

    # Hard cap on conversation length
    max_messages: int = MAX_MESSAGES_PER_SESSION

    # Daily-plan clarification state (unchanged)
    state: Literal["ready", "clarifying"] = "ready"
    clarification_generator: Optional[Any] = field(default=None, repr=False)
    pending_rag_data: Optional[dict] = field(default=None, repr=False)

    created_at: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------ #
    # Legacy accessors — keep existing code working without changes        #
    # ------------------------------------------------------------------ #

    @property
    def messages(self) -> list[Message]:
        """Backward-compat: daily-plan messages are all messages in the thread."""
        return self.conversation

    @property
    def weekly_messages(self) -> list[Message]:
        """Backward-compat: weekly messages share the unified thread."""
        return self.conversation

    @property
    def is_at_message_limit(self) -> bool:
        return len(self.conversation) >= self.max_messages

    @classmethod
    def create(cls, member_id: str, user_profile: dict) -> "Session":
        return cls(
            session_id=str(uuid.uuid4()),
            member_id=member_id,
            user_profile=user_profile,
        )
