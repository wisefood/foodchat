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
    intent: Optional[Literal["daily_plan", "weekly_plan", "refine_plan", "switch_plan_type", "chat"]] = None
    plan_id: Optional[str] = None  # references MealPlan.id or WeeklyMealPlan.id


@dataclass
class MealCourse:
    recipe_id: str
    title: str
    ingredients: str
    directions: str

    @classmethod
    def from_list(cls, data) -> "MealCourse":
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
    # Version lineage — v1 has parent_id=None; each refinement increments version
    version: int = 1
    parent_id: Optional[str] = None
    llm_score: int = 0
    llm_reasoning: str = ""
    fvs_count: int = 0
    fvs_reasoning: str = ""
    diversity_llm_score: int = 0
    diversity_llm_reasoning: str = ""
    guideline_adherence_score: int = 0
    guideline_adherence_reasoning: str = ""

    @classmethod
    def from_response(
        cls,
        meal_plan_tuple: tuple,
        reasoning: str,
        metrics: Optional[dict] = None,
        version: int = 1,
        parent_id: Optional[str] = None,
    ) -> "MealPlan":
        metrics = metrics or {}
        return cls(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            breakfast=MealCourse.from_list(meal_plan_tuple[0] if len(meal_plan_tuple) > 0 else []),
            lunch=MealCourse.from_list(meal_plan_tuple[1] if len(meal_plan_tuple) > 1 else []),
            dinner=MealCourse.from_list(meal_plan_tuple[2] if len(meal_plan_tuple) > 2 else []),
            reasoning=reasoning,
            version=version,
            parent_id=parent_id,
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
    # Version lineage
    version: int = 1
    parent_id: Optional[str] = None


@dataclass
class PlanCanvas:
    """
    Tracks the live canvas for one plan type.

    current_id — the latest version currently shown to the user
    root_id    — the first plan ever generated in this canvas lineage
    plan_type  — "daily" | "weekly"
    """
    plan_type: Literal["daily", "weekly"]
    current_id: str
    root_id: str


@dataclass
class Session:
    session_id: str
    member_id: str
    user_profile: dict

    # Unified conversation thread
    conversation: list[Message] = field(default_factory=list)

    # Plan stores (separate for typed access)
    meal_plans: list[MealPlan] = field(default_factory=list)
    weekly_meal_plans: list[WeeklyMealPlan] = field(default_factory=list)

    # Two independent canvases — one per plan type
    daily_canvas: Optional[PlanCanvas] = None
    weekly_canvas: Optional[PlanCanvas] = None

    max_messages: int = MAX_MESSAGES_PER_SESSION

    # Daily-plan clarification state (generator not serialisable)
    state: Literal["ready", "clarifying"] = "ready"
    clarification_generator: Optional[Any] = field(default=None, repr=False)
    pending_rag_data: Optional[dict] = field(default=None, repr=False)
    # Intent that triggered clarification — restored after clarification completes
    pending_intent: Optional[Literal["daily_plan", "weekly_plan", "refine_plan"]] = None

    created_at: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------ #
    # Canvas helpers                                                       #
    # ------------------------------------------------------------------ #

    @property
    def active_canvas(self) -> Optional[PlanCanvas]:
        """The most recently updated canvas — drives refine_plan routing."""
        daily_ts = self._canvas_ts(self.daily_canvas)
        weekly_ts = self._canvas_ts(self.weekly_canvas)
        if daily_ts is None and weekly_ts is None:
            return None
        if weekly_ts is None:
            return self.daily_canvas
        if daily_ts is None:
            return self.weekly_canvas
        return self.daily_canvas if daily_ts >= weekly_ts else self.weekly_canvas

    def _canvas_ts(self, canvas: Optional[PlanCanvas]) -> Optional[datetime]:
        if canvas is None:
            return None
        plan = self._find_plan(canvas.plan_type, canvas.current_id)
        return plan.created_at if plan else None

    def _find_plan(self, plan_type: str, plan_id: str):
        if plan_type == "daily":
            return next((p for p in self.meal_plans if p.id == plan_id), None)
        return next((p for p in self.weekly_meal_plans if p.id == plan_id), None)

    def get_current_daily_plan(self) -> Optional[MealPlan]:
        if self.daily_canvas is None:
            return None
        return self._find_plan("daily", self.daily_canvas.current_id)

    def get_current_weekly_plan(self) -> Optional[WeeklyMealPlan]:
        if self.weekly_canvas is None:
            return None
        return self._find_plan("weekly", self.weekly_canvas.current_id)

    # ------------------------------------------------------------------ #
    # Backward-compat shims                                                #
    # ------------------------------------------------------------------ #

    @property
    def active_context(self) -> Optional[Any]:
        """Shim used by OrchestratorService — returns duck-typed object."""
        canvas = self.active_canvas
        if canvas is None:
            return None

        class _Compat:
            def __init__(self, plan_type, plan_id):
                self.plan_type = plan_type
                self.plan_id = plan_id

        return _Compat(canvas.plan_type, canvas.current_id)

    @property
    def messages(self) -> list[Message]:
        return self.conversation

    @property
    def weekly_messages(self) -> list[Message]:
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
