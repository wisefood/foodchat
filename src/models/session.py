"""
Session domain models — conversation, meal plans, and plan canvases.

A *canvas* is the versioned lineage of plans of one type (daily or weekly)
within a session: ``current_id`` points at the version shown to the user,
``root_id`` at the first version of the lineage. Refinements create new
versions (``version`` + ``parent_id``); switching plan types freezes the old
canvas without deleting it, so history stays retrievable.

Clarification state is a plain JSON-able dict (see
``services.clarification.ClarificationState``) persisted on the session row —
never a live object — so sessions survive restarts and replicas.
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from models.recipe import CandidateRecipe

MAX_MESSAGES_PER_SESSION = int(os.getenv("SESSION_MAX_MESSAGES", "200"))

Intent = Literal[
    "daily_plan", "weekly_plan", "refine_plan", "switch_plan_type",
    "nutrition_question", "favorites_offer", "chat",
]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    intent: Optional[Intent] = None
    plan_id: Optional[str] = None  # references MealPlan.id or WeeklyMealPlan.id


@dataclass
class MealCourse:
    recipe_id: str
    title: str
    ingredients: str
    directions: str
    # M4 enrichment (RecipeWrangler batch details; None/[] when unavailable)
    nutrition: Optional[dict] = None       # {kcal, protein_g, carbs_g, fat_g, nutri_score_label}
    image_url: Optional[str] = None
    # Why this recipe was chosen — transparency chips in the UI
    match_reasons: list = field(default_factory=list)   # [{kind, label}]

    @classmethod
    def from_candidate(cls, candidate: CandidateRecipe) -> "MealCourse":
        return cls(
            recipe_id=candidate.recipe_id,
            title=candidate.title,
            ingredients=candidate.ingredients,
            directions=candidate.directions,
        )

    def to_candidate(self) -> CandidateRecipe:
        """Back-convert for patch edits (unchanged slots of a new version)."""
        return CandidateRecipe(
            recipe_id=self.recipe_id, title=self.title,
            ingredients=self.ingredients, directions=self.directions,
        )


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
    # M4 transparency: the constraint ledger and personalization counts the
    # UI renders above the plan ([{constraint, type, status, source}]).
    constraints_applied: list = field(default_factory=list)
    personalization_summary: Optional[dict] = None

    @classmethod
    def from_courses(
        cls,
        courses: list[CandidateRecipe],
        reasoning: str,
        metrics: Optional[dict] = None,
        version: int = 1,
        parent_id: Optional[str] = None,
    ) -> "MealPlan":
        """Build a plan from [breakfast, lunch, dinner] candidates + quality metrics."""
        if len(courses) != 3:
            raise ValueError(f"A daily plan needs exactly 3 courses, got {len(courses)}")
        metrics = metrics or {}
        return cls(
            id=str(uuid.uuid4()),
            created_at=datetime.now(),
            breakfast=MealCourse.from_candidate(courses[0]),
            lunch=MealCourse.from_candidate(courses[1]),
            dinner=MealCourse.from_candidate(courses[2]),
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
    """Live canvas pointer for one plan type (see module docstring)."""

    plan_type: Literal["daily", "weekly"]
    current_id: str
    root_id: str


@dataclass
class Session:
    session_id: str
    member_id: str
    user_profile: dict

    # Unified conversation thread (user + assistant turns, all intents)
    conversation: list[Message] = field(default_factory=list)

    # Plan stores (separate for typed access)
    meal_plans: list[MealPlan] = field(default_factory=list)
    weekly_meal_plans: list[WeeklyMealPlan] = field(default_factory=list)

    # Two independent canvases — one per plan type
    daily_canvas: Optional[PlanCanvas] = None
    weekly_canvas: Optional[PlanCanvas] = None

    max_messages: int = MAX_MESSAGES_PER_SESSION

    # Clarification flow: "clarifying" while a question is outstanding.
    # ``clarification`` is the serialized ClarificationState dict (or None).
    state: Literal["ready", "clarifying"] = "ready"
    clarification: Optional[dict] = None

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
