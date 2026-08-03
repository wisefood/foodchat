"""
Structured-output schemas for LLM agents.

Each schema is passed to the Groq client as a JSON-schema response format
(backend/groq.py) and parsed by the corresponding agent. Keep each schema
in sync with its prompt in prompts.py — the prompt describes the expected
JSON to the model; the schema is what gets validated.

Consumers: agents.py, services/clarification.py.
"""

from typing import Literal, Optional

from pydantic import BaseModel, conint


class ScoringSchema(BaseModel):
    """Generic 1–5 judge output (plan grading, diversity, guideline adherence)."""
    reasoning: str
    score: conint(ge=1, le=5)


class PlanGradeSchema(BaseModel):
    """One graded plan inside a batch."""
    plan_index: int
    reasoning: str
    score: conint(ge=1, le=5)


class BatchScoringSchema(BaseModel):
    """All candidate day-plans graded in ONE call.

    One combination per LLM call made grading the latency floor of every plan
    request — ten sequential Groq round-trips before a member saw anything.
    Grading the batch together also lets the judge rank comparatively, which
    is the actual task: pick the best day, not absolute-score ten days.
    """
    grades: list[PlanGradeSchema]


class QueryReconcilerSchema(BaseModel):
    """Conflicts / gaps between a plan request and the user profile."""
    missing_info: list[str]          # e.g. cooking time, difficulty, goal
    has_dietary_conflict: bool
    conflict_explanation: Optional[str] = None
    needs_clarification: bool


class QueryCheckerSchema(BaseModel):
    """Is the query specific enough to generate from? YES/NO."""
    response: Literal["YES", "NO"]


class UserProfileCheckerSchema(BaseModel):
    """Does the profile carry enough signal? If NO, what to ask about."""
    response: Literal["YES", "NO"]
    suggestions: list[str]


class QueryReformulatorSchema(BaseModel):
    """Final retrieval-ready query folded from collected clarifications."""
    reformulated_query: str


class OrchestratorSchema(BaseModel):
    """Per-turn intent classification (the single router of the pipeline)."""
    intent: Literal[
        "daily_plan", "weekly_plan", "refine_plan", "edit_plan_slot",
        "switch_plan_type", "nutrition_question", "plan_question",
        "preference_update", "chat",
    ]
    reasoning: str
    # Populated only when intent == "switch_plan_type"
    target_plan_type: Optional[Literal["daily", "weekly"]] = None


class DietaryTagsSchema(BaseModel):
    """Diet tags detected in a user query (vegan, low-carb, …)."""
    dietary_tags: list[str]


class CandidateMemorySchema(BaseModel):
    """One durable preference candidate detected in a user turn (M3).

    Candidates are SUGGESTED to the user, never written silently — the
    consent flow (memory nudges) owns durable profile writes.
    """
    kind: Literal["like", "dislike", "cuisine", "constraint", "allergy_hint",
                  "standing_seed", "dietary_goal"]
    value: str                       # canonical item, e.g. "blueberries"
    statement: str                   # user-facing nudge text
    evidence: str                    # what in the message supports this
    confidence: Literal["low", "medium", "high"]


class PreferenceExtractionSchema(BaseModel):
    memories: list[CandidateMemorySchema]


class EditCommandSchema(BaseModel):
    """A targeted single-slot plan edit parsed from the user message (M4b)."""
    meal_type: Optional[Literal["breakfast", "lunch", "dinner"]] = None
    day: Optional[conint(ge=1, le=7)] = None
    directive: str
    needs_slot_clarification: bool = False
    question: Optional[str] = None


class SeedDishSchema(BaseModel):
    """One dish the user explicitly asked to include in the plan."""
    name: str
    meal_type: Optional[Literal["breakfast", "lunch", "dinner"]] = None
    day: Optional[conint(ge=1, le=7)] = None   # 1 = Monday … 7 = Sunday


class SeedExtractionSchema(BaseModel):
    """Named dishes to anchor into the plan (empty when none requested)."""
    seeds: list[SeedDishSchema]


class MealPlateSchema(BaseModel):
    """The plates one meal is served as.

    Roles, not course types: FoodChat reasons in `main`/`side`/`dessert`/`drink`
    because that is what a plate's nutrition weight is keyed by and what
    `MealCourse.role` stores. The translation to RecipeWrangler's course-type
    taxonomy happens at the client boundary, so the model never has to learn
    two vocabularies for the same idea.
    """
    slot: Literal["breakfast", "brunch", "lunch", "dinner",
                  "snack", "dessert", "side", "drink"]
    roles: list[Literal["main", "side", "dessert", "drink"]] = ["main"]


class PlanSpecSchema(BaseModel):
    """The shape of plan the user asked for, if they said anything about it.

    `mentioned` is the field that matters. Most messages say nothing about
    shape, and "three meals, unstated" and "three meals, explicitly asked for"
    produce the same lists — only the flag tells them apart. Without it the
    extractor cannot abstain, and every vague message silently re-shapes the
    plan the user is served.
    """
    mentioned: bool = False
    num_days: conint(ge=1, le=7) = 1
    meals: list[Literal["breakfast", "brunch", "lunch", "dinner",
                        "snack", "dessert", "side", "drink"]] = []
    plates: list[MealPlateSchema] = []
