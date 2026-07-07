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
    intent: Literal["daily_plan", "weekly_plan", "refine_plan", "switch_plan_type", "chat"]
    reasoning: str
    # Populated only when intent == "switch_plan_type"
    target_plan_type: Optional[Literal["daily", "weekly"]] = None


class DietaryTagsSchema(BaseModel):
    """Diet tags detected in a user query (vegan, low-carb, …)."""
    dietary_tags: list[str]
