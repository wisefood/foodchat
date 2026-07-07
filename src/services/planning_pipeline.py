"""
Daily meal-plan generation pipeline.

Replaces the pre-M0 ``FoodChat`` class and its LangChain runnable chains
(``create_pre_clarification_chain`` / ``create_post_clarification_chain``),
which existed to serve a local Chroma/BM25 RAG stack that has been removed.
The pipeline is now three explicit, typed steps:

    generate(query, profile)    → candidates (RecipeWrangler) → LLM-graded plans

Consumers: ``services.chat_service`` (the only caller). The reconciled
profile is produced upstream by ``services.clarification`` — this pipeline
only fetches and grades.

No data files, vector stores, or embeddings are required — the service boots
with Groq + RecipeWrangler + WiseFood credentials alone.
"""

import logging
import os

from agents import DocumentGrader
from models.recipe import ScoredPlan
from services.candidates_client import CANDIDATES

logger = logging.getLogger(__name__)

# Candidates requested per meal slot from RecipeWrangler. 8 per slot bounds the
# grading space (8³ = 512 combos, sampled down by DocumentGrader) while keeping
# enough variety for refinements to find alternatives.
CANDIDATE_LIMIT = int(os.getenv("FOODCHAT_CANDIDATE_LIMIT", "8"))


class PlanningPipeline:
    """Reconcile → fetch candidates → grade combinations → ranked plans."""

    def __init__(self):
        self.grader = DocumentGrader()

    def generate(self, query: str, profile: dict) -> list[ScoredPlan]:
        """Produce ranked daily-plan combinations for the reformulated query.

        Hard constraints (allergies, diet, dislikes) are enforced server-side
        by RecipeWrangler; the LLM grader ranks the surviving combinations
        against the query and soft preferences.
        """
        candidates = CANDIDATES.fetch_candidates(
            allergens=profile.get("allergies", []),
            diet=profile.get("diet", []),
            include_ingredients=list(set(
                (profile.get("food_likes") or []) + (profile.get("include_ingredients") or [])
            )),
            exclude_ingredients=profile.get("food_dislikes") or [],
            nutrition_profile=profile.get("nutrition_profile") or None,
            limit_per_slot=CANDIDATE_LIMIT,
            randomize=False,
        )

        empty_slots = [slot for slot, recipes in candidates.items() if not recipes]
        if empty_slots:
            logger.warning("No candidates for slot(s) %s — cannot build a full plan", empty_slots)
            return []

        scored = self.grader.grade_daily_plans(query, candidates, profile)
        logger.info(
            "Pipeline produced %d scored plan(s); best score=%s",
            len(scored), scored[0].score if scored else "n/a",
        )
        return scored
