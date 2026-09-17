"""
Plan scorer — score a meal plan the member wrote themselves.

Reached through the ``score_plan`` intent of the unified chat endpoint. The
pasted plan is parsed, grounded against RecipeWrangler and built into the
same objects the planners produce, so the existing evaluation routines can
grade it. It is never written to a canvas.

    parsing.py    step 1 — free text → PastedPlan (line scanner + PlanTextParser)
    grounding.py  step 2 — each dish → matched | approximate | unresolved
    building.py   step 3 — daily ScoredPlan-shaped courses / weekly entry dicts
    scoring.py    step 4 — hard-constraint rows, measured metrics, LLM judges, caps
    service.py    the turn: the ``score_plan`` clarification kind, the summary,
                  and the payload stored with the reply (``messages.plan_score``)

Reached from chat (the ``score_plan`` intent) and from
``POST /sessions/{id}/score-plan`` (the text box). See IDEAS.md "Plan scorer".
"""

from .service import CLARIFICATION_KIND, PlanScorerService, ScoreTurn

__all__ = ["CLARIFICATION_KIND", "PlanScorerService", "ScoreTurn"]
