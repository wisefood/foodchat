"""
Deterministic constraint logic for the weekly planner (M6).

Constraints now act BEFORE a recipe is picked: ``WeeklyPlanner`` calls
``apply_hard_constraints`` (meat limit — excludes meat candidates once the
weekly limit is spent) and ``constraint_score`` (soft calorie budget —
penalizes candidates whose kcal exceeds the fair share of the remaining
weekly budget) on every slot's candidate pool. Previously this module
computed the same penalties strictly AFTER selection, so they were logged
and stored but never changed a single pick.

The former per-step Groq grading call (``get_llm_feedback``) was removed
outright: it fired once per committed slot (21 calls per weekly plan) to
grade a recipe that was already locked into the plan — pure cost and
latency with zero effect on the output. Preference steering is the
heuristic scorer's job (``planner.build_preference_scorer``); the weekly
planning loop is now fully LLM-free.

``RewardCalculator.calculate_step_reward`` is kept (same signature) so the
per-entry ``reward`` field in plan payloads and API responses stays
populated — it is now simply the negative constraint penalty at that step.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .day_summary import is_meat_meal

if TYPE_CHECKING:
    from .state_tracking import WeeklyNutritionalTracker

logger = logging.getLogger(__name__)

# Score points lost per kcal above the slot's fair share of the remaining
# weekly budget. 0.01 => a 300 kcal overshoot costs 3.0, comparable to the
# preference scorer's favorite boost (+5) without dominating it.
CALORIE_OVERAGE_WEIGHT = 0.01


def candidate_kcal(candidate: Dict[str, Any]) -> Optional[float]:
    """Per-serving kcal from an enriched candidate dict, None when unknown."""
    nutrition = candidate.get("nutrition") or {}
    for key in ("kcal", "calories"):
        value = nutrition.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def is_meat_candidate(candidate: Dict[str, Any], count_fish: bool = True) -> bool:
    return is_meat_meal(
        str(candidate.get("recipe_title") or candidate.get("title") or ""),
        str(candidate.get("recipe_ingredients") or candidate.get("ingredients") or ""),
        tags=candidate.get("tags"),
        count_fish=count_fish,
    )


def apply_hard_constraints(
    candidates: List[Dict[str, Any]], tracker: "WeeklyNutritionalTracker"
) -> List[Dict[str, Any]]:
    """Drop candidates that would break a hard constraint.

    Today that is the weekly meat limit: once ``meat_limit_left`` hits 0,
    meat candidates leave the pool. If that would empty the pool the limit
    is relaxed for the slot (with a warning) rather than failing the plan —
    a plan with one extra meat meal beats no plan at all.
    """
    status = tracker.get_status()
    if status["remaining"]["meat_limit_left"] <= 0:
        meatless = [
            c for c in candidates
            if not is_meat_candidate(c, count_fish=tracker.counts_fish_as_meat)
        ]
        if meatless:
            return meatless
        logger.warning(
            "Weekly meat limit (%s) reached but every candidate in this slot "
            "contains meat — relaxing the limit for this slot.",
            status["targets"].get("meat_limit"),
        )
    return candidates


def constraint_score(
    candidate: Dict[str, Any],
    tracker: "WeeklyNutritionalTracker",
    slots_remaining: int,
) -> float:
    """Soft calorie-budget score (<= 0), added to the preference score.

    A candidate at or under the fair per-slot share of the remaining weekly
    calorie budget scores 0; anything above is penalized proportionally.
    Once the budget is spent the fair share is 0, steering every remaining
    pick toward the lowest-calorie candidates. Candidates without nutrition
    data are neutral.
    """
    kcal = candidate_kcal(candidate)
    if kcal is None or slots_remaining <= 0:
        return 0.0
    remaining_budget = tracker.get_status()["remaining"]["calories"]
    fair_share = max(remaining_budget, 0.0) / slots_remaining
    overage = kcal - fair_share
    return -overage * CALORIE_OVERAGE_WEIGHT if overage > 0 else 0.0


class RewardCalculator:
    """
    Deterministic per-step reward for the weekly planner — the stored
    ``reward`` on each plan entry. No LLM involved (see module docstring).
    """

    def calculate_constraint_penalty(self, tracker: "WeeklyNutritionalTracker", preferences: List[str]) -> float:
        """
        Calculate penalties for violating nutritional or dietary constraints.

        Args:
            tracker: The WeeklyNutritionalTracker containing current cumulative state.
            preferences: List of user preference strings (for additional context).

        Returns:
            A non-negative penalty value (to be subtracted from reward).
        """
        status = tracker.get_status()
        remaining = status["remaining"]

        penalty = 0.0

        # 1. Meat limit penalty
        if remaining["meat_limit_left"] < 0:
            # Significant penalty for exceeding the meat limit
            penalty += abs(remaining["meat_limit_left"]) * 15.0

        # 2. Calorie constraint penalty
        if remaining["calories"] < 0:
            # Penalty increases with the amount of excess calories
            penalty += abs(remaining["calories"]) * 0.1

        return penalty

    def calculate_step_reward(self, action: Dict[str, Any], tracker: "WeeklyNutritionalTracker", preferences: List[str], user_query: Optional[str] = None) -> float:
        """Negative constraint penalty after committing ``action`` (<= 0)."""
        return -self.calculate_constraint_penalty(tracker, preferences)
