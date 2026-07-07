"""
Feedback signals (M3) — thumbs up/down finally drive recommendations.

The raw material existed since M0-era storage: feedback rows join to
assistant messages (``messages.plan_id``) which join to stored plan payloads,
yielding per-recipe ratings ACROSS the member's sessions. This service
aggregates them into the two signals the pipeline consumes:

    signals.downvoted_recipe_ids → hard exclusions in candidate fetches
    signals.history_text         → the grader prompts' feedback_history
                                   (which was a hardcoded "" until now)

Aggregation is per recipe: a recipe is excluded when its downvotes outnumber
its upvotes. Comments ride into the history text so the LLM grader can honor
qualitative feedback ("too heavy", "loved the spice").
"""

import json
import logging
from dataclasses import dataclass, field

from db import SessionLocal, db_get_member_feedback_with_plans

logger = logging.getLogger(__name__)

# Cap the history text fed into grader prompts (most recent first).
MAX_HISTORY_LINES = 12


@dataclass(frozen=True)
class FeedbackSignals:
    downvoted_recipe_ids: list = field(default_factory=list)
    history_text: str = ""


class FeedbackService:
    """Aggregates a member's message feedback into recommendation signals."""

    def get_signals(self, member_id: str) -> FeedbackSignals:
        """Best-effort: feedback is a soft signal, failures return empty."""
        try:
            db = SessionLocal()
            try:
                rows = db_get_member_feedback_with_plans(db, member_id)
            finally:
                db.close()
        except Exception as e:
            logger.warning("Feedback signals unavailable for %s: %s", member_id, e)
            return FeedbackSignals()

        votes: dict[str, dict] = {}     # recipe_id -> {"title", "up", "down"}
        lines: list[str] = []
        for row in rows:                # newest first (db helper orders desc)
            recipes = self._recipes_from_payload(row["plan_type"], row["payload"])
            verb = "Liked" if row["rating"] == "up" else "Disliked"
            titles = ", ".join(r["title"] for r in recipes if r["title"])[:160]
            if titles and len(lines) < MAX_HISTORY_LINES:
                comment = f" — “{row['comment']}”" if row.get("comment") else ""
                lines.append(f"{verb}: {titles}{comment}")
            for recipe in recipes:
                entry = votes.setdefault(
                    recipe["recipe_id"], {"title": recipe["title"], "up": 0, "down": 0}
                )
                entry["up" if row["rating"] == "up" else "down"] += 1

        downvoted = [
            rid for rid, v in votes.items()
            if rid and v["down"] > v["up"]
        ]
        if downvoted:
            logger.info("Member %s: excluding %d downvoted recipe(s)", member_id, len(downvoted))
        return FeedbackSignals(
            downvoted_recipe_ids=downvoted,
            history_text="\n".join(lines),
        )

    @staticmethod
    def _recipes_from_payload(plan_type: str, payload_json: str) -> list[dict]:
        """Extract (recipe_id, title) pairs from a stored plan payload."""
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            return []

        recipes = []
        if plan_type == "daily":
            for slot in ("breakfast", "lunch", "dinner"):
                course = payload.get(slot) or {}
                recipes.append({
                    "recipe_id": str(course.get("recipe_id", "")),
                    "title": course.get("title", ""),
                })
        else:  # weekly
            for entry in payload.get("entries") or []:
                recipe = entry.get("recipe") or {}
                recipes.append({
                    "recipe_id": str(recipe.get("recipe_id", "")),
                    "title": recipe.get("recipe_title", "") or recipe.get("title", ""),
                })
        return [r for r in recipes if r["recipe_id"]]
