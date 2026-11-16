"""Reading a chat exchange back, for someone reviewing feedback about it.

A thumbs-down on a chat message used to arrive in the shared feedback inbox as
a rating, a message id, and nothing else. The reviewer saw that somebody was
unhappy and had no way to see what they were unhappy *about* — the id points at
a row in this service's database, and the console has no route to it.

This is the route. Deliberately narrow: one message, the turns immediately
around it, and nothing else. Enough to judge the complaint, not a general
window onto anyone's chat history. Beside it, one plan by id — what FoodChat
put on the table, never what the member typed — so a session read in the
console can say what a plan was, not only that one was made.

**Authorization lives at the gateway**, which requires an admin or expert token
and records the read as expert activity. This service verifies the caller is
the gateway the same way every other cross-service read here does, and does not
apply the member-ownership check that guards the member-facing routes: a
reviewer legitimately reads conversations they do not own, which is the whole
point, and applying an ownership check would make the endpoint useless rather
than safe. That asymmetry is why this lives in its own module with its own
name, instead of a flag on the member-facing conversation endpoint where it
would be one wrong default away from leaking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Turns either side of the rated message. Four is enough to see the question
#: that prompted an answer and the correction that followed it; more starts
#: being someone's conversation rather than the context for one complaint.
DEFAULT_WINDOW = 4
MAX_WINDOW = 20

#: Chat messages are free text a person wrote. Truncated so a reviewer sees
#: the exchange without the inbox becoming a transcript store.
MAX_CONTENT_CHARS = 4000


def _to_dict(row: Any, *, rated: bool) -> Dict[str, Any]:
    content = row.content or ""
    return {
        "id": row.id,
        "role": row.role,
        "content": content[:MAX_CONTENT_CHARS],
        "truncated": len(content) > MAX_CONTENT_CHARS,
        "intent": row.intent,
        "plan_id": row.plan_id,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "attribution": _load_json(getattr(row, "attribution", None)),
        # The message the feedback was actually about, so the console does not
        # have to compare ids to highlight it.
        "is_rated": rated,
    }


def _load_json(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


class ChatReviewService:
    """One rated message with the turns around it."""

    def message_context(
        self, message_id: str, *, window: int = DEFAULT_WINDOW
    ) -> Optional[Dict[str, Any]]:
        """The exchange a piece of feedback was about, or None if it is gone.

        None rather than an error for a missing message: chat sessions are
        pruned and a complaint outlives the conversation it was about, so an
        inbox item pointing at nothing is expected rather than exceptional.
        """
        from db import MessageRow, SessionLocal

        try:
            target_id = int(message_id)
        except (TypeError, ValueError):
            # Feedback carries whatever the client sent. A non-numeric id is
            # not a message here, and is not worth a database round trip.
            return None

        span = max(1, min(int(window or DEFAULT_WINDOW), MAX_WINDOW))
        db = SessionLocal()
        try:
            rated = db.query(MessageRow).filter(MessageRow.id == target_id).first()
            if rated is None:
                return None

            # Ordered by id rather than timestamp: two turns in one second are
            # common and their order is what makes the exchange readable.
            before = (
                db.query(MessageRow)
                .filter(
                    MessageRow.session_id == rated.session_id,
                    MessageRow.id < target_id,
                )
                .order_by(MessageRow.id.desc())
                .limit(span)
                .all()
            )
            after = (
                db.query(MessageRow)
                .filter(
                    MessageRow.session_id == rated.session_id,
                    MessageRow.id > target_id,
                )
                .order_by(MessageRow.id.asc())
                .limit(span)
                .all()
            )
            total = (
                db.query(MessageRow)
                .filter(MessageRow.session_id == rated.session_id)
                .count()
            )
        except Exception:
            logger.warning("chat_review.context_failed id=%s", message_id, exc_info=True)
            return None
        finally:
            db.close()

        messages: List[Dict[str, Any]] = [
            _to_dict(row, rated=False) for row in reversed(before)
        ]
        messages.append(_to_dict(rated, rated=True))
        messages.extend(_to_dict(row, rated=False) for row in after)

        return {
            "message_id": str(target_id),
            "session_id": rated.session_id,
            "messages": messages,
            "window": span,
            # So the console can say "5 of 42 turns" rather than implying the
            # exchange shown is the whole conversation.
            "session_messages": total,
        }


    def plan_summary(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """What one meal plan put on the table, and how it was judged.

        A reviewer reading a session saw "made a daily plan, 7 items" and a
        thumbs-down, and could not tell what the plan was. This is the plan as
        FoodChat produced it: the dishes per meal, the judges' scores with
        their reasoning, and the constraint rows the member was shown.

        Only what FoodChat wrote. Left out on purpose: the personalisation
        summary, which is the member's profile restated; the title a member
        typed when saving; and each dish's ingredients and method, which are
        the recipe's and make the plan a page long.

        None when the plan is gone, which is ordinary: plans die with their
        conversation unless saved.
        """
        from db import MealPlanRow, SessionLocal

        if not plan_id:
            return None
        db = SessionLocal()
        try:
            row = db.query(MealPlanRow).filter(MealPlanRow.id == str(plan_id)).first()
        except Exception:
            logger.warning("chat_review.plan_failed id=%s", plan_id, exc_info=True)
            return None
        finally:
            db.close()
        if row is None:
            return None

        payload = _load_json(row.payload) or {}
        weekly = row.plan_type == "weekly"
        return {
            "plan_id": row.id,
            "session_id": row.session_id,
            "plan_type": row.plan_type,
            "version": row.version,
            "parent_id": row.parent_id,
            "saved": bool(row.saved),
            "saved_at": row.saved_at.isoformat() if row.saved_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "meals": _weekly_meals(payload) if weekly else _daily_meals(payload),
            "reasoning": _clip(payload.get("reasoning")),
            "scores": [] if weekly else _scores(payload),
            "metrics": payload.get("metrics") if weekly else None,
            "constraints_applied": list(payload.get("constraints_applied") or []),
        }


#: Reasoning is the model's prose about the plan. Clipped like chat content.
MAX_REASONING_CHARS = 2000


def _clip(text: Any) -> Optional[str]:
    if not text:
        return None
    text = str(text)
    return text if len(text) <= MAX_REASONING_CHARS else text[:MAX_REASONING_CHARS] + "…"


def _plate(course: Dict[str, Any]) -> Dict[str, Any]:
    """A dish by name, never by its ingredients or method."""
    nutrition = course.get("nutrition") or {}
    return {
        "recipe_id": course.get("recipe_id") or course.get("id"),
        # A day plan's course says `title`; a week plan's entry carries the
        # recipe as the planner retrieved it, which says `recipe_title`.
        "title": course.get("title") or course.get("recipe_title") or "",
        "role": course.get("role") or "main",
        "kcal": nutrition.get("kcal") if isinstance(nutrition, dict) else None,
        "match_reasons": [
            reason.get("label")
            for reason in course.get("match_reasons") or []
            if isinstance(reason, dict) and reason.get("label")
        ],
    }


def _daily_meals(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Multi-plate and multi-day plans carry `days`; a legacy plan has one
    # course per slot and nothing else.
    if payload.get("days"):
        return [
            {
                "day": day.get("day"),
                "meal_type": meal.get("meal_type") or "",
                "plates": [_plate(p) for p in meal.get("plates") or []],
            }
            for day in payload["days"]
            for meal in day.get("meals") or []
        ]
    return [
        {"day": None, "meal_type": slot, "plates": [_plate(payload[slot])]}
        for slot in ("breakfast", "lunch", "dinner")
        if isinstance(payload.get(slot), dict)
    ]


def _weekly_meals(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    meals: Dict[tuple, Dict[str, Any]] = {}
    for entry in payload.get("entries") or []:
        key = (entry.get("day"), entry.get("meal_idx"), entry.get("meal_type"))
        meal = meals.setdefault(
            key, {"day": entry.get("day"), "meal_type": entry.get("meal_type") or "", "plates": []}
        )
        plate = _plate(entry.get("recipe") or {})
        plate["role"] = entry.get("role") or plate["role"]
        meal["plates"].append(plate)
    return list(meals.values())


def _scores(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The judges' verdicts, each with the reasoning it gave."""
    judged = (
        ("overall", "llm_score", "llm_reasoning"),
        ("fruit_and_veg", "fvs_count", "fvs_reasoning"),
        ("variety", "diversity_llm_score", "diversity_llm_reasoning"),
        ("guidelines", "guideline_adherence_score", "guideline_adherence_reasoning"),
    )
    return [
        {
            "name": name,
            "score": payload.get(score),
            "reasoning": _clip(payload.get(reasoning)),
        }
        for name, score, reasoning in judged
        # Zero with no reasoning is a judge that did not run, not a verdict.
        if payload.get(score) or payload.get(reasoning)
    ]


CHAT_REVIEW_SERVICE = ChatReviewService()
