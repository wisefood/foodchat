"""
FoodChat HTTP API.

All conversational traffic goes through ONE endpoint —
``POST /foodchat/sessions/{session_id}/chat`` — which classifies intent
server-side and routes internally (see services/orchestrator_service.py).
The remaining endpoints are session lifecycle, plan/canvas reads,
paginated conversation history, and message feedback.

Authorization model: FoodChat sits behind the wisefood-api gateway, which
authenticates the Keycloak user and passes the household member's
``member_id`` as data. Every session-scoped endpoint therefore REQUIRES the
member_id and verifies it matches the session owner — this is the only
access control at this layer, so never make it optional.

Removed in M0 (see CHANGES.md): the legacy pre-orchestrator endpoints
``POST/GET /sessions/{id}/messages`` and ``POST/GET /sessions/{id}/weekly``.
Their gateway proxies were removed in the same change.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

import services
from db import SessionLocal, db_upsert_feedback, db_get_message_by_id
from models.session import MealCourse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/foodchat",
    tags=["foodchat"],
    responses={404: {"description": "Not found"}},
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    member_id: str


class SessionResponse(BaseModel):
    session_id: str
    member_id: str
    state: str
    message_count: int
    created_at: datetime


class MealCourseResponse(BaseModel):
    recipe_id: str
    title: str
    ingredients: str
    directions: str

    @classmethod
    def from_meal_course(cls, course: MealCourse) -> "MealCourseResponse":
        return cls(
            recipe_id=course.recipe_id,
            title=course.title,
            ingredients=course.ingredients,
            directions=course.directions,
        )


class MealPlanResponse(BaseModel):
    id: str
    created_at: datetime
    version: int
    parent_id: Optional[str] = None
    breakfast: MealCourseResponse
    lunch: MealCourseResponse
    dinner: MealCourseResponse
    reasoning: str
    llm_score: int | None = None
    llm_reasoning: str | None = None
    fvs_count: int | None = None
    fvs_reasoning: str | None = None
    diversity_llm_score: int | None = None
    diversity_llm_reasoning: str | None = None
    guideline_adherence_score: int | None = None
    guideline_adherence_reasoning: str | None = None

    @classmethod
    def from_meal_plan(cls, mp) -> "MealPlanResponse":
        return cls(
            id=mp.id,
            created_at=mp.created_at,
            version=mp.version,
            parent_id=mp.parent_id,
            breakfast=MealCourseResponse.from_meal_course(mp.breakfast),
            lunch=MealCourseResponse.from_meal_course(mp.lunch),
            dinner=MealCourseResponse.from_meal_course(mp.dinner),
            reasoning=mp.reasoning,
            llm_score=mp.llm_score,
            llm_reasoning=mp.llm_reasoning,
            fvs_count=mp.fvs_count,
            fvs_reasoning=mp.fvs_reasoning,
            diversity_llm_score=mp.diversity_llm_score,
            diversity_llm_reasoning=mp.diversity_llm_reasoning,
            guideline_adherence_score=mp.guideline_adherence_score,
            guideline_adherence_reasoning=mp.guideline_adherence_reasoning,
        )


class WeeklyMealPlanEntryResponse(BaseModel):
    day: int
    meal_idx: int
    meal_type: str
    recipe: dict
    reward: float


class WeeklyMealPlanResponse(BaseModel):
    id: str
    created_at: datetime
    version: int
    parent_id: Optional[str] = None
    entries: List[WeeklyMealPlanEntryResponse]

    @classmethod
    def from_weekly_meal_plan(cls, wmp) -> "WeeklyMealPlanResponse":
        return cls(
            id=wmp.id,
            created_at=wmp.created_at,
            version=wmp.version,
            parent_id=wmp.parent_id,
            entries=[WeeklyMealPlanEntryResponse(**entry) for entry in wmp.entries],
        )


class ChatRequest(BaseModel):
    content: str
    member_id: str


class ChatTurnResponse(BaseModel):
    role: str
    content: str
    intent: str
    needs_clarification: bool = False
    meal_plan: Optional[MealPlanResponse] = None
    weekly_meal_plan: Optional[WeeklyMealPlanResponse] = None
    at_message_limit: bool = False
    # Version metadata so the UI knows which canvas version was just produced
    plan_version: Optional[int] = None
    plan_parent_id: Optional[str] = None


class ConversationPage(BaseModel):
    messages: List[dict]
    has_more: bool
    next_before_id: Optional[int] = None


class FeedbackRequest(BaseModel):
    member_id: str
    rating: str          # "up" | "down"
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    message_id: int
    rating: str
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Service guard
# ---------------------------------------------------------------------------


def _require_orchestrator_service():
    """Defensive guard — the orchestrator is initialized unconditionally at
    startup since M0 (no data-file dependency), so this should never fire."""
    if services.orchestrator_service is None:
        logger.error("orchestrator_service is None — startup initialization failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator service unavailable. Check server logs.",
        )
    return services.orchestrator_service


def _require_session(session_id: str, member_id: str):
    """Load a session and enforce owner access (404 on mismatch, never 403 —
    a mismatched member must not learn that the session exists)."""
    session = services.session_service.get_session(session_id, member_id=member_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied"
        )
    return session


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@router.post(
    "/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session; fetches the member's profile from WiseFood."""
    try:
        user_profile = services.profile_service.get_member_profile(request.member_id)
        session = services.session_service.create_session(request.member_id, user_profile)
        logger.info("Session %s created for member %s.", session.session_id, request.member_id)
        return SessionResponse(
            session_id=session.session_id,
            member_id=session.member_id,
            state=session.state,
            message_count=len(session.conversation),
            created_at=session.created_at,
        )
    except Exception as e:
        logger.error("Failed to create session for member %s: %s", request.member_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Get session state and metadata. Only the owning member may read it."""
    session = _require_session(session_id, member_id)
    return SessionResponse(
        session_id=session.session_id,
        member_id=session.member_id,
        state=session.state,
        message_count=len(session.conversation),
        created_at=session.created_at,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Delete a session. Only the owning member may delete it."""
    if not services.session_service.delete_session(session_id, member_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied"
        )


@router.get("/members/{member_id}/sessions", response_model=List[SessionResponse])
async def get_member_sessions(member_id: str):
    """Get all sessions for a specific member."""
    sessions = services.session_service.get_member_sessions(member_id)
    return [
        SessionResponse(
            session_id=s.session_id,
            member_id=s.member_id,
            state=s.state,
            message_count=len(s.conversation),
            created_at=s.created_at,
        )
        for s in sessions
    ]


# ---------------------------------------------------------------------------
# Unified chat
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/chat", response_model=ChatTurnResponse)
async def unified_chat(session_id: str, request: ChatRequest):
    """
    Unified conversational endpoint.

    Accepts any message and routes to the correct sub-service based on intent:
      - daily_plan       → new daily meal plan (fresh canvas)
      - weekly_plan      → new weekly meal plan (fresh canvas)
      - refine_plan      → update the active canvas plan in-place (version++)
      - switch_plan_type → freeze current canvas type, start fresh canvas of the other type
      - chat             → general conversation

    The caller must supply member_id to prove session ownership.

    Response includes plan_version and plan_parent_id so the UI can track
    which canvas version was just produced.
    """
    orch_svc = _require_orchestrator_service()

    logger.info("[%s] /chat from member %s: %.120s", session_id, request.member_id, request.content)
    try:
        turn = orch_svc.process(session_id, request.member_id, request.content)
        logger.info(
            "[%s] /chat response — intent=%s v=%s needs_clarification=%s at_limit=%s",
            session_id, turn.intent, turn.plan_version, turn.needs_clarification, turn.at_message_limit,
        )
    except ValueError as e:
        logger.warning("[%s] /chat 404: %s", session_id, e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        logger.warning("[%s] /chat 429 (message limit): %s", session_id, e)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except Exception as e:
        logger.error("[%s] /chat 500: %s", session_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return ChatTurnResponse(
        role=turn.role,
        content=turn.content,
        intent=turn.intent,
        needs_clarification=turn.needs_clarification,
        meal_plan=MealPlanResponse.from_meal_plan(turn.meal_plan) if turn.meal_plan else None,
        weekly_meal_plan=(
            WeeklyMealPlanResponse.from_weekly_meal_plan(turn.weekly_meal_plan)
            if turn.weekly_meal_plan else None
        ),
        at_message_limit=turn.at_message_limit,
        plan_version=turn.plan_version,
        plan_parent_id=turn.plan_parent_id,
    )


@router.get("/sessions/{session_id}/conversation", response_model=ConversationPage)
async def get_conversation(
    session_id: str,
    member_id: str = Query(..., description="WiseFood member ID — must match session owner"),
    before_id: Optional[int] = Query(None, description="Cursor: return messages with DB id < this value"),
    limit: int = Query(20, ge=1, le=100, description="Number of messages to return"),
):
    """
    Cursor-based paginated conversation history for infinite scroll.

    First call: omit before_id → returns the most recent `limit` messages.
    Next page:  pass next_before_id from the previous response.
    Scroll is exhausted when has_more=False.

    Returns messages oldest-first so the UI can prepend to the top of the chat.
    """
    _require_session(session_id, member_id)

    page = services.session_service.get_messages_page(session_id, before_id=before_id, limit=limit)

    has_more = len(page) == limit
    next_cursor = page[0]["id"] if has_more else None

    return ConversationPage(
        messages=[
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "intent": m["intent"],
                "plan_id": m["plan_id"],
                "timestamp": m["timestamp"].isoformat(),
            }
            for m in page
        ],
        has_more=has_more,
        next_before_id=next_cursor,
    )


# ---------------------------------------------------------------------------
# Plan / canvas reads
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}/meal-plans", response_model=List[MealPlanResponse])
async def get_meal_plans(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Get all daily meal plan versions in this session (canvas history)."""
    session = _require_session(session_id, member_id)
    return [MealPlanResponse.from_meal_plan(mp) for mp in session.meal_plans]


@router.get("/sessions/{session_id}/meal-plans/current", response_model=Optional[MealPlanResponse])
async def get_current_meal_plan(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Get the current (latest) daily meal plan on the canvas."""
    session = _require_session(session_id, member_id)
    plan = session.get_current_daily_plan()
    return MealPlanResponse.from_meal_plan(plan) if plan else None


@router.get("/sessions/{session_id}/meal-plans/history", response_model=List[MealPlanResponse])
async def get_daily_plan_history(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """
    Full version history of daily meal plans, oldest first. Each entry has
    version and parent_id set so the caller can reconstruct the refinement chain.
    """
    _require_session(session_id, member_id)
    plans = services.session_service.get_daily_plan_history(session_id)
    return [MealPlanResponse.from_meal_plan(mp) for mp in plans]


@router.get("/sessions/{session_id}/weekly-meal-plans", response_model=List[WeeklyMealPlanResponse])
async def get_weekly_meal_plans(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Get all weekly meal plan versions in this session (canvas history)."""
    session = _require_session(session_id, member_id)
    return [WeeklyMealPlanResponse.from_weekly_meal_plan(wmp) for wmp in session.weekly_meal_plans]


@router.get("/sessions/{session_id}/weekly-meal-plans/current", response_model=Optional[WeeklyMealPlanResponse])
async def get_current_weekly_meal_plan(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Get the current (latest) weekly meal plan on the canvas."""
    session = _require_session(session_id, member_id)
    plan = session.get_current_weekly_plan()
    return WeeklyMealPlanResponse.from_weekly_meal_plan(plan) if plan else None


@router.get("/sessions/{session_id}/weekly-meal-plans/history", response_model=List[WeeklyMealPlanResponse])
async def get_weekly_plan_history(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Full version history of weekly meal plans, oldest first."""
    _require_session(session_id, member_id)
    plans = services.session_service.get_weekly_plan_history(session_id)
    return [WeeklyMealPlanResponse.from_weekly_meal_plan(wmp) for wmp in plans]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
)
async def submit_feedback(session_id: str, message_id: int, request: FeedbackRequest):
    """
    Submit thumbs up/down feedback on an assistant message.

    Calling again with the same member_id updates the existing feedback (upsert).
    """
    if request.rating not in ("up", "down"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="rating must be 'up' or 'down'",
        )

    _require_session(session_id, request.member_id)

    db = SessionLocal()
    try:
        msg_row = db_get_message_by_id(db, message_id)
        if not msg_row or msg_row.session_id != session_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
        if msg_row.role != "assistant":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Feedback can only be submitted on assistant messages",
            )
        fb = db_upsert_feedback(
            db,
            message_id=message_id,
            session_id=session_id,
            member_id=request.member_id,
            rating=request.rating,
            comment=request.comment,
        )
        return FeedbackResponse(message_id=fb.message_id, rating=fb.rating, comment=fb.comment)
    finally:
        db.close()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "foodchat"}
