import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

import services

logger = logging.getLogger(__name__)
from db import SessionLocal, db_upsert_feedback, db_get_message_by_id
from models.session import MealCourse

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


class MessageRequest(BaseModel):
    content: str


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
            version=getattr(mp, "version", 1),
            parent_id=getattr(mp, "parent_id", None),
            breakfast=MealCourseResponse.from_meal_course(mp.breakfast),
            lunch=MealCourseResponse.from_meal_course(mp.lunch),
            dinner=MealCourseResponse.from_meal_course(mp.dinner),
            reasoning=mp.reasoning,
            llm_score=getattr(mp, "llm_score", None),
            llm_reasoning=getattr(mp, "llm_reasoning", None),
            fvs_count=getattr(mp, "fvs_count", None),
            fvs_reasoning=getattr(mp, "fvs_reasoning", None),
            diversity_llm_score=getattr(mp, "diversity_llm_score", None),
            diversity_llm_reasoning=getattr(mp, "diversity_llm_reasoning", None),
            guideline_adherence_score=getattr(mp, "guideline_adherence_score", None),
            guideline_adherence_reasoning=getattr(mp, "guideline_adherence_reasoning", None),
        )


class MessageResponse(BaseModel):
    role: str
    content: str
    needs_clarification: bool = False
    meal_plan: Optional[MealPlanResponse] = None


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
            version=getattr(wmp, "version", 1),
            parent_id=getattr(wmp, "parent_id", None),
            entries=[WeeklyMealPlanEntryResponse(**entry) for entry in wmp.entries],
        )


class WeeklyMessageResponse(BaseModel):
    role: str
    content: str
    weekly_meal_plan: Optional[WeeklyMealPlanResponse] = None


class MessageHistoryItem(BaseModel):
    role: str
    content: str
    timestamp: datetime


# --- Unified chat models ---


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
# Service guards
# ---------------------------------------------------------------------------


def _require_orchestrator_service():
    if services.orchestrator_service is None:
        logger.error(
            "Request reached /chat but orchestrator_service is None. "
            "Root cause: FoodChat failed to initialize at startup (missing CSV data or embeddings). "
            "Check startup logs for 'CSV file not found' or embedding errors."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator service unavailable — recipe data not loaded at startup. Check server logs.",
        )
    return services.orchestrator_service


def _require_chat_service():
    if services.chat_service is None:
        logger.error(
            "Request reached a chat endpoint but chat_service is None. "
            "FoodChat system was not initialized — CSV data missing or embeddings failed."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service unavailable — CSV data not loaded. Check server logs.",
        )
    return services.chat_service


def _require_weekly_plan_service():
    if services.weekly_plan_service is None:
        logger.error("weekly_plan_service is None — this should not happen after startup.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weekly plan service unavailable. Check server logs.",
        )
    return services.weekly_plan_service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session for a household member."""
    try:
        user_profile = services.profile_service.get_member_profile(request.member_id)
        session = services.session_service.create_session(request.member_id, user_profile)
        logger.info("Session %s created for member %s.", session.session_id, request.member_id)
        return SessionResponse(
            session_id=session.session_id,
            member_id=session.member_id,
            state=session.state,
            message_count=len(session.messages),
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
    session = services.session_service.get_session(session_id, member_id=member_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied"
        )
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


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: str, request: MessageRequest):
    """Send a message and get a response (legacy daily-plan endpoint)."""
    chat_svc = _require_chat_service()

    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    try:
        response_text, needs_clarification, meal_plan = chat_svc.process_message(
            session_id, request.content
        )
        meal_plan_resp = MealPlanResponse.from_meal_plan(meal_plan) if meal_plan else None
        return MessageResponse(
            role="assistant",
            content=response_text,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan_resp,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sessions/{session_id}/messages", response_model=List[MessageHistoryItem])
async def get_messages(session_id: str, limit: Optional[int] = None):
    """Get message history for a session."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = session.messages
    if limit:
        messages = messages[-limit:]

    return [
        MessageHistoryItem(role=msg.role, content=msg.content, timestamp=msg.timestamp)
        for msg in messages
    ]


@router.get("/sessions/{session_id}/meal-plans", response_model=List[MealPlanResponse])
async def get_meal_plans(session_id: str):
    """Get all daily meal plan versions in this session (canvas history)."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return [MealPlanResponse.from_meal_plan(mp) for mp in session.meal_plans]


@router.get("/sessions/{session_id}/meal-plans/current", response_model=Optional[MealPlanResponse])
async def get_current_meal_plan(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Get the current (latest) daily meal plan on the canvas."""
    session = services.session_service.get_session(session_id, member_id=member_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied")
    plan = session.get_current_daily_plan()
    if plan is None:
        return None
    return MealPlanResponse.from_meal_plan(plan)


@router.get("/sessions/{session_id}/meal-plans/history", response_model=List[MealPlanResponse])
async def get_daily_plan_history(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """
    Get the full version history of daily meal plans for this session,
    ordered oldest to newest. Each entry has version and parent_id set,
    so the caller can reconstruct the refinement chain.
    """
    session = services.session_service.get_session(session_id, member_id=member_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied")
    plans = services.session_service.get_daily_plan_history(session_id)
    return [MealPlanResponse.from_meal_plan(mp) for mp in plans]


@router.post("/sessions/{session_id}/weekly", response_model=WeeklyMessageResponse)
async def send_weekly_message(session_id: str, request: MessageRequest):
    """Send a message to generate a 7-day weekly meal plan (legacy endpoint)."""
    weekly_svc = _require_weekly_plan_service()

    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    try:
        response_text, weekly_plan = weekly_svc.process_message(
            session_id, request.content
        )
        return WeeklyMessageResponse(
            role="assistant",
            content=response_text,
            weekly_meal_plan=WeeklyMealPlanResponse.from_weekly_meal_plan(weekly_plan),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sessions/{session_id}/weekly", response_model=List[MessageHistoryItem])
async def get_weekly_messages(session_id: str, limit: Optional[int] = None):
    """Get weekly message history for a session."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = session.weekly_messages
    if limit:
        messages = messages[-limit:]

    return [
        MessageHistoryItem(role=msg.role, content=msg.content, timestamp=msg.timestamp)
        for msg in messages
    ]


@router.get("/sessions/{session_id}/weekly-meal-plans", response_model=List[WeeklyMealPlanResponse])
async def get_weekly_meal_plans(session_id: str):
    """Get all weekly meal plan versions in this session (canvas history)."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return [WeeklyMealPlanResponse.from_weekly_meal_plan(wmp) for wmp in session.weekly_meal_plans]


@router.get("/sessions/{session_id}/weekly-meal-plans/current", response_model=Optional[WeeklyMealPlanResponse])
async def get_current_weekly_meal_plan(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """Get the current (latest) weekly meal plan on the canvas."""
    session = services.session_service.get_session(session_id, member_id=member_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied")
    plan = session.get_current_weekly_plan()
    if plan is None:
        return None
    return WeeklyMealPlanResponse.from_weekly_meal_plan(plan)


@router.get("/sessions/{session_id}/weekly-meal-plans/history", response_model=List[WeeklyMealPlanResponse])
async def get_weekly_plan_history(
    session_id: str,
    member_id: str = Query(..., description="Must match session owner"),
):
    """
    Get the full version history of weekly meal plans for this session,
    ordered oldest to newest. Each entry has version and parent_id set.
    """
    session = services.session_service.get_session(session_id, member_id=member_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied")
    plans = services.session_service.get_weekly_plan_history(session_id)
    return [WeeklyMealPlanResponse.from_weekly_meal_plan(wmp) for wmp in plans]


@router.get("/members/{member_id}/sessions", response_model=List[SessionResponse])
async def get_member_sessions(member_id: str):
    """Get all sessions for a specific member."""
    sessions = services.session_service.get_member_sessions(member_id)
    return [
        SessionResponse(
            session_id=s.session_id,
            member_id=s.member_id,
            state=s.state,
            message_count=len(s.messages),
            created_at=s.created_at,
        )
        for s in sessions
    ]


@router.post("/sessions/{session_id}/chat", response_model=ChatTurnResponse)
async def unified_chat(session_id: str, request: ChatRequest):
    """
    Unified conversational endpoint.

    Accepts any message and routes to the correct sub-service based on intent:
      - daily_plan       → new daily meal plan (fresh canvas)
      - weekly_plan      → new weekly meal plan (fresh canvas)
      - refine_plan      → update the active canvas plan in-place (version++)
      - switch_plan_type → freeze current canvas type, start fresh canvas of the other type
      - chat             → general nutrition / cooking conversation

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

    meal_plan_resp = None
    if turn.meal_plan is not None:
        meal_plan_resp = MealPlanResponse.from_meal_plan(turn.meal_plan)

    weekly_resp = None
    if turn.weekly_meal_plan is not None:
        weekly_resp = WeeklyMealPlanResponse.from_weekly_meal_plan(turn.weekly_meal_plan)

    return ChatTurnResponse(
        role=turn.role,
        content=turn.content,
        intent=turn.intent,
        needs_clarification=turn.needs_clarification,
        meal_plan=meal_plan_resp,
        weekly_meal_plan=weekly_resp,
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
    session = services.session_service.get_session(session_id, member_id=member_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied")

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

    session = services.session_service.get_session(session_id, member_id=request.member_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied")

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
