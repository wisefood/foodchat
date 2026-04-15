from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

import services
from models.session import MealCourse

router = APIRouter(
    prefix="/foodchat",
    tags=["foodchat"],
    responses={404: {"description": "Not found"}},
)


# --- Request/Response Models ---


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
    entries: List[WeeklyMealPlanEntryResponse]

    @classmethod
    def from_weekly_meal_plan(cls, wmp) -> "WeeklyMealPlanResponse":
        return cls(
            id=wmp.id,
            created_at=wmp.created_at,
            entries=[
                WeeklyMealPlanEntryResponse(**entry) for entry in wmp.entries
            ],
        )


class WeeklyMessageResponse(BaseModel):
    role: str
    content: str
    weekly_meal_plan: Optional[WeeklyMealPlanResponse] = None


class MessageHistoryItem(BaseModel):
    role: str
    content: str
    timestamp: datetime


def _require_chat_service():
    """Raise 503 if chat service is not available."""
    if services.chat_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service unavailable (CSV data not loaded)",
        )
    return services.chat_service


def _require_weekly_plan_service():
    """Raise 503 if weekly plan service is not available."""
    if services.weekly_plan_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weekly plan service unavailable",
        )
    return services.weekly_plan_service


# --- Endpoints ---


@router.post(
    "/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session for a household member.

    Args:
        request: Contains member_id from WiseFood

    Returns:
        Session details including the session_id for subsequent requests
    """
    try:
        # Fetch profile from WiseFood
        user_profile = services.profile_service.get_member_profile(request.member_id)

        # Create session
        session = services.session_service.create_session(request.member_id, user_profile)

        return SessionResponse(
            session_id=session.session_id,
            member_id=session.member_id,
            state=session.state,
            message_count=len(session.messages),
            created_at=session.created_at,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session state and metadata."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    return SessionResponse(
        session_id=session.session_id,
        member_id=session.member_id,
        state=session.state,
        message_count=len(session.messages),
        created_at=session.created_at,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """Delete a session."""
    if not services.session_service.delete_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: str, request: MessageRequest):
    """Send a message and get a response.

    The response includes a `needs_clarification` flag. If True, the assistant
    is asking a clarifying question and expects a follow-up message.
    """
    chat_svc = _require_chat_service()

    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    try:
        response_text, needs_clarification, meal_plan = chat_svc.process_message(
            session_id, request.content
        )

        meal_plan_resp = None
        if meal_plan is not None:
            meal_plan_resp = MealPlanResponse.from_meal_plan(meal_plan)

        return MessageResponse(
            role="assistant",
            content=response_text,
            needs_clarification=needs_clarification,
            meal_plan=meal_plan_resp,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get(
    "/sessions/{session_id}/messages", response_model=List[MessageHistoryItem]
)
async def get_messages(session_id: str, limit: Optional[int] = None):
    """Get message history for a session.

    Args:
        session_id: The session ID
        limit: Optional limit on number of messages to return (most recent)
    """
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    messages = session.messages
    if limit:
        messages = messages[-limit:]

    return [
        MessageHistoryItem(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
        )
        for msg in messages
    ]


@router.get("/sessions/{session_id}/meal-plans", response_model=List[MealPlanResponse])
async def get_meal_plans(session_id: str):
    """Get all meal plans generated in this session."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    return [MealPlanResponse.from_meal_plan(mp) for mp in session.meal_plans]


@router.post("/sessions/{session_id}/weekly", response_model=WeeklyMessageResponse)
async def send_weekly_message(session_id: str, request: MessageRequest):
    """Send a message to generate a 7-day weekly meal plan."""
    weekly_svc = _require_weekly_plan_service()

    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get(
    "/sessions/{session_id}/weekly", response_model=List[MessageHistoryItem]
)
async def get_weekly_messages(session_id: str, limit: Optional[int] = None):
    """Get weekly message history for a session."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    messages = session.weekly_messages
    if limit:
        messages = messages[-limit:]

    return [
        MessageHistoryItem(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
        )
        for msg in messages
    ]


@router.get("/sessions/{session_id}/weekly-meal-plans", response_model=List[WeeklyMealPlanResponse])
async def get_weekly_meal_plans(session_id: str):
    """Get all weekly meal plans generated in this session."""
    session = services.session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    return [WeeklyMealPlanResponse.from_weekly_meal_plan(wmp) for wmp in session.weekly_meal_plans]


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


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "foodchat"}
