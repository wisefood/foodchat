from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

import services

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


class MessageResponse(BaseModel):
    role: str
    content: str
    needs_clarification: bool = False


class MessageHistoryItem(BaseModel):
    role: str
    content: str
    timestamp: datetime


class MealPlanResponse(BaseModel):
    id: str
    created_at: datetime
    meal_plan: List[tuple]
    reasoning: str


def _require_chat_service():
    """Raise 503 if chat service is not available."""
    if services.chat_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service unavailable (CSV data not loaded)",
        )
    return services.chat_service


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
        response_text, needs_clarification = chat_svc.process_message(
            session_id, request.content
        )

        return MessageResponse(
            role="assistant",
            content=response_text,
            needs_clarification=needs_clarification,
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

    return [
        MealPlanResponse(
            id=mp.id,
            created_at=mp.created_at,
            breakfast=mp.breakfast,
            lunch=mp.lunch,
            dinner=mp.dinner,
            reasoning=mp.reasoning,
        )
        for mp in session.meal_plans
    ]


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "foodchat"}
