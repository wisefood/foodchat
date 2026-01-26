from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from foodchat import *
from main import main_run

foodchat = main_run()

router = APIRouter(
    prefix="/foodchat",
    tags=["foodchat"],
    responses={404: {"description": "Not found"}},
)

# --- Pydantic Models for Input/Output Validation ---

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None # conversation history or context

class ChatResponse(BaseModel):
    response: str


def get_foodchat_service():
    return FoodChat() 

# --- Endpoints ---

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    service: FoodChat = Depends(get_foodchat_service)
):
    """
    Endpoint to interact with the FoodChat bot.
    """
    try:
        
        answer = service.get_test_response(request.message)
        
        return ChatResponse(response=answer)
    
    except Exception as e:
        print(f"Error in foodchat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "foodchat"}