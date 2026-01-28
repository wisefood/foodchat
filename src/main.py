from fastapi import FastAPI

from foodchat_init import initialize_foodchat
from routers import foodchat_router
from services.chat_service import ChatService
from services.profile_service import ProfileService
from services.session_service import SessionService

app = FastAPI(
    title="FoodChat API",
    description="Session-based meal planning chat API",
    version="1.0.0",
)

# Initialize FoodChat system (retriever, vectorstore, etc.)
foodchat, config = initialize_foodchat()

session_service = SessionService()
profile_service = ProfileService()
chat_service = ChatService(foodchat, config, session_service)

# Initialize router with services§
foodchat_router.init_services(session_service, profile_service, chat_service)
app.include_router(foodchat_router.router)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to FoodChat API",
        "docs": "/docs",
        "health": "/foodchat/health",
    }


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
