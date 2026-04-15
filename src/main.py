from fastapi import FastAPI

from foodchat_init import initialize_foodchat
from routers import foodchat_router
from services import init_chat_service, init_weekly_plan_service

app = FastAPI(
    title="FoodChat API",
    description="Session-based meal planning chat API",
    version="1.0.0",
)

# Initialize FoodChat system (retriever, vectorstore, etc.)
foodchat, config = initialize_foodchat()

# Initialize services
init_chat_service(foodchat, config)
init_weekly_plan_service()

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
