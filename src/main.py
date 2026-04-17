import logging

from fastapi import FastAPI

from db import init_db
from foodchat_init import initialize_foodchat
from routers import foodchat_router
from services import init_chat_service, init_weekly_plan_service, init_orchestrator_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FoodChat API",
    description="Session-based meal planning chat API",
    version="1.0.0",
)

# Create DB tables
logger.info("Initializing database...")
init_db()
logger.info("Database initialized.")

# Initialize FoodChat system (retriever, vectorstore, etc.)
logger.info("Initializing FoodChat system (CSV data, vectorstore, embeddings)...")
foodchat, config = initialize_foodchat()

if foodchat is None:
    logger.error(
        "FoodChat system failed to initialize — CSV data not found or embeddings unavailable. "
        "chat_service and orchestrator_service will be UNAVAILABLE. "
        "Check CSV_HUMMUS_PATH env var and that the data file exists at that path."
    )
else:
    logger.info("FoodChat system initialized successfully.")

# Initialize services (order matters: orchestrator needs chat + weekly)
chat_svc = init_chat_service(foodchat, config)
if chat_svc is None:
    logger.error("chat_service NOT initialized (foodchat is None).")
else:
    logger.info("chat_service initialized.")

weekly_svc = init_weekly_plan_service()
logger.info("weekly_plan_service initialized.")

orch_svc = init_orchestrator_service()
if orch_svc is None:
    logger.error(
        "orchestrator_service NOT initialized — requires both chat_service and weekly_plan_service. "
        "The /chat endpoint will return 503 until this is resolved."
    )
else:
    logger.info("orchestrator_service initialized.")

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
