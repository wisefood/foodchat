"""
FoodChat API entry point.

Startup: create/migrate the database schema, then initialize the service
singletons (chat → weekly → orchestrator; see services/__init__.py).

External dependencies (all via env, see .env.example):
  - Groq API           — every LLM call (agents.py, backend/groq.py)
  - RecipeWrangler     — recipe candidates (services/candidates_client.py)
  - WiseFood API       — member profiles (backend/platform.py)
  - DATABASE_URL       — session store (SQLite file by default; PostgreSQL in M5)

No data files, vector stores, or embedding models are required to boot.
"""

import logging

from dotenv import load_dotenv

# Load .env before any module reads os.getenv at import time (agents, backends).
load_dotenv()

from fastapi import FastAPI

from db import init_db
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
    version="2.0.0",
)

logger.info("Initializing database...")
init_db()
logger.info("Database initialized.")

# Order matters: the orchestrator requires chat + weekly services.
init_chat_service()
init_weekly_plan_service()
init_orchestrator_service()
logger.info("Services initialized (chat, weekly, orchestrator).")

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
