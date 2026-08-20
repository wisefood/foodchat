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
import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before any module reads os.getenv at import time (agents, backends).
load_dotenv()

from fastapi import FastAPI

from db import init_db
from routers import foodchat_router
from services import (
    init_chat_service,
    init_weekly_plan_service,
    init_memory_service,
    init_orchestrator_service,
)

# Both Dockerfile stages set LOG_LEVEL (DEBUG in dev, INFO in prod) and nothing
# read it, so the dev image logged at INFO exactly like production. Validated
# against a fixed set rather than getattr(logging, ...): an unknown value falls
# back to INFO, because a typo'd log level must not stop the pod from booting.
_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(
    level=_LOG_LEVEL if _LOG_LEVEL in _LOG_LEVELS else "INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown for Langfuse (entirely optional — no-op without keys).

    Startup seeds the prompt registry into Langfuse in a DAEMON thread so a
    slow or down Langfuse never delays or fails pod boot. Seeding is idempotent
    (creates only missing prompts; never overwrites UI edits). Shutdown flushes
    buffered traces so the last requests aren't lost on pod termination.
    """
    def _seed_prompts() -> None:
        try:
            from prompts import sync_prompts
            result = sync_prompts()
            if any(result.values()):
                logger.info("Langfuse prompt sync: %s", result)
        except Exception as exc:  # observability must never break the app
            logger.warning("Langfuse prompt sync failed: %s", exc)

    threading.Thread(
        target=_seed_prompts, name="langfuse-prompt-sync", daemon=True
    ).start()

    yield

    from backend.observability import flush_langfuse
    flush_langfuse()


app = FastAPI(
    title="FoodChat API",
    description="Session-based meal planning chat API",
    version="2.0.0",
    lifespan=lifespan,
)

logger.info("Initializing database...")
init_db()
logger.info("Database initialized.")

# Order matters: the orchestrator requires chat + weekly + memory services.
init_chat_service()
init_weekly_plan_service()
init_memory_service()
init_orchestrator_service()
logger.info("Services initialized (chat, weekly, memory, orchestrator).")

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
    import uvicorn

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
