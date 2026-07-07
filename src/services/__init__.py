"""
Service singletons and startup wiring.

Initialization order (main.py): session/profile services are import-time
singletons; chat → weekly → orchestrator are created by the init_* functions
in that order (the orchestrator requires the other two). Since M0 there is no
data-file dependency, so all services initialize unconditionally at startup.
"""

from .session_service import SessionService
from .profile_service import ProfileService
from .chat_service import ChatService
from .weekly_plan_service import WeeklyPlanService
from .memory_service import MemoryService
from .orchestrator_service import OrchestratorService

# Import-time singletons
session_service = SessionService()
profile_service = ProfileService()

# Created at startup via the init_* functions below
chat_service: ChatService | None = None
weekly_plan_service: WeeklyPlanService | None = None
memory_service: MemoryService | None = None
orchestrator_service: OrchestratorService | None = None


def init_chat_service() -> ChatService:
    """Initialize the chat service singleton."""
    global chat_service
    chat_service = ChatService(session_service)
    return chat_service


def init_weekly_plan_service() -> WeeklyPlanService:
    """Initialize the weekly plan service singleton."""
    global weekly_plan_service
    weekly_plan_service = WeeklyPlanService(session_service)
    return weekly_plan_service


def init_memory_service() -> MemoryService:
    """Initialize the consented-memory service singleton (M3)."""
    global memory_service
    memory_service = MemoryService(session_service, profile_service)
    return memory_service


def init_orchestrator_service() -> OrchestratorService:
    """Initialize the orchestrator singleton (requires chat/weekly/memory services)."""
    global orchestrator_service
    if chat_service is None or weekly_plan_service is None:
        raise RuntimeError("init_chat_service/init_weekly_plan_service must run first")
    orchestrator_service = OrchestratorService(
        session_service=session_service,
        chat_service=chat_service,
        weekly_plan_service=weekly_plan_service,
        memory_service=memory_service,
    )
    return orchestrator_service


__all__ = [
    "SessionService",
    "ProfileService",
    "ChatService",
    "WeeklyPlanService",
    "OrchestratorService",
    "session_service",
    "profile_service",
    "chat_service",
    "weekly_plan_service",
    "orchestrator_service",
    "memory_service",
    "init_chat_service",
    "init_weekly_plan_service",
    "init_memory_service",
    "init_orchestrator_service",
]
