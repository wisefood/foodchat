from .session_service import SessionService
from .profile_service import ProfileService
from .chat_service import ChatService
from .weekly_plan_service import WeeklyPlanService
from .orchestrator_service import OrchestratorService

# Singleton instances
session_service = SessionService()
profile_service = ProfileService()

# Services initialized lazily or after other dependencies
chat_service: ChatService | None = None
weekly_plan_service: WeeklyPlanService | None = None
orchestrator_service: OrchestratorService | None = None


def init_chat_service(foodchat, config) -> ChatService | None:
    """Initialize the chat service singleton (requires foodchat to be loaded)."""
    global chat_service
    if foodchat is not None:
        chat_service = ChatService(foodchat, config, session_service)
    return chat_service


def init_weekly_plan_service() -> WeeklyPlanService:
    """Initialize the weekly plan service singleton."""
    global weekly_plan_service
    weekly_plan_service = WeeklyPlanService(session_service)
    return weekly_plan_service


def init_orchestrator_service() -> OrchestratorService | None:
    """Initialize the orchestrator service singleton (requires chat + weekly services)."""
    global orchestrator_service
    if chat_service is not None and weekly_plan_service is not None:
        orchestrator_service = OrchestratorService(
            session_service=session_service,
            chat_service=chat_service,
            weekly_plan_service=weekly_plan_service,
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
    "init_chat_service",
    "init_weekly_plan_service",
    "init_orchestrator_service",
]
