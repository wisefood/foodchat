from .session_service import SessionService
from .profile_service import ProfileService
from .chat_service import ChatService

# Singleton instances
session_service = SessionService()
profile_service = ProfileService()

# chat_service is initialized lazily since it depends on foodchat
chat_service: ChatService | None = None


def init_chat_service(foodchat, config) -> ChatService | None:
    """Initialize the chat service singleton (requires foodchat to be loaded)."""
    global chat_service
    if foodchat is not None:
        chat_service = ChatService(foodchat, config, session_service)
    return chat_service


__all__ = [
    "SessionService",
    "ProfileService",
    "ChatService",
    "session_service",
    "profile_service",
    "chat_service",
    "init_chat_service",
]
