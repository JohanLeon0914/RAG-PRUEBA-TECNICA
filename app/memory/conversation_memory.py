from abc import ABC, abstractmethod

from app.schemas import ChatMessage


class ConversationMemory(ABC):
    @abstractmethod
    def get(self, session_id: str) -> list[ChatMessage]:
        """Return recent conversation messages for a session."""

    @abstractmethod
    def add(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Store a user/assistant interaction."""


class InMemoryConversationMemory(ConversationMemory):
    def __init__(self, max_messages: int) -> None:
        self.max_messages = max_messages
        self._messages: dict[str, list[ChatMessage]] = {}

    def get(self, session_id: str) -> list[ChatMessage]:
        return self._messages.get(session_id, [])[-self.max_messages :]

    def add(self, session_id: str, user_message: str, assistant_message: str) -> None:
        messages = self._messages.setdefault(session_id, [])
        messages.append(ChatMessage(role="user", content=user_message))
        messages.append(ChatMessage(role="assistant", content=assistant_message))
        self._messages[session_id] = messages[-self.max_messages :]

