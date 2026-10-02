from abc import ABC, abstractmethod

from app.schemas import AnalyticsSummary, ChatMessage, ConversationMessage, ConversationSession


class ConversationRepository(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Create persistence schema if it does not exist."""

    @abstractmethod
    def create_session(self, session_id: str | None = None) -> ConversationSession:
        """Create a conversation session and return it."""

    @abstractmethod
    def get_session(self, session_id: str) -> ConversationSession | None:
        """Return a session by id if it exists."""

    @abstractmethod
    def get_or_create_session(self, session_id: str | None = None) -> ConversationSession:
        """Return an existing session or create a new one."""

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        supported_by_context: bool | None = None,
        embedding_latency_ms: float | None = None,
        search_latency_ms: float | None = None,
        rerank_latency_ms: float | None = None,
        llm_latency_ms: float | None = None,
        total_latency_ms: float | None = None,
        sources_count: int | None = None,
    ) -> ConversationMessage:
        """Persist one user or assistant message."""

    @abstractmethod
    def get_recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        """Return the last N messages for prompt history, oldest first."""

    @abstractmethod
    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        """Return all persisted messages in chronological order."""

    @abstractmethod
    def list_sessions(self) -> list[ConversationSession]:
        """Return sessions ordered by most recently updated."""

    @abstractmethod
    def get_analytics_summary(self) -> AnalyticsSummary:
        """Aggregate runtime conversation analytics from persisted messages."""
