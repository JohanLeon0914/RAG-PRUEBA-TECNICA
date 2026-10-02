from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class SourceDocument(BaseModel):
    document_id: str
    url: HttpUrl
    title: str
    content: str
    section: str | None = None
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    url: HttpUrl
    title: str
    chunk_index: int
    section: str | None = None
    scraped_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(DocumentChunk):
    score: float


class RerankedChunk(RetrievedChunk):
    rerank_score: float


class Source(BaseModel):
    title: str
    url: HttpUrl
    chunk_index: int | None = None
    score: float | None = None


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ConversationMessage(ChatMessage):
    id: int
    session_id: str
    supported_by_context: bool | None = None
    embedding_latency_ms: float | None = None
    search_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    llm_latency_ms: float | None = None
    total_latency_ms: float | None = None
    sources_count: int | None = None


class ConversationSession(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    supported_by_context: bool
    sources: list[Source]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    session_id: str | None = None
    answer: str
    supported_by_context: bool
    sources: list[Source]
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyticsSummary(BaseModel):
    total_sessions: int
    total_messages: int
    total_user_messages: int
    total_assistant_messages: int
    supported_answers: int
    unsupported_answers: int
    supported_answer_rate: float
    average_total_latency_ms: float | None
    average_embedding_latency_ms: float | None
    average_search_latency_ms: float | None
    average_rerank_latency_ms: float | None
    average_llm_latency_ms: float | None
    average_sources_per_supported_answer: float | None
    average_messages_per_session: float | None
    average_user_messages_per_session: float | None
    impact_indicators: dict[str, float | None] = Field(default_factory=dict)
