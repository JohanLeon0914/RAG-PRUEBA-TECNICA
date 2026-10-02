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


class ChatResponse(BaseModel):
    answer: str
    supported_by_context: bool
    sources: list[Source]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    answer: str
    supported_by_context: bool
    sources: list[Source]
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
