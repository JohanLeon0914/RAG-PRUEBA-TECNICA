from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class QueryMetrics:
    session_id_hash: str
    retrieved_chunk_ids: list[str]
    llm_provider: str
    llm_model: str
    retrieval_latency_ms: float | None = None
    reranking_latency_ms: float | None = None
    llm_latency_ms: float | None = None
    total_latency_ms: float | None = None
    similarity_scores: list[float] = field(default_factory=list)
    reranking_scores: list[float] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class MetricsRecorder:
    def record_query(self, metrics: QueryMetrics) -> None:
        raise NotImplementedError("Structured analytics will be implemented in Phase 11.")

