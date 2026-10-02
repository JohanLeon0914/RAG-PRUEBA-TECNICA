from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from app.embeddings.base import EmbeddingStrategy
from app.repositories.vector_repository import VectorRepository
from app.schemas import RetrievedChunk


@dataclass(frozen=True)
class RetrieverResult:
    chunks: list[RetrievedChunk]
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingStrategy,
        repository: VectorRepository,
        default_top_k: int = 10,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        self.embeddings = embeddings
        self.repository = repository
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retrieve_with_metadata(query=query, top_k=top_k).chunks

    def retrieve_with_metadata(self, query: str, top_k: int | None = None) -> RetrieverResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        effective_top_k = self.default_top_k if top_k is None else top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be positive")

        embedding_start = perf_counter()
        query_vector = self.embeddings.embed_query(normalized_query)
        embedding_latency_ms = (perf_counter() - embedding_start) * 1000

        search_start = perf_counter()
        chunks = self.repository.search(query_vector=query_vector, top_k=effective_top_k)
        search_latency_ms = (perf_counter() - search_start) * 1000

        return RetrieverResult(
            chunks=chunks,
            metadata={
                "top_k": effective_top_k,
                "embedding_latency_ms": embedding_latency_ms,
                "search_latency_ms": search_latency_ms,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "retrieval_scores": [chunk.score for chunk in chunks],
            },
        )
