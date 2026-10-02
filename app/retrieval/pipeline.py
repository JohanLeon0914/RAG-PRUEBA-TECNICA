from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from app.reranking.base import RerankingStrategy
from app.retrieval.retriever import Retriever
from app.schemas import RetrievedChunk


@dataclass(frozen=True)
class RetrievalPipelineResult:
    chunks: list[RetrievedChunk]
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        context_top_k: int,
        rerank_enabled: bool = False,
        reranker: RerankingStrategy | None = None,
        candidate_k: int = 15,
        min_retrieval_score: float | None = None,
    ) -> None:
        if context_top_k <= 0:
            raise ValueError("context_top_k must be positive")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if rerank_enabled and reranker is None:
            raise ValueError("reranker is required when rerank_enabled=True")

        self.retriever = retriever
        self.context_top_k = context_top_k
        self.rerank_enabled = rerank_enabled
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.min_retrieval_score = min_retrieval_score

    def retrieve(self, query: str) -> RetrievalPipelineResult:
        retrieve_top_k = self.candidate_k if self.rerank_enabled else self.context_top_k
        retrieval_result = self.retriever.retrieve_with_metadata(query, top_k=retrieve_top_k)
        candidates = self._filter_by_score(self._deduplicate(retrieval_result.chunks))
        metadata: dict[str, Any] = {
            **retrieval_result.metadata,
            "rerank_enabled": self.rerank_enabled,
            "candidate_k": retrieve_top_k,
            "context_top_k": self.context_top_k,
        }

        if not self.rerank_enabled:
            return RetrievalPipelineResult(
                chunks=candidates[: self.context_top_k],
                metadata=metadata,
            )

        rerank_start = perf_counter()
        assert self.reranker is not None
        reranked = self.reranker.rerank(query, candidates, top_n=self.context_top_k)
        rerank_latency_ms = (perf_counter() - rerank_start) * 1000
        metadata.update(
            {
                "rerank_latency_ms": rerank_latency_ms,
                "reranked_chunk_ids": [chunk.chunk_id for chunk in reranked],
                "reranker_scores": [
                    getattr(chunk, "rerank_score", None)
                    for chunk in reranked
                ],
            }
        )
        return RetrievalPipelineResult(chunks=list(reranked), metadata=metadata)

    def _filter_by_score(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.min_retrieval_score is None:
            return chunks
        return [chunk for chunk in chunks if chunk.score >= self.min_retrieval_score]

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            unique_chunks.append(chunk)
        return unique_chunks
