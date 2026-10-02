from app.reranking.base import RerankingStrategy
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas import RerankedChunk, RetrievedChunk
from tests.helpers import retrieved_chunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int | None]] = []

    def retrieve_with_metadata(self, query: str, top_k: int | None = None):
        self.calls.append((query, top_k))
        return type(
            "Result",
            (),
            {
                "chunks": self.chunks[:top_k],
                "metadata": {
                    "embedding_latency_ms": 1.0,
                    "search_latency_ms": 2.0,
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in self.chunks[:top_k]],
                    "retrieval_scores": [chunk.score for chunk in self.chunks[:top_k]],
                },
            },
        )()


class FakeReranker(RerankingStrategy):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_n: int,
    ) -> list[RerankedChunk]:
        self.calls.append((query, [document.chunk_id for document in documents], top_n))
        reranked = [
            RerankedChunk(**document.model_dump(), rerank_score=float(index))
            for index, document in enumerate(reversed(documents), start=1)
        ]
        return reranked[:top_n]


def test_retrieval_pipeline_without_reranker_uses_context_top_k() -> None:
    chunks = [retrieved_chunk(chunk_id=str(index)) for index in range(3)]
    retriever = FakeRetriever(chunks)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        context_top_k=2,
        rerank_enabled=False,
        reranker=reranker,
        candidate_k=15,
    )

    result = pipeline.retrieve("consulta")

    assert retriever.calls == [("consulta", 2)]
    assert reranker.calls == []
    assert [chunk.chunk_id for chunk in result.chunks] == ["0", "1"]
    assert result.metadata["rerank_enabled"] is False


def test_retrieval_pipeline_with_reranker_uses_candidate_k_and_top_k() -> None:
    chunks = [retrieved_chunk(chunk_id=str(index)) for index in range(4)]
    retriever = FakeRetriever(chunks)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        context_top_k=2,
        rerank_enabled=True,
        reranker=reranker,
        candidate_k=4,
    )

    result = pipeline.retrieve("consulta")

    assert retriever.calls == [("consulta", 4)]
    assert reranker.calls == [("consulta", ["0", "1", "2", "3"], 2)]
    assert [chunk.chunk_id for chunk in result.chunks] == ["3", "2"]
    assert result.metadata["rerank_enabled"] is True
    assert "rerank_latency_ms" in result.metadata


def test_retrieval_pipeline_filters_optional_score_threshold() -> None:
    chunks = [
        retrieved_chunk(chunk_id="low", score=0.2),
        retrieved_chunk(chunk_id="high", score=0.8),
    ]
    pipeline = RetrievalPipeline(
        retriever=FakeRetriever(chunks),  # type: ignore[arg-type]
        context_top_k=2,
        min_retrieval_score=0.5,
    )

    result = pipeline.retrieve("consulta")

    assert [chunk.chunk_id for chunk in result.chunks] == ["high"]
