from datetime import UTC, datetime

import pytest

from app.embeddings.base import EmbeddingStrategy
from app.repositories.vector_repository import VectorRepository
from app.retrieval.retriever import Retriever
from app.schemas import DocumentChunk, RetrievedChunk


class FakeEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0, 2.0, 3.0]

    @property
    def dimension(self) -> int:
        return 3


class FailingEmbeddingStrategy(FakeEmbeddingStrategy):
    def embed_query(self, query: str) -> list[float]:
        raise RuntimeError("embedding failed")


class FakeVectorRepository(VectorRepository):
    def __init__(self) -> None:
        self.search_calls: list[tuple[list[float], int]] = []
        self.results = [
            RetrievedChunk(
                chunk_id="doc-1:0:abc",
                document_id="doc-1",
                text="Tarjeta debito con beneficios",
                url="https://www.bancolombia.com/personas/productos/cuentas/tarjetas-debito",
                title="Tarjetas Debito",
                section="personas",
                chunk_index=0,
                scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
                metadata={"source": "test"},
                score=0.91,
            )
        ]

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        return None

    def recreate_collection(self, vector_size: int) -> None:
        return None

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        return None

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        self.search_calls.append((query_vector, top_k))
        return self.results

    def count_points(self) -> int:
        return len(self.results)


class FailingVectorRepository(FakeVectorRepository):
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        raise RuntimeError("search failed")


def test_retriever_rejects_empty_query() -> None:
    retriever = Retriever(FakeEmbeddingStrategy(), FakeVectorRepository())

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ")


def test_retriever_embeds_normalized_query_and_searches_repository() -> None:
    embeddings = FakeEmbeddingStrategy()
    repository = FakeVectorRepository()
    retriever = Retriever(embeddings, repository, default_top_k=10)

    results = retriever.retrieve("  tarjetas debito  ")

    assert embeddings.queries == ["tarjetas debito"]
    assert repository.search_calls == [([1.0, 2.0, 3.0], 10)]
    assert results == repository.results


def test_retriever_allows_top_k_override() -> None:
    repository = FakeVectorRepository()
    retriever = Retriever(FakeEmbeddingStrategy(), repository, default_top_k=10)

    retriever.retrieve("beneficios", top_k=5)

    assert repository.search_calls == [([1.0, 2.0, 3.0], 5)]


def test_retriever_rejects_invalid_top_k_values() -> None:
    retriever = Retriever(FakeEmbeddingStrategy(), FakeVectorRepository())

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.retrieve("beneficios", top_k=0)


def test_retriever_rejects_invalid_default_top_k() -> None:
    with pytest.raises(ValueError, match="default_top_k must be positive"):
        Retriever(FakeEmbeddingStrategy(), FakeVectorRepository(), default_top_k=0)


def test_retriever_does_not_silence_embedding_errors() -> None:
    retriever = Retriever(FailingEmbeddingStrategy(), FakeVectorRepository())

    with pytest.raises(RuntimeError, match="embedding failed"):
        retriever.retrieve("beneficios")


def test_retriever_does_not_silence_repository_errors() -> None:
    retriever = Retriever(FakeEmbeddingStrategy(), FailingVectorRepository())

    with pytest.raises(RuntimeError, match="search failed"):
        retriever.retrieve("beneficios")
