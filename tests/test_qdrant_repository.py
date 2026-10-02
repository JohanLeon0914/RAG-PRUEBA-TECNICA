from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from app.repositories.qdrant_repository import QdrantVectorRepository
from app.schemas import DocumentChunk


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="doc-1:0:abc",
        document_id="doc-1",
        text="Contenido útil",
        url="https://www.bancolombia.com/personas",
        title="Personas",
        section="personas",
        chunk_index=0,
        scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"source": "bancolombia.com"},
    )


class FakeQdrantClient:
    def __init__(self, exists: bool = False, vector_size: int = 3) -> None:
        self.exists = exists
        self.vector_size = vector_size
        self.created = None
        self.recreated = None
        self.upserted = []
        self.query_calls = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.created = (collection_name, vectors_config)
        self.exists = True

    def recreate_collection(self, collection_name: str, vectors_config) -> None:
        self.recreated = (collection_name, vectors_config)
        self.exists = True

    def get_collection(self, collection_name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=self.vector_size))
            )
        )

    def upsert(self, collection_name: str, points, wait: bool):
        self.upserted = points

    def count(self, collection_name: str, exact: bool):
        return SimpleNamespace(count=len(self.upserted))

    def query_points(
        self,
        collection_name: str,
        query,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ):
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.87,
                    payload={
                        "chunk_id": "doc-1:0:abc",
                        "document_id": "doc-1",
                        "text": "Contenido útil",
                        "url": "https://www.bancolombia.com/personas",
                        "title": "Personas",
                        "section": "personas",
                        "chunk_index": 0,
                        "scraped_at": "2026-01-01T00:00:00+00:00",
                        "metadata": {"source": "bancolombia.com"},
                    },
                )
            ]
        )


def test_qdrant_repository_creates_collection_when_missing() -> None:
    client = FakeQdrantClient(exists=False)
    repository = QdrantVectorRepository(
        url="http://localhost:6333",
        collection_name="test",
        client=client,
    )

    repository.ensure_collection(vector_size=3)

    assert client.created[0] == "test"
    assert client.created[1].size == 3
    assert client.created[1].distance == models.Distance.COSINE


def test_qdrant_repository_rejects_dimension_mismatch() -> None:
    client = FakeQdrantClient(exists=True, vector_size=4)
    repository = QdrantVectorRepository(
        url="http://localhost:6333",
        collection_name="test",
        client=client,
    )

    with pytest.raises(ValueError, match="vector size 4"):
        repository.ensure_collection(vector_size=3)


def test_qdrant_repository_upserts_payload_without_exposing_client_to_pipeline() -> None:
    client = FakeQdrantClient(exists=True, vector_size=3)
    repository = QdrantVectorRepository(
        url="http://localhost:6333",
        collection_name="test",
        client=client,
    )

    repository.upsert_chunks([make_chunk()], [[1.0, 0.0, 0.0]])

    assert repository.count_points() == 1
    point = client.upserted[0]
    assert point.payload["chunk_id"] == "doc-1:0:abc"
    assert point.payload["document_id"] == "doc-1"
    assert point.payload["chunk_index"] == 0
    assert point.payload["metadata"] == {"source": "bancolombia.com"}
    assert point.vector == [1.0, 0.0, 0.0]


def test_qdrant_repository_search_uses_query_points_and_preserves_scores() -> None:
    client = FakeQdrantClient(exists=True, vector_size=3)
    repository = QdrantVectorRepository(
        url="http://localhost:6333",
        collection_name="test",
        client=client,
    )

    results = repository.search(query_vector=[0.1, 0.2, 0.3], top_k=5)

    assert client.query_calls == [
        {
            "collection_name": "test",
            "query": [0.1, 0.2, 0.3],
            "limit": 5,
            "with_payload": True,
            "with_vectors": False,
        }
    ]
    assert results[0].chunk_id == "doc-1:0:abc"
    assert results[0].score == 0.87
    assert results[0].title == "Personas"
