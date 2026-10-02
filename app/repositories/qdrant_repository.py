from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.repositories.vector_repository import VectorRepository
from app.schemas import DocumentChunk, RetrievedChunk


class QdrantVectorRepository(VectorRepository):
    def __init__(
        self,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self._client = client or QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        if recreate:
            self.recreate_collection(vector_size)
            return

        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        existing_size = self._collection_vector_size()
        if existing_size != vector_size:
            raise ValueError(
                f"Collection {self.collection_name!r} has vector size {existing_size}, "
                f"but active embedding strategy produces {vector_size}. "
                "Use another QDRANT_COLLECTION or recreate explicitly."
            )

    def recreate_collection(self, vector_size: int) -> None:
        self._client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=embedding,
                payload=self._payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        return [self._point_to_retrieved_chunk(point) for point in points]

    def count_points(self) -> int:
        result = self._client.count(collection_name=self.collection_name, exact=True)
        return int(result.count)

    def _collection_vector_size(self) -> int:
        collection = self._client.get_collection(self.collection_name)
        vectors = collection.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        raise ValueError("Only single-vector Qdrant collections are supported.")

    def _payload(self, chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "url": str(chunk.url),
            "title": chunk.title,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "scraped_at": chunk.scraped_at.isoformat(),
            "metadata": chunk.metadata,
        }

    def _point_to_retrieved_chunk(self, point) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            text=payload["text"],
            url=payload["url"],
            title=payload["title"],
            section=payload.get("section"),
            chunk_index=payload["chunk_index"],
            scraped_at=payload["scraped_at"],
            metadata=payload.get("metadata", {}),
            score=point.score,
        )
