from abc import ABC, abstractmethod

from app.schemas import DocumentChunk, RetrievedChunk


class VectorRepository(ABC):
    @abstractmethod
    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        """Create the collection or verify that the existing collection matches vector_size."""

    @abstractmethod
    def recreate_collection(self, vector_size: int) -> None:
        """Create a fresh vector collection with the given embedding dimension."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks and their vectors."""

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the most similar chunks for the query vector."""

    def similarity_search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Backward-compatible alias for vector search."""
        return self.search(query_vector=query_vector, top_k=top_k)

    @abstractmethod
    def count_points(self) -> int:
        """Return the number of indexed points in the collection."""
