from abc import ABC, abstractmethod


class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for document chunks."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a user query."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector size produced by this strategy."""

