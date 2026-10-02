from abc import ABC, abstractmethod

from app.schemas import RerankedChunk, RetrievedChunk


class RerankingStrategy(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_n: int,
    ) -> list[RerankedChunk]:
        """Score retrieved chunks against the query and return the best documents."""
