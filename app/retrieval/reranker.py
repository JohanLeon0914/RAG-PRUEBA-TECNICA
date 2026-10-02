from app.reranking.base import RerankingStrategy
from app.schemas import RerankedChunk, RetrievedChunk


class Reranker:
    def __init__(self, strategy: RerankingStrategy) -> None:
        self.strategy = strategy

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_n: int,
    ) -> list[RerankedChunk]:
        return self.strategy.rerank(query=query, documents=documents, top_n=top_n)
