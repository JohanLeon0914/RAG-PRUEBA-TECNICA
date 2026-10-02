from app.reranking.base import RerankingStrategy
from app.schemas import RerankedChunk, RetrievedChunk


class GTERerankingStrategy(RerankingStrategy):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_n: int,
    ) -> list[RerankedChunk]:
        raise NotImplementedError("GTERerankingStrategy will be implemented in Phase 7.")
