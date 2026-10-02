from app.reranking.base import RerankingStrategy
from app.schemas import RerankedChunk, RetrievedChunk


class BGERerankingStrategy(RerankingStrategy):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        top_n: int,
    ) -> list[RerankedChunk]:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if not documents:
            return []

        pairs = [(query, document.text) for document in documents]
        scores = self._load_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scored_documents = [
            RerankedChunk(
                **document.model_dump(),
                rerank_score=float(score),
            )
            for document, score in zip(documents, scores, strict=True)
        ]
        return sorted(scored_documents, key=lambda item: item.rerank_score, reverse=True)[:top_n]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
        return self._model
