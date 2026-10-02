import os
from threading import Lock

from app.embeddings.base import EmbeddingStrategy


class BGEM3EmbeddingStrategy(EmbeddingStrategy):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._model_lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._load_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self._load_model().encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        model = self._load_model()
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    def _load_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=True,
                        model_kwargs={"use_safetensors": False},
                    )
        return self._model
