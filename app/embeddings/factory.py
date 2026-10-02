from app.config.settings import Settings
from app.embeddings.base import EmbeddingStrategy
from app.embeddings.bge import BGEM3EmbeddingStrategy
from app.embeddings.gte import GTEMultilingualEmbeddingStrategy


def create_embedding_strategy(settings: Settings) -> EmbeddingStrategy:
    if settings.embedding_provider == "bge":
        return BGEM3EmbeddingStrategy(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )

    if settings.embedding_provider == "gte":
        return GTEMultilingualEmbeddingStrategy(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
