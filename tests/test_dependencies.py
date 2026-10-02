from app.config.settings import Settings
from app.dependencies import build_rag_service
from app.embeddings.base import EmbeddingStrategy
from app.llm.base import LLMProvider, LLMResponse


class FakeEmbeddings(EmbeddingStrategy):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0]

    @property
    def dimension(self) -> int:
        return 1


class FakeRepository:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def search(self, query_vector: list[float], top_k: int):
        return []


class FakeLLM(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(content="ok", supported_by_context=False, usage={})


def test_build_rag_service_does_not_create_reranker_when_disabled(monkeypatch, tmp_path) -> None:
    created_rerankers = []

    class ExplodingReranker:
        def __init__(self, *args, **kwargs) -> None:
            created_rerankers.append((args, kwargs))
            raise AssertionError("reranker should not be created")

    monkeypatch.setattr(
        "app.dependencies.create_embedding_strategy",
        lambda settings: FakeEmbeddings(),
    )
    monkeypatch.setattr("app.dependencies.QdrantVectorRepository", FakeRepository)
    monkeypatch.setattr("app.dependencies.LLMFactory.create", lambda settings: FakeLLM())
    monkeypatch.setattr("app.dependencies.BGERerankingStrategy", ExplodingReranker)

    service = build_rag_service(
        Settings(
            RERANK_ENABLED=False,
            GROQ_API_KEY="test-key",
            CONVERSATION_DB_PATH=str(tmp_path / "conversations.db"),
        )
    )

    assert service.retrieval_pipeline.rerank_enabled is False
    assert created_rerankers == []
