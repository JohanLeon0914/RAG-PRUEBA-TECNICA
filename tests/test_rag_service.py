import pytest

from app.llm.base import LLMProvider, LLMResponse
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.rag.service import RAGService
from app.retrieval.pipeline import RetrievalPipelineResult
from tests.helpers import retrieved_chunk


class FakePipeline:
    def __init__(self, chunks=None) -> None:
        self.chunks = chunks if chunks is not None else [retrieved_chunk()]
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalPipelineResult:
        self.queries.append(query)
        return RetrievalPipelineResult(
            chunks=self.chunks,
            metadata={"embedding_latency_ms": 1.0, "search_latency_ms": 2.0},
        )


class FakeLLM(LLMProvider):
    def __init__(self, supported_by_context: bool = True) -> None:
        self.supported_by_context = supported_by_context
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.prompts.append((system_prompt, user_prompt))
        return LLMResponse(
            content="Respuesta fundamentada",
            supported_by_context=self.supported_by_context,
            usage={"total_tokens": 12},
        )


class FailingLLM(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise RuntimeError("provider failed")


def test_rag_service_orchestrates_retrieval_context_prompt_and_llm() -> None:
    pipeline = FakePipeline()
    llm = FakeLLM()
    service = RAGService(
        retrieval_pipeline=pipeline,  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        llm=llm,
    )

    response = service.answer("  beneficios  ")

    assert pipeline.queries == ["beneficios"]
    assert response.answer == "Respuesta fundamentada"
    assert response.supported_by_context is True
    assert len(response.sources) == 1
    assert response.sources[0].title == "Titulo"
    assert response.retrieval_metadata["embedding_latency_ms"] == 1.0
    assert response.retrieval_metadata["llm_usage"] == {"total_tokens": 12}
    assert "Contenido relevante de prueba" in llm.prompts[0][1]


def test_rag_service_deduplicates_sources_by_url() -> None:
    chunk_a = retrieved_chunk(chunk_id="a", chunk_index=0)
    chunk_b = retrieved_chunk(chunk_id="b", chunk_index=1)
    service = RAGService(
        retrieval_pipeline=FakePipeline([chunk_a, chunk_b]),  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        llm=FakeLLM(),
    )

    response = service.answer("pregunta")

    assert len(response.sources) == 1


def test_rag_service_suppresses_sources_when_llm_marks_context_unsupported() -> None:
    service = RAGService(
        retrieval_pipeline=FakePipeline([retrieved_chunk()]),  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        llm=FakeLLM(supported_by_context=False),
    )

    response = service.answer("¿Cuál es la capital de Japón?")

    assert response.supported_by_context is False
    assert response.sources == []
    assert response.retrieval_metadata["supported_by_context"] is False


def test_rag_service_rejects_empty_question() -> None:
    service = RAGService(
        retrieval_pipeline=FakePipeline(),  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        llm=FakeLLM(),
    )

    with pytest.raises(ValueError, match="question must not be empty"):
        service.answer("  ")


def test_rag_service_handles_empty_retrieval_context() -> None:
    llm = FakeLLM()
    service = RAGService(
        retrieval_pipeline=FakePipeline([]),  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        llm=llm,
    )

    response = service.answer("pregunta")

    assert response.sources == []
    assert "No se recuperó contexto útil" in llm.prompts[0][1]


def test_rag_service_preserves_provider_errors() -> None:
    service = RAGService(
        retrieval_pipeline=FakePipeline(),  # type: ignore[arg-type]
        context_builder=ContextBuilder(max_chunks=5),
        prompt_builder=PromptBuilder(),
        llm=FailingLLM(),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        service.answer("pregunta")
