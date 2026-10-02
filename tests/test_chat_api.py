import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes.chat import chat
from app.schemas import ChatRequest, RAGResponse, Source


class FakeRAGService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[str] = []

    def answer(self, question: str) -> RAGResponse:
        self.questions.append(question)
        if self.error:
            raise self.error
        return RAGResponse(
            answer="Respuesta de prueba",
            supported_by_context=True,
            sources=[
                Source(
                    title="Fuente",
                    url="https://www.bancolombia.com/personas/producto",
                    chunk_index=0,
                    score=0.91,
                )
            ],
            retrieval_metadata={"embedding_latency_ms": 1.0},
        )


def test_chat_route_returns_structured_response() -> None:
    service = FakeRAGService()

    response = chat(
        request=ChatRequest(message="¿Qué beneficios tiene?"),
        rag_service=service,  # type: ignore[arg-type]
    )

    assert response.answer == "Respuesta de prueba"
    assert response.supported_by_context is True
    assert response.sources[0].title == "Fuente"
    assert response.metadata["embedding_latency_ms"] == 1.0
    assert service.questions == ["¿Qué beneficios tiene?"]


def test_chat_request_rejects_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_route_maps_controlled_runtime_errors() -> None:
    with pytest.raises(HTTPException) as exc_info:
        chat(
            request=ChatRequest(message="pregunta"),
            rag_service=FakeRAGService(RuntimeError("provider unavailable")),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "RAG service is temporarily unavailable"
