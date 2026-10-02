import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes.chat import chat
from app.api.routes.conversations import analytics_summary, session_messages
from app.schemas import AnalyticsSummary, ChatRequest, ConversationSession, RAGResponse, Source


class FakeRAGService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[str] = []

    def answer(self, question: str) -> RAGResponse:
        return self.answer_with_session(question, None)

    def answer_with_session(self, question: str, session_id: str | None) -> RAGResponse:
        self.questions.append(question)
        if self.error:
            raise self.error
        return RAGResponse(
            session_id=session_id or "session-1",
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


class FakeRAGServiceWithSession(FakeRAGService):
    def answer(self, question: str, session_id: str | None = None) -> RAGResponse:
        return self.answer_with_session(question, session_id)


class FakeConversationRepository:
    def get_session(self, session_id: str):
        if session_id == "missing":
            return None
        return ConversationSession(
            id=session_id,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def get_messages(self, session_id: str):
        return []

    def get_analytics_summary(self):
        return AnalyticsSummary(
            total_sessions=1,
            total_messages=0,
            total_user_messages=0,
            total_assistant_messages=0,
            supported_answers=0,
            unsupported_answers=0,
            supported_answer_rate=0,
            average_total_latency_ms=None,
            average_embedding_latency_ms=None,
            average_search_latency_ms=None,
            average_rerank_latency_ms=None,
            average_llm_latency_ms=None,
            average_sources_per_supported_answer=None,
            average_messages_per_session=0,
            average_user_messages_per_session=0,
            impact_indicators={"supported_answer_rate": 0},
        )


def test_chat_route_returns_structured_response() -> None:
    service = FakeRAGServiceWithSession()

    response = chat(
        request=ChatRequest(message="¿Qué beneficios tiene?", session_id="session-x"),
        rag_service=service,  # type: ignore[arg-type]
    )

    assert response.session_id == "session-x"
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
            rag_service=FakeRAGServiceWithSession(RuntimeError("provider unavailable")),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "El servicio RAG no esta disponible temporalmente"


def test_session_messages_returns_404_for_unknown_session() -> None:
    with pytest.raises(HTTPException) as exc_info:
        session_messages("missing", FakeConversationRepository())  # type: ignore[arg-type]

    assert exc_info.value.status_code == 404


def test_analytics_summary_route_returns_repository_summary() -> None:
    summary = analytics_summary(FakeConversationRepository())  # type: ignore[arg-type]

    assert summary.total_sessions == 1
    assert summary.impact_indicators["supported_answer_rate"] == 0
