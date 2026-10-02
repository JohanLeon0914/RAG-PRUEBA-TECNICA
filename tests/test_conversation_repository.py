import pytest

from app.repositories.sqlite_conversation_repository import SQLiteConversationRepository


def test_sqlite_repository_persists_sessions_and_messages(tmp_path) -> None:
    db_path = tmp_path / "conversations.db"
    repository = SQLiteConversationRepository(str(db_path))
    repository.initialize()

    session = repository.create_session("session-a")
    repository.add_message(session.id, "user", "hola")
    repository.add_message(
        session.id,
        "assistant",
        "respuesta",
        supported_by_context=True,
        embedding_latency_ms=10,
        search_latency_ms=2,
        llm_latency_ms=100,
        total_latency_ms=130,
        sources_count=2,
    )

    reopened = SQLiteConversationRepository(str(db_path))
    reopened.initialize()
    messages = reopened.get_messages(session.id)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].supported_by_context is True
    assert messages[1].sources_count == 2


def test_get_recent_messages_respects_limit_and_chronological_order(tmp_path) -> None:
    repository = SQLiteConversationRepository(str(tmp_path / "conversations.db"))
    repository.initialize()
    session = repository.create_session("session-a")

    for index in range(5):
        repository.add_message(session.id, "user", f"mensaje {index}")

    recent = repository.get_recent_messages(session.id, limit=3)

    assert [message.content for message in recent] == [
        "mensaje 2",
        "mensaje 3",
        "mensaje 4",
    ]


def test_add_message_rejects_unknown_session(tmp_path) -> None:
    repository = SQLiteConversationRepository(str(tmp_path / "conversations.db"))
    repository.initialize()

    with pytest.raises(KeyError):
        repository.add_message("missing", "user", "hola")


def test_analytics_summary_aggregates_runtime_metrics(tmp_path) -> None:
    repository = SQLiteConversationRepository(str(tmp_path / "conversations.db"))
    repository.initialize()
    first = repository.create_session("first")
    second = repository.create_session("second")
    repository.add_message(first.id, "user", "pregunta 1")
    repository.add_message(
        first.id,
        "assistant",
        "respuesta 1",
        supported_by_context=True,
        embedding_latency_ms=10,
        search_latency_ms=2,
        llm_latency_ms=100,
        total_latency_ms=120,
        sources_count=2,
    )
    repository.add_message(second.id, "user", "pregunta 2")
    repository.add_message(
        second.id,
        "assistant",
        "respuesta 2",
        supported_by_context=False,
        embedding_latency_ms=20,
        search_latency_ms=4,
        llm_latency_ms=200,
        total_latency_ms=240,
        sources_count=0,
    )

    summary = repository.get_analytics_summary()

    assert summary.total_sessions == 2
    assert summary.total_messages == 4
    assert summary.total_user_messages == 2
    assert summary.total_assistant_messages == 2
    assert summary.supported_answers == 1
    assert summary.unsupported_answers == 1
    assert summary.supported_answer_rate == 0.5
    assert summary.average_total_latency_ms == 180
    assert summary.average_embedding_latency_ms == 15
    assert summary.average_search_latency_ms == 3
    assert summary.average_rerank_latency_ms is None
    assert summary.average_llm_latency_ms == 150
    assert summary.average_sources_per_supported_answer == 2
    assert summary.average_messages_per_session == 2
    assert summary.average_user_messages_per_session == 1
    assert summary.impact_indicators["supported_answer_rate"] == 0.5
