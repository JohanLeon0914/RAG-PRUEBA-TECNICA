from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.repositories.conversation_repository import ConversationRepository
from app.schemas import AnalyticsSummary, ChatMessage, ConversationMessage, ConversationSession


class SQLiteConversationRepository(ConversationRepository):
    def __init__(self, db_path: str, timeout_seconds: float = 10.0) -> None:
        self.db_path = Path(db_path)
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    supported_by_context INTEGER,
                    embedding_latency_ms REAL,
                    search_latency_ms REAL,
                    rerank_latency_ms REAL,
                    llm_latency_ms REAL,
                    total_latency_ms REAL,
                    sources_count INTEGER,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_created "
                "ON messages(session_id, created_at, id)"
            )

    def create_session(self, session_id: str | None = None) -> ConversationSession:
        new_session_id = session_id or str(uuid4())
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(id, created_at, updated_at) VALUES (?, ?, ?)",
                (new_session_id, now, now),
            )
        return ConversationSession(
            id=new_session_id,
            created_at=_parse_datetime(now),
            updated_at=_parse_datetime(now),
        )

    def get_session(self, session_id: str) -> ConversationSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return _session_from_row(row) if row else None

    def get_or_create_session(self, session_id: str | None = None) -> ConversationSession:
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                return existing
            return self.create_session(session_id)
        return self.create_session()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        supported_by_context: bool | None = None,
        embedding_latency_ms: float | None = None,
        search_latency_ms: float | None = None,
        rerank_latency_ms: float | None = None,
        llm_latency_ms: float | None = None,
        total_latency_ms: float | None = None,
        sources_count: int | None = None,
    ) -> ConversationMessage:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        now = _utc_now()
        supported_value = None if supported_by_context is None else int(supported_by_context)
        with self._connect() as connection:
            if self._session_row(connection, session_id) is None:
                raise KeyError(f"unknown session_id: {session_id}")
            cursor = connection.execute(
                """
                INSERT INTO messages(
                    session_id, role, content, created_at, supported_by_context,
                    embedding_latency_ms, search_latency_ms, rerank_latency_ms,
                    llm_latency_ms, total_latency_ms, sources_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    now,
                    supported_value,
                    embedding_latency_ms,
                    search_latency_ms,
                    rerank_latency_ms,
                    llm_latency_ms,
                    total_latency_ms,
                    sources_count,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return _message_from_row(row)

    def get_recent_messages(self, session_id: str, limit: int) -> list[ChatMessage]:
        if limit <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                created_at=_parse_datetime(row["created_at"]),
            )
            for row in rows
        ]

    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC, id ASC",
                (session_id,),
            ).fetchall()
        return [_message_from_row(row) for row in rows]

    def list_sessions(self) -> list[ConversationSession]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [_session_from_row(row) for row in rows]

    def get_analytics_summary(self) -> AnalyticsSummary:
        with self._connect() as connection:
            total_sessions = _scalar_int(connection, "SELECT COUNT(*) FROM sessions")
            total_messages = _scalar_int(connection, "SELECT COUNT(*) FROM messages")
            total_user_messages = _scalar_int(
                connection, "SELECT COUNT(*) FROM messages WHERE role = 'user'"
            )
            total_assistant_messages = _scalar_int(
                connection, "SELECT COUNT(*) FROM messages WHERE role = 'assistant'"
            )
            supported_answers = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM messages "
                "WHERE role = 'assistant' AND supported_by_context = 1",
            )
            unsupported_answers = _scalar_int(
                connection,
                "SELECT COUNT(*) FROM messages "
                "WHERE role = 'assistant' AND supported_by_context = 0",
            )
            average_total_latency_ms = _scalar_float(
                connection,
                "SELECT AVG(total_latency_ms) FROM messages "
                "WHERE role = 'assistant' AND total_latency_ms IS NOT NULL",
            )
            average_embedding_latency_ms = _scalar_float(
                connection,
                "SELECT AVG(embedding_latency_ms) FROM messages "
                "WHERE role = 'assistant' AND embedding_latency_ms IS NOT NULL",
            )
            average_search_latency_ms = _scalar_float(
                connection,
                "SELECT AVG(search_latency_ms) FROM messages "
                "WHERE role = 'assistant' AND search_latency_ms IS NOT NULL",
            )
            average_rerank_latency_ms = _scalar_float(
                connection,
                "SELECT AVG(rerank_latency_ms) FROM messages "
                "WHERE role = 'assistant' AND rerank_latency_ms IS NOT NULL",
            )
            average_llm_latency_ms = _scalar_float(
                connection,
                "SELECT AVG(llm_latency_ms) FROM messages "
                "WHERE role = 'assistant' AND llm_latency_ms IS NOT NULL",
            )
            average_sources_per_supported_answer = _scalar_float(
                connection,
                "SELECT AVG(sources_count) FROM messages "
                "WHERE role = 'assistant' AND supported_by_context = 1 "
                "AND sources_count IS NOT NULL",
            )

        supported_answer_rate = (
            supported_answers / total_assistant_messages if total_assistant_messages else 0.0
        )
        average_messages_per_session = (
            total_messages / total_sessions if total_sessions else None
        )
        average_user_messages_per_session = (
            total_user_messages / total_sessions if total_sessions else None
        )
        return AnalyticsSummary(
            total_sessions=total_sessions,
            total_messages=total_messages,
            total_user_messages=total_user_messages,
            total_assistant_messages=total_assistant_messages,
            supported_answers=supported_answers,
            unsupported_answers=unsupported_answers,
            supported_answer_rate=supported_answer_rate,
            average_total_latency_ms=average_total_latency_ms,
            average_embedding_latency_ms=average_embedding_latency_ms,
            average_search_latency_ms=average_search_latency_ms,
            average_rerank_latency_ms=average_rerank_latency_ms,
            average_llm_latency_ms=average_llm_latency_ms,
            average_sources_per_supported_answer=average_sources_per_supported_answer,
            average_messages_per_session=average_messages_per_session,
            average_user_messages_per_session=average_user_messages_per_session,
            impact_indicators={
                "supported_answer_rate": supported_answer_rate,
                "average_response_latency_ms": average_total_latency_ms,
                "average_sources_per_supported_answer": average_sources_per_supported_answer,
            },
        )

    def _session_row(self, connection: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT id, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _session_from_row(row: sqlite3.Row) -> ConversationSession:
    return ConversationSession(
        id=row["id"],
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _message_from_row(row: sqlite3.Row) -> ConversationMessage:
    return ConversationMessage(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        created_at=_parse_datetime(row["created_at"]),
        supported_by_context=(
            None if row["supported_by_context"] is None else bool(row["supported_by_context"])
        ),
        embedding_latency_ms=row["embedding_latency_ms"],
        search_latency_ms=row["search_latency_ms"],
        rerank_latency_ms=row["rerank_latency_ms"],
        llm_latency_ms=row["llm_latency_ms"],
        total_latency_ms=row["total_latency_ms"],
        sources_count=row["sources_count"],
    )


def _scalar_int(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def _scalar_float(connection: sqlite3.Connection, query: str) -> float | None:
    value = connection.execute(query).fetchone()[0]
    return None if value is None else float(value)
