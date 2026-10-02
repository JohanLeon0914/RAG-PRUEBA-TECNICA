from time import perf_counter
from typing import Any

from app.llm.base import LLMProvider
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.repositories.conversation_repository import ConversationRepository
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas import RAGResponse, RetrievedChunk, Source


class RAGService:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
        conversation_repository: ConversationRepository | None = None,
        history_n_messages: int = 6,
    ) -> None:
        if history_n_messages <= 0:
            raise ValueError("history_n_messages must be positive")
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.conversation_repository = conversation_repository
        self.history_n_messages = history_n_messages

    def answer(self, question: str, session_id: str | None = None) -> RAGResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        session = None
        history = []
        if self.conversation_repository is not None:
            session = self.conversation_repository.get_or_create_session(session_id)
            history = self.conversation_repository.get_recent_messages(
                session.id,
                self.history_n_messages,
            )

        total_start = perf_counter()
        retrieval_result = self.retrieval_pipeline.retrieve(normalized_question)
        context = self.context_builder.build(retrieval_result.chunks)
        prompt = self.prompt_builder.build(
            question=normalized_question,
            context=context,
            history=history,
        )

        llm_start = perf_counter()
        llm_response = self.llm.generate(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )
        llm_latency_ms = (perf_counter() - llm_start) * 1000
        total_latency_ms = (perf_counter() - total_start) * 1000

        metadata: dict[str, Any] = {
            **retrieval_result.metadata,
            "llm_latency_ms": llm_latency_ms,
            "total_latency_ms": total_latency_ms,
            "llm_usage": llm_response.usage,
            "supported_by_context": llm_response.supported_by_context,
        }
        sources = (
            self._sources(retrieval_result.chunks)
            if llm_response.supported_by_context
            else []
        )
        if self.conversation_repository is not None and session is not None:
            self.conversation_repository.add_message(
                session.id,
                "user",
                normalized_question,
            )
            self.conversation_repository.add_message(
                session.id,
                "assistant",
                llm_response.content,
                supported_by_context=llm_response.supported_by_context,
                embedding_latency_ms=_optional_float(metadata.get("embedding_latency_ms")),
                search_latency_ms=_optional_float(metadata.get("search_latency_ms")),
                rerank_latency_ms=_optional_float(metadata.get("rerank_latency_ms")),
                llm_latency_ms=llm_latency_ms,
                total_latency_ms=total_latency_ms,
                sources_count=len(sources),
            )
        return RAGResponse(
            session_id=session.id if session is not None else session_id,
            answer=llm_response.content,
            supported_by_context=llm_response.supported_by_context,
            sources=sources,
            retrieval_metadata=metadata,
        )

    def _sources(self, chunks: list[RetrievedChunk]) -> list[Source]:
        sources: list[Source] = []
        seen_urls: set[str] = set()
        for chunk in chunks:
            url = str(chunk.url)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                Source(
                    title=chunk.title,
                    url=chunk.url,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                )
            )
        return sources


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
