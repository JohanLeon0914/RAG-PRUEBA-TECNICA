from time import perf_counter
from typing import Any

from app.llm.base import LLMProvider
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas import RAGResponse, RetrievedChunk, Source


class RAGService:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm = llm

    def answer(self, question: str) -> RAGResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        total_start = perf_counter()
        retrieval_result = self.retrieval_pipeline.retrieve(normalized_question)
        context = self.context_builder.build(retrieval_result.chunks)
        prompt = self.prompt_builder.build(question=normalized_question, context=context)

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
        return RAGResponse(
            answer=llm_response.content,
            supported_by_context=llm_response.supported_by_context,
            sources=self._sources(retrieval_result.chunks)
            if llm_response.supported_by_context
            else [],
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
