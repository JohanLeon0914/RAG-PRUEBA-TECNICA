from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.embeddings.factory import create_embedding_strategy
from app.llm.factory import LLMFactory
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.rag.service import RAGService
from app.repositories.qdrant_repository import QdrantVectorRepository
from app.reranking.bge import BGERerankingStrategy
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.retriever import Retriever


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    return build_rag_service(settings)


def build_rag_service(settings: Settings) -> RAGService:
    embeddings = create_embedding_strategy(settings)
    repository = QdrantVectorRepository(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
    )
    retriever = Retriever(
        embeddings=embeddings,
        repository=repository,
        default_top_k=settings.retrieval_top_k,
    )
    reranker = None
    if settings.rerank_enabled:
        reranker = BGERerankingStrategy(
            model_name=settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            device=settings.reranker_device,
        )
    retrieval_pipeline = RetrievalPipeline(
        retriever=retriever,
        context_top_k=settings.rag_context_top_k,
        rerank_enabled=settings.rerank_enabled,
        reranker=reranker,
        candidate_k=settings.retrieval_candidate_k,
        min_retrieval_score=settings.rag_min_retrieval_score,
    )
    return RAGService(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=ContextBuilder(max_chunks=settings.rag_context_top_k),
        prompt_builder=PromptBuilder(),
        llm=LLMFactory.create(settings),
    )
