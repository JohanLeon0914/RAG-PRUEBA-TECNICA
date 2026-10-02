import argparse
import time

from app.config.settings import get_settings
from app.embeddings.base import EmbeddingStrategy
from app.embeddings.factory import create_embedding_strategy
from app.repositories.qdrant_repository import QdrantVectorRepository
from app.repositories.vector_repository import VectorRepository
from app.retrieval.retriever import Retriever
from app.schemas import DocumentChunk, RetrievedChunk


class TimedEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self, wrapped: EmbeddingStrategy) -> None:
        self.wrapped = wrapped
        self.last_query_seconds = 0.0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.wrapped.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        started_at = time.perf_counter()
        embedding = self.wrapped.embed_query(query)
        self.last_query_seconds = time.perf_counter() - started_at
        return embedding

    @property
    def dimension(self) -> int:
        return self.wrapped.dimension


class TimedVectorRepository(VectorRepository):
    def __init__(self, wrapped: VectorRepository) -> None:
        self.wrapped = wrapped
        self.last_search_seconds = 0.0

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        self.wrapped.ensure_collection(vector_size=vector_size, recreate=recreate)

    def recreate_collection(self, vector_size: int) -> None:
        self.wrapped.recreate_collection(vector_size=vector_size)

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self.wrapped.upsert_chunks(chunks=chunks, embeddings=embeddings)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        started_at = time.perf_counter()
        results = self.wrapped.search(query_vector=query_vector, top_k=top_k)
        self.last_search_seconds = time.perf_counter() - started_at
        return results

    def count_points(self) -> int:
        return self.wrapped.count_points()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual vector retrieval query.")
    parser.add_argument("query", help="Question or search query to embed and search.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=700,
        help="Maximum number of characters to print per retrieved chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    top_k = settings.retrieval_top_k if args.top_k is None else args.top_k

    embeddings = TimedEmbeddingStrategy(create_embedding_strategy(settings))
    repository = TimedVectorRepository(
        QdrantVectorRepository(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
        )
    )
    retriever = Retriever(
        embeddings=embeddings,
        repository=repository,
        default_top_k=settings.retrieval_top_k,
    )

    results = retriever.retrieve(args.query, top_k=top_k)

    print("Query:")
    print(args.query)
    print()
    print(f"Embedding: {settings.embedding_model}")
    print(f"Collection: {settings.qdrant_collection}")
    print(f"Top K: {top_k}")
    print(f"Query embedding seconds: {embeddings.last_query_seconds:.4f}")
    print(f"Vector search seconds: {repository.last_search_seconds:.4f}")
    print()
    print("Top results:")
    for index, chunk in enumerate(results, start=1):
        text = chunk.text[: args.max_chars].strip()
        if len(chunk.text) > args.max_chars:
            text = f"{text}..."
        print()
        print(f"[{index}]")
        print(f"score: {chunk.score:.6f}")
        print(f"title: {chunk.title}")
        print(f"url: {chunk.url}")
        print(f"chunk_index: {chunk.chunk_index}")
        print("text:")
        print(text)


if __name__ == "__main__":
    main()
