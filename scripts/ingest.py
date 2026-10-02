from app.config.settings import get_settings
from app.embeddings.factory import create_embedding_strategy
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pipeline import IngestionPipeline
from app.repositories.qdrant_repository import QdrantVectorRepository


def main() -> None:
    settings = get_settings()
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = create_embedding_strategy(settings)
    repository = QdrantVectorRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )
    pipeline = IngestionPipeline(
        document_path=settings.processed_data_path,
        chunker=chunker,
        embeddings=embeddings,
        repository=repository,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
    )

    result = pipeline.run()
    print(f"documents loaded: {result.documents_loaded}")
    print(f"chunks generated: {result.chunks_generated}")
    print(f"embedding provider/model: {result.embedding_provider}/{result.embedding_model}")
    print(f"vector dimension: {result.vector_dimension}")
    print(f"chunks indexed: {result.chunks_indexed}")
    print(f"elapsed seconds: {result.elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
