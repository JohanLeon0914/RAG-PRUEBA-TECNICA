from dataclasses import dataclass
from time import perf_counter

from app.embeddings.base import EmbeddingStrategy
from app.ingestion.chunker import DocumentChunker
from app.repositories.vector_repository import VectorRepository
from app.scraping.storage import load_documents_jsonl


@dataclass(frozen=True)
class IngestionResult:
    documents_loaded: int
    chunks_generated: int
    embedding_provider: str
    embedding_model: str
    vector_dimension: int
    chunks_indexed: int
    elapsed_seconds: float


class IngestionPipeline:
    def __init__(
        self,
        document_path: str,
        chunker: DocumentChunker,
        embeddings: EmbeddingStrategy,
        repository: VectorRepository,
        embedding_provider: str,
        embedding_model: str,
        recreate_collection: bool = False,
    ) -> None:
        self.document_path = document_path
        self.chunker = chunker
        self.embeddings = embeddings
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.recreate_collection = recreate_collection

    def run(self) -> IngestionResult:
        started_at = perf_counter()
        documents = load_documents_jsonl(self.document_path)
        chunks = self.chunker.chunk(documents)
        texts = [chunk.text for chunk in chunks]

        vector_dimension = self.embeddings.dimension
        self.repository.ensure_collection(
            vector_size=vector_dimension,
            recreate=self.recreate_collection,
        )
        vectors = self.embeddings.embed_documents(texts)
        self.repository.upsert_chunks(chunks, vectors)

        return IngestionResult(
            documents_loaded=len(documents),
            chunks_generated=len(chunks),
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            vector_dimension=vector_dimension,
            chunks_indexed=len(chunks),
            elapsed_seconds=perf_counter() - started_at,
        )
