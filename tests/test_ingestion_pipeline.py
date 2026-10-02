from pathlib import Path

from app.embeddings.base import EmbeddingStrategy
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pipeline import IngestionPipeline
from app.repositories.vector_repository import VectorRepository
from app.schemas import DocumentChunk, RetrievedChunk


class FakeEmbeddings(EmbeddingStrategy):
    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeRepository(VectorRepository):
    def __init__(self) -> None:
        self.ensured = None
        self.upserted_chunks: list[DocumentChunk] = []
        self.upserted_embeddings: list[list[float]] = []

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        self.ensured = (vector_size, recreate)

    def recreate_collection(self, vector_size: int) -> None:
        self.ensured = (vector_size, True)

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self.upserted_chunks = chunks
        self.upserted_embeddings = embeddings

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        return []

    def count_points(self) -> int:
        return len(self.upserted_chunks)


def test_ingestion_pipeline_orchestrates_chunking_embeddings_and_repository(tmp_path: Path) -> None:
    documents_path = tmp_path / "processed.jsonl"
    documents_path.write_text(
        '{'
        '"document_id":"doc-1",'
        '"url":"https://www.bancolombia.com/personas",'
        '"title":"Personas",'
        '"content":"Linea uno\\\\nLinea dos con informacion",'
        '"section":"personas",'
        '"scraped_at":"2026-01-01T00:00:00Z",'
        '"metadata":{"source":"bancolombia.com"}'
        '}\n',
        encoding="utf-8",
    )
    repository = FakeRepository()
    pipeline = IngestionPipeline(
        document_path=str(documents_path),
        chunker=DocumentChunker(chunk_size=200, chunk_overlap=20),
        embeddings=FakeEmbeddings(),
        repository=repository,
        embedding_provider="fake",
        embedding_model="fake-model",
    )

    result = pipeline.run()

    assert result.documents_loaded == 1
    assert result.chunks_generated == 1
    assert result.vector_dimension == 3
    assert result.chunks_indexed == 1
    assert repository.ensured == (3, False)
    assert len(repository.upserted_chunks) == 1
    assert repository.upserted_embeddings == [[1.0, 0.0, 0.0]]
