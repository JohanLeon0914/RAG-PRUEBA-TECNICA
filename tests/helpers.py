from datetime import UTC, datetime

from app.schemas import RetrievedChunk


def retrieved_chunk(
    chunk_id: str = "doc-1:0:abc",
    text: str = "Contenido relevante de prueba",
    score: float = 0.91,
    title: str = "Titulo",
    url: str = "https://www.bancolombia.com/personas/producto",
    chunk_index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        url=url,
        title=title,
        section="personas",
        chunk_index=chunk_index,
        scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"source": "test"},
        score=score,
    )
