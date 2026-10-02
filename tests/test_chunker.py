from datetime import UTC, datetime

from app.ingestion.chunker import DocumentChunker
from app.schemas import SourceDocument


def make_document(content: str, document_id: str = "doc-1") -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        url="https://www.bancolombia.com/personas/productos/tarjetas-credito",
        title="Tarjetas",
        section="personas",
        content=content,
        scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"source": "bancolombia.com"},
    )


def test_short_document_produces_single_chunk() -> None:
    chunks = DocumentChunker(chunk_size=200, chunk_overlap=20).chunk(
        [make_document("Tarjetas de crédito\nBeneficios principales")]
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Tarjetas de crédito\nBeneficios principales"
    assert chunks[0].chunk_index == 0


def test_long_document_preserves_order_and_avoids_empty_chunks() -> None:
    document = make_document(
        "\n".join(f"Parrafo {index} con informacion útil." for index in range(8))
    )

    chunks = DocumentChunker(chunk_size=90, chunk_overlap=20).chunk([document])

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text.strip() for chunk in chunks)
    assert "Parrafo 0" in chunks[0].text
    assert "Parrafo 7" in chunks[-1].text


def test_overlap_reuses_suffix_without_cutting_words() -> None:
    document = make_document(
        "Primera sección con beneficios importantes.\n"
        "Segunda sección con requisitos claros.\n"
        "Tercera sección con condiciones aplicables."
    )

    chunks = DocumentChunker(chunk_size=80, chunk_overlap=30).chunk([document])

    assert len(chunks) == 3
    assert chunks[1].text.startswith("con beneficios importantes.")
    assert not chunks[1].text.startswith(" ")


def test_chunk_preserves_metadata() -> None:
    document = make_document("Contenido útil")
    chunk = DocumentChunker(chunk_size=50, chunk_overlap=5).chunk([document])[0]

    assert chunk.document_id == document.document_id
    assert chunk.url == document.url
    assert chunk.title == document.title
    assert chunk.section == document.section
    assert chunk.scraped_at == document.scraped_at
    assert chunk.metadata["source"] == "bancolombia.com"
    assert chunk.metadata["chunk_size_unit"] == "characters"


def test_chunk_ids_are_deterministic_and_depend_on_content() -> None:
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)

    first = chunker.chunk([make_document("Contenido estable")])[0]
    second = chunker.chunk([make_document("Contenido estable")])[0]
    changed = chunker.chunk([make_document("Contenido cambiado")])[0]

    assert first.chunk_id == second.chunk_id
    assert first.chunk_id != changed.chunk_id
