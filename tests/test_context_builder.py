import pytest

from app.rag.context_builder import ContextBuilder
from tests.helpers import retrieved_chunk


def test_context_builder_formats_sources_without_debug_metadata() -> None:
    context = ContextBuilder(max_chunks=2).build(
        [
            retrieved_chunk(chunk_id="a", text="Primer contenido", score=0.99),
            retrieved_chunk(chunk_id="b", text="Segundo contenido", score=0.88),
        ]
    )

    assert "[SOURCE 1]" in context
    assert "Title: Titulo" in context
    assert "URL: https://www.bancolombia.com/personas/producto" in context
    assert "Primer contenido" in context
    assert "Segundo contenido" in context
    assert "chunk_id" not in context
    assert "0.99" not in context


def test_context_builder_respects_top_k_and_deduplicates_chunks() -> None:
    context = ContextBuilder(max_chunks=1).build(
        [
            retrieved_chunk(chunk_id="a", text="Duplicado"),
            retrieved_chunk(chunk_id="a", text="Duplicado"),
            retrieved_chunk(chunk_id="b", text="No debe entrar"),
        ]
    )

    assert context.count("[SOURCE") == 1
    assert "Duplicado" in context
    assert "No debe entrar" not in context


def test_context_builder_handles_empty_context() -> None:
    assert ContextBuilder(max_chunks=2).build([]) == ""


def test_context_builder_rejects_invalid_max_chunks() -> None:
    with pytest.raises(ValueError, match="max_chunks must be positive"):
        ContextBuilder(max_chunks=0)
