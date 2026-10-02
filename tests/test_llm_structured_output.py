import pytest

from app.llm.base import parse_structured_llm_output


def test_parse_structured_llm_output_supported() -> None:
    output = parse_structured_llm_output(
        '{"answer":"Respuesta fundamentada","supported_by_context":true}'
    )

    assert output.answer == "Respuesta fundamentada"
    assert output.supported_by_context is True


def test_parse_structured_llm_output_unsupported_with_markdown_fence() -> None:
    output = parse_structured_llm_output(
        '```json\n{"answer":"No hay evidencia suficiente","supported_by_context":false}\n```'
    )

    assert output.answer == "No hay evidencia suficiente"
    assert output.supported_by_context is False


def test_parse_structured_llm_output_rejects_invalid_json() -> None:
    with pytest.raises(RuntimeError, match="invalid structured output"):
        parse_structured_llm_output("No encontré información suficiente")


def test_parse_structured_llm_output_rejects_missing_support_flag() -> None:
    with pytest.raises(RuntimeError, match="invalid structured output"):
        parse_structured_llm_output('{"answer":"Respuesta"}')
