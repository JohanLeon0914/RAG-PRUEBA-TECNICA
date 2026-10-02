import pytest

from evaluation.retrieval_metrics import (
    aggregate_by_category,
    aggregate_metrics,
    first_relevant_rank,
    hit_at_k,
    reciprocal_rank,
    validate_dataset,
)


def make_question(**overrides):
    payload = {
        "id": "q001",
        "question": "¿Cuál es el costo?",
        "category": "cuentas",
        "relevant_chunk_ids": ["chunk-1"],
        "answer_evidence": "Evidencia",
        "source_url": "https://www.bancolombia.com/personas/cuentas",
    }
    payload.update(overrides)
    return payload


def test_validate_dataset_accepts_valid_questions() -> None:
    questions = [
        make_question(),
        make_question(
            id="q002",
            question="¿Qué beneficios tiene?",
            relevant_chunk_ids=["chunk-2"],
        ),
    ]

    validated = validate_dataset(
        questions,
        chunk_ids={"chunk-1", "chunk-2"},
        source_urls={"https://www.bancolombia.com/personas/cuentas"},
    )

    assert [question.id for question in validated] == ["q001", "q002"]


def test_validate_dataset_rejects_missing_chunk_ids() -> None:
    with pytest.raises(ValueError, match="missing chunks"):
        validate_dataset(
            [make_question(relevant_chunk_ids=["missing"])],
            chunk_ids={"chunk-1"},
            source_urls={"https://www.bancolombia.com/personas/cuentas"},
        )


def test_validate_dataset_rejects_duplicate_questions() -> None:
    with pytest.raises(ValueError, match="Duplicated questions"):
        validate_dataset(
            [make_question(id="q001"), make_question(id="q002")],
            chunk_ids={"chunk-1"},
            source_urls={"https://www.bancolombia.com/personas/cuentas"},
        )


def test_first_relevant_rank_supports_multiple_relevant_chunks() -> None:
    rank = first_relevant_rank(["a", "b", "c"], {"c", "z"})

    assert rank == 3
    assert hit_at_k(["a", "b", "c"], {"c", "z"}, 2) is False
    assert hit_at_k(["a", "b", "c"], {"c", "z"}, 3) is True
    assert reciprocal_rank(rank) == pytest.approx(1 / 3)


def test_first_relevant_rank_returns_none_without_hits() -> None:
    rank = first_relevant_rank(["a", "b"], {"x"})

    assert rank is None
    assert reciprocal_rank(rank) == 0.0


def test_aggregate_metrics_handles_hits_and_misses() -> None:
    results = [
        {
            "hit_at_1": True,
            "hit_at_3": True,
            "hit_at_5": True,
            "reciprocal_rank": 1.0,
            "embedding_latency": 0.10,
            "search_latency": 0.01,
        },
        {
            "hit_at_1": False,
            "hit_at_3": False,
            "hit_at_5": True,
            "reciprocal_rank": 0.25,
            "embedding_latency": 0.20,
            "search_latency": 0.03,
        },
    ]

    metrics = aggregate_metrics(results)

    assert metrics["hit_at_1"] == 0.5
    assert metrics["hit_at_3"] == 0.5
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 0.625
    assert metrics["mean_embedding_latency"] == 0.15
    assert metrics["median_search_latency"] == 0.02


def test_aggregate_by_category_reports_n() -> None:
    results = [
        {
            "category": "cuentas",
            "hit_at_1": True,
            "hit_at_3": True,
            "hit_at_5": True,
            "reciprocal_rank": 1.0,
            "embedding_latency": 0.10,
            "search_latency": 0.01,
        },
        {
            "category": "seguros",
            "hit_at_1": False,
            "hit_at_3": False,
            "hit_at_5": False,
            "reciprocal_rank": 0.0,
            "embedding_latency": 0.20,
            "search_latency": 0.03,
        },
    ]

    by_category = aggregate_by_category(results)

    assert by_category["cuentas"]["n"] == 1
    assert by_category["cuentas"]["hit_at_1"] == 1.0
    assert by_category["seguros"]["n"] == 1
    assert by_category["seguros"]["mrr"] == 0.0
