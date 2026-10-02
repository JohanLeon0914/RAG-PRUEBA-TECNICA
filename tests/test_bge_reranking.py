from datetime import UTC, datetime

import pytest

from app.reranking.bge import BGERerankingStrategy
from app.schemas import RetrievedChunk


class FakeCrossEncoder:
    instances = 0

    def __init__(self, model_name: str, device: str, trust_remote_code: bool) -> None:
        FakeCrossEncoder.instances += 1
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code

    def predict(self, pairs, batch_size: int, show_progress_bar: bool):
        if any(text == "boom" for _, text in pairs):
            raise RuntimeError("reranker failed")
        scores_by_text = {
            "b": 0.9,
            "a": 0.2,
            "c": 0.5,
        }
        return [scores_by_text[text] for _, text in pairs]


def chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        url="https://www.bancolombia.com/personas/cuentas",
        title="Titulo",
        section="personas",
        chunk_index=0,
        scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"source": "test"},
        score=score,
    )


def patch_cross_encoder(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoder)


def test_bge_reranker_orders_by_reranker_score_and_preserves_payload(monkeypatch) -> None:
    FakeCrossEncoder.instances = 0
    patch_cross_encoder(monkeypatch)
    strategy = BGERerankingStrategy(model_name="fake-reranker", batch_size=2, device="cpu")
    candidates = [chunk("a", "a", 0.99), chunk("b", "b", 0.10), chunk("c", "c", 0.50)]

    reranked = strategy.rerank("consulta", candidates, top_n=2)

    assert [item.chunk_id for item in reranked] == ["b", "c"]
    assert [item.rerank_score for item in reranked] == [0.9, 0.5]
    assert reranked[0].score == 0.10
    assert reranked[0].metadata == {"source": "test"}
    assert FakeCrossEncoder.instances == 1


def test_bge_reranker_returns_all_candidates_when_top_n_is_larger(monkeypatch) -> None:
    patch_cross_encoder(monkeypatch)
    strategy = BGERerankingStrategy(model_name="fake-reranker")

    reranked = strategy.rerank("consulta", [chunk("a", "a", 0.1)], top_n=5)

    assert [item.chunk_id for item in reranked] == ["a"]


def test_bge_reranker_handles_empty_candidates_without_loading_model(monkeypatch) -> None:
    FakeCrossEncoder.instances = 0
    patch_cross_encoder(monkeypatch)
    strategy = BGERerankingStrategy(model_name="fake-reranker")

    assert strategy.rerank("consulta", [], top_n=5) == []
    assert FakeCrossEncoder.instances == 0


def test_bge_reranker_rejects_invalid_top_n(monkeypatch) -> None:
    patch_cross_encoder(monkeypatch)
    strategy = BGERerankingStrategy(model_name="fake-reranker")

    with pytest.raises(ValueError, match="top_n must be positive"):
        strategy.rerank("consulta", [chunk("a", "a", 0.1)], top_n=0)


def test_bge_reranker_loads_model_once(monkeypatch) -> None:
    FakeCrossEncoder.instances = 0
    patch_cross_encoder(monkeypatch)
    strategy = BGERerankingStrategy(model_name="fake-reranker")

    strategy.rerank("consulta", [chunk("a", "a", 0.1)], top_n=1)
    strategy.rerank("consulta", [chunk("b", "b", 0.1)], top_n=1)

    assert FakeCrossEncoder.instances == 1


def test_bge_reranker_does_not_silence_model_errors(monkeypatch) -> None:
    patch_cross_encoder(monkeypatch)
    strategy = BGERerankingStrategy(model_name="fake-reranker")

    with pytest.raises(RuntimeError, match="reranker failed"):
        strategy.rerank("consulta", [chunk("x", "boom", 0.1)], top_n=1)
