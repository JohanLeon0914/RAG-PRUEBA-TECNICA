import numpy as np

from app.embeddings.bge import BGEM3EmbeddingStrategy


class FakeSentenceTransformer:
    instances = 0

    def __init__(self, model_name: str, device: str, trust_remote_code: bool, **kwargs) -> None:
        FakeSentenceTransformer.instances += 1
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.kwargs = kwargs

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])
        return np.array([[float(index), 1.0, 0.0] for index, _ in enumerate(texts)])

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_embedding_strategy_loads_model_once_and_batches(monkeypatch) -> None:
    FakeSentenceTransformer.instances = 0
    calls = []

    def fake_sentence_transformer(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeSentenceTransformer(*args, **kwargs)

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        fake_sentence_transformer,
    )

    strategy = BGEM3EmbeddingStrategy(model_name="fake-model", batch_size=2, device="cpu")

    assert strategy.dimension == 3
    assert strategy.embed_documents(["a", "b", "c"]) == [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    assert strategy.embed_query("consulta") == [1.0, 0.0, 0.0]
    assert FakeSentenceTransformer.instances == 1
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["model_kwargs"] == {"use_safetensors": False}
