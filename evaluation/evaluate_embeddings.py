import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.embeddings.base import EmbeddingStrategy
from app.embeddings.bge import BGEM3EmbeddingStrategy
from app.embeddings.gte import GTEMultilingualEmbeddingStrategy
from app.ingestion.chunker import DocumentChunker
from app.repositories.qdrant_repository import QdrantVectorRepository
from app.scraping.storage import load_documents_jsonl
from evaluation.retrieval_metrics import (
    aggregate_by_category,
    aggregate_metrics,
    first_relevant_rank,
    hit_at_k,
    reciprocal_rank,
    validate_dataset,
)

DATASET_PATH = "evaluation/retrieval_dataset.json"
SUMMARY_PATH = "evaluation/summary.json"
OUT_OF_DOMAIN_PATH = "evaluation/out_of_domain_results.json"
CORPUS_PATH = "data/corpus/processed_documents.jsonl"
MANIFEST_PATH = "data/corpus/manifest.json"
TOP_K = 5
OUT_OF_DOMAIN_QUERIES = [
    "¿Cuál es la capital de Japón?",
    "¿Cómo preparo una lasaña vegetariana?",
    "¿Qué es una función lambda en Python?",
    "¿Quién ganó el mundial de fútbol de 2014?",
    "¿Cómo se calcula la órbita de Marte?",
]


def main() -> None:
    documents = load_documents_jsonl(CORPUS_PATH)
    manifest = read_json(MANIFEST_PATH)
    chunks = DocumentChunker(chunk_size=900, chunk_overlap=150).chunk(documents)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    source_urls = {entry["url"] for entry in manifest}
    corpus_fingerprint = fingerprint_chunk_ids([chunk.chunk_id for chunk in chunks])

    dataset_payload = read_json(DATASET_PATH)
    questions = validate_dataset(dataset_payload, chunk_ids=chunk_ids, source_urls=source_urls)

    experiments = [
        {
            "name": "bge",
            "provider": "bge",
            "model": "BAAI/bge-m3",
            "collection": "bancolombia_eval_bge_m3",
            "results_path": "evaluation/results_bge.json",
            "strategy": lambda: BGEM3EmbeddingStrategy(
                model_name="BAAI/bge-m3",
                batch_size=16,
                device="cpu",
            ),
        },
        {
            "name": "gte",
            "provider": "gte",
            "model": "Alibaba-NLP/gte-multilingual-base",
            "collection": "bancolombia_eval_gte_multilingual",
            "results_path": "evaluation/results_gte.json",
            "strategy": lambda: GTEMultilingualEmbeddingStrategy(
                model_name="Alibaba-NLP/gte-multilingual-base",
                batch_size=16,
                device="cpu",
            ),
        },
    ]

    summaries: dict[str, Any] = {}
    out_of_domain_results: dict[str, list[dict[str, Any]]] = {}

    for experiment in experiments:
        print(f"Running {experiment['name']} ({experiment['model']})")
        strategy: EmbeddingStrategy = experiment["strategy"]()
        repository = QdrantVectorRepository(
            url="http://localhost:6333",
            collection_name=experiment["collection"],
        )

        model_load_started = time.perf_counter()
        dimension = strategy.dimension
        model_load_time = time.perf_counter() - model_load_started

        repository.recreate_collection(vector_size=dimension)
        document_vectors = strategy.embed_documents([chunk.text for chunk in chunks])
        repository.upsert_chunks(chunks, document_vectors)
        points = repository.count_points()

        warmup_started = time.perf_counter()
        strategy.embed_query("consulta de calentamiento para retrieval")
        warmup_time = time.perf_counter() - warmup_started

        query_results = run_queries(strategy, repository, questions)
        write_json(query_results, experiment["results_path"])

        ood_results = run_out_of_domain(strategy, repository)
        out_of_domain_results[experiment["name"]] = ood_results

        summaries[experiment["name"]] = {
            "provider": experiment["provider"],
            "model": experiment["model"],
            "collection": experiment["collection"],
            "dimension": dimension,
            "distance": "Cosine",
            "points": points,
            "model_load_time": round(model_load_time, 4),
            "warmup_time": round(warmup_time, 4),
            "metrics": aggregate_metrics(query_results),
            "metrics_by_category": aggregate_by_category(query_results),
        }

    write_json(out_of_domain_results, OUT_OF_DOMAIN_PATH)

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus_fingerprint": corpus_fingerprint,
        "number_of_documents": len(documents),
        "number_of_chunks": len(chunks),
        "chunk_size": 900,
        "chunk_overlap": 150,
        "dataset_size": len(questions),
        "dataset_distribution": category_distribution(dataset_payload),
        "experiments": summaries,
        "comparison": compare_results(
            read_json("evaluation/results_bge.json"),
            read_json("evaluation/results_gte.json"),
        ),
        "out_of_domain": out_of_domain_results,
    }
    write_json(summary, SUMMARY_PATH)
    print_summary(summary)


def run_queries(
    strategy: EmbeddingStrategy,
    repository: QdrantVectorRepository,
    questions,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for question in questions:
        embedding_started = time.perf_counter()
        query_vector = strategy.embed_query(question.question)
        embedding_latency = time.perf_counter() - embedding_started

        search_started = time.perf_counter()
        retrieved = repository.search(query_vector=query_vector, top_k=TOP_K)
        search_latency = time.perf_counter() - search_started

        retrieved_ids = [chunk.chunk_id for chunk in retrieved]
        relevant = set(question.relevant_chunk_ids)
        rank = first_relevant_rank(retrieved_ids, relevant)
        results.append(
            {
                "question_id": question.id,
                "question": question.question,
                "category": question.category,
                "relevant_chunk_ids": question.relevant_chunk_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "scores": [chunk.score for chunk in retrieved],
                "first_relevant_rank": rank,
                "hit_at_1": hit_at_k(retrieved_ids, relevant, 1),
                "hit_at_3": hit_at_k(retrieved_ids, relevant, 3),
                "hit_at_5": hit_at_k(retrieved_ids, relevant, 5),
                "reciprocal_rank": reciprocal_rank(rank),
                "embedding_latency": round(embedding_latency, 6),
                "search_latency": round(search_latency, 6),
            }
        )
    return results


def run_out_of_domain(
    strategy: EmbeddingStrategy,
    repository: QdrantVectorRepository,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for query in OUT_OF_DOMAIN_QUERIES:
        query_vector = strategy.embed_query(query)
        retrieved = repository.search(query_vector=query_vector, top_k=1)
        top = retrieved[0] if retrieved else None
        observations.append(
            {
                "query": query,
                "top_score": top.score if top else None,
                "top_chunk_id": top.chunk_id if top else None,
                "top_title": top.title if top else None,
                "top_url": str(top.url) if top else None,
            }
        )
    return observations


def compare_results(bge_results: list[dict[str, Any]], gte_results: list[dict[str, Any]]) -> dict:
    bge_by_id = {result["question_id"]: result for result in bge_results}
    gte_by_id = {result["question_id"]: result for result in gte_results}
    both_top_1: list[str] = []
    bge_better: list[str] = []
    gte_better: list[str] = []
    neither_hit_5: list[str] = []

    for question_id in sorted(bge_by_id):
        bge_rank = bge_by_id[question_id]["first_relevant_rank"]
        gte_rank = gte_by_id[question_id]["first_relevant_rank"]
        if bge_rank == 1 and gte_rank == 1:
            both_top_1.append(question_id)
        if bge_rank is None and gte_rank is None:
            neither_hit_5.append(question_id)
            continue
        if gte_rank is None or (bge_rank is not None and bge_rank < gte_rank):
            bge_better.append(question_id)
        elif bge_rank is None or (gte_rank is not None and gte_rank < bge_rank):
            gte_better.append(question_id)

    return {
        "both_top_1": both_top_1,
        "bge_better": bge_better,
        "gte_better": gte_better,
        "neither_hit_5": neither_hit_5,
    }


def fingerprint_chunk_ids(chunk_ids: list[str]) -> str:
    payload = "\n".join(sorted(chunk_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def category_distribution(questions: list[dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for question in questions:
        category = question["category"]
        distribution[category] = distribution.get(category, 0) + 1
    return dict(sorted(distribution.items()))


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Any, path: str) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"corpus fingerprint: {summary['corpus_fingerprint']}")
    print(f"documents: {summary['number_of_documents']}")
    print(f"chunks: {summary['number_of_chunks']}")
    print(f"dataset size: {summary['dataset_size']}")
    for name, experiment in summary["experiments"].items():
        metrics = experiment["metrics"]
        print(
            f"{name}: dimension={experiment['dimension']} points={experiment['points']} "
            f"hit@1={metrics['hit_at_1']} hit@3={metrics['hit_at_3']} "
            f"hit@5={metrics['hit_at_5']} mrr={metrics['mrr']}"
        )


if __name__ == "__main__":
    main()
