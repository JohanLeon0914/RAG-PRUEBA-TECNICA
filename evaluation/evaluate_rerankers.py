import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.embeddings.bge import BGEM3EmbeddingStrategy
from app.ingestion.chunker import DocumentChunker
from app.repositories.qdrant_repository import QdrantVectorRepository
from app.reranking.bge import BGERerankingStrategy
from app.scraping.storage import load_documents_jsonl
from evaluation.evaluate_embeddings import fingerprint_chunk_ids
from evaluation.retrieval_metrics import (
    aggregate_metrics,
    first_relevant_rank,
    hit_at_k,
    reciprocal_rank,
    validate_dataset,
)

DATASET_PATH = "evaluation/retrieval_dataset.json"
CORPUS_PATH = "data/corpus/processed_documents.jsonl"
MANIFEST_PATH = "data/corpus/manifest.json"
RESULTS_PATH = "evaluation/results_reranker.json"
SUMMARY_PATH = "evaluation/reranker_summary.json"
BGE_COLLECTION = "bancolombia_eval_bge_m3"


def main() -> None:
    settings = Settings()
    candidate_k = settings.retrieval_candidate_k
    final_k = settings.rerank_top_k

    documents = load_documents_jsonl(CORPUS_PATH)
    chunks = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).chunk(documents)
    corpus_fingerprint = fingerprint_chunk_ids([chunk.chunk_id for chunk in chunks])
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    manifest = read_json(MANIFEST_PATH)
    dataset_payload = read_json(DATASET_PATH)
    questions = validate_dataset(
        dataset_payload,
        chunk_ids=chunk_ids,
        source_urls={entry["url"] for entry in manifest},
    )

    embeddings = BGEM3EmbeddingStrategy(
        model_name="BAAI/bge-m3",
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )
    repository = QdrantVectorRepository(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=BGE_COLLECTION,
    )
    reranker = BGERerankingStrategy(
        model_name=settings.reranker_model,
        batch_size=settings.reranker_batch_size,
        device=settings.reranker_device,
    )

    reranker_load_started = time.perf_counter()
    reranker._load_model()
    reranker_model_load_time = time.perf_counter() - reranker_load_started

    warmup_candidates = repository.search(
        query_vector=embeddings.embed_query("consulta de calentamiento para reranking"),
        top_k=min(candidate_k, 3),
    )
    warmup_started = time.perf_counter()
    reranker.rerank("consulta de calentamiento para reranking", warmup_candidates, top_n=1)
    reranker_warmup_time = time.perf_counter() - warmup_started

    results = evaluate_questions(
        questions=questions,
        embeddings=embeddings,
        repository=repository,
        reranker=reranker,
        candidate_k=candidate_k,
        final_k=final_k,
    )
    write_json(results, RESULTS_PATH)

    baseline_results = [to_baseline_metric_result(result) for result in results]
    reranked_results = [to_reranked_metric_result(result) for result in results]
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus_fingerprint": corpus_fingerprint,
        "dataset_size": len(questions),
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": settings.reranker_model,
        "candidate_k": candidate_k,
        "final_k": final_k,
        "candidate_recall_at_15": round(
            sum(bool(result["candidate_hit"]) for result in results) / len(results),
            4,
        ),
        "baseline_metrics": aggregate_metrics(baseline_results),
        "reranked_metrics": aggregate_metrics(reranked_results),
        "latency": latency_summary(results, reranker_model_load_time, reranker_warmup_time),
        "comparison": compare_dense_vs_reranked(results),
    }
    write_json(summary, SUMMARY_PATH)
    print_summary(summary)


def evaluate_questions(
    questions,
    embeddings: BGEM3EmbeddingStrategy,
    repository: QdrantVectorRepository,
    reranker: BGERerankingStrategy,
    candidate_k: int,
    final_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for question in questions:
        total_started = time.perf_counter()
        embedding_started = time.perf_counter()
        query_vector = embeddings.embed_query(question.question)
        embedding_latency = time.perf_counter() - embedding_started

        search_started = time.perf_counter()
        dense_candidates = repository.search(query_vector=query_vector, top_k=candidate_k)
        search_latency = time.perf_counter() - search_started

        rerank_started = time.perf_counter()
        reranked = reranker.rerank(question.question, dense_candidates, top_n=final_k)
        rerank_latency = time.perf_counter() - rerank_started
        total_latency = time.perf_counter() - total_started

        dense_ids = [chunk.chunk_id for chunk in dense_candidates]
        dense_top_final_ids = dense_ids[:final_k]
        reranked_ids = [chunk.chunk_id for chunk in reranked]
        relevant = set(question.relevant_chunk_ids)
        dense_rank = first_relevant_rank(dense_ids, relevant)
        baseline_rank = first_relevant_rank(dense_top_final_ids, relevant)
        reranked_rank = first_relevant_rank(reranked_ids, relevant)

        results.append(
            {
                "question_id": question.id,
                "question": question.question,
                "category": question.category,
                "relevant_chunk_ids": question.relevant_chunk_ids,
                "dense_top15_ids": dense_ids,
                "dense_scores": [chunk.score for chunk in dense_candidates],
                "reranked_top5_ids": reranked_ids,
                "reranker_scores": [chunk.rerank_score for chunk in reranked],
                "first_relevant_dense_rank": dense_rank,
                "first_relevant_baseline_rank": baseline_rank,
                "first_relevant_reranked_rank": reranked_rank,
                "candidate_hit": dense_rank is not None,
                "baseline_hit_at_1": hit_at_k(dense_top_final_ids, relevant, 1),
                "baseline_hit_at_3": hit_at_k(dense_top_final_ids, relevant, 3),
                "baseline_hit_at_5": hit_at_k(dense_top_final_ids, relevant, 5),
                "reranked_hit_at_1": hit_at_k(reranked_ids, relevant, 1),
                "reranked_hit_at_3": hit_at_k(reranked_ids, relevant, 3),
                "reranked_hit_at_5": hit_at_k(reranked_ids, relevant, 5),
                "embedding_latency": round(embedding_latency, 6),
                "search_latency": round(search_latency, 6),
                "rerank_latency": round(rerank_latency, 6),
                "total_latency": round(total_latency, 6),
            }
        )
    return results


def to_baseline_metric_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_at_1": result["baseline_hit_at_1"],
        "hit_at_3": result["baseline_hit_at_3"],
        "hit_at_5": result["baseline_hit_at_5"],
        "reciprocal_rank": reciprocal_rank(result["first_relevant_baseline_rank"]),
        "embedding_latency": result["embedding_latency"],
        "search_latency": result["search_latency"],
    }


def to_reranked_metric_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_at_1": result["reranked_hit_at_1"],
        "hit_at_3": result["reranked_hit_at_3"],
        "hit_at_5": result["reranked_hit_at_5"],
        "reciprocal_rank": reciprocal_rank(result["first_relevant_reranked_rank"]),
        "embedding_latency": result["embedding_latency"],
        "search_latency": result["search_latency"],
    }


def latency_summary(
    results: list[dict[str, Any]],
    reranker_model_load_time: float,
    reranker_warmup_time: float,
) -> dict[str, float]:
    from statistics import mean, median

    rerank_latencies = [result["rerank_latency"] for result in results]
    total_latencies = [result["total_latency"] for result in results]
    return {
        "reranker_model_load_time": round(reranker_model_load_time, 4),
        "reranker_warmup_time": round(reranker_warmup_time, 4),
        "mean_rerank_latency": round(mean(rerank_latencies), 4),
        "median_rerank_latency": round(median(rerank_latencies), 4),
        "mean_total_latency": round(mean(total_latencies), 4),
        "median_total_latency": round(median(total_latencies), 4),
    }


def compare_dense_vs_reranked(results: list[dict[str, Any]]) -> dict[str, list[str] | int]:
    improvements: list[str] = []
    unchanged: list[str] = []
    regressions: list[str] = []
    top1_regressions: list[str] = []
    missing_candidates: list[str] = []

    for result in results:
        baseline_rank = result["first_relevant_baseline_rank"]
        reranked_rank = result["first_relevant_reranked_rank"]
        if result["first_relevant_dense_rank"] is None:
            missing_candidates.append(result["question_id"])

        if baseline_rank == 1 and reranked_rank != 1:
            top1_regressions.append(result["question_id"])

        if baseline_rank is None and reranked_rank is None:
            unchanged.append(result["question_id"])
        elif baseline_rank is None:
            improvements.append(result["question_id"])
        elif reranked_rank is None:
            regressions.append(result["question_id"])
        elif reranked_rank < baseline_rank:
            improvements.append(result["question_id"])
        elif reranked_rank > baseline_rank:
            regressions.append(result["question_id"])
        else:
            unchanged.append(result["question_id"])

    return {
        "improvements": improvements,
        "unchanged": unchanged,
        "regressions": regressions,
        "top1_regressions": top1_regressions,
        "missing_candidates": missing_candidates,
        "improvements_count": len(improvements),
        "unchanged_count": len(unchanged),
        "regressions_count": len(regressions),
    }


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Any, path: str) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    print(f"candidate_k: {summary['candidate_k']}")
    print(f"final_k: {summary['final_k']}")
    print(f"candidate recall: {summary['candidate_recall_at_15']}")
    print(f"baseline: {summary['baseline_metrics']}")
    print(f"reranked: {summary['reranked_metrics']}")
    print(f"comparison: {summary['comparison']}")


if __name__ == "__main__":
    main()
