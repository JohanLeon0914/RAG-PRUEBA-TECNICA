from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

VALID_CATEGORIES = {
    "cuentas",
    "tarjetas_credito",
    "tarjetas_debito",
    "prestamos",
    "vivienda",
    "seguros",
    "inversiones",
}


@dataclass(frozen=True)
class RetrievalQuestion:
    id: str
    question: str
    category: str
    relevant_chunk_ids: list[str]
    answer_evidence: str
    source_url: str


def validate_dataset(
    questions: list[dict[str, Any]],
    chunk_ids: set[str],
    source_urls: set[str],
) -> list[RetrievalQuestion]:
    ids = [item.get("id", "") for item in questions]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicated question ids: {duplicates}")

    question_texts = [item.get("question", "") for item in questions]
    duplicate_questions = [item for item, count in Counter(question_texts).items() if count > 1]
    if duplicate_questions:
        raise ValueError(f"Duplicated questions: {duplicate_questions}")

    validated: list[RetrievalQuestion] = []
    for item in questions:
        question_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        category = str(item.get("category", "")).strip()
        relevant_chunk_ids = item.get("relevant_chunk_ids", [])
        source_url = str(item.get("source_url", "")).strip()
        answer_evidence = str(item.get("answer_evidence", "")).strip()

        if not question_id:
            raise ValueError("Question id must not be empty")
        if not question:
            raise ValueError(f"Question {question_id} must not be empty")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Question {question_id} has invalid category: {category}")
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise ValueError(f"Question {question_id} must have relevant_chunk_ids")
        missing_chunks = sorted(set(relevant_chunk_ids) - chunk_ids)
        if missing_chunks:
            raise ValueError(f"Question {question_id} references missing chunks: {missing_chunks}")
        if source_url not in source_urls:
            raise ValueError(f"Question {question_id} source_url is not in corpus: {source_url}")
        if not answer_evidence:
            raise ValueError(f"Question {question_id} must have answer_evidence")

        validated.append(
            RetrievalQuestion(
                id=question_id,
                question=question,
                category=category,
                relevant_chunk_ids=relevant_chunk_ids,
                answer_evidence=answer_evidence,
                source_url=source_url,
            )
        )

    return validated


def first_relevant_rank(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str]) -> int | None:
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return index
    return None


def hit_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str], k: int) -> bool:
    return first_relevant_rank(retrieved_chunk_ids[:k], relevant_chunk_ids) is not None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
            "mean_embedding_latency": 0.0,
            "median_embedding_latency": 0.0,
            "mean_search_latency": 0.0,
            "median_search_latency": 0.0,
        }

    embedding_latencies = [float(result["embedding_latency"]) for result in results]
    search_latencies = [float(result["search_latency"]) for result in results]
    return {
        "hit_at_1": round(mean(float(result["hit_at_1"]) for result in results), 4),
        "hit_at_3": round(mean(float(result["hit_at_3"]) for result in results), 4),
        "hit_at_5": round(mean(float(result["hit_at_5"]) for result in results), 4),
        "mrr": round(mean(float(result["reciprocal_rank"]) for result in results), 4),
        "mean_embedding_latency": round(mean(embedding_latencies), 4),
        "median_embedding_latency": round(median(embedding_latencies), 4),
        "mean_search_latency": round(mean(search_latencies), 4),
        "median_search_latency": round(median(search_latencies), 4),
    }


def aggregate_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)

    return {
        category: {"n": len(category_results), **aggregate_metrics(category_results)}
        for category, category_results in sorted(grouped.items())
    }
