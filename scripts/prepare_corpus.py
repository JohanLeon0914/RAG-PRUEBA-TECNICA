import json
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from app.config.settings import get_settings
from app.corpus.selection import (
    chunk_size_stats,
    content_hash,
    document_size_stats,
    parse_sitemap_urls,
    select_corpus_urls,
    write_manifest,
)
from app.ingestion.chunker import DocumentChunker
from app.schemas import SourceDocument
from app.scraping.cleaner import BankContentCleaner
from app.scraping.scraper import BancolombiaScraper
from app.scraping.storage import save_documents_jsonl

SITEMAP_PERSONAS_URL = "https://www.bancolombia.com/sitemap-personas.xml"
RAW_CORPUS_PATH = "data/corpus/raw_documents.jsonl"
PROCESSED_CORPUS_PATH = "data/corpus/processed_documents.jsonl"
MANIFEST_PATH = "data/corpus/manifest.json"
REPORT_PATH = "data/corpus/quality_report.json"
PER_CATEGORY_LIMIT = 8
MAX_TOTAL_URLS = 56
MIN_USEFUL_CONTENT_CHARS = 500


def main() -> None:
    settings = get_settings()
    sitemap_xml = fetch_sitemap(settings)
    sitemap_urls = parse_sitemap_urls(sitemap_xml)
    selection = select_corpus_urls(
        sitemap_urls,
        per_category_limit=PER_CATEGORY_LIMIT,
        max_total=MAX_TOTAL_URLS,
    )
    categories_by_url = {candidate.url: candidate.category for candidate in selection.selected}

    scraper = BancolombiaScraper(
        start_urls=[candidate.url for candidate in selection.selected],
        allowed_domain=settings.bank_allowed_domain,
        max_pages=len(selection.selected),
        timeout_seconds=settings.scraper_timeout_seconds,
        rate_limit_seconds=settings.scraper_rate_limit_seconds,
        user_agent=settings.scraper_user_agent,
        accept=settings.scraper_accept,
        accept_language=settings.scraper_accept_language,
        accept_encoding=settings.scraper_accept_encoding,
        http2=settings.scraper_http2,
        disallowed_patterns=settings.scraper_disallowed_patterns,
    )
    scrape_result = scraper.scrape_exact([candidate.url for candidate in selection.selected])
    raw_documents = scrape_result.documents
    save_documents_jsonl(raw_documents, RAW_CORPUS_PATH)

    cleaner = BankContentCleaner()
    processed_documents, processing_discards, exact_duplicates = clean_and_deduplicate(
        cleaner=cleaner,
        raw_documents=raw_documents,
    )
    save_documents_jsonl(processed_documents, PROCESSED_CORPUS_PATH)
    manifest = write_manifest(processed_documents, categories_by_url, MANIFEST_PATH)

    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk(processed_documents)

    report = build_report(
        sitemap_urls=sitemap_urls,
        selected_urls=[candidate.url for candidate in selection.selected],
        candidate_count=len(selection.candidates),
        raw_documents=raw_documents,
        processed_documents=processed_documents,
        processing_discards=processing_discards,
        selection_discards=selection.discarded,
        scrape_errors=scrape_result.errors,
        scrape_duplicates=scrape_result.duplicates,
        exact_duplicates=exact_duplicates,
        manifest=manifest,
        chunks_count=len(chunks),
        chunk_stats=chunk_size_stats(chunks),
        categories_by_url=categories_by_url,
    )
    write_json(report, REPORT_PATH)

    print_summary(report)


def fetch_sitemap(settings) -> str:
    headers = {
        "User-Agent": settings.scraper_user_agent,
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": settings.scraper_accept_language,
        "Accept-Encoding": settings.scraper_accept_encoding,
    }
    with httpx.Client(
        timeout=settings.scraper_timeout_seconds,
        follow_redirects=True,
        http2=settings.scraper_http2,
        headers=headers,
    ) as client:
        response = client.get(SITEMAP_PERSONAS_URL)
        response.raise_for_status()
        return response.text


def clean_and_deduplicate(
    cleaner: BankContentCleaner,
    raw_documents: list[SourceDocument],
) -> tuple[list[SourceDocument], list[dict[str, str]], list[dict[str, str]]]:
    processed_documents: list[SourceDocument] = []
    discards: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    seen_content_hashes: dict[str, SourceDocument] = {}

    for document in raw_documents:
        cleaned_text = cleaner.clean_text(document.content)
        if not cleaned_text:
            discards.append(
                {
                    "url": str(document.url),
                    "title": document.title,
                    "reason": "empty_after_cleaning",
                }
            )
            continue

        if len(cleaned_text) < MIN_USEFUL_CONTENT_CHARS:
            discards.append(
                {
                    "url": str(document.url),
                    "title": document.title,
                    "reason": "clean_content_too_small",
                    "size": str(len(cleaned_text)),
                }
            )
            continue

        digest = content_hash(cleaned_text)
        original = seen_content_hashes.get(digest)
        if original is not None:
            duplicates.append(
                {
                    "url": str(document.url),
                    "title": document.title,
                    "duplicate_of_url": str(original.url),
                    "content_hash": digest,
                }
            )
            discards.append(
                {
                    "url": str(document.url),
                    "title": document.title,
                    "reason": "exact_clean_content_duplicate",
                }
            )
            continue

        cleaned_document = document.model_copy(update={"content": cleaned_text})
        seen_content_hashes[digest] = cleaned_document
        processed_documents.append(cleaned_document)

    return processed_documents, discards, duplicates


def build_report(
    sitemap_urls: list[str],
    selected_urls: list[str],
    candidate_count: int,
    raw_documents: list[SourceDocument],
    processed_documents: list[SourceDocument],
    processing_discards: list[dict[str, str]],
    selection_discards,
    scrape_errors,
    scrape_duplicates: list[str],
    exact_duplicates: list[dict[str, str]],
    manifest: list[dict[str, str]],
    chunks_count: int,
    chunk_stats: dict[str, float | int],
    categories_by_url: dict[str, str],
) -> dict:
    distribution = Counter(entry["category"] for entry in manifest)
    suspicious_small = [
        {
            "url": str(document.url),
            "title": document.title,
            "size": len(document.content),
            "category": categories_by_url.get(str(document.url), "unknown"),
        }
        for document in processed_documents
        if len(document.content) < MIN_USEFUL_CONTENT_CHARS
    ]

    selection_discard_summary = Counter(discard.reason for discard in selection_discards)
    samples = representative_samples(processed_documents, categories_by_url)

    return {
        "sitemap_url": SITEMAP_PERSONAS_URL,
        "sitemap_urls_found": len(sitemap_urls),
        "candidate_urls_found": candidate_count,
        "selected_urls": len(selected_urls),
        "pages_requested": len(selected_urls),
        "raw_documents": len(raw_documents),
        "processed_documents": len(processed_documents),
        "distribution_by_category": dict(sorted(distribution.items())),
        "document_size_stats": document_size_stats(processed_documents),
        "suspicious_small_documents": suspicious_small,
        "chunks_total": chunks_count,
        "chunk_size_stats": chunk_stats,
        "scrape_errors": [
            {
                "url": error.url,
                "reason": error.reason,
                "status_code": error.status_code,
                "detail": error.detail,
            }
            for error in scrape_errors
        ],
        "scrape_duplicate_urls": scrape_duplicates,
        "processing_discards": processing_discards,
        "exact_duplicates": exact_duplicates,
        "selection_discard_summary": dict(selection_discard_summary.most_common()),
        "selection_policy": {
            "source": "sitemap-personas.xml",
            "per_category_limit": PER_CATEGORY_LIMIT,
            "max_total_urls": MAX_TOTAL_URLS,
            "min_useful_content_chars": MIN_USEFUL_CONTENT_CHARS,
            "home_excluded_reason": (
                "La home se usa como navegacion/descubrimiento y tiene menor densidad "
                "de conocimiento especifico de producto que las paginas de producto."
            ),
        },
        "representative_samples": samples,
        "output_files": {
            "raw": RAW_CORPUS_PATH,
            "processed": PROCESSED_CORPUS_PATH,
            "manifest": MANIFEST_PATH,
            "report": REPORT_PATH,
        },
    }


def representative_samples(
    documents: list[SourceDocument],
    categories_by_url: dict[str, str],
) -> list[dict[str, str]]:
    by_category: dict[str, list[SourceDocument]] = defaultdict(list)
    for document in documents:
        by_category[categories_by_url.get(str(document.url), "unknown")].append(document)

    samples: list[dict[str, str]] = []
    for category in sorted(by_category):
        document = by_category[category][0]
        content = document.content
        middle_start = max(0, (len(content) // 2) - 220)
        samples.append(
            {
                "category": category,
                "title": document.title,
                "url": str(document.url),
                "start": content[:450],
                "middle": content[middle_start : middle_start + 450],
                "end": content[-450:],
            }
        )
    return samples


def write_json(payload: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(report: dict) -> None:
    print(f"sitemap urls found: {report['sitemap_urls_found']}")
    print(f"candidate urls found: {report['candidate_urls_found']}")
    print(f"pages requested: {report['pages_requested']}")
    print(f"raw documents: {report['raw_documents']}")
    print(f"processed documents: {report['processed_documents']}")
    print(f"distribution: {report['distribution_by_category']}")
    print(f"document size stats: {report['document_size_stats']}")
    print(f"suspicious small documents: {len(report['suspicious_small_documents'])}")
    print(f"scrape errors: {len(report['scrape_errors'])}")
    print(f"exact duplicates: {len(report['exact_duplicates'])}")
    print(f"chunks total: {report['chunks_total']}")
    print(f"chunk size stats: {report['chunk_size_stats']}")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
