import fnmatch
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlparse
from xml.etree import ElementTree

from app.schemas import DocumentChunk, SourceDocument

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "cuentas": (
        "/personas/cuentas",
        "/personas/a-la-mano",
        "/personas/bolsillos",
        "/personas/productos-servicios/cuentas",
    ),
    "tarjetas_credito": (
        "/personas/tarjetas-de-credito",
        "/personas/productos-servicios/tarjetas-credito",
    ),
    "tarjetas_debito": (
        "/personas/tarjetas-debito",
        "/personas/productos-servicios/tarjetas-debito",
        "/personas/productos/cuentas/tarjetas-debito",
    ),
    "prestamos": (
        "/personas/creditos/consumo",
        "/personas/creditos/vehiculo",
        "/personas/creditos/estudio",
        "/personas/creditos/negocios",
    ),
    "vivienda": (
        "/personas/creditos/vivienda",
        "/personas/vivienda",
        "/personas/cuentas/vivienda",
    ),
    "seguros": (
        "/personas/seguros",
        "/personas/productos-servicios/seguros",
    ),
    "inversiones": (
        "/personas/productos-servicios/inversiones",
        "/personas/aprender-es-facil/como-manejar-dinero/invertir",
    ),
}

EXCLUDED_PATH_PATTERNS = (
    "/",
    "/personas",
    "*/login*",
    "*/contactanos*",
    "*/buscador*",
    "*/formulario*",
    "*/solicitud-de-productos*",
    "*/preaprobados*",
    "*/preaprobados-digitales*",
    "*/oferta/*",
    "*/ofertas*",
    "*/referido*",
    "*/simulador*",
    "*/calculadora*",
    "*/src/*",
    "*.sass",
    "*.css",
    "*.js",
    "*.pdf",
    "*/documentos-legales/*",
    "*/eventos/*",
    "*/beneficios/*",
    "*/necesidades/*",
    "*/app-bancolombia/*",
    "*/canales-servicio/*",
    "*/pagos/*",
    "*/giros/*",
    "*/seguridad/*",
    "*/tarifario*",
    "*/mapa-sitio*",
)


@dataclass(frozen=True)
class UrlCandidate:
    url: str
    category: str


@dataclass(frozen=True)
class DiscardedUrl:
    url: str
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    candidates: list[UrlCandidate]
    selected: list[UrlCandidate]
    discarded: list[DiscardedUrl]


def parse_sitemap_urls(xml_text: str) -> list[str]:
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ElementTree.fromstring(xml_text)
    return [
        loc.text.strip()
        for loc in root.findall(".//sm:loc", namespace)
        if loc.text and loc.text.strip()
    ]


def classify_url(url: str) -> str | None:
    path = _normalized_path(url)
    for category, prefixes in CATEGORY_PATTERNS.items():
        if any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes):
            return category
    return None


def is_excluded_url(url: str) -> tuple[bool, str | None]:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    if parsed.query:
        return True, "query_string"
    if parsed.hostname != "www.bancolombia.com":
        return True, "outside_www_bancolombia"
    for pattern in EXCLUDED_PATH_PATTERNS:
        if fnmatch.fnmatch(path, pattern):
            return True, f"excluded_pattern:{pattern}"
    return False, None


def select_corpus_urls(
    urls: list[str],
    per_category_limit: int = 8,
    max_total: int = 56,
) -> SelectionResult:
    candidates: list[UrlCandidate] = []
    selected: list[UrlCandidate] = []
    discarded: list[DiscardedUrl] = []
    seen_urls: set[str] = set()
    selected_by_category: Counter[str] = Counter()

    for url in urls:
        normalized = url.rstrip("/")
        if normalized in seen_urls:
            discarded.append(DiscardedUrl(url=normalized, reason="duplicate_url"))
            continue
        seen_urls.add(normalized)

        excluded, reason = is_excluded_url(normalized)
        if excluded:
            discarded.append(DiscardedUrl(url=normalized, reason=reason or "excluded"))
            continue

        category = classify_url(normalized)
        if category is None:
            discarded.append(DiscardedUrl(url=normalized, reason="category_not_supported"))
            continue

        candidate = UrlCandidate(url=normalized, category=category)
        candidates.append(candidate)

        if selected_by_category[category] >= per_category_limit:
            discarded.append(DiscardedUrl(url=normalized, reason=f"category_limit:{category}"))
            continue
        if len(selected) >= max_total:
            discarded.append(DiscardedUrl(url=normalized, reason="max_total_reached"))
            continue

        selected.append(candidate)
        selected_by_category[category] += 1

    return SelectionResult(candidates=candidates, selected=selected, discarded=discarded)


def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_manifest(
    documents: list[SourceDocument],
    categories_by_url: dict[str, str],
    path: str,
) -> list[dict[str, str]]:
    entries = [
        {
            "url": str(document.url),
            "title": document.title,
            "category": categories_by_url.get(str(document.url), "unknown"),
            "document_id": document.document_id,
        }
        for document in documents
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return entries


def document_size_stats(documents: list[SourceDocument]) -> dict[str, float | int]:
    sizes = [len(document.content) for document in documents]
    if not sizes:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0.0}
    return {
        "min": min(sizes),
        "max": max(sizes),
        "avg": round(mean(sizes), 2),
        "median": round(median(sizes), 2),
    }


def chunk_size_stats(chunks: list[DocumentChunk]) -> dict[str, float | int]:
    sizes = [len(chunk.text) for chunk in chunks]
    if not sizes:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0.0}
    return {
        "min": min(sizes),
        "max": max(sizes),
        "avg": round(mean(sizes), 2),
        "median": round(median(sizes), 2),
    }


def _normalized_path(url: str) -> str:
    return urlparse(url).path.rstrip("/") or "/"
