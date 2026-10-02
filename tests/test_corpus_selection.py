from datetime import UTC, datetime

from app.corpus.selection import (
    classify_url,
    content_hash,
    is_excluded_url,
    parse_sitemap_urls,
    select_corpus_urls,
    write_manifest,
)
from app.schemas import SourceDocument


def test_parse_sitemap_urls_reads_namespaced_locs() -> None:
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://www.bancolombia.com/personas/cuentas</loc></url>
      <url><loc>https://www.bancolombia.com/personas/tarjetas-de-credito</loc></url>
    </urlset>
    """

    assert parse_sitemap_urls(xml) == [
        "https://www.bancolombia.com/personas/cuentas",
        "https://www.bancolombia.com/personas/tarjetas-de-credito",
    ]


def test_classify_url_maps_supported_categories() -> None:
    assert (
        classify_url("https://www.bancolombia.com/personas/cuentas/ahorros-y-corriente")
        == "cuentas"
    )
    assert (
        classify_url("https://www.bancolombia.com/personas/tarjetas-de-credito/visa")
        == "tarjetas_credito"
    )
    assert (
        classify_url("https://www.bancolombia.com/personas/creditos/vivienda")
        == "vivienda"
    )


def test_is_excluded_url_rejects_home_login_assets_and_forms() -> None:
    assert is_excluded_url("https://www.bancolombia.com/")[0] is True
    assert is_excluded_url("https://www.bancolombia.com/personas/login")[0] is True
    assert (
        is_excluded_url("https://www.bancolombia.com/personas/solicitud-de-productos")[0]
        is True
    )
    assert is_excluded_url("https://www.bancolombia.com/personas/cuentas/main.sass")[0] is True


def test_select_corpus_urls_limits_per_category_and_reports_discards() -> None:
    urls = [
        "https://www.bancolombia.com/personas/cuentas/uno",
        "https://www.bancolombia.com/personas/cuentas/dos",
        "https://www.bancolombia.com/personas/cuentas/tres",
        "https://www.bancolombia.com/personas/login",
        "https://www.bancolombia.com/personas/tarjetas-debito/clasica",
    ]

    result = select_corpus_urls(urls, per_category_limit=2, max_total=10)

    assert [candidate.url for candidate in result.selected] == [
        "https://www.bancolombia.com/personas/cuentas/uno",
        "https://www.bancolombia.com/personas/cuentas/dos",
        "https://www.bancolombia.com/personas/tarjetas-debito/clasica",
    ]
    assert any(discard.reason == "category_limit:cuentas" for discard in result.discarded)
    assert any(discard.reason == "excluded_pattern:*/login*" for discard in result.discarded)


def test_content_hash_normalizes_whitespace_and_case() -> None:
    assert content_hash("Tarjeta   Débito\nBancolombia") == content_hash(
        "tarjeta débito bancolombia"
    )


def test_write_manifest_persists_traceability_fields(tmp_path) -> None:
    document = SourceDocument(
        document_id="doc-1",
        url="https://www.bancolombia.com/personas/cuentas",
        title="Cuentas",
        content="Contenido",
        scraped_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    target = tmp_path / "manifest.json"

    entries = write_manifest(
        [document],
        {"https://www.bancolombia.com/personas/cuentas": "cuentas"},
        str(target),
    )

    assert entries == [
        {
            "url": "https://www.bancolombia.com/personas/cuentas",
            "title": "Cuentas",
            "category": "cuentas",
            "document_id": "doc-1",
        }
    ]
    assert target.exists()
