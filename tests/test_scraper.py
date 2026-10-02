import httpx
import pytest

from app.scraping.scraper import BancolombiaScraper


def build_scraper(transport: httpx.MockTransport, max_pages: int = 3) -> BancolombiaScraper:
    client = httpx.Client(transport=transport, follow_redirects=True)
    return BancolombiaScraper(
        start_urls=["https://www.bancolombia.com/personas"],
        allowed_domain="bancolombia.com",
        max_pages=max_pages,
        timeout_seconds=5,
        rate_limit_seconds=0,
        user_agent="test-agent",
        accept="text/html",
        accept_language="es-CO",
        accept_encoding="gzip, deflate",
        http2=True,
        disallowed_patterns=[
            "/*myportal*",
            "/rest/*",
            "/cgi/*",
            "/*formulario*",
            "/*?ofertaId*",
            "/*pdf*",
            "/*buscador*",
            "/personas/solicitud-de-productos/*",
            "*/preaprobados*",
        ],
        http_client=client,
    )


def test_scraper_stays_inside_domain_and_respects_disallowed_patterns() -> None:
    scraper = build_scraper(httpx.MockTransport(lambda request: httpx.Response(404)))

    assert scraper._is_allowed_url("https://www.bancolombia.com/personas")
    assert scraper._is_allowed_url("https://sub.bancolombia.com/personas")
    assert not scraper._is_allowed_url("https://www.other-bank.com/personas.html")
    assert not scraper._is_allowed_url("mailto:test@example.com")
    assert not scraper._is_allowed_url("https://www.bancolombia.com/rest/accounts")
    assert not scraper._is_allowed_url("https://www.bancolombia.com/personas/buscador")
    assert not scraper._is_allowed_url(
        "https://www.bancolombia.com/personas/productos?ofertaId=123"
    )
    assert not scraper._is_allowed_url(
        "https://www.bancolombia.com/personas/solicitud-de-productos/cuenta-plan-basico"
    )
    assert not scraper._is_allowed_url(
        "https://www.bancolombia.com/personas/productos/tarjetas-credito/preaprobados"
    )


def test_scraper_extracts_documents_and_follows_valid_links_once() -> None:
    html_by_path = {
        "/personas": """
            <html>
              <head><title>Personas | Bancolombia</title></head>
              <body>
                <main>
                  <h1>Personas</h1>
                  <p>Productos para personas.</p>
                  <a href="/personas/productos/tarjetas-credito">Tarjetas</a>
                  <a href="https://external.example.com/page.html">External</a>
                  <a href="/personas/buscador">Forbidden</a>
                </main>
              </body>
            </html>
        """,
        "/personas/productos/tarjetas-credito": """
            <html>
              <head><title>Tarjetas | Bancolombia</title></head>
              <body>
                <main>
                  <h1>Tarjetas</h1>
                  <p>Informacion de tarjetas.</p>
                  <a href="/personas#fragment">Volver</a>
                </main>
              </body>
            </html>
        """,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        html = html_by_path.get(request.url.path)
        if html is None:
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    documents = build_scraper(httpx.MockTransport(handler)).scrape()

    assert [document.title for document in documents] == [
        "Personas | Bancolombia",
        "Tarjetas | Bancolombia",
    ]
    assert documents[0].section == "personas"
    assert "Productos para personas." in documents[0].content
    assert "Informacion de tarjetas." in documents[1].content


def test_scraper_ignores_non_html_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"pdf")

    assert build_scraper(httpx.MockTransport(handler)).scrape() == []


def test_scraper_logs_http_status_errors(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, headers={"content-type": "text/html"}, text="Forbidden")

    with caplog.at_level("WARNING"):
        documents = build_scraper(httpx.MockTransport(handler)).scrape()

    assert documents == []
    assert "Skipping URL due to HTTP status error" in caplog.text
