import fnmatch
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from app.config.settings import Settings
from app.schemas import SourceDocument

logger = logging.getLogger(__name__)


@dataclass
class ScrapeError:
    url: str
    reason: str
    status_code: int | None = None
    detail: str | None = None


@dataclass
class ScrapeResult:
    documents: list[SourceDocument]
    errors: list[ScrapeError] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)


class BancolombiaScraper:
    def __init__(
        self,
        start_urls: list[str],
        allowed_domain: str,
        max_pages: int,
        timeout_seconds: int,
        rate_limit_seconds: float,
        user_agent: str,
        accept: str,
        accept_language: str,
        accept_encoding: str,
        http2: bool,
        disallowed_patterns: list[str],
        http_client: httpx.Client | None = None,
    ) -> None:
        self.start_urls = [self._normalize_url(url) for url in start_urls]
        self.allowed_domain = allowed_domain
        self.max_pages = max_pages
        self.timeout_seconds = timeout_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.disallowed_patterns = disallowed_patterns
        self._client = http_client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            http2=http2,
            headers={
                "User-Agent": user_agent,
                "Accept": accept,
                "Accept-Language": accept_language,
                "Accept-Encoding": accept_encoding,
            },
        )
        self._owns_client = http_client is None

    @classmethod
    def from_settings(cls, settings: Settings) -> "BancolombiaScraper":
        return cls(
            start_urls=settings.bank_start_urls,
            allowed_domain=settings.bank_allowed_domain,
            max_pages=settings.scraper_max_pages,
            timeout_seconds=settings.scraper_timeout_seconds,
            rate_limit_seconds=settings.scraper_rate_limit_seconds,
            user_agent=settings.scraper_user_agent,
            accept=settings.scraper_accept,
            accept_language=settings.scraper_accept_language,
            accept_encoding=settings.scraper_accept_encoding,
            http2=settings.scraper_http2,
            disallowed_patterns=settings.scraper_disallowed_patterns,
        )

    def scrape(self) -> list[SourceDocument]:
        queue: deque[str] = deque(self.start_urls)
        visited: set[str] = set()
        documents: list[SourceDocument] = []

        try:
            while queue and len(documents) < self.max_pages:
                url = queue.popleft()
                if url in visited or not self._is_allowed_url(url):
                    continue

                visited.add(url)
                response = self._fetch(url)
                if response is None:
                    continue

                document, links = self._parse_response(url, response.text)
                if document.content:
                    documents.append(document)

                for link in links:
                    normalized_link = self._normalize_url(link)
                    if normalized_link not in visited and self._is_allowed_url(normalized_link):
                        queue.append(normalized_link)

                if self.rate_limit_seconds > 0 and queue and len(documents) < self.max_pages:
                    time.sleep(self.rate_limit_seconds)
        finally:
            if self._owns_client:
                self._client.close()

        return documents

    def scrape_exact(self, urls: list[str]) -> ScrapeResult:
        documents: list[SourceDocument] = []
        errors: list[ScrapeError] = []
        duplicates: list[str] = []
        visited: set[str] = set()

        try:
            for index, url in enumerate(urls):
                normalized_url = self._normalize_url(url)
                if normalized_url in visited:
                    duplicates.append(normalized_url)
                    continue
                visited.add(normalized_url)

                if not self._is_allowed_url(normalized_url):
                    errors.append(ScrapeError(url=normalized_url, reason="disallowed_url"))
                    continue

                response, error = self._fetch_with_error(normalized_url)
                if error:
                    errors.append(error)
                    continue

                if response is None:
                    continue

                document, _ = self._parse_response(normalized_url, response.text)
                if document.content:
                    documents.append(document)

                if self.rate_limit_seconds > 0 and index < len(urls) - 1:
                    time.sleep(self.rate_limit_seconds)
        finally:
            if self._owns_client:
                self._client.close()

        return ScrapeResult(documents=documents, errors=errors, duplicates=duplicates)

    def _fetch(self, url: str) -> httpx.Response | None:
        response, _ = self._fetch_with_error(url)
        return response

    def _fetch_with_error(self, url: str) -> tuple[httpx.Response | None, ScrapeError | None]:
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Skipping URL due to HTTP status error: url=%s status_code=%s",
                url,
                exc.response.status_code,
            )
            return None, ScrapeError(
                url=url,
                reason="http_status_error",
                status_code=exc.response.status_code,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Skipping URL due to HTTP client error: url=%s error=%s",
                url,
                exc,
            )
            return None, ScrapeError(url=url, reason="http_client_error", detail=str(exc))

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            logger.info(
                "Skipping non-HTML URL: url=%s content_type=%s",
                url,
                content_type,
            )
            return None, ScrapeError(
                url=url,
                reason="non_html",
                detail=content_type,
            )
        return response, None

    def _parse_response(self, url: str, html: str) -> tuple[SourceDocument, list[str]]:
        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup)
        content_root = soup.find("main") or soup.find("article") or soup.body or soup

        for tag in content_root.select("script, style, noscript"):
            tag.decompose()

        text = content_root.get_text("\n", strip=True)
        links = [urljoin(url, anchor["href"]) for anchor in soup.select("a[href]")]

        document = SourceDocument(
            document_id=self._document_id(url),
            url=url,
            title=title,
            content=text,
            section=self._infer_section(url),
            scraped_at=datetime.now(UTC),
            metadata={"source": "bancolombia.com"},
        )
        return document, links

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()

        heading = soup.find(["h1", "h2"])
        if heading:
            return heading.get_text(" ", strip=True)

        return "Bancolombia"

    def _is_allowed_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False

        hostname = parsed.hostname or ""
        if hostname != self.allowed_domain and not hostname.endswith(f".{self.allowed_domain}"):
            return False

        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return not any(
            self._is_disallowed_path(path, pattern) for pattern in self.disallowed_patterns
        )

    def _is_disallowed_path(self, path: str, pattern: str) -> bool:
        if "*" in pattern:
            return fnmatch.fnmatch(path, pattern)
        return path == pattern or path.startswith(f"{pattern}/")

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")

        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
                "",
                parsed.query,
                "",
            )
        )

    def _document_id(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _infer_section(self, url: str) -> str | None:
        path_parts = [part for part in urlparse(url).path.split("/") if part]
        if not path_parts:
            return None
        return path_parts[0].removesuffix(".html")
