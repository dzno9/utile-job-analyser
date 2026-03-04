from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from providers.company_researcher import SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchConfig:
    timeout_seconds: float = 12.0
    fetch_top_n: int = 3
    max_page_chars: int = 3000


class DuckDuckGoWebSearchTool:
    """Web search adapter using the duckduckgo-search (ddgs) library.

    Fetches actual page content for the top N results to provide richer context
    for downstream LLM synthesis.
    """

    def __init__(self, *, config: WebSearchConfig | None = None) -> None:
        self.config = config or WebSearchConfig()

    def search(
        self,
        *,
        query: str,
        iteration: int,
        max_results: int,
    ) -> list[SearchResult]:
        del iteration

        # Try the newer `ddgs` package first, then fall back to `duckduckgo_search`
        DDGS = None
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError:
                logger.warning("Neither ddgs nor duckduckgo-search installed; returning empty results")
                return []

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
        except Exception:  # noqa: BLE001
            logger.warning("DuckDuckGo search failed for query: %s", query, exc_info=True)
            return []

        results: list[SearchResult] = []
        for idx, item in enumerate(raw_results):
            href = (item.get("href") or item.get("link") or "").strip()
            if not href:
                continue
            title = (item.get("title") or "Untitled").strip()
            snippet = (item.get("body") or title).strip()

            # Fetch actual page content for the top N results.
            content = snippet
            if idx < self.config.fetch_top_n:
                fetched = _fetch_page_text(href, timeout=self.config.timeout_seconds)
                if fetched and len(fetched) > len(snippet):
                    content = fetched[: self.config.max_page_chars]

            results.append(
                SearchResult(
                    title=title,
                    url=href,
                    published_date=None,
                    snippet=snippet,
                    content=content,
                )
            )

        return results


def _fetch_page_text(url: str, *, timeout: float = 10.0) -> str | None:
    """Best-effort fetch and extract visible text from a URL."""
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            },
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            if "html" not in resp.headers.get("content-type", "").lower():
                return None
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "svg", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip() or None
    except Exception:  # noqa: BLE001
        logger.debug("Failed to fetch page content from %s", url, exc_info=True)
        return None
