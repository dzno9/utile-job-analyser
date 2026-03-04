from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from models import JobContext
from providers.job_context_extractor import (
    ExtractionHints,
    JobContextExtractor,
    build_extractor,
)


TARGET_PLATFORMS = {"linkedin", "greenhouse", "workable", "lever", "ashby"}


@dataclass(frozen=True)
class FetchedPage:
    final_url: str
    status_code: int
    text: str
    content_type: str


class URLFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage:
        ...


class HttpxURLFetcher:
    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def fetch(self, url: str) -> FetchedPage:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                with httpx.Client(
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                ) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    return FetchedPage(
                        final_url=str(response.url),
                        status_code=response.status_code,
                        text=response.text,
                        content_type=content_type,
                    )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == 0:
                    continue
        raise last_exc  # type: ignore[misc]


class ScrapeResult(BaseModel):
    source_url: HttpUrl
    platform: str | None = None
    scrape_succeeded: bool
    manual_text_input_required: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_notes: list[str] = Field(default_factory=list)
    job_context: JobContext | None = None
    error: str | None = None


class JobPostingScraper:
    def __init__(
        self,
        *,
        fetcher: URLFetcher | None = None,
        extractor: JobContextExtractor | None = None,
        low_confidence_threshold: float = 0.65,
    ) -> None:
        self.fetcher = fetcher or HttpxURLFetcher()
        self.extractor = extractor or build_extractor()
        self.low_confidence_threshold = low_confidence_threshold

    def scrape(self, job_url: str) -> ScrapeResult:
        platform = detect_platform(job_url)
        try:
            page = self.fetcher.fetch(job_url)
        except Exception as exc:  # noqa: BLE001
            return ScrapeResult(
                source_url=job_url,
                platform=platform,
                scrape_succeeded=False,
                manual_text_input_required=True,
                confidence_score=0.0,
                confidence_notes=["URL fetch failed"],
                error=str(exc),
            )

        if "html" not in page.content_type.lower():
            return ScrapeResult(
                source_url=job_url,
                platform=platform,
                scrape_succeeded=False,
                manual_text_input_required=True,
                confidence_score=0.0,
                confidence_notes=["Unsupported content type"],
                error=f"Unsupported content type: {page.content_type}",
            )

        soup = BeautifulSoup(page.text, "html.parser")

        # Detect login walls (LinkedIn, etc.) before expensive extraction.
        if _is_login_wall(page.text, page.final_url, platform):
            return ScrapeResult(
                source_url=job_url,
                platform=platform,
                scrape_succeeded=False,
                manual_text_input_required=True,
                confidence_score=0.0,
                confidence_notes=[
                    "Login wall detected — this site requires authentication to view the job posting"
                ],
                error="Login wall detected. Paste the full job description manually.",
            )

        hints = extract_hints(soup, page.final_url, platform)
        page_text = extract_visible_text(soup)

        # JSON-LD JobPosting description is the richest source on JS-rendered pages.
        # It's inside a <script> tag so it's always in the initial HTML.
        jsonld_text = _extract_jsonld_description(soup)
        if jsonld_text and len(jsonld_text) > len(page_text):
            page_text = jsonld_text

        # When visible text is still too thin, enrich with meta/noscript content
        if len(page_text) < 200:
            page_text = _enrich_thin_page_text(page.text, page_text, hints)

        llm_output = self.extractor.extract(
            page_text=page_text,
            job_url=page.final_url,
            hints=hints,
        )

        job_context = build_job_context(
            source_url=page.final_url,
            page_text=page_text,
            hints=hints,
            llm_output=llm_output,
        )
        confidence_score, confidence_notes = score_confidence(
            platform=platform,
            page_text=page_text,
            hints=hints,
            job_context=job_context,
        )
        manual_required = confidence_score < self.low_confidence_threshold
        if job_context is not None and not job_context.requirements and not job_context.required_skills:
            manual_required = True
            confidence_notes.append(
                "Could not extract requirements or skills from posting; manual job text recommended"
            )

        return ScrapeResult(
            source_url=page.final_url,
            platform=platform,
            scrape_succeeded=job_context is not None,
            manual_text_input_required=manual_required or job_context is None,
            confidence_score=confidence_score,
            confidence_notes=confidence_notes,
            job_context=job_context,
            error=None if job_context is not None else "Could not validate JobContext",
        )


def detect_platform(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower()
    if "linkedin.com" in host:
        return "linkedin"
    if "greenhouse.io" in host:
        return "greenhouse"
    if "workable.com" in host:
        return "workable"
    if "lever.co" in host:
        return "lever"
    if "ashbyhq.com" in host:
        return "ashby"
    return None


def extract_hints(soup: BeautifulSoup, source_url: str, platform: str | None) -> ExtractionHints:
    json_ld = _extract_jobposting_jsonld(soup)
    job_title = _first_non_empty(
        json_ld.get("title"),
        _select_text(soup, "h1"),
        _meta_content(soup, "og:title"),
        _meta_content(soup, "twitter:title"),
    )
    company_name = _first_non_empty(
        _deep_get(json_ld, "hiringOrganization", "name"),
        _select_text(
            soup,
            "[data-test='company-name']",
            ".topcard__org-name-link",
            ".company",
            ".posting-categories .sort-by-time",
        ),
        _meta_content(soup, "og:site_name"),
    )
    company_name = _first_non_empty(company_name, _company_from_url(source_url, platform))
    location = _first_non_empty(
        _deep_get(json_ld, "jobLocation", "address", "addressLocality"),
        _deep_get(json_ld, "jobLocation", "address", "addressRegion"),
        _select_text(
            soup,
            "[data-test='location']",
            ".topcard__flavor--bullet",
            ".location",
            ".posting-categories .location",
        ),
    )
    employment_type = _first_non_empty(
        _safe_string(json_ld.get("employmentType")),
        _match_pattern(
            extract_visible_text(soup)[:2000],
            r"\b(full[- ]time|part[- ]time|contract|temporary|internship)\b",
        ),
    )

    requirements = _extract_section_list(
        soup, ("requirements", "what you bring", "qualifications", "must have")
    )
    responsibilities = _extract_section_list(
        soup, ("responsibilities", "what you'll do", "what you will do", "role")
    )
    tech_stack = _extract_tech_stack(extract_visible_text(soup))
    seniority_level = _extract_seniority(job_title or "", extract_visible_text(soup))

    # Platform-specific patch-ups where raw selectors are reliable.
    if platform == "greenhouse":
        company_from_greenhouse = _select_text(soup, "#header .company-name", ".company-name")
        company_name = _first_non_empty(company_name, company_from_greenhouse)
    elif platform == "lever":
        location = _first_non_empty(
            location,
            _select_text(soup, ".posting-categories .location"),
        )

    return ExtractionHints(
        job_title=job_title,
        company_name=company_name,
        location=location,
        employment_type=employment_type,
        requirements=requirements,
        tech_stack=tech_stack,
        responsibilities=responsibilities,
        seniority_level=seniority_level,
    )


def build_job_context(
    *,
    source_url: str,
    page_text: str,
    hints: ExtractionHints,
    llm_output,
) -> JobContext | None:
    merged = {
        "job_id": _build_job_id(source_url),
        "job_title": llm_output.job_title or hints.job_title or "Unknown role",
        "company_name": llm_output.company_name or hints.company_name or "Unknown company",
        "location": llm_output.location or hints.location,
        "job_url": source_url,
        "posting_text": page_text[:20000],
        "requirements": _clean_list(hints.requirements or []),
        "required_skills": _clean_list(llm_output.required_skills),
        "nice_to_have_skills": _clean_list(llm_output.nice_to_have_skills),
        "employment_type": llm_output.employment_type or hints.employment_type,
    }
    if not merged["required_skills"] and hints.tech_stack:
        merged["required_skills"] = hints.tech_stack[:15]

    try:
        return JobContext.model_validate(merged)
    except ValidationError:
        return None


def score_confidence(
    *,
    platform: str | None,
    page_text: str,
    hints: ExtractionHints,
    job_context: JobContext | None,
) -> tuple[float, list[str]]:
    score = 0.0
    notes: list[str] = []

    if platform in TARGET_PLATFORMS:
        score += 0.05
        notes.append("Target platform detected")

    if job_context is not None:
        if job_context.job_title and "unknown" not in job_context.job_title.lower():
            score += 0.2
        if job_context.company_name and "unknown" not in job_context.company_name.lower():
            score += 0.2
        if job_context.location:
            score += 0.1
        if len(job_context.required_skills) >= 3:
            score += 0.15
        if len(job_context.posting_text) > 1200:
            score += 0.2
        elif len(job_context.posting_text) > 400:
            score += 0.1
    else:
        notes.append("JobContext validation failed")

    if hints.requirements and len(hints.requirements) >= 2:
        score += 0.1
    if hints.responsibilities and len(hints.responsibilities) >= 2:
        score += 0.1
    if hints.tech_stack and len(hints.tech_stack) >= 2:
        score += 0.1
    if hints.seniority_level:
        score += 0.05

    if _is_non_job_like(page_text):
        score -= 0.35
        notes.append("Page content does not look like a job posting")

    score = max(0.0, min(1.0, score))
    notes.append(f"Computed confidence={score:.2f}")
    return score, notes


def extract_visible_text(soup: BeautifulSoup) -> str:
    """Extract visible text from parsed HTML.

    Re-parses from string to avoid mutating the original soup.
    Preserves <noscript> content since many JS-rendered job pages put real content there.
    """
    # Re-parse to avoid mutating the caller's soup object.
    fresh = BeautifulSoup(str(soup), "html.parser")
    # Keep noscript — JS-rendered pages often put real job content there for SEO.
    for tag in fresh(["script", "style", "svg"]):
        tag.decompose()
    # Remove nav/footer only when they're large blocks (not small location hints).
    for tag in fresh(["nav", "footer"]):
        if len(tag.get_text(strip=True)) > 300:
            tag.decompose()
    text = fresh.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _enrich_thin_page_text(
    raw_html: str, visible_text: str, hints: ExtractionHints
) -> str:
    """When visible text is too thin (JS-rendered page), build richer text from HTML metadata."""
    parts: list[str] = [visible_text] if visible_text.strip() else []

    # Re-parse raw HTML (original soup was modified by extract_visible_text)
    soup = BeautifulSoup(raw_html, "html.parser")

    title_tag = soup.find("title")
    if title_tag:
        parts.append(f"Page title: {title_tag.get_text(strip=True)}")

    for meta in soup.find_all("meta"):
        content = (meta.get("content") or "").strip()
        if content and len(content) > 30:
            name = (meta.get("name") or meta.get("property") or "").lower()
            if name in ("description", "og:description", "og:title", "twitter:description"):
                parts.append(content)

    # Extract noscript content (many SPAs include SEO content here)
    for noscript in soup.find_all("noscript"):
        text = noscript.get_text(separator="\n", strip=True)
        if text and len(text) > 50:
            parts.append(text)

    # Include structured hints extracted from HTML
    if hints.job_title:
        parts.append(f"Job title: {hints.job_title}")
    if hints.company_name:
        parts.append(f"Company: {hints.company_name}")
    if hints.location:
        parts.append(f"Location: {hints.location}")
    if hints.requirements:
        parts.append(f"Requirements: {'; '.join(hints.requirements)}")
    if hints.tech_stack:
        parts.append(f"Tech stack: {', '.join(hints.tech_stack)}")
    if hints.responsibilities:
        parts.append(f"Responsibilities: {'; '.join(hints.responsibilities)}")

    enriched = "\n\n".join(parts)
    return enriched if len(enriched) > len(visible_text) else visible_text


def _extract_jobposting_jsonld(soup: BeautifulSoup) -> dict:
    for script in soup.select("script[type='application/ld+json']"):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            parsed = __import__("json").loads(raw)
        except Exception:  # noqa: BLE001
            continue
        candidates = parsed if isinstance(parsed, list) else [parsed]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            type_value = candidate.get("@type")
            if type_value == "JobPosting":
                return candidate
    return {}


def _extract_jsonld_description(soup: BeautifulSoup) -> str | None:
    """Extract the full job description text from JSON-LD JobPosting schema.

    Most job sites (LinkedIn, Greenhouse, Lever, Indeed, etc.) include this in
    a <script type="application/ld+json"> tag, so it's available even when the
    page is JS-rendered and the visible text is thin.
    """
    jsonld = _extract_jobposting_jsonld(soup)
    description = jsonld.get("description")
    if not description or not isinstance(description, str):
        return None

    # The description is often HTML-formatted. Parse and extract text.
    desc_soup = BeautifulSoup(description, "html.parser")
    text = desc_soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Prepend title/company/location from JSON-LD for context
    parts: list[str] = []
    title = jsonld.get("title")
    if title and isinstance(title, str):
        parts.append(f"Job Title: {title.strip()}")
    org = jsonld.get("hiringOrganization")
    if isinstance(org, dict) and org.get("name"):
        parts.append(f"Company: {org['name']}")
    loc = jsonld.get("jobLocation")
    if isinstance(loc, dict):
        addr = loc.get("address")
        if isinstance(addr, dict):
            loc_parts = [addr.get("addressLocality"), addr.get("addressRegion"), addr.get("addressCountry")]
            loc_str = ", ".join(p for p in loc_parts if p and isinstance(p, str))
            if loc_str:
                parts.append(f"Location: {loc_str}")
    emp_type = jsonld.get("employmentType")
    if emp_type and isinstance(emp_type, str):
        parts.append(f"Employment Type: {emp_type}")

    if parts:
        header = "\n".join(parts)
        text = f"{header}\n\n{text}"

    return text.strip() if len(text.strip()) > 100 else None


def _extract_section_list(soup: BeautifulSoup, keywords: tuple[str, ...]) -> list[str]:
    results: list[str] = []
    heading_tags = soup.find_all(re.compile("^h[1-4]$"))
    for heading in heading_tags:
        heading_text = heading.get_text(" ", strip=True).lower()
        if not any(keyword in heading_text for keyword in keywords):
            continue
        for sibling in heading.find_all_next(limit=30):
            if sibling.name and re.match("^h[1-4]$", sibling.name):
                break
            if sibling.name in {"li", "p"}:
                text = sibling.get_text(" ", strip=True)
                if text and 20 <= len(text) <= 240:
                    results.append(text)
        if results:
            break
    return _clean_list(results)[:12]


def _extract_tech_stack(text: str) -> list[str]:
    lower = text.lower()
    known = [
        "python",
        "sql",
        "java",
        "javascript",
        "typescript",
        "react",
        "node.js",
        "aws",
        "gcp",
        "azure",
        "docker",
        "kubernetes",
        "spark",
        "airflow",
        "dbt",
        "tableau",
        "looker",
        "snowflake",
    ]
    out = []
    for item in known:
        pattern = re.escape(item).replace("\\.", r"\.?")
        if re.search(rf"\b{pattern}\b", lower):
            out.append(item.upper() if item == "sql" else item.title())
    return _clean_list(out)


def _extract_seniority(title: str, text: str) -> str | None:
    content = f"{title} {text[:1200]}".lower()
    for level in ("intern", "junior", "mid", "senior", "staff", "principal", "lead"):
        if re.search(rf"\b{level}\b", content):
            return level.title()
    return None


def _build_job_id(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parts:
        return f"{parsed.hostname}:{parts[-1]}"
    return f"{parsed.hostname}:{uuid.uuid4().hex[:8]}"


def _company_from_url(url: str, platform: str | None) -> str | None:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        return None
    # On Lever/Workable/Ashby URLs, first path segment usually maps to company slug.
    if platform in {"lever", "workable", "ashby"}:
        slug = path_parts[0]
        return _slug_to_title(slug)
    return None


def _slug_to_title(slug: str) -> str | None:
    cleaned = re.sub(r"[^a-zA-Z0-9\-]+", "", slug).strip("-")
    if not cleaned:
        return None
    words = [word for word in cleaned.split("-") if word]
    if not words:
        return None
    return " ".join(word.capitalize() for word in words)


def _meta_content(soup: BeautifulSoup, property_name: str) -> str | None:
    tag = soup.find("meta", attrs={"property": property_name}) or soup.find(
        "meta", attrs={"name": property_name}
    )
    if not tag:
        return None
    return _safe_string(tag.get("content"))


def _select_text(soup: BeautifulSoup, *selectors: str) -> str | None:
    for selector in selectors:
        tag = soup.select_one(selector)
        if tag:
            value = _safe_string(tag.get_text(" ", strip=True))
            if value:
                return value
    return None


def _deep_get(data: dict, *path: str):
    current = data
    for key in path:
        if isinstance(current, list):
            current = current[0] if current else {}
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _safe_string(current)


def _match_pattern(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return _safe_string(match.group(1) if match.groups() else match.group(0))


def _safe_string(value) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def _is_login_wall(html: str, final_url: str, platform: str | None) -> bool:
    """Detect if the fetched page is a login/auth wall instead of the actual job posting."""
    lower_html = html.lower()

    # LinkedIn auth wall detection
    if platform == "linkedin" or "linkedin.com" in (final_url or "").lower():
        login_signals = (
            "authwall",
            "sign in to linkedin",
            "join linkedin",
            "login-email",
            "sign in to view",
            "uas/login",
        )
        if sum(1 for sig in login_signals if sig in lower_html) >= 2:
            return True

    # Generic auth wall detection (works for other sites too)
    generic_signals = (
        "please sign in",
        "please log in",
        "create an account to",
        "sign up to view",
        "login to continue",
        "you need to sign in",
    )
    if sum(1 for sig in generic_signals if sig in lower_html) >= 2:
        return True

    # If URL redirected to a login path
    if final_url and any(
        segment in final_url.lower()
        for segment in ("/login", "/signin", "/authwall", "/uas/login")
    ):
        return True

    return False


def _is_non_job_like(text: str) -> bool:
    lower = text.lower()
    job_keywords = [
        "responsibilities",
        "requirements",
        "qualifications",
        "apply",
        "hiring",
        "job",
        "role",
    ]
    hits = sum(1 for token in job_keywords if token in lower)
    return hits < 2 or len(text) < 250
