from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Protocol
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from models import JobContext, VisaAssessment, VisaEvidenceTag, VisaLikelihood
from providers.company_researcher import SearchResult


_GOV_UK_REGISTER_PAGE = (
    "https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers"
)

_ENTITY_SUFFIXES = {
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "llc",
    "llp",
    "plc",
    "corp",
    "corporation",
    "company",
    "co",
    "uk",
    "holdings",
    "holding",
    "group",
    "technologies",
    "technology",
}
_SIGNIFICANT_TOKEN_MIN_LEN = 3


@dataclass(frozen=True)
class SponsorRegisterEntry:
    organisation_name: str
    source_row: dict[str, str]


class GovUkRegisterSource(Protocol):
    def fetch_html(self, *, index_url: str) -> str: ...
    def fetch_csv(self, *, csv_url: str) -> str: ...


class SponsoredRoleSearchTool(Protocol):
    def search(
        self,
        *,
        query: str,
        iteration: int,
        max_results: int,
    ) -> list[SearchResult]:
        ...


class HttpGovUkRegisterSource:
    def __init__(self, *, timeout_seconds: int = 25) -> None:
        self._client = httpx.Client(timeout=timeout_seconds, follow_redirects=True)

    def fetch_html(self, *, index_url: str) -> str:
        response = self._client.get(index_url)
        response.raise_for_status()
        return response.text

    def fetch_csv(self, *, csv_url: str) -> str:
        response = self._client.get(csv_url)
        response.raise_for_status()
        return response.text


class CachedGovUkSponsorRegister:
    def __init__(
        self,
        *,
        source: GovUkRegisterSource | None = None,
        cache_dir: str | Path = ".cache",
        ttl_days: int = 7,
        index_url: str = _GOV_UK_REGISTER_PAGE,
    ) -> None:
        self.source = source or HttpGovUkRegisterSource()
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.index_url = index_url
        self.cache_csv_path = self.cache_dir / "uk_sponsor_licence_register.csv"
        self.cache_meta_path = self.cache_dir / "uk_sponsor_licence_register.meta.json"

    def load(self) -> tuple[list[SponsorRegisterEntry], str]:
        csv_text: str
        source_url: str
        if self._is_cache_fresh():
            csv_text = self.cache_csv_path.read_text(encoding="utf-8")
            source_url = self._read_metadata().get("source_url", self.index_url)
        else:
            html = self.source.fetch_html(index_url=self.index_url)
            source_url = self._extract_csv_url(html=html)
            csv_text = self.source.fetch_csv(csv_url=source_url)
            self._write_cache(csv_text=csv_text, source_url=source_url)
        return _parse_register_csv(csv_text), source_url

    def _is_cache_fresh(self) -> bool:
        if not self.cache_csv_path.exists() or not self.cache_meta_path.exists():
            return False
        try:
            meta = self._read_metadata()
            fetched_at = datetime.fromisoformat(str(meta["fetched_at"]))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        return datetime.now(UTC) - fetched_at < timedelta(days=self.ttl_days)

    def _read_metadata(self) -> dict[str, str]:
        raw = self.cache_meta_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    def _extract_csv_url(self, *, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href", "")).strip()
            if not href:
                continue
            if ".csv" in href.lower():
                return urljoin(self.index_url, href)
        raise ValueError("Could not find CSV download link on GOV.UK sponsor register page")

    def _write_cache(self, *, csv_text: str, source_url: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_csv_path.write_text(csv_text, encoding="utf-8")
        self.cache_meta_path.write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "source_url": source_url,
                }
            ),
            encoding="utf-8",
        )


class UKVisaSponsorChecker:
    def __init__(
        self,
        *,
        register: CachedGovUkSponsorRegister | None = None,
        search_tool: SponsoredRoleSearchTool | None = None,
    ) -> None:
        self.register = register or CachedGovUkSponsorRegister()
        self.search_tool = search_tool

    def assess(self, *, job_context: JobContext) -> VisaAssessment:
        entries, source_url = self.register.load()
        company_name = job_context.company_name.strip()

        direct_match = self._find_direct_match(company_name=company_name, entries=entries)
        if direct_match is not None:
            return VisaAssessment(
                likelihood=VisaLikelihood.CONFIRMED_SPONSOR,
                evidence_tags=[VisaEvidenceTag.DIRECT_REGISTER_MATCH],
                reasoning=(
                    f"{direct_match.organisation_name} appears on the UK Skilled Worker "
                    "sponsor licence register."
                ),
                evidence=[source_url],
            )

        related_match = self._find_related_entity(company_name=company_name, entries=entries)
        if related_match is not None:
            return VisaAssessment(
                likelihood=VisaLikelihood.LIKELY,
                evidence_tags=[VisaEvidenceTag.RELATED_ENTITY_MATCH],
                reasoning=(
                    "No direct legal-entity match found, but a likely related entity "
                    f"({related_match.organisation_name}) appears on the register."
                ),
                evidence=[source_url],
            )

        sponsored_role_evidence = self._search_sponsored_role_signals(company_name=company_name)
        if sponsored_role_evidence:
            return VisaAssessment(
                likelihood=VisaLikelihood.LIKELY,
                evidence_tags=[VisaEvidenceTag.SPONSORED_ROLE_SIGNAL],
                reasoning=(
                    "No register match was found, but public hiring signals suggest prior "
                    "visa sponsorship activity."
                ),
                evidence=sponsored_role_evidence,
            )

        return VisaAssessment(
            likelihood=VisaLikelihood.UNKNOWN,
            evidence_tags=[VisaEvidenceTag.NONE],
            reasoning=(
                "No direct register match, related entity match, or sponsored-role signal "
                "was found in available public sources."
            ),
            evidence=[],
        )

    def _find_direct_match(
        self,
        *,
        company_name: str,
        entries: list[SponsorRegisterEntry],
    ) -> SponsorRegisterEntry | None:
        best_entry: SponsorRegisterEntry | None = None
        best_score = 0.0
        for entry in entries:
            score = _name_similarity(company_name, entry.organisation_name)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is None:
            return None
        if best_score >= 0.86:
            return best_entry

        # Handle brand-vs-legal variants like "Cleo" vs "Cleo AI Ltd".
        # Require the longer side to have at most 2 canonical tokens to avoid
        # false positives (e.g. "Bumble" should NOT match "Bumble Hole Foods").
        left = _canonical_name(company_name)
        right = _canonical_name(best_entry.organisation_name)
        left_tokens = left.split()
        right_tokens = right.split()
        if (
            left
            and right
            and (left in right or right in left)
            and (len(left_tokens) == 1 or len(right_tokens) == 1)
            and max(len(left_tokens), len(right_tokens)) <= 2
        ):
            return best_entry
        return None

    def _find_related_entity(
        self,
        *,
        company_name: str,
        entries: list[SponsorRegisterEntry],
    ) -> SponsorRegisterEntry | None:
        anchor_tokens = _anchor_tokens(company_name)
        # Require at least 2 anchor tokens to avoid false positives on
        # single-word brand names (e.g. "Bumble" matching "Bumble Hole Foods").
        if len(anchor_tokens) < 2:
            return None

        best_entry: SponsorRegisterEntry | None = None
        best_overlap = 0.0
        for entry in entries:
            entry_tokens = set(_tokenize(entry.organisation_name))
            if not entry_tokens:
                continue
            overlap = len(anchor_tokens & entry_tokens) / len(anchor_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_entry = entry
        if best_entry is not None and best_overlap >= 0.5:
            return best_entry
        return None

    def _search_sponsored_role_signals(self, *, company_name: str) -> list[str]:
        if self.search_tool is None:
            return []
        queries = [
            f"{company_name} visa sponsorship job UK",
            f"{company_name} Skilled Worker sponsorship role",
            f"{company_name} certificate of sponsorship hiring",
        ]

        evidence_urls: list[str] = []
        seen: set[str] = set()
        for iteration, query in enumerate(queries, start=1):
            for result in self.search_tool.search(
                query=query, iteration=iteration, max_results=5
            ):
                if result.url in seen:
                    continue
                seen.add(result.url)
                haystack = " ".join(
                    [result.title, result.snippet, result.content]
                ).lower()
                if _contains_sponsored_role_signal(haystack):
                    evidence_urls.append(result.url)
        return evidence_urls[:5]


def _parse_register_csv(csv_text: str) -> list[SponsorRegisterEntry]:
    reader = csv.DictReader(csv_text.splitlines())
    entries: list[SponsorRegisterEntry] = []
    for row in reader:
        name = _extract_organisation_name(row)
        if not name:
            continue
        entries.append(SponsorRegisterEntry(organisation_name=name, source_row=row))
    return entries


def _extract_organisation_name(row: dict[str, str]) -> str:
    for key, value in row.items():
        key_l = key.strip().lower()
        if "organisation" in key_l and "name" in key_l:
            return (value or "").strip()
        if "organization" in key_l and "name" in key_l:
            return (value or "").strip()
    if row:
        first_value = next(iter(row.values()))
        return (first_value or "").strip()
    return ""


def _canonical_name(name: str) -> str:
    tokens = _tokenize(name)
    filtered = [token for token in tokens if token not in _ENTITY_SUFFIXES]
    return " ".join(filtered).strip()


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", (text or "").lower())
    return [piece for piece in cleaned.split() if piece]


def _anchor_tokens(company_name: str) -> set[str]:
    tokens = [
        token
        for token in _tokenize(company_name)
        if token not in _ENTITY_SUFFIXES and len(token) >= _SIGNIFICANT_TOKEN_MIN_LEN
    ]
    if not tokens:
        return set()
    return set(tokens[:2])


def _name_similarity(left: str, right: str) -> float:
    left_canonical = _canonical_name(left)
    right_canonical = _canonical_name(right)
    if not left_canonical or not right_canonical:
        return 0.0
    if left_canonical == right_canonical:
        return 1.0

    left_tokens = set(left_canonical.split())
    right_tokens = set(right_canonical.split())
    token_overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    contains_bonus = 0.15 if left_canonical in right_canonical or right_canonical in left_canonical else 0.0
    return min(1.0, token_overlap + contains_bonus)


def _contains_sponsored_role_signal(text: str) -> bool:
    signals = (
        "visa sponsorship",
        "skilled worker visa",
        "certificate of sponsorship",
        "uk sponsorship",
        "sponsorship available",
        "can sponsor",
    )
    return any(signal in text for signal in signals)
