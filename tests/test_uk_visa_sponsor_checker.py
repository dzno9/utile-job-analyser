from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import tempfile
import unittest

from models import JobContext, VisaEvidenceTag, VisaLikelihood
from providers.company_researcher import SearchResult
from providers.uk_visa_sponsor_checker import (
    CachedGovUkSponsorRegister,
    GovUkRegisterSource,
    UKVisaSponsorChecker,
)


class FakeGovUkRegisterSource(GovUkRegisterSource):
    def __init__(self, *, csv_text: str, csv_url: str = "https://gov.uk/register.csv") -> None:
        self.csv_text = csv_text
        self.csv_url = csv_url
        self.html_calls = 0
        self.csv_calls = 0

    def fetch_html(self, *, index_url: str) -> str:
        self.html_calls += 1
        return f'<html><body><a href="{self.csv_url}">Download CSV</a></body></html>'

    def fetch_csv(self, *, csv_url: str) -> str:
        self.csv_calls += 1
        return self.csv_text


class FakeSearchTool:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(
        self, *, query: str, iteration: int, max_results: int
    ) -> list[SearchResult]:
        self.queries.append(query)
        query_l = query.lower()
        if "tiny startup" in query_l:
            return [
                SearchResult(
                    title="Tiny Startup Data Scientist role",
                    url="https://jobs.example.com/tiny-startup/data-scientist",
                    published_date="2026-01-20",
                    snippet="UK role with visa sponsorship available for strong candidates.",
                    content="Skilled Worker visa sponsorship available.",
                )
            ]
        return []


class TestUKVisaSponsorChecker(unittest.TestCase):
    def setUp(self) -> None:
        self.csv_text = (
            "Organisation Name,Town/City\n"
            "Monzo Bank Limited,London\n"
            "Global Parent Holdings Limited,London\n"
            "Meta Platforms UK Limited,London\n"
            "Cleo AI Ltd,London\n"
        )

    def _build_checker(
        self,
        *,
        csv_text: str | None = None,
        search_tool: FakeSearchTool | None = None,
    ) -> tuple[UKVisaSponsorChecker, FakeGovUkRegisterSource]:
        source = FakeGovUkRegisterSource(csv_text=csv_text or self.csv_text)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        register = CachedGovUkSponsorRegister(source=source, cache_dir=temp_dir.name, ttl_days=7)
        return UKVisaSponsorChecker(register=register, search_tool=search_tool), source

    def _job(self, company_name: str) -> JobContext:
        return JobContext(
            job_id=f"{company_name}-1",
            job_title="Data Analyst",
            company_name=company_name,
            location="London",
            job_url="https://jobs.example.com/test-role",
            posting_text="Role details.",
            required_skills=["SQL"],
        )

    def test_confirmed_sponsor_for_direct_register_match(self) -> None:
        checker, _ = self._build_checker()
        assessment = checker.assess(job_context=self._job("Monzo"))
        self.assertEqual(assessment.likelihood, VisaLikelihood.CONFIRMED_SPONSOR)
        self.assertEqual(
            assessment.evidence_tags, [VisaEvidenceTag.DIRECT_REGISTER_MATCH]
        )

    def test_likely_for_related_entity_match(self) -> None:
        checker, _ = self._build_checker()
        assessment = checker.assess(job_context=self._job("Global Parent Labs"))
        self.assertEqual(assessment.likelihood, VisaLikelihood.LIKELY)
        self.assertEqual(assessment.evidence_tags, [VisaEvidenceTag.RELATED_ENTITY_MATCH])

    def test_likely_for_sponsored_role_signal_when_not_in_register(self) -> None:
        search_tool = FakeSearchTool()
        checker, _ = self._build_checker(search_tool=search_tool)
        assessment = checker.assess(job_context=self._job("Tiny Startup"))
        self.assertEqual(assessment.likelihood, VisaLikelihood.LIKELY)
        self.assertEqual(
            assessment.evidence_tags, [VisaEvidenceTag.SPONSORED_ROLE_SIGNAL]
        )
        self.assertTrue(any("visa sponsorship" in query.lower() for query in search_tool.queries))

    def test_unknown_when_no_signal_exists(self) -> None:
        checker, _ = self._build_checker(search_tool=FakeSearchTool())
        assessment = checker.assess(job_context=self._job("Completely Unknown Entity"))
        self.assertEqual(assessment.likelihood, VisaLikelihood.UNKNOWN)
        self.assertEqual(assessment.evidence_tags, [VisaEvidenceTag.NONE])

    def test_fuzzy_matching_handles_common_name_variants(self) -> None:
        checker, _ = self._build_checker()
        cleo = checker.assess(job_context=self._job("Cleo"))
        meta = checker.assess(job_context=self._job("Meta Platforms Inc"))
        self.assertEqual(cleo.likelihood, VisaLikelihood.CONFIRMED_SPONSOR)
        self.assertEqual(meta.likelihood, VisaLikelihood.CONFIRMED_SPONSOR)

    def test_register_cache_is_reused_and_refreshed_weekly(self) -> None:
        source = FakeGovUkRegisterSource(csv_text=self.csv_text)
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        register = CachedGovUkSponsorRegister(source=source, cache_dir=temp_dir.name, ttl_days=7)
        checker = UKVisaSponsorChecker(register=register)

        checker.assess(job_context=self._job("Monzo"))
        checker.assess(job_context=self._job("Monzo"))
        self.assertEqual(source.html_calls, 1)
        self.assertEqual(source.csv_calls, 1)

        stale_time = (datetime.now(UTC) - timedelta(days=8)).isoformat()
        register.cache_meta_path.write_text(
            json.dumps({"fetched_at": stale_time, "source_url": source.csv_url}),
            encoding="utf-8",
        )
        checker.assess(job_context=self._job("Monzo"))
        self.assertEqual(source.html_calls, 2)
        self.assertEqual(source.csv_calls, 2)


if __name__ == "__main__":
    unittest.main()
