from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from models import (
    CompanyBrief,
    JobContext,
    MatchStrength,
    Recommendation,
    ResearchFinding,
    ScoreRow,
    Scorecard,
    VisaAssessment,
    VisaEvidenceTag,
    VisaLikelihood,
)
from stages.ui_orchestrator import PipelineOrchestrator, PipelineRequest


class FakeScrapeResult:
    def __init__(
        self,
        *,
        scrape_succeeded: bool,
        manual_text_input_required: bool,
        confidence_notes: list[str],
        job_context: JobContext | None,
    ) -> None:
        self.source_url = "https://jobs.example.com/role"
        self.platform = "example"
        self.scrape_succeeded = scrape_succeeded
        self.manual_text_input_required = manual_text_input_required
        self.confidence_score = 0.2 if manual_text_input_required else 0.9
        self.confidence_notes = confidence_notes
        self.job_context = job_context
        self.error = None if scrape_succeeded else "fetch failed"


class FakeScraper:
    def __init__(self, result: FakeScrapeResult) -> None:
        self.result = result

    def scrape(self, _: str):
        return self.result


class FakeGapResult:
    def __init__(self, recommendation: Recommendation, should_continue: bool, score: float) -> None:
        self.recommendation = recommendation
        self.should_continue_pipeline = should_continue
        self.cv_text = "Jane Doe\nSenior Analyst"
        self.scorecard = Scorecard(
            rows=[
                ScoreRow(
                    requirement_from_job_post="SQL",
                    matching_experience="Used SQL in production analytics.",
                    rationale="Direct evidence",
                    match_strength=MatchStrength.STRONG,
                )
            ],
            total_score=score,
            recommendation=recommendation,
            pipeline_should_continue=should_continue,
            risk_flags=[] if should_continue else ["Critical gaps"],
        )


class FakeGapAnalyzer:
    def __init__(self, result: FakeGapResult) -> None:
        self.result = result

    def analyze(
        self,
        *,
        job_context: JobContext,
        cv_pdf_path: str,
        candidate_notes: str,
        cv_text_override: str = "",
    ) -> FakeGapResult:
        del job_context, cv_pdf_path, candidate_notes, cv_text_override
        return self.result


class FailingGapAnalyzer:
    def analyze(
        self,
        *,
        job_context: JobContext,
        cv_pdf_path: str,
        candidate_notes: str,
        cv_text_override: str = "",
    ):
        del job_context, cv_pdf_path, candidate_notes, cv_text_override
        raise ValueError("Could not extract text from CV PDF with fitz or pdfplumber")


class FakeCompanyResearchResult:
    def __init__(self) -> None:
        self.company_brief = CompanyBrief(
            company_name="Acme",
            summary="Acme launched a new product [1] and expanded hiring [2].",
            industry="SaaS",
            sources=["https://example.com/1", "https://example.com/2"],
        )
        self.findings = [
            ResearchFinding(
                claim="Acme launched a new product.",
                source_url="https://example.com/1",
                source_date="2026-01-01",
                relevance="Product momentum",
                confidence=0.8,
            )
        ]
        self.warnings = ["1 finding(s) have low confidence and should be verified."]


class FakeCompanyResearcher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def research(self, *, job_context: JobContext, candidate_notes: str):
        del job_context, candidate_notes
        if self.fail:
            raise RuntimeError("search unavailable")
        return FakeCompanyResearchResult()


class FakeVisaChecker:
    def assess(self, *, job_context: JobContext) -> VisaAssessment:
        del job_context
        return VisaAssessment(
            likelihood=VisaLikelihood.LIKELY,
            evidence_tags=[VisaEvidenceTag.SPONSORED_ROLE_SIGNAL],
            reasoning="Public job ads mention sponsorship.",
            evidence=["https://example.com/visa"],
        )


class FailingVisaChecker:
    def assess(self, *, job_context: JobContext) -> VisaAssessment:
        del job_context
        raise RuntimeError("visa provider unavailable")


class FakeMaterialsResult:
    def __init__(self) -> None:
        from models import CoverLetter

        self.cover_letter = CoverLetter(
            candidate_name="Jane Doe",
            company_name="Acme",
            job_title="Data Analyst",
            draft_markdown="Dear team,\n\nI can contribute immediately with analytics delivery.",
            emphasis_strategy=["Lead with direct SQL impact."],
        )
        self.cv_tweaks = ["Quantify SQL impact with before/after metric."]


class FakeMaterialsGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs):
        del kwargs
        self.calls += 1
        return FakeMaterialsResult()


class FakeCVParser:
    def parse(self, _: str) -> str:
        return "Jane Doe\nSenior Analyst"


class TestPipelineOrchestrator(unittest.TestCase):
    def _job_context(self) -> JobContext:
        return JobContext(
            job_id="job-1",
            job_title="Data Analyst",
            company_name="Acme",
            location="London",
            job_url="https://jobs.example.com/acme/data-analyst",
            posting_text="Analyze data and build dashboards.",
            requirements=["SQL"],
            required_skills=["SQL"],
        )

    def _request(self) -> PipelineRequest:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake")
            path = f.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        return PipelineRequest(
            job_url="https://jobs.example.com/acme/data-analyst",
            cv_pdf_path=path,
            candidate_notes="I used their product for 2 years.",
        )

    def test_happy_path_runs_all_stages_and_generates_report(self) -> None:
        scraper = FakeScraper(
            FakeScrapeResult(
                scrape_succeeded=True,
                manual_text_input_required=False,
                confidence_notes=["Computed confidence=0.91"],
                job_context=self._job_context(),
            )
        )
        materials = FakeMaterialsGenerator()
        orchestrator = PipelineOrchestrator(
            scraper=scraper,
            gap_analyzer=FakeGapAnalyzer(
                FakeGapResult(Recommendation.CONFIDENT_APPLY, True, 85)
            ),
            company_researcher=FakeCompanyResearcher(),
            visa_checker=FakeVisaChecker(),
            materials_generator=materials,
            cv_parser=FakeCVParser(),
        )
        events = []

        result = orchestrator.run(self._request(), on_event=events.append)

        self.assertIsNotNone(result.report)
        self.assertTrue(result.cover_letter_generated)
        self.assertEqual(materials.calls, 1)
        self.assertFalse(result.needs_manual_posting_text)
        self.assertIn("Scrape: Computed confidence=0.91", result.warnings)
        self.assertTrue(any(event.stage_key == "assemble" for event in events))

    def test_scrape_failure_requires_manual_text_then_recovers(self) -> None:
        scraper = FakeScraper(
            FakeScrapeResult(
                scrape_succeeded=False,
                manual_text_input_required=True,
                confidence_notes=["URL fetch failed"],
                job_context=None,
            )
        )
        orchestrator = PipelineOrchestrator(
            scraper=scraper,
            gap_analyzer=FakeGapAnalyzer(
                FakeGapResult(Recommendation.APPLY_WITH_CAVEATS, True, 45)
            ),
            company_researcher=FakeCompanyResearcher(),
            visa_checker=FakeVisaChecker(),
            materials_generator=FakeMaterialsGenerator(),
            cv_parser=FakeCVParser(),
        )

        initial = orchestrator.run(self._request())
        self.assertTrue(initial.needs_manual_posting_text)
        self.assertIsNone(initial.report)

        request = PipelineRequest(
            job_url="https://jobs.example.com/acme/data-analyst",
            cv_pdf_path=self._request().cv_pdf_path,
            candidate_notes="I used their product for 2 years.",
            manual_job_posting_text=(
                "Data Analyst\nAcme\nRequirements\n- Strong SQL and dashboarding\n- Stakeholder collaboration"
            ),
        )
        recovered = orchestrator.run(request)
        self.assertFalse(recovered.needs_manual_posting_text)
        self.assertIsNotNone(recovered.report)

    def test_do_not_apply_skips_materials_stage(self) -> None:
        materials = FakeMaterialsGenerator()
        orchestrator = PipelineOrchestrator(
            scraper=FakeScraper(
                FakeScrapeResult(
                    scrape_succeeded=True,
                    manual_text_input_required=False,
                    confidence_notes=[],
                    job_context=self._job_context(),
                )
            ),
            gap_analyzer=FakeGapAnalyzer(
                FakeGapResult(Recommendation.DO_NOT_APPLY, False, 20)
            ),
            company_researcher=FakeCompanyResearcher(),
            visa_checker=FakeVisaChecker(),
            materials_generator=materials,
            cv_parser=FakeCVParser(),
        )
        events = []

        result = orchestrator.run(self._request(), on_event=events.append)

        self.assertIsNotNone(result.report)
        self.assertFalse(result.cover_letter_generated)
        self.assertEqual(materials.calls, 0)
        self.assertEqual(result.report.recommendation, Recommendation.DO_NOT_APPLY)
        self.assertIn("Skipped - score below threshold", result.report.cover_letter.draft_markdown)
        self.assertTrue(
            any(event.stage_key == "materials" and event.status == "skipped" for event in events)
        )

    def test_cv_extraction_failure_requests_manual_cv_text(self) -> None:
        orchestrator = PipelineOrchestrator(
            scraper=FakeScraper(
                FakeScrapeResult(
                    scrape_succeeded=True,
                    manual_text_input_required=False,
                    confidence_notes=[],
                    job_context=self._job_context(),
                )
            ),
            gap_analyzer=FailingGapAnalyzer(),
            company_researcher=FakeCompanyResearcher(),
            visa_checker=FakeVisaChecker(),
            materials_generator=FakeMaterialsGenerator(),
            cv_parser=FakeCVParser(),
        )

        result = orchestrator.run(self._request())

        self.assertTrue(result.needs_manual_cv_text)
        self.assertIsNone(result.report)
        self.assertTrue(
            any("CV text extraction failed" in warning for warning in result.warnings)
        )

    def test_missing_requirements_forces_manual_job_text(self) -> None:
        weak_job = JobContext(
            job_id="job-weak",
            job_title="Senior Product Manager",
            company_name="Pinpoint",
            location="Remote",
            job_url="https://jobs.example.com/pinpoint/senior-pm",
            posting_text="Thin posting content.",
            requirements=[],
            required_skills=[],
        )
        orchestrator = PipelineOrchestrator(
            scraper=FakeScraper(
                FakeScrapeResult(
                    scrape_succeeded=True,
                    manual_text_input_required=True,
                    confidence_notes=["Could not extract requirements or skills from posting"],
                    job_context=weak_job,
                )
            ),
            gap_analyzer=FakeGapAnalyzer(
                FakeGapResult(Recommendation.CONFIDENT_APPLY, True, 85)
            ),
            company_researcher=FakeCompanyResearcher(),
            visa_checker=FakeVisaChecker(),
            materials_generator=FakeMaterialsGenerator(),
            cv_parser=FakeCVParser(),
        )

        result = orchestrator.run(self._request())
        self.assertTrue(result.needs_manual_posting_text)
        self.assertIsNone(result.report)
        self.assertTrue(
            any("paste full job description" in warning.lower() for warning in result.warnings)
        )

    def test_user_facing_text_avoids_fallback_wording(self) -> None:
        orchestrator = PipelineOrchestrator(
            scraper=FakeScraper(
                FakeScrapeResult(
                    scrape_succeeded=False,
                    manual_text_input_required=True,
                    confidence_notes=["URL fetch failed"],
                    job_context=None,
                )
            ),
            gap_analyzer=FakeGapAnalyzer(
                FakeGapResult(Recommendation.APPLY_WITH_CAVEATS, True, 45)
            ),
            company_researcher=FakeCompanyResearcher(fail=True),
            visa_checker=FailingVisaChecker(),
            materials_generator=FakeMaterialsGenerator(),
            cv_parser=FakeCVParser(),
        )
        request = PipelineRequest(
            job_url="https://jobs.example.com/acme/data-analyst",
            cv_pdf_path=self._request().cv_pdf_path,
            candidate_notes="I used their product for 2 years.",
            manual_job_posting_text=(
                "Data Analyst\nAcme\nRequirements\n- Strong SQL and dashboarding\n- Stakeholder collaboration"
            ),
        )
        events = []

        result = orchestrator.run(request, on_event=events.append)

        self.assertIsNotNone(result.report)
        self.assertTrue(all("fallback" not in warning.lower() for warning in result.warnings))
        self.assertTrue(all("fallback" not in event.detail.lower() for event in events))

    def test_synthetic_pipeline_latency_is_below_budget(self) -> None:
        scraper = FakeScraper(
            FakeScrapeResult(
                scrape_succeeded=True,
                manual_text_input_required=False,
                confidence_notes=[],
                job_context=self._job_context(),
            )
        )
        orchestrator = PipelineOrchestrator(
            scraper=scraper,
            gap_analyzer=FakeGapAnalyzer(
                FakeGapResult(Recommendation.CONFIDENT_APPLY, True, 88)
            ),
            company_researcher=FakeCompanyResearcher(),
            visa_checker=FakeVisaChecker(),
            materials_generator=FakeMaterialsGenerator(),
            cv_parser=FakeCVParser(),
        )

        started = time.perf_counter()
        result = orchestrator.run(self._request())
        elapsed = time.perf_counter() - started

        self.assertIsNotNone(result.report)
        # Synthetic latency guard for orchestration overhead only.
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
