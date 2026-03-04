from __future__ import annotations

import unittest
from datetime import datetime
from unittest import mock

import app
from models import (
    AnalysisReport,
    CompanyBrief,
    CoverLetter,
    JobContext,
    MatchStrength,
    Recommendation,
    ScoreRow,
    Scorecard,
    VisaAssessment,
    VisaEvidenceTag,
    VisaLikelihood,
)
from stages.ui_orchestrator import StageEvent


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        del exc_type, exc, tb
        return False


class _FakePlaceholder:
    def __init__(self) -> None:
        self.markdowns: list[str] = []

    def markdown(self, text: str, **kwargs) -> None:  # noqa: ANN003
        del kwargs
        self.markdowns.append(text)


class _FakeStreamlit:
    def __init__(self) -> None:
        self.markdowns: list[str] = []
        self.dataframes = []
        self.warnings: list[str] = []
        self.expander_labels: list[str] = []

    def tabs(self, labels):  # noqa: ANN001
        return [_DummyContext() for _ in labels]

    def metric(self, *args, **kwargs):  # noqa: ANN001, ANN002
        del args, kwargs

    def dataframe(self, df, **kwargs):  # noqa: ANN001
        del kwargs
        self.dataframes.append(df)

    def markdown(self, text: str, **kwargs):  # noqa: ANN003
        del kwargs
        self.markdowns.append(text)

    def expander(self, label: str, **kwargs):  # noqa: ANN003
        del kwargs
        self.expander_labels.append(label)
        return _DummyContext()

    def warning(self, text: str) -> None:
        self.warnings.append(text)

    def info(self, text: str) -> None:
        self.markdowns.append(text)

    def write(self, text: str) -> None:
        self.markdowns.append(text)

    def caption(self, text: str) -> None:
        self.markdowns.append(text)


def _sample_report() -> AnalysisReport:
    return AnalysisReport(
        job_context=JobContext(
            job_id="job-1",
            job_title="Data Analyst",
            company_name="Acme",
            location="London",
            job_url="https://jobs.example.com/acme/data-analyst",
            posting_text="Analyze data and ship insights.",
            requirements=["SQL"],
            required_skills=["SQL"],
        ),
        company_brief=CompanyBrief(
            company_name="Acme",
            summary="Acme is growing quickly.",
            industry="SaaS",
            sources=["https://example.com/company"],
        ),
        research_findings=[],
        visa_assessment=VisaAssessment(
            likelihood=VisaLikelihood.LIKELY,
            evidence_tags=[VisaEvidenceTag.SPONSORED_ROLE_SIGNAL],
            reasoning="Signals suggest sponsorship is possible.",
            evidence=["https://example.com/visa"],
        ),
        scorecard=Scorecard(
            rows=[
                ScoreRow(
                    requirement_from_job_post="SQL",
                    matching_experience="[fallback] Dika Satria +44-7824-711960 | dzno9a@gmail",
                    rationale="[fallback] Built analytics pipelines | with strong ownership",
                    match_strength=MatchStrength.PARTIAL,
                )
            ],
            total_score=50,
            recommendation=Recommendation.APPLY_WITH_CAVEATS,
            pipeline_should_continue=True,
            risk_flags=[],
        ),
        cover_letter=CoverLetter(
            candidate_name="Dika",
            company_name="Acme",
            job_title="Data Analyst",
            draft_markdown="[fallback] Dear team, I can contribute immediately. +44-7824-711960",
            emphasis_strategy=["[fallback] Focus on SQL outcomes"],
        ),
        cv_tweaks=["Add measurable impact"],
        summary="Recommendation: ApplyWithCaveats.",
        recommendation=Recommendation.APPLY_WITH_CAVEATS,
        generated_at=datetime.utcnow(),
    )


class TestAppDisplayWiring(unittest.TestCase):
    def test_build_score_ring_svg_uses_score_tier_color(self) -> None:
        svg = app._build_score_ring_svg(50, size=120, stroke_width=8)

        self.assertIn('stroke="#d97706"', svg)
        self.assertIn("stroke-dasharray", svg)
        self.assertIn(">50<", svg)
        self.assertIn(">Partial Match<", svg)

    def test_render_results_hero_includes_recommendation_and_warning_count(self) -> None:
        fake_st = _FakeStreamlit()
        report = _sample_report()
        report.scorecard.risk_flags = ["Missing API ownership examples", "Needs domain depth"]

        with mock.patch.object(app, "st", fake_st):
            app._render_results_hero(report)

        hero_html = "\n".join(fake_st.markdowns)
        self.assertIn("Data Analyst", hero_html)
        self.assertIn("at Acme", hero_html)
        self.assertIn("Worth Applying - Review Caveats First", hero_html)
        self.assertIn("2 items need attention", hero_html)
        self.assertIn("50/100", hero_html)

    def test_render_report_uses_display_cleaners(self) -> None:
        fake_st = _FakeStreamlit()
        with (
            mock.patch.object(app, "st", fake_st),
            mock.patch.object(app, "render_copy_button"),
        ):
            app.render_report(
                _sample_report(),
                warnings=["Scrape: Computed confidence=0.75"],
                cover_letter_generated=True,
                show_recommendation_banner=True,
            )

        scorecard_html = "\n".join(fake_st.markdowns)
        # Evidence should be cleaned: no [fallback], no phone number, no email
        self.assertNotIn("[fallback]", scorecard_html)
        self.assertNotIn("+44-7824-711960", scorecard_html)
        self.assertNotIn("dzno9a@gmail", scorecard_html)
        # Cleaned rationale should appear
        self.assertIn("Built analytics pipelines with strong ownership", scorecard_html)
        self.assertIn("Data Quality Notes", fake_st.expander_labels)
        self.assertIn(
            "Job posting scraped with 75% confidence - some details may be approximate",
            fake_st.warnings,
        )
        banner_html = "\n".join(fake_st.markdowns)
        self.assertIn("Conditional Match", banner_html)
        self.assertNotIn("ApplyWithCaveats", banner_html)

    def test_update_progress_view_maps_loading_step(self) -> None:
        placeholder = _FakePlaceholder()
        progress_state = {
            "completed_steps": set(),
            "active_step": app.LOADING_STEP_ORDER[0],
        }
        event = StageEvent(
            stage_key="gap",
            stage_name="Gap Analysis",
            status="running",
            detail="[fallback] Comparing CV to role",
        )

        app._update_progress_view(event, placeholder, progress_state)

        self.assertEqual(progress_state["active_step"], "Gap Analysis")
        self.assertIn("Job Scrape", progress_state["completed_steps"])
        self.assertIn("CV Parse", progress_state["completed_steps"])
        self.assertIn("Comparing your fit", placeholder.markdowns[-1])

    def test_update_progress_view_keeps_company_step_active_until_visa_finishes(self) -> None:
        placeholder = _FakePlaceholder()
        progress_state = {
            "completed_steps": {"Job Scrape", "CV Parse", "Gap Analysis"},
            "active_step": "Company Brief",
        }
        company_complete = StageEvent(
            stage_key="company",
            stage_name="Company Brief",
            status="complete",
            detail="Research complete",
        )
        visa_complete = StageEvent(
            stage_key="visa",
            stage_name="Visa Check",
            status="complete",
            detail="Likely",
        )

        app._update_progress_view(company_complete, placeholder, progress_state)
        self.assertNotIn("Company Brief", progress_state["completed_steps"])

        app._update_progress_view(visa_complete, placeholder, progress_state)
        self.assertIn("Company Brief", progress_state["completed_steps"])
        self.assertEqual(progress_state["active_step"], "Application Materials")


if __name__ == "__main__":
    unittest.main()
