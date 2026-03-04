import unittest

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
from stages.application_materials_generator import ApplicationMaterialsGenerator


class TestApplicationMaterialsGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = ApplicationMaterialsGenerator()
        self.visa = VisaAssessment(
            likelihood=VisaLikelihood.LIKELY,
            evidence_tags=[VisaEvidenceTag.SPONSORED_ROLE_SIGNAL],
            reasoning="Sponsorship appears available in similar postings.",
            evidence=["https://jobs.example.com/visa-note"],
        )

    def test_fintech_pm_generates_confident_specific_outputs(self) -> None:
        job = JobContext(
            job_id="pm-101",
            job_title="Senior Product Manager, Fintech",
            company_name="LedgerLoop",
            location="London",
            job_url="https://jobs.example.com/ledgerloop-pm",
            posting_text="Lead payments roadmap and experimentation in regulated markets.",
            requirements=[
                "Lead roadmap for payments and onboarding flows",
                "Run experimentation strategy for conversion funnels",
                "Use SQL analytics to make KPI trade-offs",
            ],
        )
        brief = CompanyBrief(
            company_name="LedgerLoop",
            summary="LedgerLoop builds API-led financial infrastructure for SMEs.",
            industry="Fintech",
            website="https://ledgerloop.example.com",
        )
        findings = [
            ResearchFinding(
                claim="LedgerLoop raised a Series B to scale regulated SME payments across Europe.",
                source_url="https://news.example.com/ledgerloop-series-b",
                source_date="2026-01-12",
                relevance="Growth context for hiring priorities.",
                confidence=0.85,
            ),
            ResearchFinding(
                claim="LedgerLoop launched instant settlement APIs for cross-border merchants.",
                source_url="https://blog.example.com/ledgerloop-instant-settlement",
                source_date="2025-11-04",
                relevance="Product direction tied to role scope.",
                confidence=0.82,
            ),
        ]
        scorecard = Scorecard(
            rows=[
                ScoreRow(
                    requirement_from_job_post="Lead roadmap for payments and onboarding flows",
                    matching_experience="Built quarterly roadmap for wallet onboarding and KYC at FinPay.",
                    rationale="Direct product ownership in fintech onboarding flows.",
                    match_strength=MatchStrength.STRONG,
                ),
                ScoreRow(
                    requirement_from_job_post="Run experimentation strategy for conversion funnels",
                    matching_experience="Built A/B testing infrastructure using Mida and Optimizely.",
                    rationale="Hands-on experimentation platform delivery.",
                    match_strength=MatchStrength.STRONG,
                ),
                ScoreRow(
                    requirement_from_job_post="Use SQL analytics to make KPI trade-offs",
                    matching_experience="Used SQL and Amplitude to drive KPI trade-off decisions across teams.",
                    rationale="Strong analytics ownership tied to KPI choices.",
                    match_strength=MatchStrength.STRONG,
                ),
            ],
            total_score=90,
            recommendation=Recommendation.CONFIDENT_APPLY,
            pipeline_should_continue=True,
            risk_flags=[],
        )

        result = self.generator.generate(
            job_context=job,
            company_brief=brief,
            research_findings=findings,
            visa_assessment=self.visa,
            scorecard=scorecard,
            recommendation=Recommendation.CONFIDENT_APPLY,
            candidate_name="Alex Doe",
            candidate_notes=(
                "At Verto, I built onboarding instrumentation and collaborated with compliance."
            ),
            cv_content=(
                "Built quarterly roadmap for wallet onboarding and KYC at FinPay. "
                "Built A/B testing infrastructure using Mida and Optimizely. "
                "Used SQL and Amplitude to drive KPI trade-off decisions."
            ),
        )

        letter = result.cover_letter.draft_markdown
        self.assertIn("Series B", letter)
        self.assertIn("instant settlement APIs", letter)
        self.assertIn("Lead roadmap for payments and onboarding flows", letter)
        self.assertIn("Run experimentation strategy for conversion funnels", letter)
        self.assertIn("Use SQL analytics to make KPI trade-offs", letter)
        self.assertNotIn("passionate", letter.lower())
        self.assertNotIn("excited to apply", letter.lower())
        self.assertNotIn("great fit", letter.lower())

        self.assertGreaterEqual(len(result.cover_letter.emphasis_strategy), 4)
        self.assertTrue(
            any("because" in item.lower() for item in result.cover_letter.emphasis_strategy)
        )

        self.assertGreaterEqual(len(result.cv_tweaks), 3)
        self.assertTrue(any("Series B" in tweak for tweak in result.cv_tweaks))
        self.assertTrue(
            any("Reword CV evidence under" in tweak for tweak in result.cv_tweaks)
        )

    def test_borderline_growth_role_acknowledges_gaps_and_differs(self) -> None:
        job = JobContext(
            job_id="gm-202",
            job_title="Growth Marketing Manager",
            company_name="ScaleLoop",
            location="London",
            job_url="https://jobs.example.com/scaleloop-growth",
            posting_text="Own lifecycle and paid channels for growth.",
            requirements=[
                "Own lifecycle email campaigns and CRM automation",
                "Manage paid acquisition channel budget",
                "Design referral growth loops",
            ],
        )
        brief = CompanyBrief(
            company_name="ScaleLoop",
            summary="ScaleLoop builds attribution products for PLG teams.",
            industry="SaaS",
            website="https://scaleloop.example.com",
        )
        findings = [
            ResearchFinding(
                claim="ScaleLoop introduced an AI-led lifecycle optimizer for onboarding flows.",
                source_url="https://blog.example.com/scaleloop-lifecycle",
                source_date="2026-02-10",
                relevance="Signals lifecycle optimization priority.",
                confidence=0.81,
            ),
            ResearchFinding(
                claim="ScaleLoop expanded its partner ecosystem after a recent enterprise push.",
                source_url="https://news.example.com/scaleloop-partners",
                source_date="2026-01-07",
                relevance="Shows go-to-market priorities.",
                confidence=0.78,
            ),
        ]
        scorecard = Scorecard(
            rows=[
                ScoreRow(
                    requirement_from_job_post="Own lifecycle email campaigns and CRM automation",
                    matching_experience="No strong matching evidence found in CV or candidate notes.",
                    rationale="Limited direct lifecycle ownership evidence.",
                    match_strength=MatchStrength.GAP,
                ),
                ScoreRow(
                    requirement_from_job_post="Manage paid acquisition channel budget",
                    matching_experience="No strong matching evidence found in CV or candidate notes.",
                    rationale="No explicit paid budget ownership examples.",
                    match_strength=MatchStrength.GAP,
                ),
                ScoreRow(
                    requirement_from_job_post="Design referral growth loops",
                    matching_experience="Designed a referral loop at Verto that improved acquisition quality.",
                    rationale="Direct growth loop evidence.",
                    match_strength=MatchStrength.STRONG,
                ),
            ],
            total_score=40,
            recommendation=Recommendation.APPLY_WITH_CAVEATS,
            pipeline_should_continue=True,
            risk_flags=["Critical gaps: lifecycle and paid budget ownership"],
        )

        result = self.generator.generate(
            job_context=job,
            company_brief=brief,
            research_findings=findings,
            visa_assessment=self.visa,
            scorecard=scorecard,
            recommendation=Recommendation.APPLY_WITH_CAVEATS,
            candidate_name="Alex Doe",
            candidate_notes=(
                "I have light lifecycle email exposure and stronger referral experimentation experience."
            ),
            cv_content="Designed referral loop at Verto with measurable acquisition lift.",
        )

        letter = result.cover_letter.draft_markdown
        self.assertIn("I have not yet owned", letter)
        self.assertIn("Own lifecycle email campaigns and CRM automation", letter)
        self.assertIn("Manage paid acquisition channel budget", letter)
        self.assertIn("AI-led lifecycle optimizer", letter)
        self.assertTrue(any("ScaleLoop" in tweak or "lifecycle" in tweak for tweak in result.cv_tweaks))

        baseline_result = self._fintech_output()
        baseline_sentences = {s.strip() for s in baseline_result.split(".") if s.strip()}
        growth_sentences = {s.strip() for s in letter.split(".") if s.strip()}
        self.assertEqual(baseline_sentences.intersection(growth_sentences), set())

    def _fintech_output(self) -> str:
        job = JobContext(
            job_id="pm-101",
            job_title="Senior Product Manager, Fintech",
            company_name="LedgerLoop",
            location="London",
            job_url="https://jobs.example.com/ledgerloop-pm",
            posting_text="Lead payments roadmap and experimentation in regulated markets.",
            requirements=[
                "Lead roadmap for payments and onboarding flows",
                "Run experimentation strategy for conversion funnels",
                "Use SQL analytics to make KPI trade-offs",
            ],
        )
        brief = CompanyBrief(
            company_name="LedgerLoop",
            summary="LedgerLoop builds API-led financial infrastructure for SMEs.",
            industry="Fintech",
            website="https://ledgerloop.example.com",
        )
        findings = [
            ResearchFinding(
                claim="LedgerLoop raised a Series B to scale regulated SME payments across Europe.",
                source_url="https://news.example.com/ledgerloop-series-b",
                source_date="2026-01-12",
                relevance="Growth context for hiring priorities.",
                confidence=0.85,
            ),
            ResearchFinding(
                claim="LedgerLoop launched instant settlement APIs for cross-border merchants.",
                source_url="https://blog.example.com/ledgerloop-instant-settlement",
                source_date="2025-11-04",
                relevance="Product direction tied to role scope.",
                confidence=0.82,
            ),
        ]
        scorecard = Scorecard(
            rows=[
                ScoreRow(
                    requirement_from_job_post="Lead roadmap for payments and onboarding flows",
                    matching_experience="Built quarterly roadmap for wallet onboarding and KYC at FinPay.",
                    rationale="Direct product ownership in fintech onboarding flows.",
                    match_strength=MatchStrength.STRONG,
                ),
                ScoreRow(
                    requirement_from_job_post="Run experimentation strategy for conversion funnels",
                    matching_experience="Built A/B testing infrastructure using Mida and Optimizely.",
                    rationale="Hands-on experimentation platform delivery.",
                    match_strength=MatchStrength.STRONG,
                ),
                ScoreRow(
                    requirement_from_job_post="Use SQL analytics to make KPI trade-offs",
                    matching_experience="Used SQL and Amplitude to drive KPI trade-off decisions across teams.",
                    rationale="Strong analytics ownership tied to KPI choices.",
                    match_strength=MatchStrength.STRONG,
                ),
            ],
            total_score=90,
            recommendation=Recommendation.CONFIDENT_APPLY,
            pipeline_should_continue=True,
            risk_flags=[],
        )
        return self.generator.generate(
            job_context=job,
            company_brief=brief,
            research_findings=findings,
            visa_assessment=self.visa,
            scorecard=scorecard,
            recommendation=Recommendation.CONFIDENT_APPLY,
            candidate_name="Alex Doe",
            candidate_notes="Strong fintech PM experience.",
            cv_content="Built quarterly roadmap and experimentation stack.",
        ).cover_letter.draft_markdown

    def test_do_not_apply_is_blocked(self) -> None:
        job = JobContext(
            job_id="stop-1",
            job_title="Role",
            company_name="Example",
            location="Remote",
            job_url="https://jobs.example.com/stop",
            posting_text="...",
            requirements=["Requirement"],
        )
        brief = CompanyBrief(
            company_name="Example",
            summary="Example summary",
            industry="SaaS",
            website="https://example.com",
        )
        scorecard = Scorecard(
            rows=[
                ScoreRow(
                    requirement_from_job_post="Requirement",
                    matching_experience="No strong matching evidence found in CV or candidate notes.",
                    rationale="Gap",
                    match_strength=MatchStrength.GAP,
                )
            ],
            total_score=10,
            recommendation=Recommendation.DO_NOT_APPLY,
            pipeline_should_continue=False,
            risk_flags=["Critical gap"],
        )
        with self.assertRaises(ValueError):
            self.generator.generate(
                job_context=job,
                company_brief=brief,
                research_findings=[],
                visa_assessment=self.visa,
                scorecard=scorecard,
                recommendation=Recommendation.DO_NOT_APPLY,
                candidate_name="Alex Doe",
                candidate_notes="",
                cv_content="",
            )

    def test_sparse_research_context_does_not_crash_cover_letter_generation(self) -> None:
        job = JobContext(
            job_id="sparse-1",
            job_title="Data Analyst",
            company_name="SparseCo",
            location="Remote",
            job_url="https://jobs.example.com/sparse/analyst",
            posting_text="Analyze metrics.",
            requirements=["Analyze metrics with SQL"],
        )
        brief = CompanyBrief(
            company_name="SparseCo",
            summary="",
            industry="SaaS",
            website="https://sparseco.example.com",
        )
        scorecard = Scorecard(
            rows=[
                ScoreRow(
                    requirement_from_job_post="Analyze metrics with SQL",
                    matching_experience="Used SQL for KPI reporting across product teams.",
                    rationale="Direct SQL analytics evidence.",
                    match_strength=MatchStrength.STRONG,
                )
            ],
            total_score=70,
            recommendation=Recommendation.CONFIDENT_APPLY,
            pipeline_should_continue=True,
            risk_flags=[],
        )

        result = self.generator.generate(
            job_context=job,
            company_brief=brief,
            research_findings=[],
            visa_assessment=self.visa,
            scorecard=scorecard,
            recommendation=Recommendation.CONFIDENT_APPLY,
            candidate_name="Alex Doe",
            candidate_notes="",
            cv_content="Used SQL for KPI reporting.",
        )

        self.assertTrue(result.cover_letter.draft_markdown)


if __name__ == "__main__":
    unittest.main()
