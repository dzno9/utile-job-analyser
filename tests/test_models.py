import unittest

from pydantic import ValidationError

from models import (
    AnalysisReport,
    CompanyBrief,
    CoverLetter,
    JobContext,
    MatchStrength,
    Recommendation,
    ResearchFinding,
    ScoreRow,
    Scorecard,
    VisaAssessment,
    VisaEvidenceTag,
    VisaLikelihood,
    recommendation_from_score,
)


class TestModels(unittest.TestCase):
    def build_valid_report(self) -> AnalysisReport:
        job_context = JobContext(
            job_id="job-001",
            job_title="Senior Data Analyst",
            company_name="Stripe",
            location="San Francisco, CA",
            job_url="https://jobs.example.com/stripe/senior-data-analyst",
            posting_text=(
                "Build metrics, partner with product teams, and present business insights."
            ),
            requirements=[
                "Run experimentation programs",
                "Lead cross-functional analytics",
            ],
            required_skills=["SQL", "Python", "A/B testing"],
            nice_to_have_skills=["dbt", "Looker"],
            employment_type="Full-time",
        )

        company_brief = CompanyBrief(
            company_name="Stripe",
            summary="Financial infrastructure platform for internet businesses.",
            industry="Fintech",
            size="5000+",
            headquarters="San Francisco",
            website="https://stripe.com",
            sources=["https://stripe.com/about"],
        )

        findings = [
            ResearchFinding(
                claim="Stripe has a global payments footprint and product-led operating model.",
                source_url="https://stripe.com/about",
                source_date="2025-10-01",
                relevance="Provides core company context and product/culture positioning.",
                confidence=0.83,
            )
        ]

        visa = VisaAssessment(
            likelihood=VisaLikelihood.LIKELY,
            evidence_tags=[VisaEvidenceTag.SPONSORED_ROLE_SIGNAL],
            reasoning="Public role postings indicate visa sponsorship is available.",
            evidence=["https://jobs.example.com/stripe/visa-sponsorship-role"],
        )

        rows = [
            ScoreRow(
                requirement_from_job_post="Run experimentation programs",
                matching_experience="Built A/B testing framework used across growth squads.",
                rationale="Direct experimentation leadership maps well to role requirement.",
                match_strength=MatchStrength.STRONG,
            ),
            ScoreRow(
                requirement_from_job_post="Lead cross-functional analytics",
                matching_experience="Partnered with product, design, and engineering on KPI reporting.",
                rationale="Experience is relevant but scale of leadership is slightly narrower.",
                match_strength=MatchStrength.PARTIAL,
            ),
        ]

        scorecard = Scorecard(
            rows=rows,
            total_score=60,
            recommendation=Recommendation.CONFIDENT_APPLY,
            pipeline_should_continue=True,
            risk_flags=[],
        )

        cover_letter = CoverLetter(
            candidate_name="Alex Doe",
            company_name="Stripe",
            job_title="Senior Data Analyst",
            draft_markdown=(
                "Alex Doe\n\n"
                "I led experimentation programs and KPI analytics that align with this role."
            ),
            emphasis_strategy=[
                "Led with experimentation ownership because it directly matches the first requirement.",
                "Highlighted KPI analytics because it is repeatedly called out in the job posting.",
            ],
        )

        return AnalysisReport(
            job_context=job_context,
            company_brief=company_brief,
            research_findings=findings,
            visa_assessment=visa,
            scorecard=scorecard,
            cover_letter=cover_letter,
            cv_tweaks=[
                "Reword experimentation bullet to include measurable KPI impact and ownership scope."
            ],
            summary="Strong overall fit with manageable visa risk.",
            recommendation=Recommendation.CONFIDENT_APPLY,
        )

    def test_all_models_instantiate_with_valid_data(self) -> None:
        report = self.build_valid_report()
        self.assertEqual(
            report.scorecard.recommendation, Recommendation.CONFIDENT_APPLY
        )
        self.assertEqual(recommendation_from_score(20), Recommendation.DO_NOT_APPLY)
        self.assertEqual(
            recommendation_from_score(35), Recommendation.APPLY_WITH_CAVEATS
        )
        self.assertEqual(
            recommendation_from_score(56), Recommendation.CONFIDENT_APPLY
        )

    def test_invalid_match_strength_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ScoreRow(
                requirement_from_job_post="Role Fit",
                matching_experience="Relevant experience",
                rationale="Bad enum",
                match_strength="excellent",
            )

    def test_pipeline_flag_must_match_recommendation(self) -> None:
        with self.assertRaises(ValidationError):
            Scorecard(
                rows=[
                    ScoreRow(
                        requirement_from_job_post="Need iOS",
                        matching_experience="No evidence",
                        rationale="Gap",
                        match_strength=MatchStrength.GAP,
                    )
                ],
                total_score=20,
                recommendation=Recommendation.DO_NOT_APPLY,
                pipeline_should_continue=True,
            )

    def test_job_context_requirements_fallback_from_required_skills(self) -> None:
        ctx = JobContext(
            job_id="job-002",
            job_title="Data Scientist",
            company_name="Example",
            location="Remote",
            job_url="https://jobs.example.com/role",
            posting_text="Build ML models",
            required_skills=["Python", "SQL"],
        )
        self.assertEqual(ctx.requirements, ["Python", "SQL"])

    def test_unknown_visa_likelihood_requires_none_tag(self) -> None:
        with self.assertRaises(ValidationError):
            VisaAssessment(
                likelihood=VisaLikelihood.UNKNOWN,
                evidence_tags=[VisaEvidenceTag.SPONSORED_ROLE_SIGNAL],
                reasoning="No evidence.",
                evidence=[],
            )

    def test_cover_letter_rejects_generic_phrase(self) -> None:
        with self.assertRaises(ValidationError):
            CoverLetter(
                candidate_name="Alex Doe",
                company_name="Stripe",
                job_title="Senior Data Analyst",
                draft_markdown="I am passionate about this role.",
                emphasis_strategy=[],
            )


if __name__ == "__main__":
    unittest.main()
