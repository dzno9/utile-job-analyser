import tempfile
import unittest
import importlib.util
from pathlib import Path

from models import JobContext, MatchStrength, Recommendation
from providers.cv_parser import CVParser
from providers.gap_matcher import RuleBasedGapMatcher
from stages.gap_analyzer import GapAnalyzer


class StubCVParser:
    def __init__(self, text: str) -> None:
        self.text = text

    def parse(self, pdf_path: str | Path) -> str:
        return self.text


class TestCVParser(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        parser = CVParser()
        with self.assertRaises(FileNotFoundError):
            parser.parse("/tmp/this-file-does-not-exist.pdf")

    def test_fallback_to_pdfplumber_when_fitz_empty(self) -> None:
        class TestableCVParser(CVParser):
            def _parse_with_fitz(self, path: Path) -> str:
                return ""

            def _parse_with_pdfplumber(self, path: Path) -> str:
                return "Extracted by pdfplumber"

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            parser = TestableCVParser()
            result = parser.parse(tmp.name)
            self.assertEqual(result, "Extracted by pdfplumber")

    @unittest.skipUnless(
        importlib.util.find_spec("fitz") is not None,
        "PyMuPDF (fitz) is required for real PDF integration parsing test",
    )
    def test_parse_real_pdf_end_to_end_with_fitz(self) -> None:
        import fitz  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                (72, 72),
                "Real PDF CV sample: Built A/B testing infrastructure using Mida.",
            )
            doc.save(tmp.name)
            doc.close()

            parser = CVParser()
            extracted = parser.parse(tmp.name)
            self.assertIn("Built A/B testing infrastructure using Mida", extracted)


class TestGapAnalyzerDecisionGate(unittest.TestCase):
    def setUp(self) -> None:
        self.cv_text = (
            "Product Manager with 7 years in fintech and payments products. "
            "At FinPay, led roadmap for wallet, KYC, and fraud initiatives with engineering and design. "
            "Built A/B testing infrastructure using Mida and Optimizely to run experimentation at scale. "
            "Used SQL, Amplitude, and Looker to build KPI dashboards and cohort analysis for decision making. "
            "Partnered cross-functionally with sales and operations to launch growth features."
        )
        self.candidate_notes = (
            "At Verto, I designed a referral system that became a core growth loop for acquisition. "
            "I have light experience running lifecycle email onboarding campaigns but no large paid budget ownership."
        )
        self.analyzer = GapAnalyzer(
            cv_parser=StubCVParser(self.cv_text),
            gap_matcher=RuleBasedGapMatcher(),
        )

    def test_fintech_pm_role_scores_high_and_continues(self) -> None:
        job = JobContext(
            job_id="pm-001",
            job_title="Senior Product Manager, Fintech",
            company_name="BankFlow",
            location="London",
            job_url="https://jobs.example.com/pm-fintech",
            posting_text="...",
            requirements=[
                "Lead product roadmap for fintech payments platform",
                "Own experimentation platform strategy",
                "Use SQL and analytics to guide KPI decisions",
                "Drive cross-functional delivery with engineering and design",
            ],
        )

        result = self.analyzer.analyze(
            job_context=job,
            cv_pdf_path="/tmp/ignored.pdf",
            candidate_notes=self.candidate_notes,
        )

        self.assertGreaterEqual(result.scorecard.total_score, 65)
        self.assertEqual(result.recommendation, Recommendation.CONFIDENT_APPLY)
        self.assertTrue(result.should_continue_pipeline)

    def test_senior_ios_role_scores_low_and_stops(self) -> None:
        job = JobContext(
            job_id="ios-001",
            job_title="Senior iOS Engineer",
            company_name="AppWave",
            location="Remote",
            job_url="https://jobs.example.com/ios",
            posting_text="...",
            requirements=[
                "Ship production iOS apps with SwiftUI and UIKit",
                "Optimize app performance with Xcode instruments",
                "Publish mobile releases to the App Store",
                "Deep experience with Objective-C interoperability",
            ],
        )

        result = self.analyzer.analyze(
            job_context=job,
            cv_pdf_path="/tmp/ignored.pdf",
            candidate_notes=self.candidate_notes,
        )

        self.assertLess(result.scorecard.total_score, 35)
        self.assertEqual(result.recommendation, Recommendation.DO_NOT_APPLY)
        self.assertFalse(result.should_continue_pipeline)
        self.assertGreaterEqual(len(result.scorecard.risk_flags), 1)

    def test_growth_marketing_role_is_borderline_with_caveats(self) -> None:
        job = JobContext(
            job_id="gm-001",
            job_title="Growth Marketing Manager",
            company_name="ScaleLoop",
            location="London",
            job_url="https://jobs.example.com/growth",
            posting_text="...",
            requirements=[
                "Own lifecycle email campaigns and CRM automation",
                "Design referral growth loops and acquisition experiments",
                "Manage paid acquisition channel budget",
                "Build attribution models and growth dashboards in SQL",
                "Drive technical SEO optimization strategy",
                "Own creative production with external agencies",
            ],
        )

        result = self.analyzer.analyze(
            job_context=job,
            cv_pdf_path="/tmp/ignored.pdf",
            candidate_notes=self.candidate_notes,
        )

        self.assertGreaterEqual(result.scorecard.total_score, 35)
        self.assertLessEqual(result.scorecard.total_score, 55)
        self.assertEqual(result.recommendation, Recommendation.APPLY_WITH_CAVEATS)
        self.assertTrue(result.should_continue_pipeline)
        self.assertGreaterEqual(len(result.scorecard.risk_flags), 1)

    def test_semantic_non_literal_alignment_experimentation(self) -> None:
        job = JobContext(
            job_id="sem-001",
            job_title="PM",
            company_name="Example",
            location="Remote",
            job_url="https://jobs.example.com/semantic",
            posting_text="...",
            requirements=["Experience with experimentation platforms"],
        )

        result = self.analyzer.analyze(
            job_context=job,
            cv_pdf_path="/tmp/ignored.pdf",
            candidate_notes=self.candidate_notes,
        )

        row = result.scorecard.rows[0]
        self.assertEqual(row.match_strength, MatchStrength.STRONG)
        self.assertIn("Mida", row.matching_experience)

    def test_candidate_notes_used_as_matching_evidence_when_relevant(self) -> None:
        job = JobContext(
            job_id="note-001",
            job_title="Growth PM",
            company_name="Example",
            location="Remote",
            job_url="https://jobs.example.com/notes",
            posting_text="...",
            requirements=["Design referral growth loops"],
        )

        result = self.analyzer.analyze(
            job_context=job,
            cv_pdf_path="/tmp/ignored.pdf",
            candidate_notes=self.candidate_notes,
        )

        row = result.scorecard.rows[0]
        self.assertIn("[candidate_notes]", row.matching_experience)
        self.assertIn("Verto", row.matching_experience)

    def test_missing_requirements_uses_fallback_row(self) -> None:
        job = JobContext(
            job_id="fallback-001",
            job_title="Operations Analyst",
            company_name="Example",
            location="Remote",
            job_url="https://jobs.example.com/fallback",
            posting_text="General role context with sparse structure.",
            requirements=[],
            required_skills=[],
        )

        result = self.analyzer.analyze(
            job_context=job,
            cv_pdf_path="/tmp/ignored.pdf",
            candidate_notes=self.candidate_notes,
        )

        self.assertGreaterEqual(len(result.scorecard.rows), 1)
        self.assertIn("role-relevant impact", result.scorecard.rows[0].requirement_from_job_post)


if __name__ == "__main__":
    unittest.main()
