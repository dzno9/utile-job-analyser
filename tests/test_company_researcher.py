from __future__ import annotations

import unittest

from models import JobContext
from providers.company_researcher import CompanyResearcher, SearchResult


class FakeWebSearchTool:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(
        self,
        *,
        query: str,
        iteration: int,
        max_results: int,
    ) -> list[SearchResult]:
        self.queries.append(query)
        q = query.lower()
        if "fca" in q or "compliance" in q:
            return [
                SearchResult(
                    title="Cleo appears on FCA register",
                    url="https://fca.org.uk/register/cleo",
                    published_date="2025-08-14",
                    snippet="FCA register entry confirms authorization details.",
                    content=(
                        "Cleo is listed on the FCA register and has active compliance disclosures."
                    ),
                ),
                SearchResult(
                    title="Fintech compliance update mentions Cleo",
                    url="https://finextra.com/news/cleo-compliance-update",
                    published_date="2025-11-19",
                    snippet="KYC/AML controls expanded in latest update.",
                    content="Industry report covers regulatory updates and KYC remediation.",
                ),
            ]
        if "funding" in q or "investors" in q:
            return [
                SearchResult(
                    title="AcmeAI raises seed round",
                    url="https://techcrunch.com/acmeai-seed-round",
                    published_date="2025-09-02",
                    snippet="Seed round led by Northzone with angel participation.",
                    content="AcmeAI raised a $4M seed round and plans to double hiring.",
                ),
                SearchResult(
                    title="AcmeAI investors and runway",
                    url="https://sifted.eu/articles/acmeai-investors-runway",
                    published_date="2025-10-01",
                    snippet="Investor mix includes operator angels and micro-VC funds.",
                    content="Investors include Northzone Scout and AI-focused operators.",
                ),
            ]
        if "headcount" in q or "growth" in q:
            return [
                SearchResult(
                    title="AcmeAI team growth profile",
                    url="https://wellfound.com/company/acmeai",
                    published_date="2025-12-10",
                    snippet="Headcount grew from 8 to 20 in 12 months.",
                    content="Hiring trajectory shows sharp growth in engineering.",
                )
            ]
        if "earnings" in q or "layoffs" in q or "reorg" in q or "leadership" in q:
            return [
                SearchResult(
                    title="Meta earnings call highlights AI strategy",
                    url="https://investor.fb.com/earnings-q4-2025",
                    published_date="2026-01-29",
                    snippet="Leadership reiterated infra investment and product priorities.",
                    content="Meta discussed strategic priorities, cost focus, and org direction.",
                ),
                SearchResult(
                    title="Meta reorg and product direction",
                    url="https://www.theverge.com/meta-reorg-product-direction",
                    published_date="2025-11-08",
                    snippet="Reports focus on product direction and org shape.",
                    content="The report links leadership changes with platform strategy.",
                ),
            ]
        if "segment" in q or "data stack" in q:
            return [
                SearchResult(
                    title="Cleo engineering blog on customer data stack",
                    url="https://web.meetcleo.com/blog/data-stack",
                    published_date="2025-07-03",
                    snippet="Explains Segment-based event pipeline and warehouse sync.",
                    content=(
                        "Engineering shared Segment instrumentation patterns and downstream "
                        "analytics modeling for experimentation."
                    ),
                )
            ]
        if "blog" in q or "product launch" in q or "culture" in q:
            return [
                SearchResult(
                    title="Company blog: new product launch",
                    url="https://company.example.com/blog/new-launch",
                    published_date="2025-06-22",
                    snippet="Launch post with product messaging and culture references.",
                    content="Latest launch post signals roadmap and product focus.",
                )
            ]
        return [
            SearchResult(
                title="General company profile",
                url=f"https://news.example.com/{iteration}",
                published_date="2025-01-01",
                snippet="General context source.",
                content="General company update.",
            )
        ]


class TestCompanyResearcher(unittest.TestCase):
    def setUp(self) -> None:
        self.search_tool = FakeWebSearchTool()
        self.researcher = CompanyResearcher(search_tool=self.search_tool)

    def test_fintech_context_includes_regulatory_focus(self) -> None:
        job = JobContext(
            job_id="cleo-1",
            job_title="Senior Product Analyst",
            company_name="Cleo",
            location="London",
            job_url="https://jobs.example.com/cleo/senior-product-analyst",
            posting_text="Join our fintech team improving credit and compliance outcomes.",
            requirements=["Build fraud and risk analytics"],
            required_skills=["SQL", "Python"],
        )
        result = self.researcher.research(job_context=job, candidate_notes="")

        self.assertEqual(result.strategy_profile, "fintech")
        self.assertTrue(
            any("fca" in query.lower() or "compliance" in query.lower() for query in result.queries_executed)
        )
        self.assertTrue(
            any("regulatory" in finding.relevance.lower() or "compliance" in finding.relevance.lower() for finding in result.findings)
        )

    def test_startup_context_includes_funding_and_investor_focus(self) -> None:
        job = JobContext(
            job_id="startup-1",
            job_title="Founding Data Engineer",
            company_name="AcmeAI",
            location="Remote",
            job_url="https://jobs.example.com/acmeai/founding-data-engineer",
            posting_text="Early-stage startup hiring across product and engineering after seed stage.",
            requirements=["Ship quickly in a small team"],
            required_skills=["Python", "SQL"],
        )
        result = self.researcher.research(job_context=job, candidate_notes="")

        self.assertEqual(result.strategy_profile, "startup")
        self.assertTrue(
            any("funding" in query.lower() or "investors" in query.lower() for query in result.queries_executed)
        )
        self.assertTrue(
            any("funding" in finding.relevance.lower() or "investor" in finding.relevance.lower() for finding in result.findings)
        )

    def test_enterprise_context_includes_strategic_moves_focus(self) -> None:
        job = JobContext(
            job_id="meta-1",
            job_title="Analytics Lead",
            company_name="Meta",
            location="London",
            job_url="https://jobs.example.com/meta/analytics-lead",
            posting_text="Join our enterprise analytics org focused on product strategy.",
            requirements=["Partner with leadership on strategic metrics"],
            required_skills=["SQL", "Python"],
        )
        result = self.researcher.research(job_context=job, candidate_notes="")

        self.assertEqual(result.strategy_profile, "enterprise")
        self.assertTrue(
            any(
                any(keyword in query.lower() for keyword in ("earnings", "layoffs", "reorg", "leadership"))
                for query in result.queries_executed
            )
        )
        self.assertTrue(
            any("strategic moves" in finding.relevance.lower() for finding in result.findings)
        )

    def test_candidate_notes_influence_queries_when_relevant(self) -> None:
        job = JobContext(
            job_id="cleo-2",
            job_title="Product Analyst",
            company_name="Cleo",
            location="London",
            job_url="https://jobs.example.com/cleo/product-analyst",
            posting_text="Fintech analytics role.",
            requirements=["Instrumentation and experimentation"],
            required_skills=["SQL"],
        )
        result = self.researcher.research(
            job_context=job,
            candidate_notes="I know they use Segment for event tracking.",
        )

        self.assertTrue(any("segment" in query.lower() for query in result.queries_executed))

    def test_limits_and_required_finding_fields(self) -> None:
        job = JobContext(
            job_id="google-1",
            job_title="Data Scientist",
            company_name="Google",
            location="London",
            job_url="https://jobs.example.com/google/data-scientist",
            posting_text="Public company role in product analytics.",
            requirements=["Run analyses and experiments"],
            required_skills=["Python", "SQL"],
        )
        result = self.researcher.research(job_context=job, candidate_notes="")

        self.assertLessEqual(result.search_iterations_used, 4)
        self.assertLessEqual(result.synthesized_source_count, 8)
        self.assertLessEqual(len(result.company_brief.sources), 8)
        self.assertEqual(result.company_brief.company_name, "Google")
        for finding in result.findings:
            self.assertTrue(str(finding.source_url))
            self.assertTrue(finding.source_date)
            self.assertGreaterEqual(finding.confidence, 0.0)
            self.assertLessEqual(finding.confidence, 1.0)

    def test_low_confidence_findings_are_flagged(self) -> None:
        job = JobContext(
            job_id="unknown-1",
            job_title="Analyst",
            company_name="UnknownCo",
            location="Remote",
            job_url="https://jobs.example.com/unknown/analyst",
            posting_text="General analyst role.",
            requirements=["Analyze data"],
            required_skills=["SQL"],
        )
        result = self.researcher.research(job_context=job, candidate_notes="")
        self.assertTrue(
            any("low confidence" in warning.lower() for warning in result.warnings)
        )


if __name__ == "__main__":
    unittest.main()
