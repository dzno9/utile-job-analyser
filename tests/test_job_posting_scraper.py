import unittest

from providers import LLMExtraction
from stages import FetchedPage, JobPostingScraper


class FakeFetcher:
    def __init__(self, payload_by_url: dict[str, str], content_type: str = "text/html"):
        self.payload_by_url = payload_by_url
        self.content_type = content_type

    def fetch(self, url: str) -> FetchedPage:
        if url not in self.payload_by_url:
            raise RuntimeError("network error")
        return FetchedPage(
            final_url=url,
            status_code=200,
            text=self.payload_by_url[url],
            content_type=self.content_type,
        )


class FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def extract(self, *, page_text: str, job_url: str, hints):
        self.calls.append({"page_text": page_text, "job_url": job_url, "hints": hints})
        required = list(hints.tech_stack or []) + ["Communication", "Problem Solving"]
        return LLMExtraction(
            job_title=hints.job_title,
            company_name=hints.company_name,
            location=hints.location,
            required_skills=required[:8],
            nice_to_have_skills=["Mentorship", "System Design"],
            employment_type=hints.employment_type or "Full-time",
            confidence_note="fake extractor",
        )


class EmptyExtractor:
    def extract(self, *, page_text: str, job_url: str, hints):
        del page_text, job_url
        return LLMExtraction(
            job_title=hints.job_title,
            company_name=hints.company_name,
            location=hints.location,
            required_skills=[],
            nice_to_have_skills=[],
            employment_type=hints.employment_type,
            confidence_note="empty extractor",
        )


LINKEDIN_HTML = """
<html><head>
<meta property="og:title" content="Senior Software Engineer">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"JobPosting","title":"Senior Software Engineer","hiringOrganization":{"name":"OpenAI"},"jobLocation":{"address":{"addressLocality":"San Francisco","addressRegion":"CA"}},"employmentType":"FULL_TIME"}
</script></head>
<body>
<h1>Senior Software Engineer</h1>
<a class="topcard__org-name-link">OpenAI</a>
<span class="topcard__flavor--bullet">San Francisco, CA</span>
<h2>Responsibilities</h2>
<ul>
  <li>Build resilient backend services for model APIs and internal platforms.</li>
  <li>Partner with product and research teams to deliver end-to-end features.</li>
</ul>
<h2>Requirements</h2>
<ul>
  <li>5+ years writing Python services in production environments.</li>
  <li>Strong SQL fundamentals and distributed systems knowledge.</li>
  <li>Experience with AWS and Kubernetes.</li>
</ul>
<p>Nice to have: React, dbt, or data platform experience.</p>
</body></html>
"""

GREENHOUSE_HTML = """
<html><head>
<script type="application/ld+json">
{"@type":"JobPosting","title":"Data Engineer","hiringOrganization":{"name":"Example Labs"},"jobLocation":{"address":{"addressLocality":"London"}}}
</script></head>
<body>
<h1>Data Engineer</h1>
<div class="company-name">Example Labs</div>
<div class="location">London</div>
<h3>What you'll do</h3>
<ul>
  <li>Design and maintain robust ETL pipelines for product analytics.</li>
  <li>Own data modeling standards across event and warehouse layers.</li>
</ul>
<h3>Qualifications</h3>
<ul>
  <li>Expert in SQL and Python for data transformations.</li>
  <li>Hands-on with Airflow, dbt, and Snowflake.</li>
</ul>
</body></html>
"""

WORKABLE_HTML = """
<html><head><meta property="og:title" content="Backend Engineer"></head>
<body>
<h1>Backend Engineer</h1>
<div data-test="company-name">Contoso</div>
<div data-test="location">Remote - UK</div>
<h2>Responsibilities</h2>
<ul>
  <li>Develop microservices and APIs in Python and Node.js to support product growth.</li>
  <li>Collaborate with data and frontend teams to ship new user-facing capabilities.</li>
</ul>
<h2>Requirements</h2>
<ul>
  <li>Strong Python, SQL, Docker, and cloud platform experience in production systems.</li>
  <li>Experience deploying services on AWS with observability and incident response ownership.</li>
  <li>Excellent communication skills and experience mentoring engineers across teams.</li>
</ul>
<p>
  You will own backend services from design through operation, improve reliability, and
  continuously improve latency, availability, and developer productivity for the wider platform.
</p>
</body></html>
"""

LEVER_HTML = """
<html><body>
<h1>Machine Learning Engineer</h1>
<div class="posting-categories">
  <span class="location">New York, NY</span>
</div>
<h3>What you will do</h3>
<ul>
  <li>Deploy and monitor machine learning models in production systems.</li>
  <li>Build experimentation pipelines and offline evaluation tools.</li>
</ul>
<h3>Requirements</h3>
<ul>
  <li>Proficiency in Python and modern ML tooling.</li>
  <li>Experience with Spark, AWS, and Kubernetes at scale.</li>
</ul>
</body></html>
"""

ASHBY_HTML = """
<html><body>
<h1>Staff Platform Engineer</h1>
<div class="company">Fabrikam</div>
<div class="location">Berlin</div>
<h2>What you bring</h2>
<ul><li>Deep experience with distributed systems and platform APIs.</li></ul>
<h2>What you'll do</h2>
<ul><li>Drive architecture for high throughput services and team enablement.</li></ul>
</body></html>
"""

BLOG_HTML = """
<html><body>
<h1>10 Tips For Better Product Meetings</h1>
<p>This article covers facilitation and writing better agendas.</p>
<p>Teams can improve communication by preparing in advance.</p>
</body></html>
"""

THIN_JOB_HTML = """
<html><head>
<meta property="og:title" content="Senior Product Manager">
<meta property="og:site_name" content="Pinpoint">
</head>
<body>
<h1>Senior Product Manager</h1>
<div class="location">Remote - UK</div>
<p>
You will partner across engineering, design, and analytics to shape roadmap priorities and improve
customer outcomes through structured product discovery and execution.
</p>
<p>
This role collaborates with leadership and cross-functional teams to scale product impact and ship
high-quality customer-facing improvements across the platform.
</p>
</body></html>
"""


class TestJobPostingScraper(unittest.TestCase):
    def setUp(self) -> None:
        self.urls = {
            "https://www.linkedin.com/jobs/view/4187493642/": LINKEDIN_HTML,
            "https://www.linkedin.com/jobs/view/4191122334/": LINKEDIN_HTML.replace(
                "Senior Software Engineer", "Senior Data Engineer"
            ),
            "https://boards.greenhouse.io/example/jobs/5071771004": GREENHOUSE_HTML,
            "https://apply.workable.com/example/j/ABC12345/": WORKABLE_HTML,
            "https://jobs.lever.co/example/80bd01bb-4f5d-4a08-a2fd-1d9fe91f27e1": LEVER_HTML,
            "https://jobs.ashbyhq.com/example/1234-5678": ASHBY_HTML,
            "https://example.com/blog/product-meeting-tips": BLOG_HTML,
            "https://jobs.example.com/pinpoint/senior-product-manager": THIN_JOB_HTML,
        }
        self.extractor = FakeExtractor()
        self.scraper = JobPostingScraper(
            fetcher=FakeFetcher(self.urls),
            extractor=self.extractor,
            low_confidence_threshold=0.65,
        )

    def test_target_platform_batch_returns_valid_job_context(self) -> None:
        target_urls = [
            "https://www.linkedin.com/jobs/view/4187493642/",
            "https://www.linkedin.com/jobs/view/4191122334/",
            "https://boards.greenhouse.io/example/jobs/5071771004",
            "https://apply.workable.com/example/j/ABC12345/",
            "https://jobs.lever.co/example/80bd01bb-4f5d-4a08-a2fd-1d9fe91f27e1",
        ]

        for url in target_urls:
            result = self.scraper.scrape(url)
            self.assertTrue(result.scrape_succeeded, url)
            self.assertFalse(result.manual_text_input_required, url)
            self.assertGreaterEqual(result.confidence_score, 0.65, url)
            self.assertIsNotNone(result.job_context, url)
            self.assertTrue(result.job_context.job_title, url)
            self.assertTrue(result.job_context.company_name, url)
            self.assertTrue(result.job_context.location, url)
            self.assertGreater(len(result.job_context.required_skills), 0, url)

    def test_unreachable_url_returns_failure_flag(self) -> None:
        result = self.scraper.scrape("https://broken.example.com/missing-role")
        self.assertFalse(result.scrape_succeeded)
        self.assertTrue(result.manual_text_input_required)
        self.assertEqual(result.confidence_score, 0.0)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.job_context)

    def test_non_job_page_is_low_confidence(self) -> None:
        result = self.scraper.scrape("https://example.com/blog/product-meeting-tips")
        self.assertTrue(result.scrape_succeeded)
        self.assertTrue(result.manual_text_input_required)
        self.assertLess(result.confidence_score, 0.65)

    def test_ashby_platform_is_supported(self) -> None:
        result = self.scraper.scrape("https://jobs.ashbyhq.com/example/1234-5678")
        self.assertTrue(result.scrape_succeeded)
        self.assertEqual(result.platform, "ashby")
        self.assertIsNotNone(result.job_context)

    def test_llm_extractor_invoked(self) -> None:
        self.scraper.scrape("https://boards.greenhouse.io/example/jobs/5071771004")
        self.assertEqual(len(self.extractor.calls), 1)
        call = self.extractor.calls[0]
        self.assertIn("Data Engineer", call["page_text"])
        self.assertIn("greenhouse.io", call["job_url"])

    def test_missing_requirements_and_skills_forces_manual_text_fallback(self) -> None:
        scraper = JobPostingScraper(
            fetcher=FakeFetcher(self.urls),
            extractor=EmptyExtractor(),
            low_confidence_threshold=0.65,
        )
        result = scraper.scrape("https://jobs.example.com/pinpoint/senior-product-manager")
        self.assertTrue(result.scrape_succeeded)
        self.assertTrue(result.manual_text_input_required)
        self.assertTrue(
            any(
                "Could not extract requirements or skills" in note
                for note in result.confidence_notes
            )
        )


if __name__ == "__main__":
    unittest.main()
