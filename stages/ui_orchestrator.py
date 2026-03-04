from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from models import (
    AnalysisReport,
    CompanyBrief,
    CoverLetter,
    JobContext,
    Recommendation,
    ResearchFinding,
    VisaAssessment,
    VisaEvidenceTag,
    VisaLikelihood,
)
from providers import CompanyResearcher, CVParser, UKVisaSponsorChecker
from providers.job_context_extractor import ExtractionHints, RuleBasedJobContextExtractor
from stages.application_materials_generator import ApplicationMaterialsGenerator
from stages.gap_analyzer import GapAnalyzer
from stages.job_posting_scraper import JobPostingScraper, ScrapeResult


STAGE_SCRAPE = "scrape"
STAGE_GAP = "gap"
STAGE_COMPANY = "company"
STAGE_VISA = "visa"
STAGE_MATERIALS = "materials"
STAGE_ASSEMBLE = "assemble"


@dataclass(frozen=True)
class PipelineRequest:
    job_url: str
    cv_pdf_path: str | Path
    candidate_notes: str
    manual_job_posting_text: str = ""
    manual_cv_text: str = ""


@dataclass(frozen=True)
class StageEvent:
    stage_key: str
    stage_name: str
    status: str
    detail: str


@dataclass
class PipelineResult:
    report: AnalysisReport | None = None
    scrape_result: ScrapeResult | None = None
    needs_manual_posting_text: bool = False
    needs_manual_cv_text: bool = False
    warnings: list[str] = field(default_factory=list)
    cover_letter_generated: bool = False


class PipelineOrchestrator:
    def __init__(
        self,
        *,
        scraper: JobPostingScraper,
        gap_analyzer: GapAnalyzer,
        company_researcher: CompanyResearcher,
        visa_checker: UKVisaSponsorChecker,
        materials_generator: ApplicationMaterialsGenerator,
        cv_parser: CVParser | None = None,
    ) -> None:
        self.scraper = scraper
        self.gap_analyzer = gap_analyzer
        self.company_researcher = company_researcher
        self.visa_checker = visa_checker
        self.materials_generator = materials_generator
        self.cv_parser = cv_parser or CVParser()

    def run(
        self,
        request: PipelineRequest,
        *,
        on_event: Callable[[StageEvent], None] | None = None,
    ) -> PipelineResult:
        warnings: list[str] = []

        self._emit(on_event, STAGE_SCRAPE, "Job Scrape", "running", "Scraping job URL")
        scrape = self.scraper.scrape(request.job_url)
        job_context = scrape.job_context
        if scrape.confidence_notes:
            warnings.extend([f"Scrape: {note}" for note in scrape.confidence_notes])

        if not scrape.scrape_succeeded:
            self._emit(
                on_event,
                STAGE_SCRAPE,
                "Job Scrape",
                "warning",
                "Scrape failed - manual job description required",
            )
            if not request.manual_job_posting_text.strip():
                warnings.append("Scrape confidence low - consider pasting job description.")
                return PipelineResult(
                    scrape_result=scrape,
                    needs_manual_posting_text=True,
                    warnings=warnings,
                )

        if scrape.manual_text_input_required and request.manual_job_posting_text.strip():
            job_context = _build_manual_job_context(
                job_url=request.job_url,
                posting_text=request.manual_job_posting_text,
                fallback_company=scrape.job_context.company_name if scrape.job_context else None,
                fallback_title=scrape.job_context.job_title if scrape.job_context else None,
            )
            self._emit(
                on_event,
                STAGE_SCRAPE,
                "Job Scrape",
                "warning",
                "Using pasted job description text",
            )
        elif scrape.manual_text_input_required:
            hard_require_manual = job_context is None or (
                job_context is not None
                and not job_context.requirements
                and not job_context.required_skills
            )
            if hard_require_manual:
                warnings.append(
                    "Could not reliably extract role requirements; paste full job description to continue."
                )
                self._emit(
                    on_event,
                    STAGE_SCRAPE,
                    "Job Scrape",
                    "warning",
                    "Requirements missing - manual job text required",
                )
                return PipelineResult(
                    scrape_result=scrape,
                    needs_manual_posting_text=True,
                    warnings=warnings,
                )
            warnings.append("Scrape confidence low - consider pasting job description.")
            self._emit(
                on_event,
                STAGE_SCRAPE,
                "Job Scrape",
                "warning",
                "Scrape confidence low",
            )

        if job_context is None:
            if not request.manual_job_posting_text.strip():
                return PipelineResult(
                    scrape_result=scrape,
                    needs_manual_posting_text=True,
                    warnings=warnings,
                )
            job_context = _build_manual_job_context(
                job_url=request.job_url,
                posting_text=request.manual_job_posting_text,
            )

        self._emit(on_event, STAGE_SCRAPE, "Job Scrape", "complete", "Job context ready")

        self._emit(on_event, STAGE_GAP, "Gap Analysis", "running", "Comparing CV to role")
        try:
            gap_result = self.gap_analyzer.analyze(
                job_context=job_context,
                cv_pdf_path=request.cv_pdf_path,
                candidate_notes=request.candidate_notes,
                cv_text_override=request.manual_cv_text,
            )
        except ValueError as exc:
            message = str(exc)
            if "Could not extract text from CV PDF" in message:
                warnings.append(
                    "CV text extraction failed. Paste CV text manually to continue."
                )
                self._emit(
                    on_event,
                    STAGE_GAP,
                    "Gap Analysis",
                    "warning",
                    "CV extraction failed - manual CV text required",
                )
                return PipelineResult(
                    scrape_result=scrape,
                    needs_manual_cv_text=True,
                    warnings=warnings,
                )
            raise
        self._emit(
            on_event,
            STAGE_GAP,
            "Gap Analysis",
            "complete",
            f"Score {gap_result.scorecard.total_score:.0f} ({gap_result.recommendation.value})",
        )

        self._emit(on_event, STAGE_COMPANY, "Company Brief", "running", "Researching company")
        company_warnings: list[str] = []
        try:
            company_result = self.company_researcher.research(
                job_context=job_context,
                candidate_notes=request.candidate_notes,
            )
            company_brief = company_result.company_brief
            findings = company_result.findings
            company_warnings.extend(company_result.warnings)
            self._emit(on_event, STAGE_COMPANY, "Company Brief", "complete", "Research complete")
        except Exception as exc:  # noqa: BLE001
            company_brief = CompanyBrief(
                company_name=job_context.company_name,
                summary=(
                    "Public web research is currently unavailable. Base this application on the job post and interview validation."
                ),
                industry="Unknown",
                sources=[],
            )
            findings = []
            company_warnings.append(f"Company research substitute used: {exc}")
            self._emit(
                on_event,
                STAGE_COMPANY,
                "Company Brief",
                "warning",
                "Research unavailable - using summary brief",
            )

        warnings.extend(company_warnings)

        self._emit(on_event, STAGE_VISA, "Visa Check", "running", "Checking sponsorship evidence")
        visa_warnings: list[str] = []
        try:
            visa_assessment = self.visa_checker.assess(job_context=job_context)
            self._emit(on_event, STAGE_VISA, "Visa Check", "complete", visa_assessment.likelihood.value)
        except Exception as exc:  # noqa: BLE001
            visa_assessment = VisaAssessment(
                likelihood=VisaLikelihood.UNKNOWN,
                evidence_tags=[VisaEvidenceTag.NONE],
                reasoning="Visa assessment unavailable due to source lookup failure.",
                evidence=[],
            )
            visa_warnings.append(f"Visa checker substitute used: {exc}")
            self._emit(on_event, STAGE_VISA, "Visa Check", "warning", "Visa lookup unavailable")
        warnings.extend(visa_warnings)

        cover_letter_generated = False
        cover_letter: CoverLetter
        cv_tweaks: list[str]

        if gap_result.should_continue_pipeline:
            self._emit(on_event, STAGE_MATERIALS, "Application Materials", "running", "Drafting cover letter and CV tweaks")
            cv_text = gap_result.cv_text
            candidate_name = _extract_candidate_name(cv_text)
            materials = self.materials_generator.generate(
                job_context=job_context,
                company_brief=company_brief,
                research_findings=findings,
                visa_assessment=visa_assessment,
                scorecard=gap_result.scorecard,
                recommendation=gap_result.recommendation,
                candidate_name=candidate_name,
                candidate_notes=request.candidate_notes,
                cv_content=cv_text,
            )
            cover_letter = materials.cover_letter
            cv_tweaks = materials.cv_tweaks
            cover_letter_generated = True
            self._emit(on_event, STAGE_MATERIALS, "Application Materials", "complete", "Materials generated")
        else:
            cover_letter = _build_skipped_cover_letter(job_context=job_context)
            cv_tweaks = [
                "Focus next on role-targeted evidence before applying. Prioritize closing the highest-impact scorecard gaps.",
            ]
            self._emit(
                on_event,
                STAGE_MATERIALS,
                "Application Materials",
                "skipped",
                "Skipped - score below threshold",
            )

        self._emit(on_event, STAGE_ASSEMBLE, "Assemble Report", "running", "Packaging final analysis")
        report = AnalysisReport(
            job_context=job_context,
            company_brief=company_brief,
            research_findings=findings,
            visa_assessment=visa_assessment,
            scorecard=gap_result.scorecard,
            cover_letter=cover_letter,
            cv_tweaks=cv_tweaks,
            summary=_build_summary(gap_result.recommendation, gap_result.scorecard.total_score, warnings),
            recommendation=gap_result.recommendation,
        )
        self._emit(on_event, STAGE_ASSEMBLE, "Assemble Report", "complete", "Analysis complete")

        return PipelineResult(
            report=report,
            scrape_result=scrape,
            needs_manual_posting_text=False,
            needs_manual_cv_text=False,
            warnings=warnings,
            cover_letter_generated=cover_letter_generated,
        )

    def _emit(
        self,
        on_event: Callable[[StageEvent], None] | None,
        stage_key: str,
        stage_name: str,
        status: str,
        detail: str,
    ) -> None:
        if on_event is None:
            return
        on_event(
            StageEvent(
                stage_key=stage_key,
                stage_name=stage_name,
                status=status,
                detail=detail,
            )
        )


def _build_summary(recommendation: Recommendation, score: float, warnings: list[str]) -> str:
    warning_suffix = ""
    if warnings:
        warning_suffix = f" Warnings: {len(warnings)} check(s) flagged for review."
    return (
        f"Recommendation: {recommendation.value}. "
        f"Role-match score: {score:.0f}/100.{warning_suffix}"
    )


def _extract_candidate_name(cv_text: str) -> str:
    import re

    # Common PII patterns to strip from name candidates
    _email_re = re.compile(r"\S+@\S+\.\S+")
    _phone_re = re.compile(r"[\+]?[\d\s\-().]{7,20}")
    _url_re = re.compile(r"https?://\S+|www\.\S+|linkedin\.com\S*|github\.com\S*", re.IGNORECASE)

    for line in cv_text.splitlines():
        cleaned = " ".join(line.split()).strip()
        if not cleaned or len(cleaned) < 3:
            continue
        # Skip lines that look like contact info, not names
        if _email_re.search(cleaned) or _url_re.search(cleaned):
            continue
        # Strip phone numbers from the line
        name_candidate = _phone_re.sub("", cleaned).strip(" |,·•-")
        name_candidate = _email_re.sub("", name_candidate).strip(" |,·•-")
        # A valid name should be mostly letters, not digits
        alpha_ratio = sum(1 for c in name_candidate if c.isalpha()) / max(len(name_candidate), 1)
        if alpha_ratio < 0.6:
            continue
        if 2 <= len(name_candidate) <= 60:
            return name_candidate
    return "Candidate"


def _build_skipped_cover_letter(*, job_context: JobContext) -> CoverLetter:
    return CoverLetter(
        candidate_name="Candidate",
        company_name=job_context.company_name,
        job_title=job_context.job_title,
        draft_markdown="Skipped - score below threshold.",
        emphasis_strategy=["Pipeline gate blocked cover letter generation due to low role-fit score."],
    )


def _build_manual_job_context(
    *,
    job_url: str,
    posting_text: str,
    fallback_company: str | None = None,
    fallback_title: str | None = None,
) -> JobContext:
    extractor = RuleBasedJobContextExtractor()
    lines = [line.strip(" -*•\t") for line in posting_text.splitlines() if line.strip()]
    hint_title = fallback_title or (lines[0] if lines else "Unknown role")

    host = (urlparse(job_url).hostname or "").replace("www.", "")
    slug = host.split(".")[0] if host else "unknown"
    hint_company = fallback_company or slug.replace("-", " ").title()

    extraction = extractor.extract(
        page_text=posting_text,
        job_url=job_url,
        hints=ExtractionHints(job_title=hint_title, company_name=hint_company),
    )

    requirements = [line for line in lines if len(line) >= 25][:12]
    if not requirements:
        requirements = [
            "Demonstrate role-relevant impact with measurable outcomes.",
            "Collaborate across stakeholders to deliver priorities.",
        ]

    parsed = urlparse(job_url)
    tail = parsed.path.rstrip("/").split("/")[-1] if parsed.path else "manual"
    job_id = f"{parsed.hostname or 'manual'}:{tail or 'manual'}"

    return JobContext(
        job_id=job_id,
        job_title=extraction.job_title or hint_title,
        company_name=extraction.company_name or hint_company,
        location=extraction.location,
        job_url=job_url,
        posting_text=posting_text,
        requirements=requirements,
        required_skills=extraction.required_skills,
        nice_to_have_skills=extraction.nice_to_have_skills,
        employment_type=extraction.employment_type,
    )
