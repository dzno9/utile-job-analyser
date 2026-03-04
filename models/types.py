from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Recommendation(str, Enum):
    CONFIDENT_APPLY = "ConfidentApply"
    APPLY_WITH_CAVEATS = "ApplyWithCaveats"
    DO_NOT_APPLY = "DoNotApply"


class MatchStrength(str, Enum):
    STRONG = "strong"
    PARTIAL = "partial"
    GAP = "gap"


class VisaRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VisaLikelihood(str, Enum):
    CONFIRMED_SPONSOR = "Confirmed sponsor"
    LIKELY = "Likely"
    UNKNOWN = "Unknown"


class VisaEvidenceTag(str, Enum):
    DIRECT_REGISTER_MATCH = "direct_register_match"
    RELATED_ENTITY_MATCH = "related_entity_match"
    SPONSORED_ROLE_SIGNAL = "sponsored_role_signal"
    NONE = "none"


class JobContext(BaseModel):
    job_id: str
    job_title: str
    company_name: str
    location: str | None = None
    job_url: HttpUrl
    posting_text: str
    requirements: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    employment_type: str | None = None

    @model_validator(mode="after")
    def ensure_requirements(self) -> "JobContext":
        if not self.requirements and self.required_skills:
            self.requirements = list(self.required_skills)
        return self


class CompanyBrief(BaseModel):
    company_name: str
    summary: str
    industry: str
    size: str | None = None
    headquarters: str | None = None
    website: HttpUrl | None = None
    sources: list[HttpUrl] = Field(default_factory=list)


class ResearchFinding(BaseModel):
    claim: str
    source_url: HttpUrl
    source_date: str
    relevance: str
    confidence: float = Field(ge=0.0, le=1.0)


class VisaAssessment(BaseModel):
    likelihood: VisaLikelihood
    evidence_tags: list[VisaEvidenceTag] = Field(default_factory=list)
    reasoning: str
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_tags(self) -> "VisaAssessment":
        deduped = list(dict.fromkeys(self.evidence_tags))
        self.evidence_tags = deduped
        if self.likelihood == VisaLikelihood.UNKNOWN and deduped != [VisaEvidenceTag.NONE]:
            raise ValueError("Unknown likelihood must include only evidence tag 'none'")
        if self.likelihood != VisaLikelihood.UNKNOWN and VisaEvidenceTag.NONE in deduped:
            raise ValueError("Evidence tag 'none' can only be used with Unknown likelihood")
        return self


class ScoreRow(BaseModel):
    requirement_from_job_post: str
    matching_experience: str
    rationale: str
    match_strength: MatchStrength


class Scorecard(BaseModel):
    rows: list[ScoreRow] = Field(min_length=1)
    total_score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    pipeline_should_continue: bool
    risk_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_recommendation(self) -> "Scorecard":
        expected = recommendation_from_score(self.total_score)
        if self.recommendation != expected:
            msg = (
                "recommendation does not match total_score thresholds: "
                "<35 => DoNotApply, 35-64 => ApplyWithCaveats, >=65 => ConfidentApply"
            )
            raise ValueError(msg)
        expected_continue = self.recommendation != Recommendation.DO_NOT_APPLY
        if self.pipeline_should_continue != expected_continue:
            raise ValueError(
                "pipeline_should_continue must be False only for DoNotApply recommendation"
            )
        return self


class CoverLetter(BaseModel):
    candidate_name: str
    company_name: str
    job_title: str
    draft_markdown: str
    emphasis_strategy: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_non_generic_language(self) -> "CoverLetter":
        banned = ("passionate", "excited to apply", "great fit")
        lowered = self.draft_markdown.lower()
        if any(phrase in lowered for phrase in banned):
            raise ValueError(
                "cover letter draft_markdown contains banned generic phrases"
            )
        return self


class AnalysisReport(BaseModel):
    job_context: JobContext
    company_brief: CompanyBrief
    research_findings: list[ResearchFinding]
    visa_assessment: VisaAssessment
    scorecard: Scorecard
    cover_letter: CoverLetter
    cv_tweaks: list[str] = Field(default_factory=list)
    summary: str
    recommendation: Recommendation
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_recommendation(self) -> "AnalysisReport":
        if self.recommendation != self.scorecard.recommendation:
            raise ValueError(
                "analysis report recommendation must match scorecard recommendation"
            )
        return self


def recommendation_from_score(score: float) -> Recommendation:
    if score < 35:
        return Recommendation.DO_NOT_APPLY
    if score < 65:
        return Recommendation.APPLY_WITH_CAVEATS
    return Recommendation.CONFIDENT_APPLY
