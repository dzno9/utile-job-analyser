from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models import (
    JobContext,
    MatchStrength,
    Recommendation,
    ScoreRow,
    Scorecard,
    recommendation_from_score,
)
from providers.cv_parser import CVParser
from providers.gap_matcher import GapMatcher, RequirementMatch, build_gap_matcher


@dataclass(frozen=True)
class GapAnalyzerResult:
    scorecard: Scorecard
    recommendation: Recommendation
    should_continue_pipeline: bool
    cv_text: str


class GapAnalyzer:
    def __init__(
        self,
        *,
        cv_parser: CVParser | None = None,
        gap_matcher: GapMatcher | None = None,
    ) -> None:
        self.cv_parser = cv_parser or CVParser()
        self.gap_matcher = gap_matcher or build_gap_matcher()

    def analyze(
        self,
        *,
        job_context: JobContext,
        cv_pdf_path: str | Path,
        candidate_notes: str,
        cv_text_override: str = "",
    ) -> GapAnalyzerResult:
        cv_text = cv_text_override.strip() or self.cv_parser.parse(cv_pdf_path)
        matches = self.gap_matcher.match(
            job_context=job_context,
            cv_text=cv_text,
            candidate_notes=candidate_notes,
        )
        if not matches:
            matches = _fallback_matches(
                job_context=job_context,
                cv_text=cv_text,
                candidate_notes=candidate_notes,
            )

        rows = [self._to_score_row(match) for match in matches]
        total_score = _compute_total_score(matches)
        recommendation = recommendation_from_score(total_score)
        risk_flags = _build_risk_flags(matches, recommendation)
        should_continue = recommendation != Recommendation.DO_NOT_APPLY

        scorecard = Scorecard(
            rows=rows,
            total_score=total_score,
            recommendation=recommendation,
            pipeline_should_continue=should_continue,
            risk_flags=risk_flags,
        )
        return GapAnalyzerResult(
            scorecard=scorecard,
            recommendation=scorecard.recommendation,
            should_continue_pipeline=scorecard.pipeline_should_continue,
            cv_text=cv_text,
        )

    def _to_score_row(self, match: RequirementMatch) -> ScoreRow:
        return ScoreRow(
            requirement_from_job_post=match.requirement_from_job_post,
            matching_experience=match.matching_experience,
            match_strength=match.match_strength,
            rationale=match.rationale,
        )


def _compute_total_score(matches: list[RequirementMatch]) -> float:
    if not matches:
        return 0.0

    points = {
        "strong": 100.0,
        "partial": 50.0,
        "gap": 0.0,
    }
    weights = {
        "strong": 1.0,
        "partial": 1.0,
        "gap": 1.0,
    }

    weighted_scores: list[float] = []
    weight_values: list[float] = []
    for match in matches:
        key = match.match_strength.value
        weighted_scores.append(points[key] * weights[key])
        weight_values.append(weights[key])

    score = sum(weighted_scores) / max(sum(weight_values), 1.0)
    return round(score, 2)


def _build_risk_flags(
    matches: list[RequirementMatch], recommendation: Recommendation
) -> list[str]:
    if recommendation == Recommendation.CONFIDENT_APPLY:
        return []

    flags: list[str] = []
    gaps = [m.requirement_from_job_post for m in matches if m.match_strength.value == "gap"]
    partials = [
        m.requirement_from_job_post
        for m in matches
        if m.match_strength.value == "partial"
    ]

    if gaps:
        flags.append(f"Critical gaps: {', '.join(gaps[:4])}")
    if partials:
        flags.append(f"Partial coverage: {', '.join(partials[:4])}")

    if not flags:
        flags.append("Insufficient evidence quality in CV and candidate notes.")

    return flags


def _fallback_matches(
    *,
    job_context: JobContext,
    cv_text: str,
    candidate_notes: str,
) -> list[RequirementMatch]:
    requirement = _fallback_requirement(job_context)
    evidence = _first_non_empty_sentence(candidate_notes) or _first_non_empty_sentence(cv_text)
    if evidence:
        return [
            RequirementMatch(
                requirement_from_job_post=requirement,
                matching_experience=f"[derived] {evidence}",
                match_strength=MatchStrength.PARTIAL,
                rationale=(
                    "Job requirements could not be extracted from the posting, so this row "
                    "uses a derived role-fit signal from available candidate evidence."
                ),
            )
        ]
    return [
        RequirementMatch(
            requirement_from_job_post=requirement,
            matching_experience="No strong matching evidence found in CV or candidate notes.",
            match_strength=MatchStrength.GAP,
            rationale=(
                "Job requirements could not be extracted and no supporting evidence was found "
                "in CV text or candidate notes."
            ),
        )
    ]


def _fallback_requirement(job_context: JobContext) -> str:
    if job_context.required_skills:
        skills = ", ".join(job_context.required_skills[:3])
        return f"Demonstrate core skills relevant to this role ({skills})"
    if job_context.job_title.strip():
        return f"Demonstrate role-relevant impact for {job_context.job_title.strip()}"
    return "Demonstrate role-relevant impact for this job"


def _first_non_empty_sentence(text: str) -> str:
    for part in text.replace("\n", " ").split("."):
        sentence = " ".join(part.split()).strip()
        if sentence:
            return sentence
    return ""
