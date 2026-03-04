from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from anthropic import Anthropic

from config import load_settings
from models import JobContext, MatchStrength


@dataclass(frozen=True)
class RequirementMatch:
    requirement_from_job_post: str
    matching_experience: str
    match_strength: MatchStrength
    rationale: str


class GapMatcher(Protocol):
    def match(
        self,
        *,
        job_context: JobContext,
        cv_text: str,
        candidate_notes: str,
    ) -> list[RequirementMatch]:
        ...


class RuleBasedGapMatcher:
    _CONCEPTS: dict[str, tuple[str, ...]] = {
        "experimentation": (
            "experimentation",
            "experiment",
            "a/b",
            "ab test",
            "split test",
            "multivariate",
            "mida",
            "optimizely",
            "feature flag",
        ),
        "product_management": (
            "product manager",
            "pm",
            "roadmap",
            "stakeholder",
            "discovery",
            "prioritization",
            "kpi",
            "okrs",
        ),
        "growth_marketing": (
            "growth",
            "acquisition",
            "retention",
            "crm",
            "email",
            "funnel",
            "attribution",
            "campaign",
            "lifecycle",
        ),
        "analytics": (
            "analytics",
            "sql",
            "dashboard",
            "looker",
            "amplitude",
            "mixpanel",
            "cohort",
            "reporting",
        ),
        "mobile_ios": (
            "ios",
            "swift",
            "swiftui",
            "uikit",
            "objective-c",
            "xcode",
            "app store",
            "mobile engineer",
        ),
        "engineering": (
            "python",
            "backend",
            "api",
            "microservice",
            "system design",
            "kubernetes",
            "aws",
            "distributed",
        ),
        "leadership": (
            "cross-functional",
            "team leadership",
            "people management",
            "mentored",
            "hiring",
            "stakeholder alignment",
        ),
        "fintech": (
            "fintech",
            "payments",
            "banking",
            "kyc",
            "fraud",
            "credit",
            "wallet",
            "money transfer",
        ),
    }

    def match(
        self,
        *,
        job_context: JobContext,
        cv_text: str,
        candidate_notes: str,
    ) -> list[RequirementMatch]:
        requirements = job_context.requirements or job_context.required_skills
        evidence_pool = _extract_evidence_sentences(cv_text, candidate_notes)
        output: list[RequirementMatch] = []

        for requirement in requirements:
            best_sentence, semantic_score, source_label = self._best_evidence(
                requirement=requirement,
                evidence_pool=evidence_pool,
            )
            strength = _strength_from_score(semantic_score)

            if best_sentence:
                matching_experience = f"[{source_label}] {best_sentence}"
            else:
                matching_experience = "No strong matching evidence found in CV or candidate notes."

            rationale = _build_rationale(
                requirement=requirement,
                strength=strength,
                evidence=best_sentence,
            )
            output.append(
                RequirementMatch(
                    requirement_from_job_post=requirement,
                    matching_experience=matching_experience,
                    match_strength=strength,
                    rationale=rationale,
                )
            )

        return output

    def _best_evidence(
        self,
        *,
        requirement: str,
        evidence_pool: list[tuple[str, str]],
    ) -> tuple[str, float, str]:
        req_concepts = _concepts_for_text(requirement, self._CONCEPTS)
        req_tokens = _tokens(requirement)

        best_sentence = ""
        best_score = 0.0
        best_source = "CV"

        for source, sentence in evidence_pool:
            sentence_concepts = _concepts_for_text(sentence, self._CONCEPTS)
            sentence_tokens = _tokens(sentence)

            concept_overlap = _overlap_ratio(req_concepts, sentence_concepts)
            token_overlap = _overlap_ratio(req_tokens, sentence_tokens)

            semantic_score = (concept_overlap * 0.7) + (token_overlap * 0.3)
            if _contains_weakness_language(sentence):
                semantic_score = max(0.0, semantic_score - 0.25)
            if semantic_score > best_score:
                best_score = semantic_score
                best_sentence = sentence
                best_source = source

        return best_sentence, best_score, best_source


class AnthropicGapMatcher:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def match(
        self,
        *,
        job_context: JobContext,
        cv_text: str,
        candidate_notes: str,
    ) -> list[RequirementMatch]:
        requirements = job_context.requirements or job_context.required_skills
        prompt = _build_prompt(requirements, cv_text, candidate_notes)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1800,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "text", None)
        )

        payload = _extract_json_object(raw_text)
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return RuleBasedGapMatcher().match(
                job_context=job_context,
                cv_text=cv_text,
                candidate_notes=candidate_notes,
            )

        output: list[RequirementMatch] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            requirement = _safe_str(item.get("requirement_from_job_post"))
            evidence = _safe_str(item.get("matching_experience"))
            rationale = _safe_str(item.get("rationale"))
            strength = _safe_str(item.get("match_strength"))
            if not (requirement and evidence and rationale and strength):
                continue
            if strength not in {"strong", "partial", "gap"}:
                continue
            output.append(
                RequirementMatch(
                    requirement_from_job_post=requirement,
                    matching_experience=evidence,
                    match_strength=MatchStrength(strength),
                    rationale=rationale,
                )
            )

        if output:
            return output

        return RuleBasedGapMatcher().match(
            job_context=job_context,
            cv_text=cv_text,
            candidate_notes=candidate_notes,
        )


class OpenAIGapMatcher:
    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required when LLM_PROVIDER=openai"
            ) from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def match(
        self,
        *,
        job_context: JobContext,
        cv_text: str,
        candidate_notes: str,
    ) -> list[RequirementMatch]:
        requirements = job_context.requirements or job_context.required_skills
        prompt = _build_prompt(requirements, cv_text, candidate_notes)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1800,
            temperature=0.0,
        )
        raw_text = (response.choices[0].message.content or "") if response.choices else ""

        payload = _extract_json_object(raw_text)
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return RuleBasedGapMatcher().match(
                job_context=job_context,
                cv_text=cv_text,
                candidate_notes=candidate_notes,
            )

        output: list[RequirementMatch] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            requirement = _safe_str(item.get("requirement_from_job_post"))
            evidence = _safe_str(item.get("matching_experience"))
            rationale = _safe_str(item.get("rationale"))
            strength = _safe_str(item.get("match_strength"))
            if not (requirement and evidence and rationale and strength):
                continue
            if strength not in {"strong", "partial", "gap"}:
                continue
            output.append(
                RequirementMatch(
                    requirement_from_job_post=requirement,
                    matching_experience=evidence,
                    match_strength=MatchStrength(strength),
                    rationale=rationale,
                )
            )

        if output:
            return output

        return RuleBasedGapMatcher().match(
            job_context=job_context,
            cv_text=cv_text,
            candidate_notes=candidate_notes,
        )


def build_gap_matcher() -> GapMatcher:
    settings = load_settings()
    if settings.llm_provider == "rule_based":
        return RuleBasedGapMatcher()
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return AnthropicGapMatcher(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIGapMatcher(api_key=settings.openai_api_key, model=settings.llm_model)
    raise ValueError(f"Unsupported provider: {settings.llm_provider}")


def _extract_evidence_sentences(cv_text: str, candidate_notes: str) -> list[tuple[str, str]]:
    sentences: list[tuple[str, str]] = []
    for sentence in _split_sentences(cv_text):
        sentences.append(("CV", sentence))
    for sentence in _split_sentences(candidate_notes):
        sentences.append(("candidate_notes", sentence))
    return sentences


def _split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    if not collapsed:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", collapsed)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 20]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9+/.-]+", text.lower()) if len(token) > 2}


def _concepts_for_text(text: str, concepts: dict[str, tuple[str, ...]]) -> set[str]:
    lower = text.lower()
    matches: set[str] = set()
    for concept, keywords in concepts.items():
        for keyword in keywords:
            if keyword in lower:
                matches.add(concept)
                break
    return matches


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _strength_from_score(score: float) -> MatchStrength:
    if score >= 0.60:
        return MatchStrength.STRONG
    if score >= 0.25:
        return MatchStrength.PARTIAL
    return MatchStrength.GAP


def _build_rationale(requirement: str, strength: MatchStrength, evidence: str) -> str:
    if strength == MatchStrength.STRONG:
        return (
            f"Strong alignment for '{requirement}' based on specific evidence: "
            f"{_truncate(evidence, 130)}"
        )
    if strength == MatchStrength.PARTIAL:
        return (
            f"Partial alignment for '{requirement}'. The evidence is relevant but does not fully "
            "cover depth or scope expected by the role."
        )
    return (
        f"Gap detected for '{requirement}'. Available CV and candidate notes do not show direct "
        "or equivalent experience yet."
    )


def _contains_weakness_language(sentence: str) -> bool:
    lower = sentence.lower()
    patterns = (
        "light experience",
        "limited experience",
        "no ",
        "not ",
        "haven't",
        "have not",
    )
    return any(token in lower for token in patterns)


def _truncate(text: str, max_len: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _build_prompt(requirements: list[str], cv_text: str, candidate_notes: str) -> str:
    schema = {
        "rows": [
            {
                "requirement_from_job_post": "string",
                "matching_experience": "string",
                "match_strength": "strong|partial|gap",
                "rationale": "string",
            }
        ]
    }
    return (
        "Perform semantic matching between job requirements and candidate background. "
        "Do not rely on exact keyword matching.\n"
        "Rules:\n"
        "- For each requirement, return one row.\n"
        "- matching_experience must cite specific evidence from CV or candidate notes.\n"
        "- rationale must be specific and 1-2 sentences.\n"
        "- match_strength must be one of: strong, partial, gap.\n"
        f"Return ONLY JSON with this schema: {json.dumps(schema)}\n"
        f"Requirements: {json.dumps(requirements)}\n"
        f"CV: {cv_text[:12000]}\n"
        f"Candidate notes: {candidate_notes[:3000]}"
    )


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _safe_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
