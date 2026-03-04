from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from models import (
    CompanyBrief,
    CoverLetter,
    JobContext,
    Recommendation,
    ResearchFinding,
    Scorecard,
    VisaAssessment,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationMaterialsResult:
    cover_letter: CoverLetter
    cv_tweaks: list[str]


class LLMClient(Protocol):
    def generate(self, *, prompt: str, max_tokens: int) -> str:
        ...


class AnthropicLLMClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, *, prompt: str, max_tokens: int) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.15,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "text", None)
        )


class OpenAILLMClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, *, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.15,
        )
        if response.choices:
            return response.choices[0].message.content or ""
        return ""


def build_materials_generator() -> ApplicationMaterialsGenerator:
    from config import load_settings

    settings = load_settings()
    llm_client: LLMClient | None = None
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        llm_client = AnthropicLLMClient(
            api_key=settings.anthropic_api_key, model=settings.llm_model
        )
    elif settings.llm_provider == "openai" and settings.openai_api_key:
        llm_client = OpenAILLMClient(
            api_key=settings.openai_api_key, model=settings.llm_model
        )
    return ApplicationMaterialsGenerator(llm_client=llm_client)


class ApplicationMaterialsGenerator:
    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        *,
        job_context: JobContext,
        company_brief: CompanyBrief,
        research_findings: list[ResearchFinding],
        visa_assessment: VisaAssessment,
        scorecard: Scorecard,
        recommendation: Recommendation,
        candidate_name: str,
        candidate_notes: str,
        cv_content: str,
    ) -> ApplicationMaterialsResult:
        if recommendation == Recommendation.DO_NOT_APPLY:
            raise ValueError(
                "Application materials must not be generated for DoNotApply recommendations"
            )

        key_findings = _pick_key_findings(research_findings, company_brief)
        mapped_rows = _pick_mappable_rows(scorecard)
        mapped_rows = mapped_rows[: max(3, min(len(mapped_rows), 5))]

        if self.llm_client is not None:
            result = self._generate_with_llm(
                job_context=job_context,
                company_brief=company_brief,
                visa_assessment=visa_assessment,
                scorecard=scorecard,
                recommendation=recommendation,
                candidate_name=candidate_name,
                candidate_notes=candidate_notes,
                cv_content=cv_content,
                key_findings=key_findings,
                mapped_rows=mapped_rows,
            )
            if result is not None:
                return result
            logger.warning("LLM cover letter generation failed; falling back to template")

        # Fallback: template-based generation
        cover_letter = CoverLetter(
            candidate_name=candidate_name,
            company_name=job_context.company_name,
            job_title=job_context.job_title,
            draft_markdown=_build_cover_letter_markdown(
                candidate_name=candidate_name,
                job_context=job_context,
                recommendation=recommendation,
                key_findings=key_findings,
                mapped_rows=mapped_rows,
                scorecard=scorecard,
                candidate_notes=candidate_notes,
            ),
            emphasis_strategy=_build_emphasis_strategy(
                recommendation=recommendation,
                mapped_rows=mapped_rows,
                key_findings=key_findings,
                scorecard=scorecard,
                visa_assessment=visa_assessment,
            ),
        )

        cv_tweaks = _build_cv_tweaks(
            mapped_rows=mapped_rows,
            scorecard=scorecard,
            key_findings=key_findings,
            candidate_notes=candidate_notes,
            cv_content=cv_content,
        )
        return ApplicationMaterialsResult(cover_letter=cover_letter, cv_tweaks=cv_tweaks)

    def _generate_with_llm(
        self,
        *,
        job_context: JobContext,
        company_brief: CompanyBrief,
        visa_assessment: VisaAssessment,
        scorecard: Scorecard,
        recommendation: Recommendation,
        candidate_name: str,
        candidate_notes: str,
        cv_content: str,
        key_findings: list[str],
        mapped_rows: list[tuple[str, str, str]],
    ) -> ApplicationMaterialsResult | None:
        assert self.llm_client is not None

        prompt = _build_llm_prompt(
            job_context=job_context,
            company_brief=company_brief,
            visa_assessment=visa_assessment,
            scorecard=scorecard,
            recommendation=recommendation,
            candidate_name=candidate_name,
            candidate_notes=candidate_notes,
            cv_content=cv_content,
            key_findings=key_findings,
            mapped_rows=mapped_rows,
        )

        try:
            raw = self.llm_client.generate(prompt=prompt, max_tokens=2500)
        except Exception:  # noqa: BLE001
            logger.warning("LLM call failed for cover letter", exc_info=True)
            return None

        parsed = _extract_json_object(raw)
        if not parsed:
            return None

        draft = (parsed.get("cover_letter") or "").strip()
        if not draft or len(draft) < 100:
            return None

        # Ensure the cover letter has paragraph breaks.
        # LLMs often return the text as one continuous block in JSON despite instructions.
        draft = _ensure_paragraph_breaks(draft)

        strategy_raw = parsed.get("emphasis_strategy")
        if isinstance(strategy_raw, list):
            strategy = [str(s).strip() for s in strategy_raw if str(s).strip()]
        elif isinstance(strategy_raw, str):
            strategy = [s.strip() for s in strategy_raw.split("\n") if s.strip()]
        else:
            strategy = ["LLM-generated cover letter with role-specific positioning."]

        tweaks_raw = parsed.get("cv_tweaks")
        if isinstance(tweaks_raw, list):
            cv_tweaks = [str(t).strip() for t in tweaks_raw if str(t).strip()]
        else:
            cv_tweaks = []

        if not cv_tweaks:
            cv_tweaks = _build_cv_tweaks(
                mapped_rows=mapped_rows,
                scorecard=scorecard,
                key_findings=key_findings,
                candidate_notes=candidate_notes,
                cv_content=cv_content,
            )

        cover_letter = CoverLetter(
            candidate_name=candidate_name,
            company_name=job_context.company_name,
            job_title=job_context.job_title,
            draft_markdown=draft,
            emphasis_strategy=strategy if strategy else ["LLM-generated positioning."],
        )

        return ApplicationMaterialsResult(cover_letter=cover_letter, cv_tweaks=cv_tweaks)


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------


def _build_llm_prompt(
    *,
    job_context: JobContext,
    company_brief: CompanyBrief,
    visa_assessment: VisaAssessment,
    scorecard: Scorecard,
    recommendation: Recommendation,
    candidate_name: str,
    candidate_notes: str,
    cv_content: str,
    key_findings: list[str],
    mapped_rows: list[tuple[str, str, str]],
) -> str:
    tier = recommendation.value
    tier_instruction = (
        "The candidate is a STRONG fit. Write with confident, direct positioning. "
        "Lead with strongest evidence and map experience to requirements precisely."
        if recommendation == Recommendation.CONFIDENT_APPLY
        else (
            "The candidate is a STRETCH fit. Write honestly — lead with transferable strengths, "
            "explicitly acknowledge gaps, and frame them as areas for fast ramp-up. "
            "Do NOT oversell or claim experience the candidate does not have."
        )
    )

    scorecard_summary = []
    for row in scorecard.rows:
        scorecard_summary.append(
            f"- {row.requirement_from_job_post}: {row.match_strength.value} "
            f"| Evidence: {row.matching_experience[:200]}"
        )

    evidence_map = []
    for req, evidence, rationale in mapped_rows[:5]:
        evidence_map.append(f"- Requirement: {req}\n  Evidence: {evidence}\n  Rationale: {rationale}")

    schema = {
        "cover_letter": "string (the full cover letter in markdown)",
        "emphasis_strategy": ["string (why each positioning choice was made)"],
        "cv_tweaks": ["string (specific, actionable CV improvement suggestions)"],
    }

    return (
        "You are an expert career strategist writing a cover letter and CV improvement suggestions.\n\n"
        "COVER LETTER FORMAT:\n"
        "- Structure the letter in 3-4 SHORT paragraphs separated by blank lines (\\n\\n).\n"
        "- Paragraph 1: Opening — state the role, why you're writing, and one sentence connecting you to the company.\n"
        "- Paragraph 2-3: Body — each paragraph has a clear thesis sentence followed by 1-2 supporting sentences.\n"
        "  Map your strongest evidence to the role's key requirements. One theme per paragraph.\n"
        "- Final paragraph: Closing — brief forward-looking statement and sign-off.\n"
        "- Write like a real human — vary sentence length, use natural transitions, avoid bullet points.\n"
        "- Do NOT write the entire letter as one continuous block. Paragraphs MUST be separated by \\n\\n.\n\n"
        "RULES:\n"
        "- Reference specific company context from the research findings — do NOT use generic phrases.\n"
        "- NEVER use phrases like 'passionate about', 'excited to apply', 'great fit', 'thrilled', 'eager', 'driven by'.\n"
        "- ANTI-HALLUCINATION: Every claim MUST cite specific evidence from the CV or candidate notes below.\n"
        "  If evidence does not exist for a skill or experience, do NOT mention it. Omit rather than fabricate.\n"
        "- Do NOT invent metrics, percentages, dollar amounts, or outcomes not present in the CV.\n"
        "- Do NOT assume the candidate has skills, tools, or experience not explicitly stated in the CV.\n"
        "- If the CV lacks evidence for a job requirement, skip it — do not fill gaps with made-up experience.\n"
        "- Keep the letter concise: 250-350 words max.\n"
        "- Address it to the hiring team at the company.\n"
        "- Sign off with the candidate's name.\n"
        f"- Recommendation tier: {tier}. {tier_instruction}\n"
        "- For CV tweaks: provide 3-6 specific, actionable suggestions with before/after examples where possible.\n"
        "  Each tweak should reference a specific job requirement and explain WHY the change helps.\n\n"
        f"Return ONLY valid JSON with this schema: {json.dumps(schema)}\n"
        "IMPORTANT: In the cover_letter JSON value, use \\n\\n between paragraphs so they render as separate blocks.\n\n"
        f"## Candidate\nName: {candidate_name}\n\n"
        f"## Job\nTitle: {job_context.job_title}\n"
        f"Company: {job_context.company_name}\n"
        f"Location: {job_context.location or 'Not specified'}\n\n"
        f"## Company Research\nIndustry: {company_brief.industry}\n"
        f"Summary: {company_brief.summary}\n"
        f"Key findings:\n" + "\n".join(f"- {f}" for f in key_findings) + "\n\n"
        f"## Scorecard ({scorecard.total_score:.0f}/100)\n"
        + "\n".join(scorecard_summary)
        + "\n\n"
        f"## Strongest Evidence Mapping\n"
        + "\n".join(evidence_map)
        + "\n\n"
        f"## Candidate Notes\n{candidate_notes[:2000]}\n\n"
        f"## CV Content (excerpt)\n{cv_content[:6000]}\n\n"
        f"## Visa Context\n{visa_assessment.likelihood.value}: {visa_assessment.reasoning[:300]}"
    )


# ---------------------------------------------------------------------------
# Template-based fallback helpers (used when LLM is unavailable)
# ---------------------------------------------------------------------------


def _pick_key_findings(
    research_findings: list[ResearchFinding], company_brief: CompanyBrief
) -> list[str]:
    extracted = [f.claim.strip() for f in research_findings if f.claim.strip()]
    if len(extracted) >= 2:
        return extracted[:3]
    fallback = [
        company_brief.summary.strip(),
        f"{company_brief.company_name} operates in {company_brief.industry}.",
    ]
    merged = [item for item in extracted + fallback if item]
    deduped: list[str] = []
    for claim in merged:
        if claim not in deduped:
            deduped.append(claim)
    fallback_defaults = [
        f"Public context is limited; focus on {company_brief.company_name}'s role priorities.",
        "Validate strategic context in interview conversations.",
    ]
    for fallback_line in fallback_defaults:
        if len(deduped) >= 2:
            break
        deduped.append(fallback_line)
    return deduped[:3]


def _pick_mappable_rows(scorecard: Scorecard) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for row in scorecard.rows:
        if "No strong matching evidence found" in row.matching_experience:
            continue
        rows.append(
            (
                row.requirement_from_job_post.strip(),
                row.matching_experience.replace("[CV] ", "").replace(
                    "[candidate_notes] ", ""
                ),
                row.rationale.strip(),
            )
        )
    if rows:
        return rows
    return [
        (
            row.requirement_from_job_post.strip(),
            row.matching_experience.strip(),
            row.rationale.strip(),
        )
        for row in scorecard.rows[:3]
    ]


def _build_cover_letter_markdown(
    *,
    candidate_name: str,
    job_context: JobContext,
    recommendation: Recommendation,
    key_findings: list[str],
    mapped_rows: list[tuple[str, str, str]],
    scorecard: Scorecard,
    candidate_notes: str,
) -> str:
    tone_line = (
        f"I can add immediate leverage in the {job_context.job_title} role through direct experience mapped to your current priorities."
        if recommendation == Recommendation.CONFIDENT_APPLY
        else (
            f"I can contribute to the {job_context.job_title} remit now while being explicit about where I will ramp quickly."
        )
    )

    findings_sentence = (
        f"Your current context is clear from public signals: {key_findings[0]}. {key_findings[1]}."
        if len(key_findings) >= 2
        else f"Based on available context: {key_findings[0] if key_findings else 'information about your team'}."
    )

    mapping_lines: list[str] = []
    for requirement, evidence, rationale in mapped_rows[:3]:
        mapping_lines.append(
            f"- {requirement}: {evidence} ({_trim_sentence(rationale)})."
        )

    caveat_line = ""
    if recommendation == Recommendation.APPLY_WITH_CAVEATS:
        gaps = [
            row.requirement_from_job_post
            for row in scorecard.rows
            if row.match_strength.value == "gap"
        ][:2]
        transferable = _first_sentence(candidate_notes) or "I bring adjacent operating experience that transfers to this scope."
        if gaps:
            caveat_line = (
                f"I have not yet owned {', '.join(gaps)} end to end; "
                f"my transferable base is: {transferable}."
            )

    body_lines = [
        f"Dear Hiring Team at {job_context.company_name},",
        "",
        tone_line,
        findings_sentence,
        "",
        f"Direct evidence against {job_context.job_title} requirements:",
        *mapping_lines,
    ]
    if caveat_line:
        body_lines.extend(["", caveat_line])
    body_lines.extend(
        [
            "",
            f"{candidate_name} | {job_context.job_title}",
        ]
    )
    return "\n".join(body_lines).strip()


def _build_emphasis_strategy(
    *,
    recommendation: Recommendation,
    mapped_rows: list[tuple[str, str, str]],
    key_findings: list[str],
    scorecard: Scorecard,
    visa_assessment: VisaAssessment,
) -> list[str]:
    strategy: list[str] = []
    for requirement, evidence, _ in mapped_rows[:3]:
        strategy.append(
            f"Emphasized '{requirement}' because '{_trim_sentence(evidence)}' is direct proof of delivery."
        )
    if len(key_findings) >= 1:
        strategy.append(
            f"Referenced company context ('{_trim_sentence(key_findings[0])}') to make the letter specific to this employer."
        )
    if len(key_findings) >= 2:
        strategy.append(
            f"Referenced company context ('{_trim_sentence(key_findings[1])}') to align examples with current business priorities."
        )
    if recommendation == Recommendation.APPLY_WITH_CAVEATS:
        gaps = [
            row.requirement_from_job_post
            for row in scorecard.rows
            if row.match_strength.value == "gap"
        ][:2]
        strategy.append(
            f"Addressed stretch areas directly ({', '.join(gaps) if gaps else 'identified gaps'}) to keep the narrative credible."
        )
    if visa_assessment.evidence:
        strategy.append(
            "Avoided unsupported relocation claims by only using provided visa assessment evidence."
        )
    return strategy


def _build_cv_tweaks(
    *,
    mapped_rows: list[tuple[str, str, str]],
    scorecard: Scorecard,
    key_findings: list[str],
    candidate_notes: str,
    cv_content: str,
) -> list[str]:
    tweaks: list[str] = []
    context_a = _trim_sentence(key_findings[0]) if key_findings else "company priorities"
    context_b = _trim_sentence(key_findings[1]) if len(key_findings) >= 2 else context_a

    for requirement, evidence, _ in mapped_rows[:3]:
        evidence_excerpt = _trim_sentence(evidence)
        rewrite = (
            f"Reword CV evidence under '{requirement}' to '{_rewrite_fragment(evidence_excerpt)}' "
            f"- aligns with '{context_a}' and reinforces this requirement against current company priorities."
        )
        tweaks.append(rewrite)

    gap_rows = [row for row in scorecard.rows if row.match_strength.value == "gap"][:2]
    note_seed = _first_sentence(candidate_notes) or _first_sentence(cv_content)
    for row in gap_rows:
        tweak = (
            f"Add a targeted bullet for '{row.requirement_from_job_post}' with a measurable result from adjacent work "
            f"(e.g., '{_trim_sentence(note_seed)}') - frame it against '{context_b}' so the transferability is explicit."
        )
        tweaks.append(tweak)

    return tweaks[:6]


def _ensure_paragraph_breaks(text: str) -> str:
    """Post-process a cover letter to guarantee proper paragraph separation.

    If the LLM already inserted double-newlines we leave them alone.
    Otherwise we split on structural cues (greeting, sign-off, topic shifts)
    so the letter always renders as distinct paragraphs.
    """
    # Already has paragraph breaks — trust the LLM output.
    if "\n\n" in text:
        return text

    # Normalise whitespace first.
    text = " ".join(text.split())

    # Split on common structural boundaries.
    # 1. After greeting line  ("Dear …,")
    text = re.sub(r"(Dear\s[^,]+,)\s*", r"\1\n\n", text)

    # 2. Before sign-off phrases
    for signoff in (
        "Best regards",
        "Kind regards",
        "Sincerely",
        "Yours faithfully",
        "Yours sincerely",
        "Thank you for",
        "I would welcome",
        "I look forward",
    ):
        pattern = rf"\.\s+({re.escape(signoff)})"
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, rf".\n\n\1", text, count=1, flags=re.IGNORECASE)
            break

    # 3. If still no breaks after greeting, split the body into roughly equal paragraphs.
    parts = text.split("\n\n")
    rebuilt: list[str] = []
    for part in parts:
        sentences = re.split(r"(?<=[.!?])\s+", part.strip())
        if len(sentences) <= 4:
            rebuilt.append(part.strip())
            continue
        # Group sentences into chunks of 3-4 for natural paragraphs.
        chunk_size = max(3, len(sentences) // 3)
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i : i + chunk_size])
            if chunk.strip():
                rebuilt.append(chunk.strip())
    return "\n\n".join(rebuilt)


def _rewrite_fragment(text: str) -> str:
    compact = _trim_sentence(text)
    if compact.startswith("Built "):
        return compact.replace("Built ", "Designed and shipped ", 1)
    if compact.startswith("Managed "):
        return compact.replace("Managed ", "Owned and optimized ", 1)
    return f"Delivered {compact[0].lower() + compact[1:]}" if compact else "Delivered measurable business impact"


def _first_sentence(text: str) -> str:
    for chunk in text.replace("\n", " ").split("."):
        sentence = chunk.strip()
        if sentence:
            return sentence
    return ""


def _trim_sentence(text: str, max_len: int = 180) -> str:
    compact = " ".join(text.split()).strip().rstrip(".")
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
