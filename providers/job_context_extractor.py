from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic

from config import load_settings


@dataclass(frozen=True)
class ExtractionHints:
    job_title: str | None = None
    company_name: str | None = None
    location: str | None = None
    employment_type: str | None = None
    requirements: list[str] | None = None
    tech_stack: list[str] | None = None
    responsibilities: list[str] | None = None
    seniority_level: str | None = None


@dataclass(frozen=True)
class LLMExtraction:
    job_title: str | None
    company_name: str | None
    location: str | None
    required_skills: list[str]
    nice_to_have_skills: list[str]
    employment_type: str | None
    confidence_note: str = ""


class JobContextExtractor(Protocol):
    def extract(
        self,
        *,
        page_text: str,
        job_url: str,
        hints: ExtractionHints,
    ) -> LLMExtraction:
        ...


class RuleBasedJobContextExtractor:
    _SKILL_TOKENS = (
        "python",
        "sql",
        "java",
        "javascript",
        "typescript",
        "react",
        "node",
        "aws",
        "gcp",
        "azure",
        "docker",
        "kubernetes",
        "spark",
        "airflow",
        "dbt",
        "tableau",
        "looker",
        "excel",
        "pandas",
    )

    def extract(
        self,
        *,
        page_text: str,
        job_url: str,
        hints: ExtractionHints,
    ) -> LLMExtraction:
        text = page_text.lower()
        required = []
        for token in self._SKILL_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", text):
                required.append(token.upper() if token == "sql" else token.title())

        if hints.tech_stack:
            required.extend(hints.tech_stack)

        nice_to_have = []
        if "nice to have" in text or "preferred" in text:
            nice_to_have = required[3:6]

        return LLMExtraction(
            job_title=hints.job_title,
            company_name=hints.company_name,
            location=hints.location,
            required_skills=_dedupe_keep_order(required)[:15],
            nice_to_have_skills=_dedupe_keep_order(nice_to_have),
            employment_type=hints.employment_type,
            confidence_note="rule-based extraction",
        )


class AnthropicJobContextExtractor:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def extract(
        self,
        *,
        page_text: str,
        job_url: str,
        hints: ExtractionHints,
    ) -> LLMExtraction:
        prompt = _build_prompt(job_url=job_url, page_text=page_text, hints=hints)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "text", None)
        )
        payload = _extract_json_object(raw_text)

        return LLMExtraction(
            job_title=_safe_str(payload.get("job_title")),
            company_name=_safe_str(payload.get("company_name")),
            location=_safe_str(payload.get("location")),
            required_skills=_safe_str_list(payload.get("required_skills")),
            nice_to_have_skills=_safe_str_list(payload.get("nice_to_have_skills")),
            employment_type=_safe_str(payload.get("employment_type")),
            confidence_note="anthropic extraction",
        )


class OpenAIJobContextExtractor:
    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required when LLM_PROVIDER=openai"
            ) from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract(
        self,
        *,
        page_text: str,
        job_url: str,
        hints: ExtractionHints,
    ) -> LLMExtraction:
        prompt = _build_prompt(job_url=job_url, page_text=page_text, hints=hints)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
        )
        raw_text = (response.choices[0].message.content or "") if response.choices else ""
        payload = _extract_json_object(raw_text)

        return LLMExtraction(
            job_title=_safe_str(payload.get("job_title")),
            company_name=_safe_str(payload.get("company_name")),
            location=_safe_str(payload.get("location")),
            required_skills=_safe_str_list(payload.get("required_skills")),
            nice_to_have_skills=_safe_str_list(payload.get("nice_to_have_skills")),
            employment_type=_safe_str(payload.get("employment_type")),
            confidence_note="openai extraction",
        )


def build_extractor() -> JobContextExtractor:
    settings = load_settings()
    if settings.llm_provider == "rule_based":
        return RuleBasedJobContextExtractor()
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return AnthropicJobContextExtractor(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIJobContextExtractor(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
        )
    raise ValueError(f"Unsupported provider: {settings.llm_provider}")


def _build_prompt(job_url: str, page_text: str, hints: ExtractionHints) -> str:
    hints_payload: dict[str, Any] = {
        "job_title": hints.job_title,
        "company_name": hints.company_name,
        "location": hints.location,
        "employment_type": hints.employment_type,
        "requirements": hints.requirements or [],
        "tech_stack": hints.tech_stack or [],
        "responsibilities": hints.responsibilities or [],
        "seniority_level": hints.seniority_level,
    }
    schema = {
        "job_title": "string",
        "company_name": "string",
        "location": "string|null",
        "required_skills": ["string"],
        "nice_to_have_skills": ["string"],
        "employment_type": "string|null",
    }
    return (
        "Extract structured job context from this job posting text.\n"
        "Return ONLY valid JSON with this schema:\n"
        f"{json.dumps(schema)}\n"
        f"Job URL: {job_url}\n"
        f"Hints: {json.dumps(hints_payload)}\n"
        "Text:\n"
        f"{page_text[:12000]}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _safe_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _safe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
    return _dedupe_keep_order(result)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output
