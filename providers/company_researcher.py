from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from typing import Protocol

from anthropic import Anthropic

from config import load_settings
from models import CompanyBrief, JobContext, ResearchFinding


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    published_date: str | None
    snippet: str
    content: str = ""


@dataclass(frozen=True)
class ResearchPlan:
    profile: str
    queries: list[str]


@dataclass(frozen=True)
class CompanyResearchResult:
    company_brief: CompanyBrief
    findings: list[ResearchFinding]
    warnings: list[str]
    strategy_profile: str
    queries_executed: list[str]
    search_iterations_used: int
    synthesized_source_count: int


class WebSearchTool(Protocol):
    def search(
        self,
        *,
        query: str,
        iteration: int,
        max_results: int,
    ) -> list[SearchResult]:
        ...


class ResearchStrategist(Protocol):
    def plan(
        self,
        *,
        job_context: JobContext,
        candidate_notes: str,
        max_iterations: int,
    ) -> ResearchPlan:
        ...


class RuleBasedResearchStrategist:
    _FINTECH_HINTS = (
        "fintech",
        "bank",
        "payments",
        "fca",
        "kyc",
        "aml",
        "compliance",
        "credit",
        "lending",
    )
    _STARTUP_HINTS = (
        "seed",
        "series a",
        "series b",
        "early-stage",
        "early stage",
        "startup",
        "founding",
        "runway",
    )
    _ENTERPRISE_HINTS = (
        "enterprise",
        "public company",
        "fortune 500",
        "earnings",
        "reorg",
        "layoff",
        "global org",
    )
    _ENTERPRISE_COMPANIES = {
        "google",
        "meta",
        "microsoft",
        "amazon",
        "apple",
        "salesforce",
        "oracle",
    }
    _FINTECH_COMPANIES = {
        "cleo",
        "monzo",
        "revolut",
        "starling",
        "stripe",
        "wise",
    }
    _TECH_STACK_TERMS = (
        "segment",
        "snowflake",
        "databricks",
        "kafka",
        "dbt",
        "amplitude",
        "mixpanel",
        "looker",
        "bigquery",
        "airflow",
    )

    def plan(
        self,
        *,
        job_context: JobContext,
        candidate_notes: str,
        max_iterations: int,
    ) -> ResearchPlan:
        company = job_context.company_name.strip()
        profile = self._classify(job_context)

        queries: list[str] = []
        if profile == "fintech":
            queries.extend(
                [
                    f"{company} FCA register regulatory status",
                    f"{company} compliance enforcement news AML KYC",
                ]
            )
        elif profile == "startup":
            queries.extend(
                [
                    f"{company} funding round investors valuation",
                    f"{company} headcount growth hiring trajectory",
                ]
            )
        elif profile == "enterprise":
            queries.extend(
                [
                    f"{company} recent earnings strategic priorities",
                    f"{company} layoffs reorg leadership changes product direction",
                ]
            )

        note_signal = self._extract_tech_signal(candidate_notes)
        if note_signal:
            queries.append(f"{company} {note_signal} data stack architecture")

        queries.extend(
            [
                f"{company} blog product launch announcement",
                f"{company} engineering culture values",
            ]
        )

        deduped = _dedupe_keep_order(queries)
        return ResearchPlan(profile=profile, queries=deduped[:max_iterations])

    def _classify(self, job_context: JobContext) -> str:
        text_blob = " ".join(
            [
                job_context.company_name,
                job_context.job_title,
                job_context.posting_text,
                " ".join(job_context.requirements),
                " ".join(job_context.required_skills),
            ]
        ).lower()
        company_lower = job_context.company_name.lower().strip()

        if company_lower in self._FINTECH_COMPANIES or _contains_any(
            text_blob, self._FINTECH_HINTS
        ):
            return "fintech"
        if company_lower in self._ENTERPRISE_COMPANIES or _contains_any(
            text_blob, self._ENTERPRISE_HINTS
        ):
            return "enterprise"
        if _contains_any(text_blob, self._STARTUP_HINTS):
            return "startup"
        return "general"

    def _extract_tech_signal(self, candidate_notes: str) -> str:
        text = (candidate_notes or "").lower()
        for term in self._TECH_STACK_TERMS:
            if term in text:
                return term

        # Fallback for "they use X" phrasing.
        match = re.search(r"\buse\s+([A-Za-z0-9.+_-]{3,20})\b", candidate_notes or "")
        if match:
            return match.group(1)
        return ""


class AnthropicResearchStrategist:
    def __init__(self, *, api_key: str, model: str) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.fallback = RuleBasedResearchStrategist()

    def plan(
        self,
        *,
        job_context: JobContext,
        candidate_notes: str,
        max_iterations: int,
    ) -> ResearchPlan:
        prompt = _build_strategy_prompt(
            job_context=job_context,
            candidate_notes=candidate_notes,
            max_iterations=max_iterations,
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "text", None)
        )
        parsed = _extract_json_object(raw_text)
        plan = _plan_from_payload(parsed=parsed, max_iterations=max_iterations)
        if plan is not None:
            return plan
        return self.fallback.plan(
            job_context=job_context,
            candidate_notes=candidate_notes,
            max_iterations=max_iterations,
        )


class OpenAIResearchStrategist:
    def __init__(self, *, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required when LLM_PROVIDER=openai"
            ) from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fallback = RuleBasedResearchStrategist()

    def plan(
        self,
        *,
        job_context: JobContext,
        candidate_notes: str,
        max_iterations: int,
    ) -> ResearchPlan:
        prompt = _build_strategy_prompt(
            job_context=job_context,
            candidate_notes=candidate_notes,
            max_iterations=max_iterations,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0,
        )
        raw_text = (response.choices[0].message.content or "") if response.choices else ""
        parsed = _extract_json_object(raw_text)
        plan = _plan_from_payload(parsed=parsed, max_iterations=max_iterations)
        if plan is not None:
            return plan
        return self.fallback.plan(
            job_context=job_context,
            candidate_notes=candidate_notes,
            max_iterations=max_iterations,
        )


class _LLMSynthesizer:
    """Synthesize a company brief from raw search findings using an LLM."""

    def __init__(self, *, client, model: str, provider: str) -> None:
        self.client = client
        self.model = model
        self.provider = provider

    def synthesize(
        self,
        *,
        company_name: str,
        profile: str,
        findings: list[ResearchFinding],
        job_context: JobContext | None = None,
    ) -> str | None:
        findings_text = "\n".join(
            f"- [{f.source_url}] {f.claim}" for f in findings[:8]
        ) if findings else "(no web search results available)"

        job_hint = ""
        if job_context:
            job_hint = (
                f"\nJob context for reference: {job_context.job_title} at {company_name}"
                f" ({job_context.location or 'location unknown'})."
                f" Skills mentioned: {', '.join(job_context.required_skills[:8]) or 'none listed'}."
            )

        prompt = (
            f"Write a company brief (4-6 sentences) for **{company_name}**.\n\n"
            "Use your training knowledge as the primary source. Supplement with the web search snippets below "
            "for recent or specific details.\n\n"
            "Cover:\n"
            "1. What the company does (products/services, core business)\n"
            "2. Industry, approximate size, and headquarters if known\n"
            "3. Notable recent activity (funding, launches, acquisitions, leadership changes)\n"
            "4. Anything relevant for a job applicant to know\n\n"
            "Rules:\n"
            "- Be specific and factual. If you're unsure about something, omit it rather than guess.\n"
            "- Clearly distinguish between what you know from training data vs. web search snippets.\n"
            "- Do NOT use generic filler phrases like 'dynamic company' or 'innovative leader'.\n"
            "- If the company is genuinely obscure, say so and focus on what IS available.\n"
            f"- Company profile category: {profile}\n"
            f"{job_hint}\n\n"
            f"Web search snippets:\n{findings_text}\n\n"
            "Return ONLY the brief text, no JSON or markdown formatting."
        )

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=400,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(
                    block.text for block in response.content if getattr(block, "text", None)
                ).strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.2,
                )
                if response.choices:
                    return (response.choices[0].message.content or "").strip()
                return None
        except Exception:  # noqa: BLE001
            return None


class CompanyResearcher:
    def __init__(
        self,
        *,
        search_tool: WebSearchTool,
        strategist: ResearchStrategist | None = None,
        synthesizer: _LLMSynthesizer | None = None,
        max_search_iterations: int = 4,
        max_synthesized_sources: int = 8,
        max_source_chars: int = 1200,
    ) -> None:
        self.search_tool = search_tool
        self.strategist = strategist or RuleBasedResearchStrategist()
        self.synthesizer = synthesizer
        self.max_search_iterations = max_search_iterations
        self.max_synthesized_sources = max_synthesized_sources
        self.max_source_chars = max_source_chars

    def research(
        self,
        *,
        job_context: JobContext,
        candidate_notes: str,
    ) -> CompanyResearchResult:
        plan = self.strategist.plan(
            job_context=job_context,
            candidate_notes=candidate_notes,
            max_iterations=self.max_search_iterations,
        )

        gathered: list[SearchResult] = []
        queries_executed: list[str] = []
        for iteration, query in enumerate(plan.queries[: self.max_search_iterations], start=1):
            queries_executed.append(query)
            gathered.extend(
                self.search_tool.search(query=query, iteration=iteration, max_results=4)
            )

        deduped_sources = self._select_sources(
            company_name=job_context.company_name,
            profile=plan.profile,
            sources=gathered,
        )
        findings = [
            self._to_finding(
                source=source,
                profile=plan.profile,
                company_name=job_context.company_name,
            )
            for source in deduped_sources
        ]
        brief = self._to_company_brief(
            company_name=job_context.company_name,
            findings=findings,
            profile=plan.profile,
            job_context=job_context,
        )
        warnings = self._build_warnings(findings=findings)

        return CompanyResearchResult(
            company_brief=brief,
            findings=findings,
            warnings=warnings,
            strategy_profile=plan.profile,
            queries_executed=queries_executed,
            search_iterations_used=len(queries_executed),
            synthesized_source_count=len(deduped_sources),
        )

    def _build_warnings(self, *, findings: list[ResearchFinding]) -> list[str]:
        warnings: list[str] = []
        if not findings:
            warnings.append(
                "Limited public evidence found; company brief may be incomplete."
            )
            return warnings

        low_confidence = [finding for finding in findings if finding.confidence <= 0.65]
        if low_confidence:
            warnings.append(
                f"{len(low_confidence)} finding(s) have low confidence and should be verified."
            )
        return warnings

    def _select_sources(
        self,
        *,
        company_name: str,
        profile: str,
        sources: list[SearchResult],
    ) -> list[SearchResult]:
        seen_urls: set[str] = set()
        ranked: list[tuple[float, SearchResult]] = []
        for source in sources:
            if source.url in seen_urls:
                continue
            seen_urls.add(source.url)
            score = _score_source(source=source, company_name=company_name, profile=profile)
            ranked.append((score, source))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [source for _, source in ranked[: self.max_synthesized_sources]]

    def _to_finding(
        self,
        *,
        source: SearchResult,
        profile: str,
        company_name: str,
    ) -> ResearchFinding:
        raw_text = source.content or source.snippet or source.title
        truncated = _truncate(raw_text, self.max_source_chars)
        # Use the full truncated text as the claim when we have rich content,
        # not just the first sentence.
        claim = truncated if len(truncated) > 80 else (_first_sentence(truncated) or source.title)
        relevance = _build_relevance(source=source, profile=profile)
        confidence = _score_confidence(source=source, company_name=company_name)
        source_date = (source.published_date or "").strip() or datetime.now(
            UTC
        ).date().isoformat()
        return ResearchFinding(
            claim=claim,
            source_url=source.url,
            source_date=source_date,
            relevance=relevance,
            confidence=confidence,
        )

    def _to_company_brief(
        self,
        *,
        company_name: str,
        findings: list[ResearchFinding],
        profile: str,
        job_context: JobContext | None = None,
    ) -> CompanyBrief:
        industry = {
            "fintech": "Financial services technology",
            "startup": "Technology startup",
            "enterprise": "Enterprise technology",
            "general": "Technology",
        }.get(profile, "Technology")
        size = {
            "fintech": "Unknown",
            "startup": "Early-stage",
            "enterprise": "Large enterprise/public",
            "general": None,
        }.get(profile)

        # Try LLM synthesis — works even with zero findings since it uses training knowledge.
        summary = None
        if self.synthesizer:
            summary = self.synthesizer.synthesize(
                company_name=company_name,
                profile=profile,
                findings=findings,
                job_context=job_context,
            )

        # Fallback: concatenate top claims
        if not summary:
            top_claims = [f.claim for f in findings[:3]]
            summary = (
                "; ".join(top_claims)
                if top_claims
                else "Limited public sources found; use role-specific interview questions."
            )

        return CompanyBrief(
            company_name=company_name,
            summary=summary,
            industry=industry,
            size=size,
            sources=[finding.source_url for finding in findings],
        )


def build_company_researcher(*, search_tool: WebSearchTool) -> CompanyResearcher:
    settings = load_settings()
    strategist: ResearchStrategist
    synthesizer: _LLMSynthesizer | None = None

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        client = Anthropic(api_key=settings.anthropic_api_key)
        strategist = AnthropicResearchStrategist(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
        )
        synthesizer = _LLMSynthesizer(
            client=client, model=settings.llm_model, provider="anthropic"
        )
    elif settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=settings.openai_api_key)
        strategist = OpenAIResearchStrategist(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
        )
        synthesizer = _LLMSynthesizer(
            client=client, model=settings.llm_model, provider="openai"
        )
    else:
        strategist = RuleBasedResearchStrategist()

    return CompanyResearcher(
        search_tool=search_tool, strategist=strategist, synthesizer=synthesizer
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


# Domains that return generic/noisy content unrelated to specific company research.
_NOISY_DOMAINS = (
    "microsoft.com",
    "support.microsoft.com",
    "learn.microsoft.com",
    "answers.microsoft.com",
    "wikipedia.org",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "youtube.com",
    "amazon.com",
    "ebay.com",
    "indeed.com",
    "glassdoor.com",
    "linkedin.com/jobs",
)


def _score_source(*, source: SearchResult, company_name: str, profile: str) -> float:
    haystack = " ".join(
        [source.title, source.snippet, source.content, source.url]
    ).lower()
    url_lower = source.url.lower()
    score = 0.0

    # Penalize noisy/generic domains that rarely have useful company research.
    if any(domain in url_lower for domain in _NOISY_DOMAINS):
        score -= 3.0

    if company_name.lower() in haystack:
        score += 2.0
    # Bonus if the company name is in the domain (likely their own site).
    company_slug = company_name.lower().replace(" ", "").replace("-", "")
    if company_slug in url_lower.replace("-", "").replace(".", ""):
        score += 3.0
    if profile == "fintech" and _contains_any(
        haystack, ("fca", "regulatory", "compliance", "kyc", "aml")
    ):
        score += 2.0
    if profile == "startup" and _contains_any(
        haystack, ("funding", "investor", "series", "seed", "headcount", "growth")
    ):
        score += 2.0
    if profile == "enterprise" and _contains_any(
        haystack, ("earnings", "layoff", "reorg", "leadership", "strategy")
    ):
        score += 2.0
    if _contains_any(haystack, ("blog", "launch", "announcement", "culture")):
        score += 1.0
    # Boost news and tech press sources.
    if _contains_any(url_lower, ("techcrunch", "crunchbase", "bloomberg", "reuters", "ft.com", "sifted")):
        score += 1.5
    if source.published_date:
        score += 0.5
    return score


def _build_relevance(*, source: SearchResult, profile: str) -> str:
    text = " ".join([source.title, source.snippet]).lower()
    if profile == "fintech" and _contains_any(
        text, ("fca", "regulatory", "compliance", "kyc", "aml")
    ):
        return "Supports regulatory/compliance assessment for fintech risk."
    if profile == "startup" and _contains_any(
        text, ("funding", "investor", "series", "seed", "headcount", "growth")
    ):
        return "Supports funding, investor backing, and growth trajectory assessment."
    if profile == "enterprise" and _contains_any(
        text, ("earnings", "layoff", "reorg", "leadership", "strategy", "product direction")
    ):
        return "Supports recent strategic moves and org direction assessment."
    if _contains_any(text, ("blog", "launch", "announcement", "culture")):
        return "Provides product launch and culture signal context."
    return "Provides general company context relevant to role fit."


def _score_confidence(*, source: SearchResult, company_name: str) -> float:
    base = 0.55
    combined = f"{source.title} {source.snippet} {source.content}".lower()
    if company_name.lower() in combined:
        base += 0.15
    if source.published_date:
        base += 0.1
    if len((source.content or source.snippet).strip()) > 240:
        base += 0.1
    return max(0.0, min(1.0, round(base, 2)))


def _truncate(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    for part in parts:
        stripped = part.strip()
        if len(stripped) >= 20:
            return stripped
    return text.strip()


def _build_strategy_prompt(
    *,
    job_context: JobContext,
    candidate_notes: str,
    max_iterations: int,
) -> str:
    return (
        "Plan adaptive company research queries.\n"
        "Return JSON only with keys: profile, queries.\n"
        "Rules:\n"
        "- profile must be one of: fintech, startup, enterprise, general\n"
        f"- include at most {max_iterations} queries\n"
        "- Always include at least one general query for blog/product/culture signals\n"
        "- If candidate_notes mention stack/tooling, include a targeted stack query\n"
        "- Fintech should include regulatory/compliance/FCA angle\n"
        "- Startup should include funding/investor/growth angle\n"
        "- Enterprise should include earnings/layoffs/reorg/leadership/product direction angle\n\n"
        f"job_context={job_context.model_dump_json()}\n"
        f"candidate_notes={candidate_notes}"
    )


def _extract_json_object(text: str) -> dict[str, object] | None:
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


def _plan_from_payload(
    *,
    parsed: dict[str, object] | None,
    max_iterations: int,
) -> ResearchPlan | None:
    if parsed is None:
        return None
    raw_profile = str(parsed.get("profile", "")).strip().lower()
    if raw_profile not in {"fintech", "startup", "enterprise", "general"}:
        return None
    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        return None
    queries = [str(item).strip() for item in raw_queries if str(item).strip()]
    if not queries:
        return None
    return ResearchPlan(
        profile=raw_profile,
        queries=_dedupe_keep_order(queries)[:max_iterations],
    )
