from __future__ import annotations

import html
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from display_utils import (
    clean_text,
    humanize_warning,
    map_loading_step,
    map_recommendation,
    score_tier,
)
from models import AnalysisReport, Recommendation
from providers import UKVisaSponsorChecker, build_company_researcher
from providers.web_search import DuckDuckGoWebSearchTool
from stages import GapAnalyzer, JobPostingScraper
from stages.application_materials_generator import build_materials_generator
from stages.ui_orchestrator import PipelineOrchestrator, PipelineRequest, StageEvent

VIEW_INPUT = "input"
VIEW_LOADING = "loading"
VIEW_RESULTS = "results"
LOADING_STEP_ORDER = [
    "Job Scrape",
    "CV Parse",
    "Gap Analysis",
    "Company Brief",
    "Application Materials",
]
LOADING_STEP_BY_STAGE_KEY = {
    "scrape": "Job Scrape",
    "gap": "Gap Analysis",
    "company": "Company Brief",
    "visa": "Company Brief",
    "materials": "Application Materials",
    "assemble": "Application Materials",
}


st.set_page_config(page_title="Utile Job Analyzer", layout="wide")


def main() -> None:
    _inject_styles()
    _init_state()
    _render_sidebar()

    if st.session_state["current_view"] == VIEW_LOADING:
        _render_loading_page()
    elif (
        st.session_state["current_view"] == VIEW_RESULTS
        and st.session_state.get("analysis_result") is not None
    ):
        _render_results_page()
    else:
        _render_input_page()


def _render_sidebar() -> None:
    from config import load_settings

    env_settings = load_settings()

    with st.sidebar:
        st.header("Settings")
        providers = ["anthropic", "openai"]
        default_idx = providers.index(env_settings.llm_provider) if env_settings.llm_provider in providers else 0
        provider = st.selectbox(
            "LLM Provider",
            providers,
            index=default_idx,
            format_func=str.title,
        )
        model_options: dict[str, list[str]] = {
            "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
            "openai": ["gpt-4o", "gpt-4o-mini"],
        }
        options = model_options.get(provider, [])
        # If the .env model matches this provider and isn't in the list, prepend it.
        if provider == env_settings.llm_provider and env_settings.llm_model not in options:
            options = [env_settings.llm_model] + options
        # Default to .env model if it's in the list.
        default_model_idx = options.index(env_settings.llm_model) if env_settings.llm_model in options else 0
        model = st.selectbox("Model", options, index=default_model_idx)
        st.session_state["llm_provider"] = provider
        st.session_state["llm_model"] = model

        key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        if os.getenv(key_name, ""):
            st.caption("API key: configured")
        else:
            st.warning(f"Set {key_name} in .env")


def _render_input_page() -> None:
    st.title("Utile Job Analyzer")
    st.write(
        "Paste a job URL and upload your CV. We'll analyze the match and generate your "
        "application materials."
    )

    # CV file uploader outside the form for reliable file handling across reruns.
    cv_upload = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
    if cv_upload is not None:
        st.session_state["cv_bytes"] = cv_upload.getvalue()
        st.session_state["cv_filename"] = cv_upload.name

    if st.session_state.get("cv_bytes"):
        st.caption(f"CV ready: {st.session_state.get('cv_filename', 'document.pdf')}")

    if st.session_state.get("cv_parse_failed"):
        _render_inline_alert(
            "We couldn't read your PDF. It may be a scanned document. Paste your CV text below."
        )
        st.text_area(
            "Paste CV text",
            key="manual_cv_text",
            placeholder="Paste CV text extracted from your document.",
            height=180,
        )

    with st.form("input_form"):
        job_url = st.text_input(
            "Job URL",
            value=st.session_state.get("job_url", ""),
            placeholder="https://jobs.example.com/...",
        )
        if st.session_state.get("scrape_failed"):
            _render_inline_alert(
                "We couldn't load this job posting. Check the URL or paste the job description below."
            )
            st.text_area(
                "Paste job description",
                key="manual_job_text",
                placeholder="Paste the full job post text here.",
                height=180,
            )

        candidate_notes = st.text_area(
            "Candidate notes (optional)",
            value=st.session_state.get("candidate_notes", ""),
            placeholder=(
                "Add context not in your CV - e.g. 'I met the hiring manager at a conference' "
                "or 'I've used their product for 2 years'"
            ),
            height=140,
        )
        action_label = (
            "Continue Analysis"
            if st.session_state.get("scrape_failed") or st.session_state.get("cv_parse_failed")
            else "Analyze Match"
        )
        submitted = st.form_submit_button(action_label, type="primary", use_container_width=True)

    if not submitted:
        return

    cv_bytes = st.session_state.get("cv_bytes", b"")
    if not job_url.strip():
        st.error("Enter a Job URL to continue.")
        return
    if not cv_bytes:
        st.error("Upload a CV PDF to continue.")
        return

    st.session_state["job_url"] = job_url.strip()
    st.session_state["candidate_notes"] = candidate_notes.strip()
    st.session_state["analysis_result"] = None
    st.session_state["events"] = []

    # Only forward manual fallback text when that fallback path is active.
    manual_job_posting_text = (
        st.session_state.get("manual_job_text", "")
        if st.session_state.get("scrape_failed")
        else ""
    )
    manual_cv_text = (
        st.session_state.get("manual_cv_text", "")
        if st.session_state.get("cv_parse_failed")
        else ""
    )
    st.session_state["pending_manual_job_text"] = manual_job_posting_text
    st.session_state["pending_manual_cv_text"] = manual_cv_text
    st.session_state["current_view"] = VIEW_LOADING
    st.rerun()


def _render_loading_page() -> None:
    if not st.session_state.get("job_url") or not st.session_state.get("cv_bytes"):
        st.session_state["current_view"] = VIEW_INPUT
        st.rerun()
        return

    _execute_pipeline(
        manual_job_posting_text=st.session_state.get("pending_manual_job_text", ""),
        manual_cv_text=st.session_state.get("pending_manual_cv_text", ""),
    )


def _render_results_page() -> None:
    result = st.session_state.get("analysis_result")
    st.session_state["manual_job_text"] = ""
    st.session_state["manual_cv_text"] = ""
    if result is None:
        st.warning("No completed report yet. Start a new analysis.")
        if st.button("Go Back", type="primary"):
            st.session_state["current_view"] = VIEW_INPUT
            st.rerun()
        return

    if result.report is not None:
        _render_results_hero(result.report)
        render_report(
            result.report,
            warnings=result.warnings,
            cover_letter_generated=result.cover_letter_generated,
            show_recommendation_banner=False,
        )
    else:
        st.error("The analysis completed without a full report for this job.")
        if result.warnings:
            with st.expander("Data Quality Notes", expanded=False):
                for warning in result.warnings:
                    st.warning(humanize_warning(warning))

    with st.expander("Processing Details", expanded=False):
        _render_event_timeline(st.session_state.get("events", []))

    if st.button("Analyze Another Job", use_container_width=True):
        _reset_flow()
        st.rerun()


def _execute_pipeline(*, manual_job_posting_text: str = "", manual_cv_text: str = "") -> None:
    # Apply user's model selection from sidebar before building pipeline components.
    os.environ["LLM_PROVIDER"] = st.session_state.get("llm_provider", "anthropic")
    os.environ["LLM_MODEL"] = st.session_state.get("llm_model", "claude-sonnet-4-20250514")

    progress_placeholder = st.empty()
    progress_state: dict[str, Any] = {
        "completed_steps": set(),
        "active_step": LOADING_STEP_ORDER[0],
    }
    _render_loading_progress(progress_placeholder, progress_state)
    events: list[StageEvent] = []

    def on_event(event: StageEvent) -> None:
        events.append(event)
        _update_progress_view(event, progress_placeholder, progress_state)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_cv:
        temp_cv.write(st.session_state["cv_bytes"])
        cv_path = temp_cv.name

    try:
        request = PipelineRequest(
            job_url=st.session_state["job_url"],
            cv_pdf_path=cv_path,
            candidate_notes=st.session_state.get("candidate_notes", ""),
            manual_job_posting_text=manual_job_posting_text,
            manual_cv_text=manual_cv_text,
        )

        orchestrator = build_orchestrator()
        result = orchestrator.run(request, on_event=on_event)
    finally:
        Path(cv_path).unlink(missing_ok=True)

    if result.needs_manual_posting_text:
        st.session_state["events"] = events
        st.session_state["scrape_failed"] = True
        st.session_state["pending_manual_job_text"] = ""
        st.session_state["pending_manual_cv_text"] = ""
        st.session_state["current_view"] = VIEW_INPUT
        st.rerun()

    if result.needs_manual_cv_text:
        st.session_state["events"] = events
        st.session_state["cv_parse_failed"] = True
        st.session_state["pending_manual_job_text"] = ""
        st.session_state["pending_manual_cv_text"] = ""
        st.session_state["current_view"] = VIEW_INPUT
        st.rerun()

    st.session_state["events"] = events
    st.session_state["analysis_result"] = result
    st.session_state["scrape_failed"] = False
    st.session_state["cv_parse_failed"] = False
    st.session_state["pending_manual_job_text"] = ""
    st.session_state["pending_manual_cv_text"] = ""
    st.session_state["current_view"] = VIEW_RESULTS
    st.rerun()


def _render_event_timeline(events: list[StageEvent]) -> None:
    if not events:
        st.caption("No stage events yet.")
        return

    for event in events:
        icon = {
            "running": "⏳",
            "complete": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "skipped": "⏭️",
        }.get(event.status, "•")
        step_label = map_loading_step(event.stage_name)
        st.markdown(f"{icon} **{step_label}** - {clean_text(event.detail)}")


def build_orchestrator() -> PipelineOrchestrator:
    search_tool = DuckDuckGoWebSearchTool()
    company_researcher = build_company_researcher(search_tool=search_tool)
    visa_checker = UKVisaSponsorChecker(search_tool=search_tool)
    return PipelineOrchestrator(
        scraper=JobPostingScraper(),
        gap_analyzer=GapAnalyzer(),
        company_researcher=company_researcher,
        visa_checker=visa_checker,
        materials_generator=build_materials_generator(),
    )


def render_report(
    report: AnalysisReport,
    *,
    warnings: list[str],
    cover_letter_generated: bool,
    show_recommendation_banner: bool = True,
) -> None:
    if show_recommendation_banner:
        _render_recommendation_banner(report)

    tabs = st.tabs(["Scorecard", "Company Brief", "Visa", "Cover Letter", "CV Tweaks"])

    with tabs[0]:
        score_style = score_tier(report.scorecard.total_score)
        st.metric(
            "Role-match score",
            f"{report.scorecard.total_score:.0f}/100",
            score_style["label"],
        )
        _render_scorecard_cards(report.scorecard.rows)

    with tabs[1]:
        st.markdown(report.company_brief.summary)
        if report.company_brief.sources:
            st.markdown("**Sources**")
            for i, source in enumerate(report.company_brief.sources, start=1):
                st.markdown(f"{i}. [{source}]({source})")
        else:
            st.caption("No external sources available in this run.")

    with tabs[2]:
        likelihood_color = {
            "Confirmed sponsor": "#16a34a",
            "Likely": "#d97706",
            "Unknown": "#dc2626",
        }.get(report.visa_assessment.likelihood.value, "#6b7280")
        st.markdown(
            f"<span style='display:inline-block;padding:6px 12px;border-radius:999px;background:{likelihood_color};color:white;font-weight:600;'>"
            f"{report.visa_assessment.likelihood.value}</span>",
            unsafe_allow_html=True,
        )
        st.write(report.visa_assessment.reasoning)
        if report.visa_assessment.evidence:
            st.markdown("**Evidence**")
            for item in report.visa_assessment.evidence:
                if item.startswith("http"):
                    st.markdown(f"- [{item}]({item})")
                else:
                    st.markdown(f"- {item}")

    with tabs[3]:
        if cover_letter_generated and report.recommendation != Recommendation.DO_NOT_APPLY:
            cleaned_cover_letter = clean_text(report.cover_letter.draft_markdown, preserve_paragraphs=True)
            st.markdown(cleaned_cover_letter)
            render_copy_button(cleaned_cover_letter, key="cover_letter_copy")
            with st.expander("Emphasis strategy"):
                for line in report.cover_letter.emphasis_strategy:
                    st.markdown(f"- {clean_text(line)}")
        else:
            st.info("Skipped - score below threshold")

    with tabs[4]:
        for idx, tweak in enumerate(report.cv_tweaks, start=1):
            st.markdown(f"{idx}. {tweak}")

    if warnings:
        with st.expander("Data Quality Notes", expanded=False):
            for warning in warnings:
                st.warning(humanize_warning(warning))


def _render_scorecard_cards(rows: list) -> None:
    match_styles = {
        "strong": ("#059669", "#ECFDF5", "Strong"),
        "partial": ("#d97706", "#FFFBEB", "Partial"),
        "gap": ("#dc2626", "#FEF2F2", "Gap"),
    }
    cards: list[str] = []
    for row in rows:
        color, bg, label = match_styles.get(
            row.match_strength.value, ("#6b7280", "#f3f4f6", "\u2014")
        )
        req = html.escape(clean_text(row.requirement_from_job_post))
        evidence_raw = clean_text(row.matching_experience or "")
        evidence = html.escape(evidence_raw) if evidence_raw.strip() else "Based on overall profile"
        rationale = html.escape(clean_text(row.rationale or ""))
        cards.append(
            f'<div style="background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px 20px;margin-bottom:12px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></span>'
            f'<span style="font-size:15px;font-weight:600;color:#111827;">{req}</span>'
            f'</div>'
            f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;background:{bg};color:{color};font-size:12px;font-weight:600;white-space:nowrap;">{label}</span>'
            f'</div>'
            f'<div style="margin-bottom:8px;">'
            f'<div style="font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Your evidence</div>'
            f'<div style="font-size:14px;color:#374151;line-height:1.6;">{evidence}</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Assessment</div>'
            f'<div style="font-size:14px;color:#374151;line-height:1.6;">{rationale}</div>'
            f'</div>'
            f'</div>'
        )
    st.markdown("\n".join(cards), unsafe_allow_html=True)


def _build_score_ring_svg(score: int | float, *, size: int = 120, stroke_width: int = 8) -> str:
    tier = score_tier(score)
    color = tier["color"]
    label = html.escape(tier["label"])
    clamped_score = max(0.0, min(100.0, float(score)))
    rounded_score = int(round(clamped_score))

    radius = (size - stroke_width) / 2
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - clamped_score / 100)

    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{size / 2}" cy="{size / 2}" r="{radius}"
              fill="none" stroke="#e5e7eb" stroke-width="{stroke_width}"/>
      <circle cx="{size / 2}" cy="{size / 2}" r="{radius}"
              fill="none" stroke="{color}" stroke-width="{stroke_width}"
              stroke-dasharray="{circumference:.2f}"
              stroke-dashoffset="{offset:.2f}"
              stroke-linecap="round"
              transform="rotate(-90 {size / 2} {size / 2})"/>
      <text x="50%" y="48%" text-anchor="middle" dominant-baseline="middle"
            font-family="JetBrains Mono, monospace" font-size="32" font-weight="700"
            fill="#111827">{rounded_score}</text>
      <text x="50%" y="68%" text-anchor="middle" dominant-baseline="middle"
            font-family="DM Sans, sans-serif" font-size="11" fill="{color}" font-weight="600">{label}</text>
    </svg>
    """


def _render_results_hero(report: AnalysisReport) -> None:
    recommendation = map_recommendation(report.recommendation)
    recommendation_color = recommendation["color"]
    recommendation_text = html.escape(
        clean_text(recommendation["description"] or recommendation["label"])
    )
    score_style = score_tier(report.scorecard.total_score)
    score_label = html.escape(score_style["label"])
    score_color = score_style["color"]
    score_value = int(round(max(0.0, min(100.0, float(report.scorecard.total_score)))))
    job_title = html.escape(clean_text(report.job_context.job_title))
    company_name = html.escape(clean_text(report.job_context.company_name))

    warning_count = len(report.scorecard.risk_flags)
    warning_html = ""
    if warning_count > 0:
        noun = "item" if warning_count == 1 else "items"
        warning_html = (
            '<div style="margin-top:16px;font-size:16px;font-weight:600;color:#b45309;">'
            f"&#9888; {warning_count} {noun} need attention"
            "</div>"
        )

    score_ring_svg = _build_score_ring_svg(report.scorecard.total_score).strip()
    hero_parts = [
        '<div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:32px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.07);margin-bottom:32px;">',
        '<div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">',
        '<div style="flex:0 0 180px;text-align:center;">',
        score_ring_svg,
        f'<div style="font-family:JetBrains Mono, monospace;font-size:18px;font-weight:700;color:#111827;">{score_value}/100</div>',
        f'<div style="font-size:14px;font-weight:600;color:{score_color};margin-top:4px;">{score_label}</div>',
        '</div>',
        '<div style="flex:1 1 320px;min-width:260px;">',
        f'<div style="font-size:24px;font-weight:700;color:#111827;line-height:1.2;">{job_title}</div>',
        f'<div style="margin-top:6px;font-size:16px;color:#6b7280;">at {company_name}</div>',
        f'<div style="margin-top:18px;font-size:18px;font-weight:600;color:{recommendation_color};">{recommendation_text}</div>',
        warning_html,
        '</div>',
        '</div>',
        '</div>',
    ]
    st.markdown("\n".join(hero_parts), unsafe_allow_html=True)


def render_copy_button(text: str, *, key: str) -> None:
    safe = html.escape(text)
    components.html(
        f"""
        <div style="margin: 8px 0 16px 0;">
          <textarea id="{key}" style="position:absolute;left:-9999px;top:-9999px;">{safe}</textarea>
          <button
            onclick="navigator.clipboard.writeText(document.getElementById('{key}').value).then(() => this.innerText='Copied');"
            style="padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 8px; background: #ffffff; cursor: pointer;"
          >Copy cover letter</button>
        </div>
        """,
        height=52,
    )


def _render_recommendation_banner(report: AnalysisReport) -> None:
    recommendation = map_recommendation(report.recommendation)
    accent = recommendation["color"]
    st.markdown(
        f"""
        <div style="padding:14px 16px;border-radius:12px;background:{accent}22;color:#111827;border:1px solid {accent}66;margin-bottom:12px;">
          <div style="font-size:20px;font-weight:700;color:{accent};">{recommendation["label"]}</div>
          <div style="margin-top:6px;">{recommendation["description"] or clean_text(report.summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _update_progress_view(
    event: StageEvent,
    placeholder,
    progress_state: dict[str, Any],
) -> None:
    loading_step = LOADING_STEP_BY_STAGE_KEY.get(event.stage_key)
    if loading_step is None and event.stage_name in LOADING_STEP_ORDER:
        loading_step = event.stage_name
    if loading_step is None:
        return

    step_index = LOADING_STEP_ORDER.index(loading_step)
    completed_steps: set[str] = progress_state["completed_steps"]

    for prior_step in LOADING_STEP_ORDER[:step_index]:
        completed_steps.add(prior_step)

    if event.status == "running":
        progress_state["active_step"] = loading_step
    elif event.status in {"complete", "warning", "skipped"}:
        if event.stage_key in {"company", "materials"}:
            progress_state["active_step"] = loading_step
        else:
            completed_steps.add(loading_step)
            progress_state["active_step"] = _next_uncompleted_loading_step(completed_steps)
    elif event.status == "failed":
        progress_state["active_step"] = loading_step

    _render_loading_progress(placeholder, progress_state)


def _next_uncompleted_loading_step(completed_steps: set[str]) -> str | None:
    for step in LOADING_STEP_ORDER:
        if step not in completed_steps:
            return step
    return None


def _render_loading_progress(placeholder, progress_state: dict[str, Any]) -> None:
    completed_steps: set[str] = progress_state.get("completed_steps", set())
    active_step = progress_state.get("active_step")

    rows: list[str] = []
    for step in LOADING_STEP_ORDER:
        label = html.escape(map_loading_step(step))
        if step in completed_steps:
            rows.append(
                f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">'
                f'<span style="color:#059669;font-weight:700;width:18px;text-align:center;">&#10003;</span>'
                f'<span style="font-size:15px;color:#9ca3af;">{label}</span>'
                f'</div>'
            )
        elif step == active_step:
            rows.append(
                f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">'
                f'<span style="color:#1a56db;font-weight:700;width:18px;text-align:center;">&#8226;</span>'
                f'<span style="font-size:15px;color:#111827;font-weight:500;">{label}...</span>'
                f'</div>'
            )
        else:
            rows.append(
                f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">'
                f'<span style="color:#d1d5db;width:18px;text-align:center;">&#9675;</span>'
                f'<span style="font-size:15px;color:#d1d5db;">{label}</span>'
                f'</div>'
            )

    parts = [
        '<div style="max-width:480px;margin:48px auto;padding:28px 24px;background:white;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 10px 30px rgba(17,24,39,0.08);">',
        '<div style="font-size:20px;font-weight:600;color:#111827;margin-bottom:16px;">Analyzing your match...</div>',
        *rows,
        '</div>',
    ]
    placeholder.markdown("\n".join(parts), unsafe_allow_html=True)


def _init_state() -> None:
    st.session_state.setdefault("current_view", VIEW_INPUT)
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("job_url", "")
    st.session_state.setdefault("candidate_notes", "")
    st.session_state.setdefault("cv_bytes", b"")
    st.session_state.setdefault("cv_filename", "")
    st.session_state.setdefault("scrape_failed", False)
    st.session_state.setdefault("cv_parse_failed", False)
    st.session_state.setdefault("manual_job_text", "")
    st.session_state.setdefault("manual_cv_text", "")
    st.session_state.setdefault("pending_manual_job_text", "")
    st.session_state.setdefault("pending_manual_cv_text", "")


def _reset_flow() -> None:
    st.session_state["current_view"] = VIEW_INPUT
    st.session_state["analysis_result"] = None
    st.session_state["events"] = []
    st.session_state["job_url"] = ""
    st.session_state["candidate_notes"] = ""
    st.session_state["cv_bytes"] = b""
    st.session_state["cv_filename"] = ""
    st.session_state["scrape_failed"] = False
    st.session_state["cv_parse_failed"] = False
    st.session_state["manual_job_text"] = ""
    st.session_state["manual_cv_text"] = ""
    st.session_state["pending_manual_job_text"] = ""
    st.session_state["pending_manual_cv_text"] = ""


def _render_inline_alert(message: str) -> None:
    st.markdown(
        f"""
        <div style="background:#FEF3C7;border-left:4px solid #d97706;padding:12px 16px;border-radius:6px;margin:8px 0 16px 0;">
          {html.escape(message)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_styles() -> None:
    components.html(
        '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700'
        '&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">',
        height=0,
    )
    st.markdown(
        '<style>'
        '#MainMenu,footer,header{visibility:hidden}'
        '.stApp{background-color:#fafafa}'
        'html,body,.stApp{font-family:"DM Sans",sans-serif}'
        '.stButton>button[kind="primary"]{background-color:#1a56db;border-radius:6px;font-weight:500;border:1px solid #1a56db}'
        '.stButton>button[kind="primary"]:hover{background-color:#1342a8;border-color:#1342a8}'
        '</style>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
