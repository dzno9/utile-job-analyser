from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
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
VIEW_RESULTS = "results"


st.set_page_config(page_title="Utile Job Analyzer", layout="wide")


def main() -> None:
    _inject_styles()
    _init_state()
    _render_sidebar()

    if (
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
    _execute_pipeline(
        manual_job_posting_text=manual_job_posting_text,
        manual_cv_text=manual_cv_text,
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
        recommendation = map_recommendation(result.report.recommendation)
        score_style = score_tier(result.report.scorecard.total_score)
        st.subheader("Analysis Summary")
        st.markdown(
            f"<span style='display:inline-block;padding:6px 12px;border-radius:999px;background:{score_style['color']};"
            "color:white;font-weight:600;'>"
            f"Role-match score: {result.report.scorecard.total_score:.0f}/100 ({score_style['label']})</span>",
            unsafe_allow_html=True,
        )
        st.write(f"Recommendation: **{recommendation['label']}**")
        if recommendation["description"]:
            st.caption(recommendation["description"])

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

    progress_placeholder = st.container()
    stage_views: dict[str, Any] = {}
    events: list[StageEvent] = []

    def on_event(event: StageEvent) -> None:
        events.append(event)
        _update_progress_view(event, progress_placeholder, stage_views)

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
        st.session_state["current_view"] = VIEW_INPUT
        st.rerun()

    if result.needs_manual_cv_text:
        st.session_state["events"] = events
        st.session_state["cv_parse_failed"] = True
        st.session_state["current_view"] = VIEW_INPUT
        st.rerun()

    st.session_state["events"] = events
    st.session_state["analysis_result"] = result
    st.session_state["scrape_failed"] = False
    st.session_state["cv_parse_failed"] = False
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
        rows = [
            {
                "Requirement": row.requirement_from_job_post,
                "Match": row.match_strength.value,
                "Evidence": clean_text(row.matching_experience),
                "Rationale": clean_text(row.rationale),
            }
            for row in report.scorecard.rows
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
    stage_views: dict[str, Any],
) -> None:
    state_map = {
        "running": "running",
        "complete": "complete",
        "warning": "complete",
        "failed": "error",
        "skipped": "complete",
    }
    label = f"{map_loading_step(event.stage_name)}: {clean_text(event.detail)}"

    if event.stage_key not in stage_views:
        stage_views[event.stage_key] = placeholder.status(
            label,
            state=state_map.get(event.status, "running"),
            expanded=True,
        )
        return

    stage_views[event.stage_key].update(
        label=label,
        state=state_map.get(event.status, "running"),
        expanded=True,
    )


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
        """
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
        :root {
          --color-primary: #1a56db;
          --color-primary-hover: #1342a8;
          --color-primary-light: #e8effc;
          --color-strong: #059669;
          --color-partial: #d97706;
          --color-weak: #dc2626;
          --color-bg: #fafafa;
          --color-surface: #ffffff;
          --color-border: #e5e7eb;
          --color-text: #111827;
          --color-text-secondary: #6b7280;
          --color-text-muted: #9ca3af;
        }

        html, body, [class*="st-"] {
          font-family: 'DM Sans', sans-serif !important;
        }

        #MainMenu, footer, header {
          visibility: hidden;
        }

        .stApp {
          background-color: var(--color-bg);
        }

        .stButton > button[kind="primary"] {
          background-color: var(--color-primary);
          border-radius: 6px;
          font-weight: 500;
          border: 1px solid var(--color-primary);
        }

        .stButton > button[kind="primary"]:hover {
          background-color: var(--color-primary-hover);
          border-color: var(--color-primary-hover);
        }
        </style>
        """,
        height=0,
    )


if __name__ == "__main__":
    main()
