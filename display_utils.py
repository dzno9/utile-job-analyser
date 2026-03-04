from __future__ import annotations

import re
from typing import Any

RECOMMENDATION_MAP = {
    "StrongApply": {
        "label": "Strong Match",
        "description": "Apply With Confidence",
        "color": "#059669",
    },
    "ConfidentApply": {
        "label": "Strong Match",
        "description": "Apply With Confidence",
        "color": "#059669",
    },
    "ApplyWithCaveats": {
        "label": "Conditional Match",
        "description": "Worth Applying - Review Caveats First",
        "color": "#d97706",
    },
    "DoNotApply": {
        "label": "Not Recommended",
        "description": "Significant Gaps Identified",
        "color": "#dc2626",
    },
}

STEP_MAP = {
    "Job Scrape": "Reading job requirements",
    "CV Parse": "Parsing your experience",
    "Gap Analysis": "Comparing your fit",
    "Company Brief": "Researching the company",
    "Visa Check": "Checking visa requirements",
    "Cover Letter": "Drafting your cover letter",
    "CV Tweaks": "Suggesting CV improvements",
    "Application Materials": "Drafting your cover letter and CV improvements",
}


def map_recommendation(enum_value: Any) -> dict[str, str]:
    value = getattr(enum_value, "value", enum_value)
    if value in RECOMMENDATION_MAP:
        return RECOMMENDATION_MAP[str(value)]
    key = str(value)
    return RECOMMENDATION_MAP.get(key, {"label": key, "description": "", "color": "#6b7280"})


def score_tier(score: int | float) -> dict[str, str]:
    if score >= 70:
        return {"label": "Strong Match", "color": "#059669"}
    if score >= 40:
        return {"label": "Partial Match", "color": "#d97706"}
    return {"label": "Weak Match", "color": "#dc2626"}


def clean_text(text: str | None, *, preserve_paragraphs: bool = False) -> str:
    if not text:
        return "Based on overall profile"
    cleaned = re.sub(r"\[fallback\]\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\+?\d[\d\s\-]{8,}\d", "", cleaned)
    cleaned = re.sub(r"\S+@\S+", "", cleaned)
    cleaned = re.sub(r"\s*\|\s*", " ", cleaned)
    if preserve_paragraphs:
        # Normalise paragraph breaks to exactly \n\n, then clean whitespace within each paragraph.
        cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
        paragraphs = cleaned.split("\n\n")
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
        cleaned = "\n\n".join(p for p in paragraphs if p)
    else:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "Based on overall profile"


def humanize_warning(warning: str) -> str:
    if not warning:
        return "Data quality note is unavailable."

    confidence_match = re.search(r"Scrape:\s*Computed confidence=(\d+(?:\.\d+)?)", warning)
    if confidence_match:
        confidence = float(confidence_match.group(1))
        percent = round(confidence * 100)
        return (
            f"Job posting scraped with {percent}% confidence - some details may be approximate"
        )

    if "Limited public evidence found" in warning:
        return "Limited public info - prepare role-specific questions for your interview"

    return clean_text(warning)


def map_loading_step(backend_name: str) -> str:
    return STEP_MAP.get(backend_name, backend_name)
