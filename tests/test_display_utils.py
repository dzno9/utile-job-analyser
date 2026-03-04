from __future__ import annotations

import unittest

from display_utils import (
    clean_text,
    humanize_warning,
    map_loading_step,
    map_recommendation,
    score_tier,
)
from models import Recommendation


class TestDisplayUtils(unittest.TestCase):
    def test_map_recommendation_for_known_enum(self) -> None:
        mapped = map_recommendation(Recommendation.APPLY_WITH_CAVEATS)
        self.assertEqual(mapped["label"], "Conditional Match")
        self.assertEqual(mapped["color"], "#d97706")

    def test_map_recommendation_for_unknown_value(self) -> None:
        mapped = map_recommendation("UnexpectedValue")
        self.assertEqual(mapped["label"], "UnexpectedValue")
        self.assertEqual(mapped["description"], "")
        self.assertEqual(mapped["color"], "#6b7280")

    def test_score_tier_boundaries(self) -> None:
        self.assertEqual(score_tier(70)["label"], "Strong Match")
        self.assertEqual(score_tier(40)["label"], "Partial Match")
        self.assertEqual(score_tier(39)["label"], "Weak Match")

    def test_clean_text_removes_fallback_phone_email_and_pipes(self) -> None:
        dirty = "[fallback] Dika Satria +44-7824-711960 | dzno9a@gmail"
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "Dika Satria")

    def test_clean_text_returns_default_for_empty_result(self) -> None:
        cleaned = clean_text("[fallback]  +44-7824-711960 | dzno9a@gmail")
        self.assertEqual(cleaned, "Based on overall profile")

    def test_humanize_warning_for_scrape_confidence(self) -> None:
        warning = "Scrape: Computed confidence=0.75"
        humanized = humanize_warning(warning)
        self.assertEqual(
            humanized,
            "Job posting scraped with 75% confidence - some details may be approximate",
        )

    def test_humanize_warning_for_limited_public_evidence(self) -> None:
        warning = "Limited public evidence found for this company profile."
        humanized = humanize_warning(warning)
        self.assertEqual(
            humanized,
            "Limited public info - prepare role-specific questions for your interview",
        )

    def test_map_loading_step_known_and_unknown(self) -> None:
        self.assertEqual(map_loading_step("Visa Check"), "Checking visa requirements")
        self.assertEqual(map_loading_step("Unmapped"), "Unmapped")


if __name__ == "__main__":
    unittest.main()
