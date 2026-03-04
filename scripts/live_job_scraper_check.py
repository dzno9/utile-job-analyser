from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from stages import JobPostingScraper, detect_platform


@dataclass
class CheckOutcome:
    url: str
    platform: str | None
    passed: bool
    expected: str
    actual: str
    confidence_score: float
    manual_text_input_required: bool
    scrape_succeeded: bool
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live integration checks for job posting scraper."
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Job posting URL. Provide 5 target URLs (2 LinkedIn, 1 Greenhouse, 1 Workable, 1 Lever).",
    )
    parser.add_argument(
        "--broken-url",
        required=True,
        help="Deliberately broken/unreachable URL to validate graceful failure.",
    )
    parser.add_argument(
        "--non-job-url",
        required=True,
        help="Non-job page URL (for example a blog post) to validate low confidence fallback.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum confidence threshold expected for valid target job pages.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path to write full JSON report.",
    )
    return parser.parse_args()


def validate_url_mix(urls: list[str]) -> list[str]:
    errors: list[str] = []
    if len(urls) != 5:
        errors.append(f"Expected exactly 5 target URLs, got {len(urls)}.")

    platform_counts: dict[str, int] = {"linkedin": 0, "greenhouse": 0, "workable": 0, "lever": 0}
    for url in urls:
        platform = detect_platform(url)
        if platform in platform_counts:
            platform_counts[platform] += 1

    if platform_counts["linkedin"] < 2:
        errors.append("Expected at least 2 LinkedIn URLs.")
    if platform_counts["greenhouse"] < 1:
        errors.append("Expected at least 1 Greenhouse URL.")
    if platform_counts["workable"] < 1:
        errors.append("Expected at least 1 Workable URL.")
    if platform_counts["lever"] < 1:
        errors.append("Expected at least 1 Lever URL.")

    return errors


def evaluate_target(scraper: JobPostingScraper, url: str, min_confidence: float) -> CheckOutcome:
    result = scraper.scrape(url)
    ctx = result.job_context

    has_required_fields = bool(
        ctx
        and ctx.job_title
        and ctx.company_name
        and ctx.location
        and len(ctx.required_skills) > 0
    )
    passed = (
        result.scrape_succeeded
        and not result.manual_text_input_required
        and result.confidence_score >= min_confidence
        and has_required_fields
    )
    actual = (
        f"scrape_succeeded={result.scrape_succeeded}, "
        f"manual_text_input_required={result.manual_text_input_required}, "
        f"confidence={result.confidence_score:.2f}, "
        f"required_fields={has_required_fields}"
    )
    return CheckOutcome(
        url=url,
        platform=result.platform,
        passed=passed,
        expected=(
            f"scrape_succeeded=True, manual_text_input_required=False, "
            f"confidence>={min_confidence:.2f}, required_fields=True"
        ),
        actual=actual,
        confidence_score=result.confidence_score,
        manual_text_input_required=result.manual_text_input_required,
        scrape_succeeded=result.scrape_succeeded,
        error=result.error,
    )


def evaluate_broken(scraper: JobPostingScraper, broken_url: str) -> CheckOutcome:
    result = scraper.scrape(broken_url)
    passed = (not result.scrape_succeeded) and result.manual_text_input_required
    actual = (
        f"scrape_succeeded={result.scrape_succeeded}, "
        f"manual_text_input_required={result.manual_text_input_required}, "
        f"error={bool(result.error)}"
    )
    return CheckOutcome(
        url=broken_url,
        platform=result.platform,
        passed=passed,
        expected="scrape_succeeded=False, manual_text_input_required=True, no crash",
        actual=actual,
        confidence_score=result.confidence_score,
        manual_text_input_required=result.manual_text_input_required,
        scrape_succeeded=result.scrape_succeeded,
        error=result.error,
    )


def evaluate_non_job(scraper: JobPostingScraper, non_job_url: str, min_confidence: float) -> CheckOutcome:
    result = scraper.scrape(non_job_url)
    passed = result.manual_text_input_required or result.confidence_score < min_confidence
    actual = (
        f"manual_text_input_required={result.manual_text_input_required}, "
        f"confidence={result.confidence_score:.2f}"
    )
    return CheckOutcome(
        url=non_job_url,
        platform=result.platform,
        passed=passed,
        expected=f"manual_text_input_required=True OR confidence<{min_confidence:.2f}",
        actual=actual,
        confidence_score=result.confidence_score,
        manual_text_input_required=result.manual_text_input_required,
        scrape_succeeded=result.scrape_succeeded,
        error=result.error,
    )


def render_console_report(outcomes: list[CheckOutcome]) -> None:
    print("\nLive scraper integration results:")
    for idx, outcome in enumerate(outcomes, start=1):
        status = "PASS" if outcome.passed else "FAIL"
        print(f"{idx}. [{status}] {outcome.url}")
        print(f"   expected: {outcome.expected}")
        print(f"   actual:   {outcome.actual}")
        if outcome.error:
            print(f"   error:    {outcome.error}")


def main() -> int:
    args = parse_args()
    validation_errors = validate_url_mix(args.url)
    if validation_errors:
        print("Input validation failed:")
        for err in validation_errors:
            print(f"- {err}")
        return 2

    scraper = JobPostingScraper(low_confidence_threshold=args.min_confidence)
    outcomes: list[CheckOutcome] = []

    for url in args.url:
        outcomes.append(evaluate_target(scraper, url, args.min_confidence))
    outcomes.append(evaluate_broken(scraper, args.broken_url))
    outcomes.append(evaluate_non_job(scraper, args.non_job_url, args.min_confidence))

    render_console_report(outcomes)
    passed = [item for item in outcomes if item.passed]
    failed = [item for item in outcomes if not item.passed]

    summary = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(outcomes),
        "passed": len(passed),
        "failed": len(failed),
        "outcomes": [asdict(item) for item in outcomes],
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nWrote JSON report to {args.json_output}")

    if failed:
        print(f"\nResult: FAIL ({len(failed)} failing checks)")
        return 1

    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
