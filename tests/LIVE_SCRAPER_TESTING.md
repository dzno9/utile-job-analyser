# Live Job Scraper Integration Run

Use this in a network-enabled environment to validate real URLs for CUS-29 acceptance.

## Command

```bash
.venv/bin/python scripts/live_job_scraper_check.py \
  --url "https://www.linkedin.com/jobs/view/<id-1>/" \
  --url "https://www.linkedin.com/jobs/view/<id-2>/" \
  --url "https://boards.greenhouse.io/<company>/jobs/<id>" \
  --url "https://apply.workable.com/<company>/j/<id>/" \
  --url "https://jobs.lever.co/<company>/<id>" \
  --broken-url "https://this-should-fail.example.com/nope" \
  --non-job-url "https://example.com/blog-post" \
  --min-confidence 0.65 \
  --json-output /tmp/live-scraper-report.json
```

## Expected behavior

1. 5 target job URLs should pass with:
   - `scrape_succeeded=True`
   - `manual_text_input_required=False`
   - confidence >= threshold
   - required fields populated in `JobContext`
2. Broken URL should pass failure-path check with:
   - `scrape_succeeded=False`
   - `manual_text_input_required=True`
3. Non-job URL should pass fallback check with:
   - `manual_text_input_required=True` or confidence below threshold

## Exit codes

- `0`: all checks passed
- `1`: one or more checks failed
- `2`: invalid input URL mix (not matching 2 LinkedIn + 1 Greenhouse + 1 Workable + 1 Lever)
