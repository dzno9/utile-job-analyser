# Utile -- AI Job Application Analyzer

Applying for jobs takes hours per application: reading the posting, researching the company, tailoring a cover letter, adjusting your CV. Most of that work is pattern-matching that an LLM can do in seconds.

**Utile** is a Streamlit app that takes a job URL and your CV, then produces:

- **Fit score** (0-100) with per-requirement breakdown and match strength
- **Company intel** synthesized from web search + LLM training knowledge
- **UK visa sponsorship check** against the official sponsor register
- **Tailored cover letter** grounded in your actual experience (no "passionate" filler)
- **CV improvement suggestions** prioritized by impact

## How It Works

```
Job URL + CV (PDF)
       |
       v
 +-----------------+
 | Job Scraper      |  HTML + JSON-LD extraction, login-wall detection
 +-----------------+
       |
       v
 +-----------------+
 | Gap Analyzer     |  LLM compares CV to each job requirement
 +-----------------+
       |
       v
 +-------------------+  +---------------------+
 | Company Researcher |  | UK Visa Checker      |
 | (web search + LLM) |  | (sponsor register)   |
 +-------------------+  +---------------------+
       |                        |
       v                        v
 +-----------------+
 | Materials Gen    |  Cover letter + CV tweaks
 +-----------------+
       |
       v
   Analysis Report
```

Each stage is independently testable. Data flows through **Pydantic models** that enforce type safety and business rules -- for example, the cover letter model rejects generic phrases like "passionate" or "excited to apply", and the scorecard model enforces that recommendation thresholds match the numeric score.

## Technical Decisions

| Decision | Why |
|----------|-----|
| **Streamlit** | Fastest path from Python pipeline to usable UI. No frontend build step. |
| **Pydantic contracts** | LLM output is unpredictable. Typed models with validators catch bad data before it reaches the user. |
| **Protocol-based DI** | Providers implement Python protocols, so swapping LLM backends (Anthropic/OpenAI) or search tools requires zero pipeline changes. |
| **JSON-LD extraction** | Many job sites render via JavaScript, but the JSON-LD `JobPosting` schema is always in the initial HTML. This is the most reliable scraping strategy. |
| **DuckDuckGo search** | No API key required, no rate-limit billing surprises. Good enough for company research context. |
| **LLM + web hybrid research** | Company briefs use LLM training knowledge as the primary source, supplemented by live web results. Pure web search returns too much noise. |

## Quick Start

```bash
# Clone and enter
git clone https://github.com/dzno9/utile-job-analyser.git
cd utile-job-analyser

# Set up Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your Anthropic or OpenAI API key

# Run
./run.sh
```

The app opens at `http://localhost:8501`. Paste a job URL, upload your CV as PDF, and click **Analyze Match**.

## Project Structure

```
app.py                     # Streamlit UI + page routing
display_utils.py           # Text cleaning, enum mapping, PII stripping
models/types.py            # Pydantic data contracts with validators
config/config.py           # Settings loader (.env + defaults)
stages/
  job_posting_scraper.py   # HTML/JSON-LD job extraction
  gap_analyzer.py          # LLM-based CV-to-job comparison
  application_materials_generator.py  # Cover letter + CV tweaks
  ui_orchestrator.py       # Pipeline sequencing + event stream
providers/
  company_researcher.py    # Web search + LLM synthesis
  uk_visa_sponsor_checker.py  # UK sponsor register lookup
  web_search.py            # DuckDuckGo search adapter
  cv_parser.py             # PDF text extraction (PyMuPDF + pdfplumber)
  gap_matcher.py           # Protocol-based gap matching
  job_context_extractor.py # LLM + rule-based job parsing
tests/                     # Unit tests for all modules
```

## Roadmap

- [ ] UI redesign: score visualization, card-based layouts, loading states
- [ ] Export analysis as formatted report
- [ ] Multi-country visa check support
- [ ] Hosted deployment

## License

MIT
