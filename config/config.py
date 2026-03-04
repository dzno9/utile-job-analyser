from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Load .env file from project root if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    # Shell-exported variables should win over .env values.
    load_dotenv(_env_path, override=False)
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: Literal["anthropic", "openai", "rule_based"]
    llm_model: str
    anthropic_api_key: str
    openai_api_key: str


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "rule_based").strip().lower()
    if provider not in {"anthropic", "openai", "rule_based"}:
        raise ValueError(
            "Unsupported LLM_PROVIDER. Expected one of: anthropic, openai, rule_based"
        )

    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )
