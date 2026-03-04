#!/usr/bin/env bash
# Quick launcher — no need to manually activate .venv every time.
# Usage: ./run.sh
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py "$@"
