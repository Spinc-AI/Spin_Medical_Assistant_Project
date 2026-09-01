#!/usr/bin/env bash
# Start the orchestrator on Linux. STT and Core_LLM must be reachable
# (set STT_URL / LLM_URL in .env if not at the defaults).
set -e
pip install -r requirements.txt
python main.py
