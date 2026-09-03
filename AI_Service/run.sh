#!/usr/bin/env bash
# Start AI_Service on Linux. Core_LLM must be reachable (set CORE_LLM_URL in
# .env if it isn't at the default http://localhost:8001) -- AI_Service loads
# no model of its own, so it needs no GPU.
set -e
pip install -r requirements.txt
python main.py
