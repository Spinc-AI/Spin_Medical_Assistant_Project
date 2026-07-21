#!/usr/bin/env bash
# Start the Core_LLM HTTP service on Linux. Ensure Ollama is reachable.
set -e
pip install -r requirements.txt
python main.py
