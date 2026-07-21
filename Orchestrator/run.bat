@echo off
REM Start the orchestrator. The STT and Core_LLM servers must be running/reachable
REM (set STT_URL / LLM_URL in .env if they are not at the defaults).
pip install -r requirements.txt
python orchestrator.py
