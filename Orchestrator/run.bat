@echo off
REM Start the orchestrator. STT and Core_LLM must be running/reachable
REM (set STT_URL / LLM_URL in .env if they are not at the defaults).
pip install -r requirements.txt
python main.py
