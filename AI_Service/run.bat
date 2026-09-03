@echo off
REM Start AI_Service. Core_LLM must be running/reachable (set CORE_LLM_URL in
REM .env if it is not at the default http://127.0.0.1:8001). No GPU needed
REM here -- this module never loads a model.
pip install -r requirements.txt
python main.py
