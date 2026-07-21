#!/usr/bin/env bash
# Needs a display. On Linux, tkinter is a system package: sudo apt install python3-tk
set -e
pip install -r requirements.txt
python3 app.py
