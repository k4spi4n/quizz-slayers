@echo off
setlocal
cd /d "%~dp0EDUX-LIVE-QUESTION"
python -m pytest -s --headed --browser chromium tests/live_solver.py
endlocal
