@echo off
setlocal
cd /d "%~dp0EDUX-TEST-SOLVER"
python -m pytest -s --headed --browser chromium
endlocal
