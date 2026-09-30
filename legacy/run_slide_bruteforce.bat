@echo off
setlocal
cd /d "%~dp0EDUX-SLIDE-BRUTEFORCE"
python -m pytest -s --headed --browser chromium
endlocal
