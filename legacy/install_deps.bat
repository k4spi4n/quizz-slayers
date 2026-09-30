@echo off
setlocal
cd /d "%~dp0EDUX-TEST-SOLVER"
python -m pip install -r requirements.txt
python -m playwright install chromium

cd /d "%~dp0EDUX-SLIDE-BRUTEFORCE"
python -m pip install -r requirements.txt
python -m playwright install chromium
endlocal
