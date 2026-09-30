@echo off
rem EDUX Slayers - Cap nhat extension, giu nguyen cau hinh AI. Xem update.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %* & exit /b
