@echo off
title Face Masker - Web UI
echo ========================================================
echo   Launching Face Masker Web Application...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
