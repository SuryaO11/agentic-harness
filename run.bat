@echo off
title Agentic Harness v1.2 Dashboard
echo Starting Agentic Harness Dashboard...
python run_dashboard.py
if errorlevel 1 (
    echo.
    echo Dashboard stopped with error.
    pause
)
