@echo off
title LLM Inference Benchmark & Optimization UI
cls
echo ================================================================
echo   LLM Inference Benchmark ^& Optimization Study - UI Launcher
echo ================================================================
echo.
echo Select which interface to launch:
echo.
echo   [1] Launch Streamlit Interactive Dashboard (Full ML Apps ^& Live Tests)
echo   [2] Open Standalone Web Showcase in Browser (Instant, Zero Server)
echo   [3] Run Automated Smoke Tests (Pytest)
echo   [4] Exit
echo.
set /p choice="Enter choice [1-4]: "

if "%choice%"=="1" (
    echo.
    echo Starting Streamlit Dashboard on http://localhost:8501 ...
    python -m streamlit run app.py
    pause
    exit /b
)

if "%choice%"=="2" (
    echo.
    echo Opening Standalone Web Showcase in your default browser...
    start "" "%~dp0ui\index.html"
    exit /b
)

if "%choice%"=="3" (
    echo.
    echo Running Pytest Suite...
    python -m pytest tests/ -v
    pause
    exit /b
)

if "%choice%"=="4" (
    exit /b
)

echo Invalid choice.
pause
