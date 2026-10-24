@echo off
rem One-click launcher for QuizForge (Windows). Double-click this file.
rem It creates the virtual environment on first run, starts Ollama if needed, then opens the app.
cd /d "%~dp0"
title QuizForge

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the Python environment ^(this takes a few minutes^)...
  py -3.12 -m venv .venv 2>nul
  if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv 2>nul
  if not exist ".venv\Scripts\python.exe" python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Could not create the environment. Please install Python 3.11 or newer from python.org and try again.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Installing the dependencies failed. Check your internet connection and run this file again.
    pause
    exit /b 1
  )
)

if not exist ".env" copy ".env.example" ".env" >nul

powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:11434/api/version | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  where ollama >nul 2>nul
  if errorlevel 1 (
    echo Ollama was not found. The app will start but show "Disconnected" until Ollama is installed and running.
  ) else (
    echo Starting Ollama...
    start "" /min ollama serve
    timeout /t 4 /nobreak >nul
  )
)

echo Opening QuizForge at http://localhost:8501  ^(close this window to stop the app^)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
