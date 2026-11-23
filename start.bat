@echo off
rem One-click launcher for QuizForge (Windows). Double-click this file.
rem First run: creates the Python environment and installs the packages. Then starts Ollama if needed and opens the app.
cd /d "%~dp0"
title QuizForge

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the Python environment ^(this takes a few minutes^)...
  py -3.12 -m venv .venv 2>nul
  if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv 2>nul
  if not exist ".venv\Scripts\python.exe" python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Could not create the environment. Install Python 3.11 or newer from python.org ^(tick "Add python.exe to PATH"^) and try again.
    pause
    exit /b 1
  )
)

rem install or update the packages when something is missing (also after an update from GitHub)
".venv\Scripts\python.exe" -c "import flask, waitress, pydantic, pandas, pymupdf, docx, requests, dotenv" 2>nul
if errorlevel 1 (
  echo Installing the Python packages...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo Installing the packages failed. Check your internet connection and run this file again.
    pause
    exit /b 1
  )
)

if not exist ".env" copy ".env.example" ".env" >nul

curl.exe -s -m 3 http://127.0.0.1:11434/api/version >nul 2>nul
if not errorlevel 1 goto ollama_ok
set "OLLAMA_EXE="
where ollama >nul 2>nul && set "OLLAMA_EXE=ollama"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA_EXE (
  echo Ollama was not found. The app opens but shows "Ollama not running" until you install it from https://ollama.com/download
  goto ollama_ok
)
echo Starting Ollama...
start "" /min "%OLLAMA_EXE%" serve
timeout /t 4 /nobreak >nul
:ollama_ok

echo.
echo QuizForge is starting at http://127.0.0.1:5000 - your browser opens by itself.
echo Keep this window open while you use the app. Close it to stop QuizForge.
echo.
".venv\Scripts\python.exe" app.py
pause
