@echo off
rem Installs the AI models QuizForge uses. Run it once per computer, after installing Ollama.
rem   1. the base models qwen2.5:3b and qwen2.5:7b from the Ollama library (it asks before each download)
rem   2. the fine-tuned models quizforge-3b and quizforge-7b from the .gguf files in model_files\ (no download)
cd /d "%~dp0"
title QuizForge - model setup

set "OLLAMA_EXE="
where ollama >nul 2>nul && set "OLLAMA_EXE=ollama"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA_EXE (
  echo Ollama is not installed. Download it from https://ollama.com/download, install it, then run this file again.
  pause
  exit /b 1
)

curl.exe -s -m 3 http://127.0.0.1:11434/api/version >nul 2>nul
if not errorlevel 1 goto ollama_ok
echo Starting Ollama...
start "" /min "%OLLAMA_EXE%" serve
timeout /t 5 /nobreak >nul
:ollama_ok

if exist ".venv\Scripts\python.exe" goto python_ok
echo Creating the Python environment first ^(a few minutes^)...
py -3.12 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo Could not create the environment. Install Python 3.11 or newer from python.org and try again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
:python_ok

echo.
echo === Fast models (about 2 GB each) ===
call :setup 3b "about 2 GB"
echo.
echo === Quality models (about 4.7 GB each, need about 8 GB of free RAM) ===
call :setup 7b "about 4.7 GB"

echo.
echo Installed models:
"%OLLAMA_EXE%" list
echo.
echo Done. Start QuizForge with start.bat. The app picks the fine-tuned model automatically when it is installed.
pause
exit /b 0

rem ------------------------------------------------------------------ :setup <3b|7b> <size>
:setup
"%OLLAMA_EXE%" show qwen2.5:%1 >nul 2>nul
if not errorlevel 1 goto base_ok
choice /C YN /M "Download the base model qwen2.5:%1 (%~2) now"
if errorlevel 2 (
  echo Skipped qwen2.5:%1 - the fine-tuned %1 model needs it too, so it is skipped as well.
  exit /b 0
)
"%OLLAMA_EXE%" pull qwen2.5:%1
if errorlevel 1 (
  echo The download failed. Check your internet connection and run this file again.
  exit /b 0
)
:base_ok
echo qwen2.5:%1 is installed.
if not exist "model_files\quizforge-%1.Q4_K_M.gguf" (
  echo model_files\quizforge-%1.Q4_K_M.gguf was not found - put the fine-tuned file there to install quizforge-%1.
  exit /b 0
)
"%OLLAMA_EXE%" show quizforge-%1 >nul 2>nul
if not errorlevel 1 (
  echo quizforge-%1 is already installed. To replace it, delete it first: ollama rm quizforge-%1
  exit /b 0
)
echo Importing quizforge-%1 from model_files ^(a minute or two^)...
".venv\Scripts\python.exe" -m finetune.import_to_ollama --gguf "model_files\quizforge-%1.Q4_K_M.gguf" --name quizforge-%1 --base qwen2.5:%1
exit /b 0
