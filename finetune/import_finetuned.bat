@echo off
rem Registers the fine-tuned models downloaded from the Colab notebook with Ollama.
rem Put quizforge-3b.Q4_K_M.gguf and/or quizforge-7b.Q4_K_M.gguf in finetune\exports\ first.
cd /d "%~dp0.."
title QuizForge - import fine-tuned models

if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. Double-click start.bat once to create it, then run this file again.
  pause
  exit /b 1
)

set FOUND=0
if exist "finetune\exports\quizforge-3b.Q4_K_M.gguf" (
  set FOUND=1
  ".venv\Scripts\python.exe" -m finetune.import_to_ollama --gguf finetune\exports\quizforge-3b.Q4_K_M.gguf --name quizforge-3b --base qwen2.5:3b
)
if exist "finetune\exports\quizforge-7b.Q4_K_M.gguf" (
  set FOUND=1
  ".venv\Scripts\python.exe" -m finetune.import_to_ollama --gguf finetune\exports\quizforge-7b.Q4_K_M.gguf --name quizforge-7b --base qwen2.5:7b
)
if "%FOUND%"=="0" echo No .gguf files found in finetune\exports\ - download them from the Colab notebook first.

echo.
echo Next: finetune\compare_finetuned.bat compares each fine-tuned model with its base model.
pause
