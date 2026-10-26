@echo off
rem Runs the complete LLM performance evaluation and writes the reports. Works from anywhere: double-click it, or pass options:
rem     run_evaluation.bat --suite metrics --models qwen2.5:3b          (only the 12 metrics, one model)
rem     run_evaluation.bat --models qwen2.5:3b qwen2.5:7b --metrics-limit 3 --max-passages 1 --limit 10   (quick check)
rem With no options it evaluates both Qwen models on all three tests - the use-case test (writing questions), the
rem question-answering benchmark and the 12 performance metrics - each model judged by the other.
rem That takes roughly 45-90 minutes on a CPU; the window shows progress the whole time.
cd /d "%~dp0"
title QuizForge - LLM evaluation

if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. Double-click start.bat once to create it, then run this file again.
  pause
  exit /b 1
)

if "%~1"=="" (
  ".venv\Scripts\python.exe" evaluation\evaluate_llm.py --models qwen2.5:3b qwen2.5:7b
) else (
  ".venv\Scripts\python.exe" evaluation\evaluate_llm.py %*
)

echo.
echo Reports are saved in:
echo     %~dp0evaluation\results\comparison_^<time^>.md       (both models side by side)
echo     %~dp0evaluation\results\^<model name^>\report_^<time^>.md / .json
pause
