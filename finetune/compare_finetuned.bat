@echo off
rem Before vs after: each fine-tuned model against its base model, on all three tests, with the SAME judge for both
rem (so the comparison is fair). Writes evaluation\results\comparison_<time>.md for each pair.
rem Extra options are passed on, e.g.:  compare_finetuned.bat --metrics-limit 3 --max-passages 1 --limit 10   (quick check)
cd /d "%~dp0.."
title QuizForge - base vs fine-tuned

if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. Double-click start.bat once to create it, then run this file again.
  pause
  exit /b 1
)

echo === Fast model: qwen2.5:3b vs quizforge-3b (judge: qwen2.5:7b) ===
".venv\Scripts\python.exe" evaluation\evaluate_llm.py --models qwen2.5:3b quizforge-3b --judge qwen2.5:7b %*

echo.
echo === Quality model: qwen2.5:7b vs quizforge-7b (judge: qwen2.5:3b) ===
".venv\Scripts\python.exe" evaluation\evaluate_llm.py --models qwen2.5:7b quizforge-7b --judge qwen2.5:3b %*

echo.
echo Comparisons are saved in %~dp0..\evaluation\results\comparison_^<time^>.md
pause
