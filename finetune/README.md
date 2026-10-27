# Fine-tuning Qwen 2.5 (3B and 7B) for QuizForge

Goal: make both models better at **this app's exact task**: writing multiple-choice questions as JSON from source material,
with one correct option, plausible distractors and a word-for-word supporting excerpt.

Method: **QLoRA** (4-bit base model + LoRA adapters, rank 16, all attention and MLP projections) with Unsloth, one epoch,
loss on the assistant reply only. Training runs on a free Kaggle or Google Colab T4 GPU; the finished models run locally in Ollama
like the base models. The app never calls a cloud API.

## The training data

| Source | What it is | How it is made the app's format |
|---|---|---|
| **SciQ** (`allenai/sciq`, 11.7k rows, CC BY-NC 3.0) | Science question + correct answer + 3 distractors + the paragraph that supports it | 1-3 rows per example: their paragraphs become the `<material>`, the options are shuffled into A-D (sometimes 3 options), the supporting sentence becomes `source_excerpt`, varied difficulty/style/explanation settings |
| **Teacher** (`finetune/distill.py`) | `qwen2.5:7b` running the app's real generator on held-out SciQ passages | Kept only if the app's validator accepts the question **and** `qwen2.5:3b` answers it the same way blind |

Every example is a full conversation built with the app's own `SYSTEM_PROMPT` and `build_user_prompt()`, and every
target question is re-checked with the app's `validate_question()` against its material. Anything the app would reject
never becomes training data. Teacher examples are weighted x2 in training (they are closest to real requests); the
validation split never shares an example with training.

## Steps

The notebook runs on **Kaggle** or **Google Colab** and detects which one it is on.

1. **Code.** The notebook clones this GitHub repository. (Without internet access to GitHub: run
   `.venv\Scripts\python.exe -m finetune.make_colab_bundle` and upload `finetune\quizforge_colab.zip`, then set
   `USE_ZIP = True` in the notebook.)
2. **Train** (about 2-3 hours):
   - **Kaggle** (free, ~30 GPU hours a week): [kaggle.com](https://www.kaggle.com/) -> *Create* -> *New Notebook* ->
     *File* -> *Import Notebook* -> upload `finetune/QuizForge_finetune.ipynb` (or paste its GitHub URL). In the right-hand
     panel set *Accelerator* -> **GPU T4 x2** and *Internet* -> **On** (needs a verified phone number), then *Run All*.
     The heavy work happens in `/tmp` (the 7B export needs ~35 GB at peak; `/kaggle/working` holds only 20 GB) and the
     finished files are copied to `/kaggle/working`.
   - **Colab:** open the notebook, *Runtime -> Change runtime type -> T4 GPU*, *Run all*. Finished files are copied to
     `MyDrive/quizforge_finetune`.
3. **Download** `quizforge-3b.Q4_K_M.gguf` (~2 GB), `quizforge-7b.Q4_K_M.gguf` (~4.7 GB) and the two `-report.json`
   files into `finetune\exports\` - on Kaggle from the *Output* panel, **before** you end the session (files of an
   interactive session are deleted when it stops); on Colab from Google Drive.
4. **Import** - double-click `finetune\import_finetuned.bat`. It creates the Ollama models `quizforge-3b` and
   `quizforge-7b`, reusing the base models' chat template (the format the models were trained on).
5. **Measure** - double-click `finetune\compare_finetuned.bat`. It runs all three tests of `run_evaluation.bat` on
   base vs fine-tuned, with the **same judge** for both, and writes `evaluation\results\comparison_<time>.md`.
6. **Use** - Settings -> Model -> *Other* -> custom tag `quizforge-3b` (or `quizforge-7b`).

## What to report

- `quizforge-*-report.json` (from the notebook): validation loss before and after training, and the *format check*: share of
  replies to validation prompts that pass the app's validator **without** a JSON-schema constraint.
- The comparison reports from step 5: acceptance rate, valid-JSON rate, ungrounded-candidate rate, seconds per question,
  answer accuracy, faithfulness, hallucination, instruction-following, safety attack success rate, latency.

## Honest limitations

- SciQ questions are short science recall questions; the teacher data adds longer passages and all difficulties, but
  the model can still drift towards SciQ's style. Compare on your own notes before switching the app's default.
- Fine-tuning can reduce safety behaviour as a side effect: check the *Safety Attack Success Rate* in step 5.
- The teacher filter (two Qwen models agreeing) reduces wrong answer keys but does not prove them right.
- The evaluation passages in `evaluation/passages.json` are not used for training, but SciQ covers similar school
  science topics (e.g. photosynthesis), so a gain on those passages partly reflects topic familiarity.
- Licences: SciQ CC BY-NC 3.0, Qwen2.5-3B-Instruct Qwen Research Licence (both non-commercial), Qwen2.5-7B Apache 2.0.

## Files

| File | Runs on | Role |
|---|---|---|
| `QuizForge_finetune.ipynb` | Kaggle / Colab | The whole pipeline, cell by cell |
| `prepare_data.py` | anywhere | SciQ -> app-format examples, validated with the app's validator |
| `distill.py` | Kaggle / Colab (Ollama) | Teacher examples from the app's real generator, filtered |
| `build_dataset.py` | anywhere | `train.jsonl`, `val.jsonl`, `dataset_info.json` |
| `train.py` | Kaggle / Colab (GPU) | QLoRA training, validation loss, format check, GGUF export (removes the export's intermediate files) |
| `import_to_ollama.py`, `import_finetuned.bat` | laptop | Register the GGUF files with Ollama |
| `compare_finetuned.bat` | laptop | Base vs fine-tuned on all evaluation tests |
| `make_colab_bundle.py` | laptop | Code-only zip for the notebook (option B) |
