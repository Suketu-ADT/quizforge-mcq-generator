# Fine-tuned model files

Put the two fine-tuned model files here, then run `setup_models.bat` from the project folder:

| File | Size | Becomes the Ollama model |
|---|---|---|
| `quizforge-3b.Q4_K_M.gguf` | ~1.9 GB | `quizforge-3b` ("Tuned Fast") |
| `quizforge-7b.Q4_K_M.gguf` | ~4.7 GB | `quizforge-7b` ("Tuned Quality") |

The files are too large for GitHub, so the team shares them separately (for example on Google Drive).
They come from `finetune/QuizForge_finetune.ipynb` (QLoRA fine-tuning of Qwen 2.5 on a Kaggle GPU).
`.gguf` files in this folder are ignored by git.

The fine-tuned models are optional: without them QuizForge uses the base Qwen 2.5 models.
