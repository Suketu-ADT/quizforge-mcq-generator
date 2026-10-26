# LLM performance evaluation

> **Three tests.** `run_evaluation.bat` runs `evaluate_llm.py`, which now has a third test, `--suite metrics`: the 12
> performance metrics (accuracy, F1, relevance, faithfulness, hallucination, instruction-following, safety, task success,
> TTFT, latency, cost, error rate). They appear as the *12 performance metrics* section of every report - see
> [`../docs/EVALUATION.md`](../docs/EVALUATION.md).

Measures how well the local model (Qwen 2.5 via Ollama) does **the job this project exists for** - turning source material into
multiple-choice questions - and prints every calculation step by step so you can explain it.

## How to run it (Windows)

The most common mistake is running the command from the wrong folder (for example `GEN AI` instead of the project folder).
Use **either** of these:

**A. Double-click `run_evaluation.bat`** in the project folder `mcq-quiz-generator`. It evaluates both models on all three tests and keeps
the window open at the end. (First run `start.bat` once if the `.venv` folder does not exist.)

**B. From PowerShell** - go to the project folder first:
```powershell
cd "C:\Users\Arpit singh\OneDrive\Desktop\GEN AI\mcq-quiz-generator"
.venv\Scripts\python.exe evaluation\evaluate_llm.py --models qwen2.5:3b qwen2.5:7b
```
(or use the full path to the script from anywhere:
`.\mcq-quiz-generator\.venv\Scripts\python.exe .\mcq-quiz-generator\evaluation\evaluate_llm.py ...`)

Ollama must be running and the models installed (`ollama pull qwen2.5:3b`, `ollama pull qwen2.5:7b`).
A full run takes roughly 45-90 minutes on a CPU (writing real questions and judging them is the slow part).

| What you want | Command (after `cd` to the project folder) |
|---|---|
| Everything, both models | `.venv\Scripts\python.exe evaluation\evaluate_llm.py --models qwen2.5:3b qwen2.5:7b` |
| One model only | `... evaluate_llm.py --models qwen2.5:3b` |
| Only the use-case test | `... evaluate_llm.py --suite generation` |
| Only the quick benchmark | `... evaluate_llm.py --suite answering` |
| Only the 12 performance metrics | `... evaluate_llm.py --suite metrics` (add `--metrics-limit 3` for a quick check) |
| Fast smoke test | `... evaluate_llm.py --suite generation --max-passages 1 --models qwen2.5:3b` |
| More questions per passage | `... evaluate_llm.py --gen-count 5` |
| Your own passages / questions | `--passages my_passages.json` / `--dataset my_questions.json` |
| Show the maths without Ollama | `... evaluate_llm.py --demo` (**simulated** - never quote as a result) |

## Where the reports are saved
Every run writes **one folder per model** under `evaluation/results/`:

```
evaluation/results/
|-- qwen2.5_3b/
|   |-- report_20261024_133000.md      <- readable report with the calculations (open this one)
|   |-- report_20261024_133000.json    <- all raw predictions, questions and metrics
|   `-- labels_20261024_133000.csv     <- for human review of the answer keys (see below)
|-- qwen2.5_7b/  ...
`-- comparison_20261024_133000.md      <- side-by-side table when two or more models were run
```
The path of every file is printed at the end of the run.

## The two tests

### 1. Use-case test: generating questions from source material (`--suite generation`)
This is the project's real task. The app's own generator (prompts, JSON schema, validator, repair loop) is asked for `--gen-count`
questions (default 3) from each of the 4 passages in `passages.json`. Then every accepted answer key is re-answered **blind**
(without seeing the key) by the same model and by a second **judge** model (with two models, each judges the other's questions).

| Metric | Calculation | What it tells you | Better |
|---|---|---|---|
| Delivery rate | accepted / requested | Did the app deliver the number of questions asked? | Higher |
| Acceptance rate | accepted / (accepted + rejected candidates) | How many written questions survived validation | Higher |
| Valid-JSON rate | usable replies / all replies | Can the model follow the output format? | Higher |
| Model calls per accepted question | replies / accepted | Repair overhead (ideal is about 0.33: 3 questions per call) | Lower |
| Ungrounded-candidate rate | candidates whose quoted source sentence is not in the text / candidates | Hallucinated support | Lower |
| Duplicate-candidate rate | duplicates / candidates | Repetition | Lower |
| Format-defect rate | structurally broken candidates / candidates | Wrong option count, invalid answer, options inside the question... | Lower |
| Seconds per accepted question | total time / accepted | Real cost of one question | Lower |
| Generation speed | eval_count / (eval_duration / 1e9) | Tokens per second from Ollama's counters | Higher |
| Blind-answer agreement | questions where the checker's answer equals the key / checked | A hint about key correctness - **not proof** | Higher |

**Key accuracy needs a human.** After the run, open `labels_<time>.csv` (Excel works), write `yes` or `no` in the
`human_key_correct` column for each question (is the marked answer really correct?), save it and run:
```powershell
.venv\Scripts\python.exe evaluation\evaluate_llm.py --score-labels evaluation\results\qwen2.5_3b\labels_<time>.csv
```
You then get **Key accuracy** (right keys / reviewed) and, treating "the check disagreed with the key" as a prediction that the key is wrong,
**Precision, Recall, F1, Accuracy, MCC and Cohen's kappa** of that check - which shows whether the automatic double-check is useful.

### 2. Question-answering benchmark (`--suite answering`)
## How the test works
1. `dataset.json` holds 40 questions with a single correct option (20 passage-based from the Unit II notes, 20 closed-book; the correct letter is shuffled with a fixed seed so it is not always "A").
2. For each question the model gets the passage (if any), the question and the options, at **temperature 0**, and must answer with one letter.
3. Ollama returns **log-probabilities** for the first token. For each option letter we take its probability and renormalise over the letters, giving a real probability distribution such as `A 0.01, B 0.97, C 0.01, D 0.01`.
4. Predictions and probabilities are compared with the known answers to compute the metrics below.

> A reply with no readable letter counts as **wrong** and uses equal probabilities (no information). It is shown in a `?` column of the confusion matrix.

## The metrics, how they are calculated, and how to explain them
Let N = number of questions, classes = option letters A-D.

| Metric | Calculation | One-line explanation | Better |
|---|---|---|---|
| **Accuracy** | correct / N | How often the model picked the right option. | Higher |
| **Exact Match (EM)** | answers identical to the expected answer / N | Strict string match. Here answers are single letters, so EM = accuracy. | Higher |
| **Precision** | TP / (TP + FP), per letter, then averaged | When the model says "B", how often is it really B? | Higher |
| **Recall** | TP / (TP + FN), per letter, then averaged | Of all questions whose answer is B, how many did it find? | Higher |
| **F1-score** | 2PR / (P + R) | Harmonic mean of precision and recall; low if either is low. | Higher |
| **Macro-F1** | mean of the per-letter F1 | Every letter counts equally. | Higher |
| **Micro-F1** | F1 from TP, FP, FN pooled over all letters | Equals accuracy when every reply is parsed. | Higher |
| **Weighted-F1** | per-letter F1 weighted by how many questions have that answer | Macro-F1 that respects class sizes. | Higher |
| **Top-k accuracy** | correct option among the k most probable / N | Was the right answer at least in the model's top k? (top-2 shown) | Higher |
| **MCC** | (c·s − Σ p_k t_k) / √((s² − Σ p_k²)(s² − Σ t_k²)) | Balanced agreement: 1 perfect, 0 chance, −1 always wrong. | Higher |
| **Cohen's kappa** | (p_o − p_e) / (1 − p_e) | Agreement beyond what random guessing with the same answer mix would give. | Higher |
| **Log loss** | −(1/N) Σ ln p(correct option) | Punishes confident wrong answers heavily. Equal guessing gives ln 4 = 1.386. | Lower |
| **Brier score** | (1/N) Σ Σ_k (p_k − y_k)² | Squared error of the probabilities. Equal guessing gives 0.75. | Lower |
| **ECE** | Σ over 10 confidence bins (n_bin/N) · \|accuracy_bin − confidence_bin\| | Does "90% confident" really mean right 90% of the time? | Lower |

(TP/FP/FN are counted per option letter; c = correct answers, s = N, p_k = how often letter k was predicted, t_k = how often it is correct, p_o = observed agreement, p_e = agreement expected by chance = Σ p_k t_k / N².)

### A tiny worked example (3 questions, options A/B/C)
Correct answers: `A, B, C`; the model answers `A, C, C`.

| Letter | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| A | 1 | 0 | 0 | 1/1 = 1.00 | 1/1 = 1.00 | 1.00 |
| B | 0 | 0 | 1 | 0/0 → 0 | 0/1 = 0.00 | 0.00 |
| C | 1 | 1 | 0 | 1/2 = 0.50 | 1/1 = 1.00 | 0.667 |

Accuracy = 2/3 = 0.667. Macro-F1 = (1.00 + 0.00 + 0.667)/3 = 0.556. Micro-F1: TP = 2, FP = 1, FN = 1 → P = R = 2/3 → 0.667 (same as accuracy).
Weighted-F1 = (1·1.00 + 1·0.00 + 1·0.667)/3 = 0.556 (every letter has one question).

### Speed
- **Mean time per question** = sum of per-question latencies / N (model load time excluded).
- **Prompt processing speed** = prompt tokens / prompt-evaluation seconds.
- **Generation speed** = `eval_count / (eval_duration / 1e9)` on a ~128-token generation (what "tokens per second" means).

## How to present the results honestly
- Quote the **sample size**. With 40 questions the accuracy has a wide confidence interval (the report prints a 95% Wilson interval); a 5-point difference between two models may be noise.
- A model can be accurate but badly calibrated (high ECE, high log loss) - which is why several metrics are reported.
- This measures the model as a **question answerer**. How well it **writes** questions is measured by the generator's own statistics (acceptance rate, grounding rate) and, for answer-key correctness, by human review.
- `--demo` uses a simulated model. Never quote demo numbers as results.
