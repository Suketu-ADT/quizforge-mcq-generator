# LLM Performance Evaluation

QuizForge measures the local model (Qwen 2.5 through Ollama) on **12 metrics**, computed only from stored, sample-level
results. Nothing is estimated, hardcoded or copied from a model card: if the data a metric needs is missing, the metric says
so instead of showing a number.

Everything runs on this computer. No cloud API is called; a judge model on another host is refused unless you opt in.

---

## 1. Quick start

1. Make sure Ollama is running and the models are installed:
   ```
   ollama pull qwen2.5:3b
   ollama pull qwen2.5:7b
   ```
2. Double-click **`run_evaluation.bat`** in the project folder. With no options it evaluates both models on all three tests -
   the use-case test (writing questions), the question-answering benchmark and the **12 performance metrics** - and each model
   is judged by the other one. The window shows progress for every item.
3. When it finishes, open the reports:
   - `evaluation\results\comparison_<time>.md` - both models side by side, one table per test (the 12 metrics are the
     section *12 performance metrics*);
   - `evaluation\results\<model>\report_<time>.md` - everything for one model: status and sample count per metric, run
     totals, failures and the step-by-step calculation of every metric;
   - `evaluation\results\<model>\report_<time>.json` - the same data including every sample (input, output, request
     timings, grading), for re-checking.

The same from PowerShell (in the project folder):

```
.venv\Scripts\python.exe evaluation\evaluate_llm.py --models qwen2.5:3b qwen2.5:7b                 # everything
.venv\Scripts\python.exe evaluation\evaluate_llm.py --suite metrics --models qwen2.5:3b qwen2.5:7b  # only the 12 metrics
.venv\Scripts\python.exe evaluation\evaluate_llm.py --suite metrics --metrics-limit 3 --models qwen2.5:3b qwen2.5:7b  # quick check
```

A full run takes roughly 45-90 minutes on a CPU (the 12-metric test alone: about 15-30 minutes per model). `--metrics-limit N`
uses only the first N items of each dataset. The evaluation is not part of the web app; it only runs from these commands.

---

## 2. Setup and configuration

**Storage.** While a run is in progress its samples are written one by one to a temporary SQLite file (tables
`eval_datasets`, `eval_runs`, `eval_samples`), which is deleted when the run ends - the report files are the output. The
app's own database (`data/quizzes.db`) is never touched. Nothing needs to be migrated.

**Environment variables** (all optional; see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint for the evaluated model. |
| `EVAL_JUDGE_BASE_URL` | `OLLAMA_BASE_URL` | Endpoint for the judge. |
| `EVAL_ALLOW_EXTERNAL_JUDGE` | *(unset)* | Must be `1` before a judge on a non-local host is used. Otherwise the run is refused. |
| `EVAL_PRICING_FILE` | `evaluation/pricing.json` | Pricing registry used for cost. |

**Pricing registry** (`evaluation/pricing.json`). Prices are per 1,000,000 tokens and versioned:

```json
{ "version": "2026-10-26", "currency": "USD",
  "models": { "qwen2.5:3b": { "input_per_million": 0.0, "output_per_million": 0.0, "compute_cost_per_hour": null } } }
```

Local models are priced at 0 per token, with a note that hardware and electricity are not included. Set
`compute_cost_per_hour` to charge measured model time at that hourly rate (the cost is then labelled **estimated**). A model
missing from the registry gets an **unknown** cost; it is never counted as zero.

**Judge.** The judge is chosen per run: `run_evaluation.bat` uses the other model, or pass `--judge <tag>` (`--judge none`
for no judge). Rules:
- No judge → relevance, faithfulness, hallucination and the judged safety cases are reported as not applicable / needing review.
- A judge equal to the evaluated model is not used (the judge-based metrics are then not applicable).
- Judge replies must match a Pydantic schema (for example `score` must be an integer 1–5). Invalid or failed replies are stored as
  `invalid` / `judge_error` and excluded from the metric, with the count shown; they never become a score.
- Rubrics are versioned (`relevance-v1`, `claim-v1`, `instruction-v1`, `answer-v1`, `safety-judge-v1`) and stored with each run.

---

## 3. Datasets

Three built-in datasets are installed automatically. All are **synthetic** and labelled as such in the UI.

| Dataset | Task type | Items | What it tests |
|---|---|---|---|
| QuizForge MCQ answering v1 (`evaluation/dataset.json`) | `mcq_answer` | 40 | Answering labelled MCQs (accuracy, F1). |
| QuizForge generation passages v1 (`evaluation/passages.json`) | `mcq_generation` | 4 | The real use case: the app's own generator writes MCQs from source text. |
| QuizForge safety suite v1 (`evaluation/safety_cases.json`) | `safety` | 11 | Prompt injection, jailbreak, system-prompt disclosure, unsafe compliance, plus benign controls. |

The schemas below are what the dataset validator accepts (JSON array, JSON Lines or CSV); invalid rows are rejected with the reason.

| Task type | Required fields | Optional fields |
|---|---|---|
| `mcq_answer` | `id`, `question`, `answer` | `options` (`{"A": "...", ...}` or CSV columns `option_A`…`option_F`); without options the item is free-text; `match` = `exact` / `normalized`; `topic`, `difficulty`, `tags` |
| `mcq_generation` | `id`, `text` (≥ 80 characters) | `title`, `count` (3), `num_options` (4), `difficulty` (Medium), `include_explanations`, `required_terms`, `forbidden_terms`, `max_question_words`, `min_questions` |
| `safety` | `id`, `category`, `kind` (`attack` / `benign`), `text`, `expected_behavior`, `grader` | grader types: `canary` (+`canary`), `disclosure` (+`markers`), `forbidden_terms` (+`terms`, `min_hits`), `judge` (+`policy`), `should_comply` |

---

## 4. The 12 metrics

Common rules:
- A metric is computed from the stored samples of one run. Cancelled samples are excluded from the quality, safety and cost
  metrics (their timing is listed separately under latency).
- Each metric stores value, unit, **status**, formula, numerator, denominator, sample count, evaluator, version and limitations.
- **Status**: `available` (computed), `insufficient_data` (not enough observations, e.g. p95 with fewer than 20 values),
  `not_applicable` (the run had nothing this metric measures, e.g. no judge or no safety cases), `evaluation_failed`
  (the evaluator itself failed, e.g. every judge reply was invalid). Only `available` has a value; the others show "—".
- Proportions are shown with a 95 % **Wilson** confidence interval.
- **Percentiles** use linear interpolation between closest ranks (Hyndman–Fan type 7, the same as NumPy's default).
  Minimum observations: p50 ≥ 1, p90 ≥ 10, p95 ≥ 20, p99 ≥ 100. Below that the percentile is `insufficient_data`.

| # | Metric | Formula | Data source | Evaluator |
|---|---|---|---|---|
| 1 | **Answer Accuracy** (%) | correct evaluable answers / evaluable answers × 100 | `mcq_answer` items with a reference; generated questions only after a human confirms or rejects the key | Exact letter match (MCQ), exact or normalised text match (free text), human label (generated keys) |
| 2 | **F1-score** (macro, 0–1) | per option class: P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R); macro = mean over classes | Multiple-choice answering items | Confusion matrix; unreadable replies form a `?` column and count as wrong; 0/0 is 0 (zero-division = 0). Micro- and weighted-F1 shown in details |
| 3 | **Answer Relevance** (1–5) | Σ judge scores / number of valid scores; pass rate = scores ≥ threshold (default 4) / valid scores | Each generated question vs. the request (passage + instructions) | Judge, rubric `relevance-v1`, schema-validated integer 1–5 |
| 4 | **Faithfulness** (%) | supported claims / (supported + contradicted + unsupported) × 100 | Claims from generated questions: the answer key ("the correct answer to Q is X") and each explanation sentence, checked against the source passage | Claim decomposition + judge, rubric `claim-v1`; the judge's evidence quote must appear in the passage. `unverifiable` claims are shown but excluded |
| 5 | **Hallucination Rate** (%) | (unsupported + contradicted) / (supported + contradicted + unsupported) × 100 | Same claims as faithfulness | Same verifier. It is an evidence-based indicator against the given source, not proof that a statement is false in the world |
| 6 | **Instruction-Following** (%) | satisfied requirements / evaluable requirements × 100 | Generation tasks: valid JSON schema, exact question count, option count, exactly one correct answer, no "all/none of the above", options not inside the stem, explanation policy, verbatim source excerpt, difficulty label, dataset constraints (required/forbidden terms, max words) | Deterministic validators (the judge is used only for subjective instructions when configured). Per-instruction pass rates in details |
| 7 | **Safety Attack Success Rate** (%) | successful attacks / gradable adversarial cases × 100 | Safety suite `attack` cases | Deterministic graders (canary token, disclosure markers, forbidden terms, refusal detection); policy cases use the judge or wait for human review. Benign controls give the separate **over-refusal** rate |
| 8 | **Task Success Rate** (%) | successful tasks / tasks with a decided outcome × 100 | All tasks | `mcq_answer`: correct answer. `mcq_generation`: no technical error, at least `min_questions` accepted questions, required constraints met. `safety`: resisted / safe refusal (attacks) or complied (benign) |
| 9 | **TTFT p95** (ms) | 95th percentile of (first generated token − request start) | Successful streamed candidate requests | Monotonic clock in the request wrapper. Load time and prompt-processing time are reported alongside. Total latency is never substituted |
| 10 | **E2E Latency p95** (ms) | 95th percentile of (task end − task start) over completed tasks | Every task, including repairs and retries | Monotonic clock. Judge time is excluded. Failed and cancelled task times are shown separately |
| 11 | **Cost per Successful Task** (USD) | total cost of all tasks (successful and failed) / successful tasks | Measured prompt and output token counts from Ollama × registry price | `insufficient_data` when any task lacks usage or a price, or nothing succeeded |
| 12 | **Error Rate** (%) | failed candidate requests / attempted candidate requests × 100 | Request traces | Timeouts, connection errors, missing model and provider errors. Wrong answers, refusals and invalid JSON are **not** technical errors (they belong to the quality metrics); invalid-output replies, repair calls and judge failures are counted separately |

The per-model report shows, for every metric, the formula, the calculation with the run's real numbers substituted, the
breakdown (by task, dataset, category or instruction) and its limitations.

---

## 5. How a run works

```
RunConfig ──prepare()──> eval_runs (queued; stores datasets + content hashes, prompt/rubric/evaluator/metric versions,
                                     pricing version, decoding settings, judge)
          ──execute()──> for every item:
                           InstrumentedClient (records each request: TTFT, latency, tokens, status, error category)
                           ├─ mcq_answer      → answer with log-probs (temperature 0) → grade
                           ├─ mcq_generation  → the app's real QuestionGenerator → validators → judge (relevance, claims)
                           └─ safety          → generator on adversarial material → safety grader (→ judge for policy cases)
                           save the sample immediately (eval_samples)
                         compute_all(samples) → 12 MetricResults stored on the run
```

- Samples are saved one by one, so a failure in one item never loses the others; a technical failure is recorded on the
  sample (and counted by the error rate) instead of stopping the run.
- Safety cases that the automatic graders cannot decide are marked *needs review* and excluded from the attack success rate;
  the report lists them under failures.

## 6. Code layout

| File | Role |
|---|---|
| `run_evaluation.bat` | Entry point: runs `evaluation/evaluate_llm.py` for both models |
| `evaluation/evaluate_llm.py` | Runs the three tests, prints the results, writes the reports and the comparison |
| `evaluation/metrics_suite.py` | Test 3: runs the engine and formats the 12 metrics for the reports |
| `services/evaluation/engine.py` | Runs every dataset item through the model (and judge), records request telemetry |
| `services/evaluation/metrics.py` | The 12 metric calculations, from stored samples only |
| `services/evaluation/evaluators.py`, `judge.py` | Deterministic graders and the schema-validated judge |
| `services/evaluation/telemetry.py`, `pricing.py`, `stats.py` | Request timing, cost registry, percentiles and confidence intervals |
| `services/evaluation/service.py`, `database/evaluation_repository.py` | Run bookkeeping and storage of runs and samples |
| `evaluation/dataset.json`, `passages.json`, `safety_cases.json`, `pricing.json` | Built-in datasets and the price registry |

## 7. Reproducing and comparing runs

Each per-model `.json` report stores everything needed to reproduce the run: dataset names, versions and content hashes,
prompt versions (hashes of the prompt text), rubric and evaluator versions, metric version, pricing version, temperature and
context size, the judge and the per-dataset limit. Compare reports only when these match apart from the model.

Answering is done at temperature 0, so it is close to deterministic. Generation uses the app's normal temperature (0.4 by
default), so generation metrics vary between runs; run several times or use more passages before drawing conclusions.
`--predictions <report.json>` re-renders a saved report without Ollama.

## 8. Interpreting metrics that are not available

| You see | Meaning | What to do |
|---|---|---|
| Not applicable | The run contained nothing this metric measures (e.g. no judge, no safety dataset) | Configure a judge / add the dataset |
| Insufficient data | Too few observations (p95 needs 20), or cost/usage missing, or no successful task | Use more items (no `--metrics-limit`); add the model to the pricing registry |
| Evaluation failed | The evaluator could not produce valid results (e.g. every judge reply was invalid) | Check the judge model; see the sample details |
| Needs review | A safety case the automatic graders could not decide | Read it in the .json report and judge it by hand |

## 9. Limitations

- The built-in datasets are small and synthetic. Confidence intervals are wide; treat results as indicative, not as a benchmark.
- An LLM judge can be wrong. Its verdicts are schema-checked and evidence-anchored, but read the failures in the report
  before relying on a number. Two Qwen 2.5 models are related, so judging one with the other is not fully independent.
- Hallucination is measured against the supplied source text only.
- Safety coverage is limited to the attack types in the suite; a 0 % attack success rate does not prove the model is safe.
- Local models cost nothing per token; the real cost is hardware time, which is only included if `compute_cost_per_hour` is set.
- Speed depends on the computer, on other running programs and on whether the model was already loaded (load time is shown
  separately in TTFT details).

## 10. Tests

```
.venv\Scripts\python.exe -m pytest -q -m "not integration"
```

- `tests/test_eval_metrics.py` — every metric: normal values, edge cases, missing data, zero denominators, statuses.
- `tests/test_eval_components.py` — validators, claim/safety graders, judge schema validation and configuration rules, pricing,
  request telemetry, dataset parsing and validation.
- `tests/test_eval_service.py` — integration: starting runs, persistence, retrieval, paging and filters, comparison, trends,
  exports, timeouts, invalid judge output, cancellation, interrupted runs, human review, dataset import.
- `tests/test_metrics_suite.py` — the 12-metric test as run by `run_evaluation.bat`: progress, `n/a` handling, per-model
  report, comparison report layout and the files written.

All of these use fake model and judge clients (`tests/eval_fakes.py`), so they need neither Ollama nor network access.
