"""Test 3 - the 12 performance metrics, computed by the evaluation engine (services/evaluation).

The engine runs three built-in datasets through the model the way the app uses it:
    mcq_answer      40 labelled questions, answered at temperature 0           -> accuracy, F1
    mcq_generation  4 passages through the app's real question generator       -> relevance, faithfulness, hallucination,
                                                                                    instruction-following
    safety          11 adversarial / benign cases fed to the generator         -> attack success rate, over-refusal
and records every request (TTFT, latency, tokens, errors), so it can also report task success, TTFT p95, end-to-end latency
p95, cost per successful task and error rate. Relevance, faithfulness and hallucination need a judge model (the other model);
without one they are reported as "not applicable", never estimated.

Results live in a throw-away SQLite file while the run is in progress; the report files are the output.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from database.db import Database
from database.evaluation_repository import failure_tags
from evaluation.common import Metric
from services.evaluation.service import EvaluationService

STATUS_TEXT = {"available": "measured", "insufficient_data": "insufficient data", "not_applicable": "not applicable",
               "evaluation_failed": "evaluation failed"}
UNIT_SUFFIX = {"%": "%", "ms": "ms", "USD": "USD", "score 1-5": "1-5", "score 0-1": "0-1"}


class _Inline:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


def run_metrics_suite(model: str, judge: str | None, base_url: str, timeout_s: int, limit: int | None = None,
                      temperature: float = 0.4, log=print, client_factory=None, judge_client_factory=None) -> dict:
    """Run the engine on the built-in datasets and return the stored run (metrics, totals, config) plus every sample."""
    folder = Path(tempfile.mkdtemp(prefix="quizforge_eval_"))
    try:
        svc = EvaluationService(Database(folder / "run.db"), client_factory=client_factory, judge_client_factory=judge_client_factory,
                                executor=_Inline())
        datasets = [d for d in svc.list_datasets() if d["source"] == "builtin"]
        total = sum(min(d["item_count"], limit) if limit else d["item_count"] for d in datasets)
        log(f"\n=== 12 performance metrics: {model} on {total} items"
            + (f" (judge: {judge})" if judge else " (no judge: relevance / faithfulness / hallucination will be N/A)") + " ===")

        def progress(sample: dict, position: int) -> None:
            outcome = (sample.get("outcomes") or {}).get("task") or {}
            state = sample["status"] if sample["status"] != "completed" else ("ok" if outcome.get("success") else "fail")
            secs = ((sample.get("timing") or {}).get("e2e_ms") or 0) / 1000
            log(f"  [{position:>3}/{total}] {sample['task_type']:<15} {sample.get('item_id', ''):<10} {state:<9} {secs:6.1f}s")

        svc.engine.on_sample = progress
        out = svc.create_run({"model": model, "dataset_ids": [d["id"] for d in datasets], "judge_model": judge,
                              "limit_per_dataset": limit, "base_url": base_url, "timeout_s": timeout_s, "temperature": temperature},
                             background=False)
        run = svc.get_run(out["run_id"])
        samples = svc.repo.all_samples(run["id"])
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    keep = ("id", "model", "status", "started_at", "finished_at", "duration_s", "sample_count", "completed_count", "config", "totals", "error")
    return {"model": model, "judge": judge, "run": {k: run.get(k) for k in keep}, "metrics": run["metrics"], "samples": samples}


def to_metrics(result: dict) -> list[Metric]:
    """The engine's MetricResults in the report format used by the other two tests."""
    out = []
    for m in result["metrics"]:
        unit = UNIT_SUFFIX.get(m.get("unit", ""), m.get("unit", ""))
        name = f"{m['name']} ({unit})" if unit else m["name"]
        status = STATUS_TEXT.get(m["status"], m["status"])
        note = f"{status}, n = {m.get('sample_count', 0)}"
        if m["value"] is None and m.get("limitations"):
            note += f" - {m['limitations'][0]}"
        working = list(m.get("working") or []) + [f"! {x}" for x in (m.get("limitations") or [])[(0 if m["value"] is not None else 1):]]
        out.append(Metric(name=name, value=m["value"], better="higher" if m.get("higher_is_better", True) else "lower",
                          formula=m.get("formula", ""), working=working, note=note))
    return out


def status_table(result: dict) -> list[str]:
    rows = ["| Metric | Value | n | Status | Better |", "|---|---:|---:|---|---|"]
    for metric, raw in zip(to_metrics(result), result["metrics"]):
        value = "n/a" if metric.value is None else f"{metric.value:.4f}"
        rows.append(f"| {metric.name} | {value} | {raw.get('sample_count', 0)} | {STATUS_TEXT.get(raw['status'], raw['status'])} "
                    f"| {metric.better} |")
    return rows


def failure_lines(result: dict, limit: int = 25) -> list[str]:
    lines = []
    for s in result["samples"]:
        tags = failure_tags(s)
        if not tags:
            continue
        task = (s.get("outcomes") or {}).get("task") or {}
        reason = task.get("failure_reason") or (s.get("error") or {}).get("message", "")
        lines.append(f"{s['task_type']:<15} {s.get('item_id', ''):<10} {', '.join(tags):<45} {reason[:110]}")
    if not lines:
        return ["no failures"]
    return lines[:limit] + ([f"... and {len(lines) - limit} more (see the .json report)"] if len(lines) > limit else [])


def totals_lines(result: dict) -> list[str]:
    t = result["run"].get("totals") or {}
    run = result["run"]
    cost = "unknown" if t.get("total_cost") is None else f"{t['total_cost']:.6f} {t.get('currency', 'USD')}"
    return [f"samples: {t.get('samples', 0)} {t.get('by_status', {})}, run time {run.get('duration_s') or 0:.0f} s",
            f"model requests: {t.get('candidate_requests', 0)} (input tokens {t.get('input_tokens', 0)}, output tokens "
            f"{t.get('output_tokens', 0)}); judge requests: {t.get('judge_requests', 0)} ({t.get('judge_tokens', 0)} tokens)",
            f"total cost: {cost} (pricing version {run['config'].get('pricing_version')})"]


def metrics_markdown(result: dict, metrics: list[Metric]) -> list[str]:
    from evaluation.common import markdown_calculations

    cfg = result["run"]["config"]
    data = ", ".join(f"{d['name']} ({d['task_type']})" for d in cfg.get("datasets", []))
    out = ["## 12 performance metrics", "",
           f"Model `{result['model']}`" + (f", judge `{result['judge']}`" if result.get("judge") else ", no judge model")
           + f". Datasets: {data} - synthetic test data shipped with the project.", "",
           "A value is `n/a` when the data it needs was not available; the status column says why (it is never replaced by 0).", ""]
    out += status_table(result)
    out += ["", "**Run totals**", ""] + [f"- {line}" for line in totals_lines(result)]
    out += ["", "**Failures** (task, item, categories, reason)", "", "```"] + failure_lines(result) + ["```", ""]
    out += [f"*Versions: metrics {cfg.get('metrics_version')}, rubrics {cfg.get('rubric_versions')}, "
            f"percentiles = linear interpolation (Hyndman-Fan type 7), p95 needs at least 20 values.*", ""]
    return out + markdown_calculations(metrics) + [""]


def print_metrics_report(result: dict, metrics: list[Metric]) -> None:
    from evaluation.common import print_calculations

    bar = "=" * 100
    print(f"\n{bar}\n12 PERFORMANCE METRICS   model: {result['model']}" + (f"   judge: {result['judge']}" if result.get("judge") else "")
          + f"\n{bar}")
    print("\n".join("  " + line for line in status_table(result)))
    print("\nRUN TOTALS")
    print("\n".join("  " + line for line in totals_lines(result)))
    print("\nFAILURES")
    print("\n".join("  " + line for line in failure_lines(result)))
    print(f"\n{bar}\nTHE CALCULATIONS, STEP BY STEP\n{bar}")
    print_calculations(metrics)
