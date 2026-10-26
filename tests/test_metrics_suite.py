"""The 12-metric suite of evaluate_llm.py / run_evaluation.bat: run, report formatting and the saved comparison (no Ollama)."""
import json
from pathlib import Path

import pytest

from evaluation import evaluate_llm
from evaluation import metrics_suite as ms
from tests.eval_fakes import FakeJudge, FakeModel


def run_fake(model="fake:3b", judge="fake:judge", limit=2, **model_kw):
    fake = FakeModel(model=model, **model_kw)
    return ms.run_metrics_suite(model, judge, "http://localhost:11434", 60, limit=limit, log=lambda *_: None,
                                client_factory=lambda *a: fake, judge_client_factory=(lambda *a: FakeJudge()) if judge else None)


def test_suite_runs_all_three_datasets_and_returns_twelve_metrics():
    lines = []
    fake = FakeModel()
    result = ms.run_metrics_suite("fake:3b", "fake:judge", "http://localhost:11434", 60, limit=2, log=lines.append,
                                  client_factory=lambda *a: fake, judge_client_factory=lambda *a: FakeJudge())
    assert result["run"]["status"] == "completed" and len(result["metrics"]) == 12 and len(result["samples"]) == 6
    assert {s["task_type"] for s in result["samples"]} == {"mcq_answer", "mcq_generation", "safety"}
    assert sum("[" in line and "/6]" in line for line in lines) == 6  # one progress line per item
    assert result["run"]["config"]["judge_model"] == "fake:judge"


def test_without_a_judge_the_judge_metrics_are_na_not_zero():
    result = run_fake(judge=None)
    metrics = {m.name.split(" (")[0]: m for m in ms.to_metrics(result)}
    for name in ("Answer Relevance", "Faithfulness", "Hallucination Rate"):
        assert metrics[name].value is None and "not applicable" in metrics[name].note
    assert metrics["Error Rate"].value == 0.0


def test_metric_names_carry_units_and_direction():
    names = {m.name: m.better for m in ms.to_metrics(run_fake())}
    assert names["Answer Accuracy (%)"] == "higher" and names["TTFT p95 (ms)"] == "lower"
    assert names["Answer Relevance (1-5)"] == "higher" and names["Hallucination Rate (%)"] == "lower"


def test_per_model_markdown_has_table_totals_failures_and_calculations():
    result = run_fake()
    md = "\n".join(ms.metrics_markdown(result, ms.to_metrics(result)))
    assert "## 12 performance metrics" in md and "| Metric | Value | n | Status | Better |" in md
    assert "insufficient data" in md  # p95 with 6 tasks
    assert "**Run totals**" in md and "**Failures**" in md and "Step-by-step calculations" in md


def test_comparison_report_has_the_same_layout_with_a_metrics_section():
    a, b = run_fake("fake:3b"), run_fake("fake:7b", answers={"": "B"})
    bundles = [{"model": "fake:3b", "metrics": a}, {"model": "fake:7b", "metrics": b}]
    md = evaluate_llm.comparison_markdown(bundles, [[], []], [[], []], [ms.to_metrics(a), ms.to_metrics(b)])
    assert md.startswith("# Model comparison: fake:3b vs fake:7b")
    assert "## 12 performance metrics" in md and "| Metric | fake:3b | fake:7b | Better |" in md
    assert "| TTFT p95 (ms) | n/a | n/a | lower |" in md
    assert "## Question-answering benchmark" not in md  # sections without results are left out


def test_bat_entry_point_writes_comparison_and_per_model_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_llm, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(evaluate_llm, "run_metrics", lambda model, judge, args: run_fake(model, judge and "fake:judge"))
    assert evaluate_llm.main(["--suite", "metrics", "--models", "fake:3b", "fake:7b"]) == 0
    comparison = list(tmp_path.glob("comparison_*.md"))
    assert len(comparison) == 1 and "## 12 performance metrics" in comparison[0].read_text(encoding="utf-8")
    for slug in ("fake_3b", "fake_7b"):
        md = next((tmp_path / slug).glob("report_*.md")).read_text(encoding="utf-8")
        data = json.loads(next((tmp_path / slug).glob("report_*.json")).read_text(encoding="utf-8"))
        assert "## 12 performance metrics" in md
        assert len(data["bundles"][0]["metrics"]["metrics"]) == 12 and data["bundles"][0]["metrics"]["samples"]


def test_saved_report_can_be_rerendered_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_llm, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(evaluate_llm, "run_metrics", lambda model, judge, args: run_fake(model, None))
    evaluate_llm.main(["--suite", "metrics", "--models", "fake:3b"])
    saved = next((tmp_path / "fake_3b").glob("report_*.json"))
    assert evaluate_llm.main(["--predictions", str(saved)]) == 0


@pytest.mark.parametrize("value,expected", [(None, "n/a"), (0.0, "0.0000"), (12.5, "12.5000")])
def test_missing_values_print_as_na(value, expected):
    from evaluation.common import val

    assert val(value) == expected
