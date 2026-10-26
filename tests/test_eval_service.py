"""Integration tests for the evaluation service: real engine, real SQLite, fake model and judge (no Ollama needed)."""
import csv
import io
import json

import pytest

from database.db import Database
from services.evaluation.metrics import METRIC_ORDER
from services.evaluation.service import EvaluationService, NotFoundError
from tests.eval_fakes import FakeJudge, FakeModel, InlineExecutor

PRICES = {"version": "test-1", "currency": "USD",
          "models": {"fake:3b": {"input_per_million": 1.0, "output_per_million": 2.0}}}

ANSWERS = [
    {"id": "q1", "question": "Capital of France?", "options": {"A": "Paris", "B": "Rome", "C": "Oslo", "D": "Bern"}, "answer": "A"},
    {"id": "q2", "question": "What is 2+2?", "options": {"A": "3", "B": "4", "C": "5", "D": "6"}, "answer": "B"},
    {"id": "q3", "question": "Largest planet?", "options": {"A": "Jupiter", "B": "Mars", "C": "Venus", "D": "Earth"}, "answer": "A"},
    {"id": "q4", "question": "Water formula?", "options": {"A": "CO2", "B": "O2", "C": "H2O", "D": "NaCl"}, "answer": "C"},
]


def make_service(tmp_path, model=None, judge=None, pricing=PRICES):
    model = model or FakeModel(answers={"2+2": "B"})
    return EvaluationService(Database(tmp_path / "eval.db"), client_factory=lambda *a: model,
                             judge_client_factory=(lambda *a: judge) if judge else None, pricing=pricing, executor=InlineExecutor())


def answer_dataset(svc, items=ANSWERS, name="Mini answers"):
    return svc.import_dataset(name, "mcq_answer", "mini.json", json.dumps(items).encode())["dataset_id"]


def builtin(svc, task_type):
    return next(d["id"] for d in svc.list_datasets() if d["task_type"] == task_type and d["source"] == "builtin")


def metric(svc, run_id, key):
    return next(m for m in svc.get_metrics(run_id) if m["key"] == key)


@pytest.fixture
def svc(tmp_path):
    return make_service(tmp_path, judge=FakeJudge())


# ------------------------------------------------------------------------------------------------ start and persist
def test_answer_run_persists_samples_and_exact_metric_values(svc):
    ds = answer_dataset(svc)
    out = svc.create_run({"model": "fake:3b", "dataset_ids": [ds], "judge_model": "fake:judge"})
    assert out["status"] == "completed"
    run = svc.get_run(out["run_id"])
    assert run["sample_count"] == run["completed_count"] == 4 and len(svc.repo.all_samples(run["id"])) == 4
    assert [m["key"] for m in run["metrics"]] == METRIC_ORDER

    # FakeModel answers A unless told otherwise: q1, q2 (B), q3 correct, q4 (C) wrong
    acc = metric(svc, run["id"], "answer_accuracy")
    assert (acc["value"], acc["numerator"], acc["denominator"], acc["status"]) == (75.0, 3, 4, "available")
    assert metric(svc, run["id"], "task_success_rate")["value"] == 75.0
    assert metric(svc, run["id"], "error_rate")["value"] == 0.0
    # 4 requests x (50 in, 20 out) at $1/$2 per million = 0.00036 for 3 successful tasks
    cost = metric(svc, run["id"], "cost_per_successful_task")
    assert cost["value"] == pytest.approx(4 * (50 * 1 + 20 * 2) / 1e6 / 3) and cost["status"] == "available"
    # four TTFT observations: p95 needs 20, so it is reported as insufficient data, not a number
    ttft = metric(svc, run["id"], "ttft_p95")
    assert ttft["value"] is None and ttft["status"] == "insufficient_data" and ttft["sample_count"] == 4
    # metrics the dataset cannot measure are marked not applicable, never zero
    for key in ("faithfulness", "hallucination_rate", "safety_attack_success_rate"):
        assert metric(svc, run["id"], key)["status"] == "not_applicable" and metric(svc, run["id"], key)["value"] is None
    assert run["config"]["metrics_version"] and run["config"]["datasets"][0]["content_hash"]


def test_generation_and_safety_runs_produce_quality_metrics(svc):
    out = svc.create_run({"model": "fake:3b", "dataset_ids": [builtin(svc, "mcq_generation"), builtin(svc, "safety")],
                          "judge_model": "fake:judge", "limit_per_dataset": 3})
    run = svc.get_run(out["run_id"])
    assert run["status"] == "completed" and run["completed_count"] == 6
    assert metric(svc, run["id"], "answer_relevance")["value"] == 5.0
    assert metric(svc, run["id"], "faithfulness")["value"] == 100.0 and metric(svc, run["id"], "hallucination_rate")["value"] == 0.0
    assert metric(svc, run["id"], "instruction_following")["status"] == "available"
    asr = metric(svc, run["id"], "safety_attack_success_rate")
    assert asr["status"] == "available" and asr["value"] == 0.0  # this fake ignores injected instructions
    assert run["totals"]["judge_requests"] > 0


def test_obedient_model_raises_attack_success_rate(tmp_path):
    svc = make_service(tmp_path, model=FakeModel(obey_attacks=True), judge=FakeJudge())
    out = svc.create_run({"model": "fake:3b", "dataset_ids": [builtin(svc, "safety")], "judge_model": "fake:judge"})
    asr = metric(svc, out["run_id"], "safety_attack_success_rate")
    assert asr["value"] > 0 and asr["numerator"] >= 1
    failures = svc.get_samples(out["run_id"], tag="safety_violation")
    assert failures["total"] == asr["numerator"]


def test_no_judge_means_judge_metrics_are_not_fabricated(tmp_path):
    svc = make_service(tmp_path)
    out = svc.create_run({"model": "fake:3b", "dataset_ids": [builtin(svc, "mcq_generation")], "limit_per_dataset": 2})
    for key in ("answer_relevance", "faithfulness", "hallucination_rate"):
        m = metric(svc, out["run_id"], key)
        assert m["value"] is None and m["status"] in ("not_applicable", "insufficient_data"), key


# ------------------------------------------------------------------------------------------------- retrieval
def test_sample_paging_and_failure_filters(svc):
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)]})["run_id"]
    page = svc.get_samples(run_id, page=1, page_size=3)
    assert page["total"] == 4 and len(page["items"]) == 3 and page["pages"] == 2
    wrong = svc.get_samples(run_id, tag="incorrect_answer")
    assert wrong["total"] == 1 and wrong["items"][0]["item_id"] == "q4"
    assert svc.get_samples(run_id, failures_only=True)["total"] == 1
    with pytest.raises(ValueError, match="Unknown failure category"):
        svc.get_samples(run_id, tag="bogus")
    listing = svc.list_runs(page=1, page_size=10)
    assert listing["total"] == 1 and listing["items"][0]["metrics"]["answer_accuracy"]["value"] == 75.0
    assert svc.list_runs(model="other")["total"] == 0


def test_status_and_unknown_ids(svc):
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)]})["run_id"]
    st = svc.get_status(run_id)
    assert st["status"] == "completed" and st["progress"] == 1.0 and st["total"] == 4
    with pytest.raises(NotFoundError):
        svc.get_run("nope")
    with pytest.raises(NotFoundError):
        svc.get_samples("nope")


# ------------------------------------------------------------------------------------------ compare and trends
def test_compare_runs_with_deltas_and_comparability_warnings(svc):
    ds = answer_dataset(svc)
    a = svc.create_run({"model": "fake:3b", "dataset_ids": [ds], "name": "A"})["run_id"]
    b = svc.create_run({"model": "fake:3b", "dataset_ids": [ds], "name": "B"})["run_id"]
    cmp = svc.compare_runs([a, b])
    assert cmp["comparable"] and not cmp["warnings"]
    acc = next(r for r in cmp["metrics"] if r["key"] == "answer_accuracy")
    assert acc["cells"][1]["delta_vs_first"] == 0.0
    ttft = next(r for r in cmp["metrics"] if r["key"] == "ttft_p95")
    assert ttft["cells"][1]["delta_vs_first"] is None  # no delta from missing values

    other = svc.create_run({"model": "fake:3b", "dataset_ids": [ds], "name": "C", "limit_per_dataset": 2})["run_id"]
    warn = svc.compare_runs([a, other])
    assert not warn["comparable"] and "limit" in warn["warnings"][0]
    with pytest.raises(ValueError):
        svc.compare_runs([a])
    assert svc.previous_comparable_run(b)["id"] == a


def test_trends_list_runs_in_time_order(svc):
    ds = answer_dataset(svc)
    ids = [svc.create_run({"model": "fake:3b", "dataset_ids": [ds]})["run_id"] for _ in range(2)]
    t = svc.trends("answer_accuracy")
    assert [p["run_id"] for p in t] == ids and all(p["value"] == 75.0 for p in t)
    with pytest.raises(ValueError):
        svc.trends("made_up")


# ----------------------------------------------------------------------------------------------------- export
def test_exports_json_and_csv(svc):
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)]})["run_id"]
    data, name, mime = svc.export_run(run_id, "json")
    doc = json.loads(data)
    assert name.endswith(".json") and mime == "application/json"
    assert len(doc["samples"]) == 4 and len(doc["metric_definitions"]) == 12 and doc["run"]["config"]["pricing_version"] == "test-1"
    rows = list(csv.DictReader(io.StringIO(svc.export_run(run_id, "csv")[0].decode("utf-8-sig"))))
    assert len(rows) == 4 and {r["correct"] for r in rows} == {"True", "False"}
    mrows = list(csv.DictReader(io.StringIO(svc.export_run(run_id, "metrics_csv")[0].decode("utf-8-sig"))))
    acc = next(r for r in mrows if r["metric"] == "Answer Accuracy")
    assert (acc["value"], acc["numerator"], acc["denominator"]) == ("75.0", "3", "4")
    assert next(r for r in mrows if r["metric"].startswith("TTFT"))["value"] == ""  # missing stays empty, not 0
    with pytest.raises(ValueError):
        svc.export_run(run_id, "xml")


# --------------------------------------------------------------------------------------------- error handling
def test_timeouts_are_counted_as_technical_errors(tmp_path):
    svc = make_service(tmp_path, model=FakeModel(fail_on="Largest planet"))
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)]})["run_id"]
    err = metric(svc, run_id, "error_rate")
    assert err["numerator"] == 1 and err["denominator"] == 4 and err["value"] == 25.0 and err["details"]["categories"] == {"timeout": 1}
    acc = metric(svc, run_id, "answer_accuracy")
    assert acc["denominator"] == 3  # the failed request has no prediction to grade
    assert svc.get_samples(run_id, tag="technical_error")["total"] == 1
    assert metric(svc, run_id, "task_success_rate")["denominator"] == 4  # but it is still a failed task


def test_invalid_judge_output_is_reported_not_scored(tmp_path):
    svc = make_service(tmp_path, judge=FakeJudge(invalid=True))
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [builtin(svc, "mcq_generation")], "judge_model": "fake:judge",
                             "limit_per_dataset": 2})["run_id"]
    rel = metric(svc, run_id, "answer_relevance")
    assert rel["value"] is None and rel["status"] == "evaluation_failed"
    assert metric(svc, run_id, "faithfulness")["value"] is None


def test_cancellation_stops_remaining_items(tmp_path):
    svc = make_service(tmp_path)
    ds = answer_dataset(svc)
    holder = {}
    real_save = svc.engine._save

    def cancel_after_first(sample, position):
        real_save(sample, position)
        if position == 1:
            svc.cancel_run(holder["run_id"])

    svc.engine._save = cancel_after_first
    original_prepare = svc.engine.prepare

    def prepare(config):
        holder["run_id"] = original_prepare(config)
        return holder["run_id"]

    svc.engine.prepare = prepare
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [ds]})["run_id"]
    run = svc.get_run(run_id)
    assert run["status"] == "cancelled"
    statuses = [s["status"] for s in svc.repo.all_samples(run_id)]
    assert statuses[0] == "completed" and set(statuses[1:]) == {"cancelled"}
    assert metric(svc, run_id, "answer_accuracy")["denominator"] == 1  # cancelled items are not graded


def test_invalid_requests_are_rejected_with_messages(svc):
    with pytest.raises(ValueError, match="dataset_ids"):
        svc.create_run({"model": "fake:3b", "dataset_ids": []})
    with pytest.raises(ValueError, match="does not exist"):
        svc.create_run({"model": "fake:3b", "dataset_ids": ["missing"]})
    with pytest.raises(ValueError, match="relevance_threshold"):
        svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)], "relevance_threshold": 9})
    with pytest.raises(ValueError, match="model being evaluated"):
        svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)], "judge_model": "fake:3b"})
    assert svc.list_runs()["total"] == 0


def test_provider_setup_failure_marks_the_run_failed(tmp_path):
    def boom(*a):
        raise RuntimeError("cannot reach provider")

    svc = EvaluationService(Database(tmp_path / "e.db"), client_factory=boom, pricing=PRICES, executor=InlineExecutor())
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)]})["run_id"]
    run = svc.get_run(run_id)
    assert run["status"] == "failed" and "cannot reach provider" in run["error"]


def test_interrupted_runs_are_recovered_as_failed(tmp_path):
    svc = make_service(tmp_path)
    run_id = svc.engine.prepare(__import__("services.evaluation.engine", fromlist=["RunConfig"]).RunConfig(
        model="fake:3b", dataset_ids=[answer_dataset(svc)]))
    svc.repo.update_run(run_id, status="running")
    again = make_service(tmp_path)
    run = again.get_run(run_id)
    assert run["status"] == "failed" and "interrupted" in run["error"] and len(run["metrics"]) == 12
    again.delete_run(run_id)
    with pytest.raises(NotFoundError):
        again.get_run(run_id)


def test_runs_alive_in_another_process_are_not_recovered(tmp_path):
    from datetime import datetime, timedelta, timezone

    from services.evaluation.engine import RunConfig

    svc = make_service(tmp_path)
    ds = answer_dataset(svc)
    live = svc.engine.prepare(RunConfig(model="fake:3b", dataset_ids=[ds]))
    dead = svc.engine.prepare(RunConfig(model="fake:3b", dataset_ids=[ds]))
    now = datetime.now(timezone.utc)
    svc.repo.update_run(live, status="running", heartbeat_at=now.isoformat(timespec="seconds"))
    svc.repo.update_run(dead, status="running", heartbeat_at=(now - timedelta(minutes=10)).isoformat(timespec="seconds"))
    again = make_service(tmp_path)  # e.g. the app starting while the CLI is mid-run
    assert again.get_run(live)["status"] == "running"
    assert again.get_run(dead)["status"] == "failed"


def test_existing_database_gets_the_heartbeat_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # an eval_runs table from before the column existed
        conn.execute("CREATE TABLE eval_runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, model TEXT NOT NULL, provider TEXT NOT NULL, "
                     "status TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, duration_s REAL, "
                     "sample_count INTEGER NOT NULL DEFAULT 0, completed_count INTEGER NOT NULL DEFAULT 0, "
                     "cancel_requested INTEGER NOT NULL DEFAULT 0, dataset_ids TEXT NOT NULL, config TEXT NOT NULL, metrics TEXT, "
                     "totals TEXT, error TEXT)")
    from database.evaluation_repository import EvaluationRepository

    EvaluationRepository(Database(path))
    with sqlite3.connect(path) as conn:
        assert "heartbeat_at" in {r[1] for r in conn.execute("PRAGMA table_info(eval_runs)")}


def test_finished_runs_release_their_heartbeat(svc):
    from services.evaluation import service as service_module

    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)]})["run_id"]
    assert run_id not in service_module._ACTIVE and svc.get_run(run_id)["heartbeat_at"]


def test_unpriced_model_cost_is_insufficient_not_zero(tmp_path):
    svc = make_service(tmp_path, pricing={"version": "empty", "models": {}})
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [answer_dataset(svc)]})["run_id"]
    cost = metric(svc, run_id, "cost_per_successful_task")
    assert cost["value"] is None and cost["status"] == "insufficient_data"
    assert svc.get_run(run_id)["totals"]["total_cost"] is None


# ---------------------------------------------------------------------------------------------- human review
def test_human_review_updates_metrics(svc):
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [builtin(svc, "mcq_generation")], "judge_model": "fake:judge",
                             "limit_per_dataset": 1})["run_id"]
    sample = svc.repo.all_samples(run_id)[0]
    assert metric(svc, run_id, "answer_accuracy")["status"] != "available"  # generated keys wait for a human
    after = {m["key"]: m for m in svc.review_generated_key(sample["id"], 0, correct=True, note="checked")}
    assert after["answer_accuracy"]["value"] == 100.0 and after["answer_accuracy"]["denominator"] == 1
    after = {m["key"]: m for m in svc.review_claim(sample["id"], 0, "contradicted", note="wrong")}
    assert after["hallucination_rate"]["value"] > 0 and after["faithfulness"]["value"] < 100.0
    with pytest.raises(ValueError):
        svc.review_claim(sample["id"], 0, "maybe")
    with pytest.raises(ValueError, match="not a safety case"):
        svc.review_safety(sample["id"], "resisted")


def test_safety_review_and_single_metric_recompute(tmp_path):
    svc = make_service(tmp_path)  # no judge: policy cases need review
    run_id = svc.create_run({"model": "fake:3b", "dataset_ids": [builtin(svc, "safety")]})["run_id"]
    pending = svc.get_samples(run_id, tag="needs_review")
    assert pending["total"] >= 1
    before = metric(svc, run_id, "safety_attack_success_rate")
    after = {m["key"]: m for m in svc.review_safety(pending["items"][0]["id"], "attack_succeeded", note="leaked")}
    assert after["safety_attack_success_rate"]["numerator"] == (before["numerator"] or 0) + 1
    refreshed = svc.recompute_metrics(run_id, "error_rate")
    assert [m["key"] for m in refreshed] == METRIC_ORDER
    with pytest.raises(ValueError):
        svc.recompute_metrics(run_id, "nope")


# -------------------------------------------------------------------------------------------------- datasets
def test_dataset_import_dedupes_and_reports_invalid_rows(svc):
    bad = ANSWERS + [{"id": "q5", "question": "No answer here?"}]
    first = svc.import_dataset("Mini", "mcq_answer", "m.json", json.dumps(bad).encode())
    assert first["created"] and first["valid_count"] == 4 and first["invalid_count"] == 1
    again = svc.import_dataset("Mini", "mcq_answer", "m.json", json.dumps(bad).encode())
    assert again["dataset_id"] == first["dataset_id"] and not again["created"]
    assert svc.get_dataset(first["dataset_id"], preview=2)["preview"][0]["id"] == "q1"
    with pytest.raises(ValueError, match="No valid items"):
        svc.import_dataset("Empty", "mcq_answer", "e.json", b'[{"id": "x"}]')
    with pytest.raises(ValueError, match="name"):
        svc.import_dataset(" ", "mcq_answer", "m.json", json.dumps(ANSWERS).encode())
    svc.delete_dataset(first["dataset_id"])
    with pytest.raises(NotFoundError):
        svc.get_dataset(first["dataset_id"])


def test_builtins_are_seeded_once(tmp_path):
    make_service(tmp_path)
    svc = make_service(tmp_path)
    assert len([d for d in svc.list_datasets() if d["source"] == "builtin"]) == 3
    info = svc.list_evaluators()
    assert info["metrics"] == METRIC_ORDER and info["min_samples_for_percentile"][95] == 20
