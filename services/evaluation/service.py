"""EvaluationService - the evaluation "API".

This project has no HTTP layer (Streamlit calls Python services directly), so the suggested REST routes are provided as
service methods with the same responsibilities and consistent, JSON-serialisable responses:

    POST /api/evaluations               -> create_run(request)              (runs in a background thread)
    GET  /api/evaluations               -> list_runs(page, page_size, filters)
    GET  /api/evaluations/{id}          -> get_run(id)
    GET  /api/evaluations/{id}/samples  -> get_samples(id, page, page_size, filters)
    GET  /api/evaluations/{id}/metrics  -> get_metrics(id) / recompute_metrics(id, metric_key)
    GET  /api/evaluations/{id}/status   -> get_status(id)
    GET  /api/evaluations/compare       -> compare_runs([ids])
    GET  /api/evaluations/trends        -> trends(metric_key, filters)
    GET  /api/evaluations/{id}/export   -> export_run(id, "json" | "csv" | "metrics_csv")
    models / datasets / evaluators      -> list_models(), list_datasets(), list_evaluators(), import_dataset(...)
    human spot-checks                   -> review_generated_key(), review_claim(), review_safety()

The app is a local single-user tool with no authentication system, so there is nothing to protect these calls with;
an HTTP wrapper added later should put its own auth in front of them.
"""
from __future__ import annotations

import csv
import io
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ValidationError, field_validator

from database.db import Database
from database.evaluation_repository import FAILURE_TAGS, EvaluationRepository
from services.evaluation import datasets as dsmod
from services.evaluation.contracts import RUN_FAILED, RUN_QUEUED, RUN_RUNNING, TASK_TYPES, MetricResult
from services.evaluation.engine import PROMPT_VERSIONS, EvaluationEngine, RunConfig
from services.evaluation.evaluators import EVALUATOR_VERSIONS
from services.evaluation.judge import RUBRIC_VERSIONS
from services.evaluation.metrics import METRIC_ORDER, compute_all
from services.evaluation.stats import MIN_SAMPLES_FOR_PERCENTILE, PERCENTILE_METHOD
from services.ollama_client import OllamaClient, OllamaError

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-eval")  # one evaluation at a time on a local GPU/CPU
_ACTIVE: dict[str, EvaluationRepository] = {}  # runs this process owns (queued or running) -> where to write their heartbeat
_LOCK = threading.Lock()
_BEAT = {"thread": None}
HEARTBEAT_EVERY_S = 10
STALE_AFTER_S = 90  # a queued/running run without a heartbeat for this long belongs to a process that has stopped


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _beat_once() -> None:
    with _LOCK:
        owned = list(_ACTIVE.items())
    for run_id, repo in owned:
        try:
            repo.update_run(run_id, heartbeat_at=_now())
        except Exception:  # a locked or deleted database must not kill the heartbeat
            pass


def _heartbeat_loop() -> None:
    """Marks this process's runs as alive, so another app window or the CLI never mistakes them for interrupted runs."""
    while True:
        _beat_once()
        time.sleep(HEARTBEAT_EVERY_S)
        with _LOCK:
            if not _ACTIVE:
                _BEAT["thread"] = None
                return


def _own(run_id: str, repo: EvaluationRepository) -> None:
    with _LOCK:
        _ACTIVE[run_id] = repo
        start = _BEAT["thread"] is None
        if start:
            _BEAT["thread"] = threading.Thread(target=_heartbeat_loop, name="llm-eval-heartbeat", daemon=True)
    repo.update_run(run_id, heartbeat_at=_now())
    if start:
        _BEAT["thread"].start()


def _is_stale(run: dict) -> bool:
    beat = run.get("heartbeat_at")
    if not beat:
        return True
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(beat)).total_seconds()
    except ValueError:
        return True
    return age > STALE_AFTER_S


class NotFoundError(LookupError):
    pass


class CreateRunRequest(BaseModel):
    model: str = Field(min_length=1)
    dataset_ids: list[str] = Field(min_length=1)
    name: str = ""
    base_url: str = "http://localhost:11434"
    judge_model: str | None = None
    allow_self_judge: bool = False
    temperature: float = Field(default=0.4, ge=0, le=2)
    timeout_s: int = Field(default=300, ge=10, le=3600)
    limit_per_dataset: int | None = Field(default=None, ge=1)
    relevance_threshold: int = Field(default=4, ge=1, le=5)

    @field_validator("judge_model")
    @classmethod
    def _blank_is_none(cls, v):
        return v.strip() or None if isinstance(v, str) else v


def _err(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())


class EvaluationService:
    def __init__(self, db: Database, client_factory=None, judge_client_factory=None, pricing: dict | None = None, executor=None):
        self.repo = EvaluationRepository(db)
        self.engine = EvaluationEngine(self.repo, client_factory=client_factory, judge_client_factory=judge_client_factory, pricing=pricing)
        self.executor = executor or _EXECUTOR
        self.ensure_builtin_datasets()
        self._recover_interrupted_runs()

    # ------------------------------------------------------------------------------------------------ datasets
    def ensure_builtin_datasets(self) -> None:
        for ds in dsmod.load_builtin():
            self.repo.upsert_dataset(ds["name"], ds["task_type"], ds["version"], dsmod.content_hash(ds["task_type"], ds["items"]),
                                     ds["items"], source="builtin", synthetic=True)

    def list_datasets(self) -> list[dict]:
        return self.repo.list_datasets()

    def get_dataset(self, ds_id: str, preview: int = 5) -> dict:
        ds = self.repo.get_dataset(ds_id)
        if ds is None:
            raise NotFoundError(f"Dataset {ds_id} not found.")
        return {k: v for k, v in ds.items() if k != "items"} | {"preview": ds["items"][:preview]}

    def validate_dataset_file(self, task_type: str, filename: str, data: bytes) -> dict:
        """Parse and validate without storing - used for the upload preview."""
        rows = dsmod.parse_file(filename, data)
        items, errors = dsmod.validate_items(task_type, rows)
        return {"valid_count": len(items), "invalid_count": len(errors), "errors": errors[:50], "preview": items[:5], "items": items}

    def import_dataset(self, name: str, task_type: str, filename: str, data: bytes, version: str = "1") -> dict:
        if not name.strip():
            raise ValueError("Give the dataset a name.")
        if task_type not in TASK_TYPES:
            raise ValueError(f"Task type must be one of {TASK_TYPES}.")
        checked = self.validate_dataset_file(task_type, filename, data)
        if not checked["items"]:
            raise ValueError("No valid items: " + "; ".join(checked["errors"][:5]))
        ds_id, created = self.repo.upsert_dataset(name.strip(), task_type, version, dsmod.content_hash(task_type, checked["items"]),
                                                  checked["items"])
        return {"dataset_id": ds_id, "created": created, "valid_count": checked["valid_count"],
                "invalid_count": checked["invalid_count"], "errors": checked["errors"], "preview": checked["preview"]}

    def delete_dataset(self, ds_id: str) -> None:
        self.repo.delete_dataset(ds_id)

    # ---------------------------------------------------------------------------------------------------- runs
    def create_run(self, request: dict | CreateRunRequest, background: bool = True) -> dict:
        try:
            req = request if isinstance(request, CreateRunRequest) else CreateRunRequest(**request)
        except ValidationError as exc:
            raise ValueError(f"Invalid evaluation request: {_err(exc)}") from exc
        config = RunConfig(model=req.model.strip(), dataset_ids=req.dataset_ids, name=req.name.strip(), base_url=req.base_url.rstrip("/"),
                           judge_model=req.judge_model, allow_self_judge=req.allow_self_judge, temperature=req.temperature,
                           timeout_s=req.timeout_s, limit_per_dataset=req.limit_per_dataset,
                           relevance_threshold=req.relevance_threshold)
        run_id = self.engine.prepare(config)
        _own(run_id, self.repo)
        if background:
            self.executor.submit(self._execute_safely, run_id)
        else:
            self._execute_safely(run_id)
        return {"run_id": run_id, "status": self.get_status(run_id)["status"]}

    def _execute_safely(self, run_id: str) -> None:
        try:
            self.engine.execute(run_id)
        except Exception as exc:  # the run must never stay "running" forever
            self.repo.update_run(run_id, status=RUN_FAILED, error=f"{type(exc).__name__}: {exc}"[:500], finished_at=_now())
        finally:
            with _LOCK:
                _ACTIVE.pop(run_id, None)

    def _recover_interrupted_runs(self) -> None:
        """Runs left 'running'/'queued' by a closed app can never finish: mark them failed (their samples are kept).
        A run whose heartbeat is recent belongs to another live process (a second app window, the CLI) and is left alone."""
        rows, _ = self.repo.list_runs(page=1, page_size=200)
        for r in rows:
            if r["status"] in (RUN_RUNNING, RUN_QUEUED) and r["id"] not in _ACTIVE and _is_stale(r):
                samples = self.repo.all_samples(r["id"])
                self.repo.update_run(r["id"], status=RUN_FAILED, error="interrupted: the application stopped while this run was in progress",
                                     metrics=[m.to_dict() for m in compute_all(samples, r["config"].get("relevance_threshold", 4))])

    def get_status(self, run_id: str) -> dict:
        run = self._run(run_id)
        total = run["sample_count"] or 0
        return {"run_id": run_id, "status": run["status"], "completed": run["completed_count"], "total": total,
                "progress": (run["completed_count"] / total) if total else 0.0, "error": run.get("error"),
                "cancel_requested": bool(run["cancel_requested"])}

    def cancel_run(self, run_id: str) -> None:
        self._run(run_id)
        self.repo.update_run(run_id, cancel_requested=1)

    def delete_run(self, run_id: str) -> None:
        if self._run(run_id)["status"] in (RUN_RUNNING, RUN_QUEUED):
            raise ValueError("Cancel the run before deleting it.")
        self.repo.delete_run(run_id)

    def _run(self, run_id: str) -> dict:
        run = self.repo.get_run(run_id)
        if run is None:
            raise NotFoundError(f"Evaluation run {run_id} not found.")
        return run

    def get_run(self, run_id: str) -> dict:
        return self._run(run_id)

    def list_runs(self, page: int = 1, page_size: int = 20, model: str = "", status: str = "", dataset_id: str = "",
                  date_from: str = "", date_to: str = "") -> dict:
        rows, total = self.repo.list_runs(page, page_size, model, status, dataset_id, date_from, date_to)
        items = [{"id": r["id"], "name": r["name"], "model": r["model"], "status": r["status"], "created_at": r["created_at"],
                  "duration_s": r["duration_s"], "sample_count": r["sample_count"], "completed_count": r["completed_count"],
                  "datasets": [d["name"] for d in r["config"].get("datasets", [])], "judge_model": r["config"].get("judge_model"),
                  "metrics": {m["key"]: {"value": m["value"], "status": m["status"], "unit": m["unit"]} for m in r["metrics"]}}
                 for r in rows]
        return {"items": items, "total": total, "page": page, "page_size": page_size, "pages": max(1, math.ceil(total / page_size))}

    def get_metrics(self, run_id: str) -> list[dict]:
        return self._run(run_id)["metrics"]

    def recompute_metrics(self, run_id: str, metric_key: str | None = None) -> list[dict]:
        """Recalculate from the stored samples (e.g. after human review). With metric_key only that metric is replaced."""
        run = self._run(run_id)
        fresh = [m.to_dict() for m in compute_all(self.repo.all_samples(run_id), run["config"].get("relevance_threshold", 4))]
        if metric_key:
            if metric_key not in METRIC_ORDER:
                raise ValueError(f"Unknown metric '{metric_key}'.")
            old = {m["key"]: m for m in run["metrics"]}
            old[metric_key] = next(m for m in fresh if m["key"] == metric_key)
            fresh = [old.get(k) or next(m for m in fresh if m["key"] == k) for k in METRIC_ORDER]
        self.repo.update_run(run_id, metrics=fresh)
        return fresh

    def get_samples(self, run_id: str, page: int = 1, page_size: int = 20, task_type: str = "", failures_only: bool = False,
                    tag: str = "", status: str = "") -> dict:
        self._run(run_id)
        if tag and tag not in FAILURE_TAGS:
            raise ValueError(f"Unknown failure category '{tag}'.")
        rows, total = self.repo.list_samples(run_id, page, page_size, task_type, failures_only, tag, status)
        return {"items": rows, "total": total, "page": page, "page_size": page_size, "pages": max(1, math.ceil(total / page_size))}

    # -------------------------------------------------------------------------------------- compare and trends
    @staticmethod
    def signature(run: dict) -> dict:
        """What must match for two runs to be directly comparable."""
        cfg = run["config"]
        return {"datasets": sorted(d["content_hash"] for d in cfg.get("datasets", [])), "limit": cfg.get("limit_per_dataset"),
                "judge": cfg.get("judge_model"), "relevance_threshold": cfg.get("relevance_threshold"),
                "metrics_version": cfg.get("metrics_version")}

    def compare_runs(self, run_ids: list[str]) -> dict:
        if len(run_ids) < 2:
            raise ValueError("Select at least two runs to compare.")
        runs = [self._run(r) for r in run_ids]
        base_sig = self.signature(runs[0])
        warnings = []
        for r in runs[1:]:
            diff = [k for k, v in self.signature(r).items() if v != base_sig[k]]
            if diff:
                warnings.append(f"'{r['name']}' differs from '{runs[0]['name']}' in: {', '.join(diff)} - differences may reflect the "
                                "evaluation set-up, not only the model.")
        rows = []
        for key in METRIC_ORDER:
            ms = [next((m for m in r["metrics"] if m["key"] == key), None) for r in runs]
            first = ms[0]
            cells = []
            for m in ms:
                delta = (m["value"] - first["value"]) if (m and first and m["value"] is not None and first["value"] is not None) else None
                cells.append({"value": m["value"] if m else None, "status": m["status"] if m else "missing",
                              "sample_count": m["sample_count"] if m else 0, "delta_vs_first": delta})
            ref = first or next((m for m in ms if m), {})
            rows.append({"key": key, "name": ref.get("name", key), "unit": ref.get("unit", ""), "higher_is_better": ref.get("higher_is_better", True),
                         "group": ref.get("group", ""), "cells": cells})
        return {"runs": [{"id": r["id"], "name": r["name"], "model": r["model"], "created_at": r["created_at"],
                          "judge_model": r["config"].get("judge_model")} for r in runs],
                "metrics": rows, "comparable": not warnings, "warnings": warnings}

    def trends(self, metric_key: str, model: str = "", date_from: str = "", date_to: str = "") -> list[dict]:
        if metric_key not in METRIC_ORDER:
            raise ValueError(f"Unknown metric '{metric_key}'.")
        rows, _ = self.repo.list_runs(1, 200, model=model, date_from=date_from, date_to=date_to)
        out = []
        for r in rows:
            m = next((m for m in r["metrics"] if m["key"] == metric_key), None)
            if m is None:
                continue
            sig = self.signature(r)
            out.append({"run_id": r["id"], "name": r["name"], "model": r["model"], "created_at": r["created_at"], "value": m["value"],
                        "status": m["status"], "sample_count": m["sample_count"],
                        "scope": "|".join(sig["datasets"])[:16] + f"|judge={sig['judge']}"})
        return sorted(out, key=lambda x: x["created_at"])

    def previous_comparable_run(self, run_id: str) -> dict | None:
        run = self._run(run_id)
        sig = self.signature(run)
        rows, _ = self.repo.list_runs(1, 200, model=run["model"])
        for r in rows:
            if r["created_at"] < run["created_at"] and r["status"] == "completed" and self.signature(r) == sig:
                return r
        return None

    # --------------------------------------------------------------------------------------------- reference
    def list_models(self, base_url: str = "http://localhost:11434") -> dict:
        try:
            return {"models": OllamaClient(base_url).list_models(), "error": None}
        except OllamaError as exc:
            return {"models": [], "error": str(exc)}

    @staticmethod
    def list_evaluators() -> dict:
        return {"metrics_version": MetricResult.__dataclass_fields__["version"].default, "metrics": METRIC_ORDER,
                "evaluators": EVALUATOR_VERSIONS, "judge_rubrics": RUBRIC_VERSIONS, "prompt_versions": PROMPT_VERSIONS,
                "percentile_method": PERCENTILE_METHOD, "min_samples_for_percentile": MIN_SAMPLES_FOR_PERCENTILE,
                "failure_categories": list(FAILURE_TAGS)}

    # ------------------------------------------------------------------------------------- human spot-checks
    def _sample(self, sample_id: str) -> dict:
        s = self.repo.get_sample(sample_id)
        if s is None:
            raise NotFoundError(f"Sample {sample_id} not found.")
        return s

    def _store_review(self, sample: dict) -> list[dict]:
        position = sample.pop("_position")
        self.repo.save_sample(sample, position)
        return self.recompute_metrics(sample["run_id"])

    def review_generated_key(self, sample_id: str, question_index: int, correct: bool, note: str = "") -> list[dict]:
        """A human confirms (or rejects) the answer key of a generated question -> it becomes an evaluable accuracy item."""
        s = self._sample(sample_id)
        entries = s.get("outcomes", {}).get("accuracy", [])
        entry = next((e for e in entries if e.get("question_index") == question_index), None)
        if entry is None:
            raise ValueError("That sample has no generated question with this index.")
        entry.update(evaluable=True, correct=bool(correct), method="human_label", reference="(human review)", note=note[:300])
        return self._store_review(s)

    def review_claim(self, sample_id: str, claim_index: int, verdict: str, note: str = "") -> list[dict]:
        if verdict not in ("supported", "contradicted", "unsupported", "unverifiable"):
            raise ValueError("Verdict must be supported, contradicted, unsupported or unverifiable.")
        s = self._sample(sample_id)
        claims = s.get("outcomes", {}).get("claims", [])
        if not 0 <= claim_index < len(claims):
            raise ValueError("Claim index out of range.")
        claims[claim_index].update(verdict=verdict, status="ok", method="human_review", human_note=note[:300],
                                   judge_verdict=claims[claim_index].get("verdict"))
        return self._store_review(s)

    def review_safety(self, sample_id: str, outcome: str, note: str = "") -> list[dict]:
        allowed = ("attack_succeeded", "resisted", "safe_refusal", "over_refusal", "complied")
        if outcome not in allowed:
            raise ValueError(f"Outcome must be one of {allowed}.")
        s = self._sample(sample_id)
        safety = s.get("outcomes", {}).get("safety")
        if not safety:
            raise ValueError("That sample is not a safety case.")
        safety.update(outcome=outcome, detail=f"human review: {note[:200]}", grader="human")
        s["outcomes"]["task"] = {"success": outcome in ("resisted", "safe_refusal", "complied"), "method": "human review",
                                 "failure_reason": "" if outcome in ("resisted", "safe_refusal", "complied") else outcome}
        return self._store_review(s)

    # ------------------------------------------------------------------------------------------------- export
    def export_run(self, run_id: str, fmt: str = "json") -> tuple[bytes, str, str]:
        run = self._run(run_id)
        samples = self.repo.all_samples(run_id)
        stamp = run["created_at"][:19].replace(":", "").replace("-", "")
        base = f"evaluation_{run['model'].replace(':', '_')}_{stamp}"
        definitions = [{k: m.get(k) for k in ("key", "name", "unit", "formula", "description", "evaluator", "version", "limitations",
                                                "higher_is_better")} for m in run["metrics"]]
        if fmt == "json":
            doc = {"run": {k: run[k] for k in ("id", "name", "model", "provider", "status", "created_at", "started_at", "finished_at",
                                                  "duration_s", "sample_count", "completed_count", "config", "totals", "error")},
                   "metric_definitions": definitions, "metrics": run["metrics"], "samples": samples}
            return json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8"), base + ".json", "application/json"
        if fmt == "metrics_csv":
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(["run_id", "model", "metric", "value", "unit", "status", "numerator", "denominator", "sample_count", "formula",
                        "evaluator", "version"])
            for m in run["metrics"]:
                w.writerow([run["id"], run["model"], m["name"], m["value"], m["unit"], m["status"], m.get("numerator"),
                            m.get("denominator"), m["sample_count"], m["formula"], m.get("evaluator", ""), m.get("version", "")])
            return buf.getvalue().encode("utf-8-sig"), base + "_metrics.csv", "text/csv"
        if fmt == "csv":
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(["run_id", "model", "sample_id", "dataset", "item_id", "task_type", "status", "task_success", "failure_reason",
                        "e2e_ms", "first_ttft_ms", "candidate_requests", "input_tokens", "output_tokens", "cost", "cost_status",
                        "error_category", "prediction", "reference", "correct", "safety_outcome", "instructions_satisfied",
                        "instructions_evaluable", "claims_supported", "claims_decided", "mean_relevance"])
            for s in samples:
                o = s.get("outcomes", {})
                cand = [r for r in s.get("requests", []) if r.get("role") == "candidate"]
                acc = next(iter(o.get("accuracy", [])), {})
                ins = [r for r in o.get("instructions", []) if r.get("satisfied") is not None]
                dec = [c for c in o.get("claims", []) if c.get("verdict") in ("supported", "contradicted", "unsupported")]
                rel = [e["score"] for e in o.get("relevance", []) if isinstance(e.get("score"), int)]
                w.writerow([run["id"], run["model"], s["id"], s.get("dataset_name", s.get("dataset_id")), s.get("item_id"), s["task_type"],
                            s["status"], (o.get("task") or {}).get("success"), (o.get("task") or {}).get("failure_reason", ""),
                            (s.get("timing") or {}).get("e2e_ms"), next((r["ttft_ms"] for r in cand if r.get("ttft_ms") is not None), None),
                            len(cand), sum(r.get("prompt_tokens") or 0 for r in cand), sum(r.get("completion_tokens") or 0 for r in cand),
                            (s.get("cost") or {}).get("amount"), (s.get("cost") or {}).get("status"), (s.get("error") or {}).get("category"),
                            acc.get("prediction"), acc.get("reference"), acc.get("correct") if acc.get("evaluable") else None,
                            (o.get("safety") or {}).get("outcome"), sum(1 for r in ins if r["satisfied"]), len(ins),
                            sum(1 for c in dec if c["verdict"] == "supported"), len(dec),
                            (sum(rel) / len(rel)) if rel else None])
            return buf.getvalue().encode("utf-8-sig"), base + "_samples.csv", "text/csv"
        raise ValueError("Export format must be json, csv or metrics_csv.")
