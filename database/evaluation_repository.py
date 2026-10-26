"""SQL for the LLM evaluation tables. Same conventions as repository.py: parameterised SQL, one transaction per call."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from database.db import Database
from models.schemas import new_id, utcnow_iso


EVAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_datasets (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    task_type    TEXT NOT NULL,
    version      TEXT NOT NULL DEFAULT '1',
    content_hash TEXT NOT NULL UNIQUE,
    item_count   INTEGER NOT NULL,
    source       TEXT NOT NULL DEFAULT 'upload',
    synthetic    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    items        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    model            TEXT NOT NULL,
    provider         TEXT NOT NULL,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    duration_s       REAL,
    sample_count     INTEGER NOT NULL DEFAULT 0,
    completed_count  INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    dataset_ids      TEXT NOT NULL,
    config           TEXT NOT NULL,
    metrics          TEXT,
    totals           TEXT,
    error            TEXT,
    heartbeat_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON eval_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_eval_runs_model ON eval_runs(model, created_at);
CREATE TABLE IF NOT EXISTS eval_samples (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,
    dataset_id   TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    task_type    TEXT NOT NULL,
    status       TEXT NOT NULL,
    task_success INTEGER,
    has_failure  INTEGER NOT NULL DEFAULT 0,
    failure_tags TEXT NOT NULL DEFAULT '',
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_samples_run ON eval_samples(run_id, position);
CREATE INDEX IF NOT EXISTS idx_eval_samples_failures ON eval_samples(run_id, has_failure, task_type);
"""


def _created_now() -> str:
    """Run creation time with microseconds, so runs started in the same second still have a defined order."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _loads(v, default):
    return json.loads(v) if v else default


class EvaluationRepository:
    def __init__(self, db: Database):
        self.db = db
        with db.connection() as conn:  # the evaluation keeps its own tables; created on first use, never touches quiz data
            conn.executescript(EVAL_SCHEMA)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(eval_runs)")}
            if "heartbeat_at" not in cols:  # additive migration for databases created before the column existed
                conn.execute("ALTER TABLE eval_runs ADD COLUMN heartbeat_at TEXT")

    # ---------------------------------------------------------------------------------------------- datasets
    def upsert_dataset(self, name: str, task_type: str, version: str, content_hash: str, items: list[dict],
                       source: str = "upload", synthetic: bool = False) -> tuple[str, bool]:
        """Returns (dataset_id, created). An identical dataset (same content hash) is never stored twice."""
        with self.db.connection() as conn:
            row = conn.execute("SELECT id FROM eval_datasets WHERE content_hash = ?", (content_hash,)).fetchone()
            if row:
                return row["id"], False
            ds_id = new_id()
            conn.execute(
                "INSERT INTO eval_datasets (id, name, task_type, version, content_hash, item_count, source, synthetic, created_at, items) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ds_id, name, task_type, version, content_hash, len(items), source, int(synthetic), utcnow_iso(),
                 json.dumps(items, ensure_ascii=False)),
            )
        return ds_id, True

    def list_datasets(self) -> list[dict]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT id, name, task_type, version, content_hash, item_count, source, synthetic, created_at "
                                "FROM eval_datasets ORDER BY source = 'builtin' DESC, created_at").fetchall()
        return [dict(r) for r in rows]

    def get_dataset(self, ds_id: str) -> dict | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM eval_datasets WHERE id = ?", (ds_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["items"] = _loads(d["items"], [])
        return d

    def delete_dataset(self, ds_id: str) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM eval_datasets WHERE id = ? AND source <> 'builtin'", (ds_id,))

    # -------------------------------------------------------------------------------------------------- runs
    def create_run(self, name: str, model: str, provider: str, dataset_ids: list[str], config: dict, sample_count: int) -> str:
        run_id = new_id()
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO eval_runs (id, name, model, provider, status, created_at, sample_count, dataset_ids, config) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, name, model, provider, "queued", _created_now(), sample_count, json.dumps(dataset_ids), json.dumps(config)),
            )
        return run_id

    def update_run(self, run_id: str, **fields) -> None:
        allowed = {"status", "started_at", "finished_at", "duration_s", "completed_count", "metrics", "totals", "error",
                   "cancel_requested", "name", "sample_count", "heartbeat_at"}
        sets, params = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"cannot update field {k}")
            sets.append(f"{k} = ?")
            params.append(json.dumps(v) if k in ("metrics", "totals") else v)
        if not sets:
            return
        with self.db.connection() as conn:
            conn.execute(f"UPDATE eval_runs SET {', '.join(sets)} WHERE id = ?", (*params, run_id))

    def _run_row(self, row) -> dict:
        d = dict(row)
        d["dataset_ids"] = _loads(d["dataset_ids"], [])
        d["config"] = _loads(d["config"], {})
        d["metrics"] = _loads(d.get("metrics"), [])
        d["totals"] = _loads(d.get("totals"), {})
        return d

    def get_run(self, run_id: str) -> dict | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        return self._run_row(row) if row else None

    def list_runs(self, page: int = 1, page_size: int = 20, model: str = "", status: str = "", dataset_id: str = "",
                  date_from: str = "", date_to: str = "") -> tuple[list[dict], int]:
        where, params = ["1=1"], []
        if model:
            where.append("model = ?")
            params.append(model)
        if status:
            where.append("status = ?")
            params.append(status)
        if dataset_id:
            where.append("dataset_ids LIKE ?")
            params.append(f'%"{dataset_id}"%')
        if date_from:
            where.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            where.append("created_at <= ?")
            params.append(date_to)
        page, page_size = max(1, page), max(1, min(page_size, 200))
        clause = " AND ".join(where)
        with self.db.connection() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS n FROM eval_runs WHERE {clause}", params).fetchone()["n"]
            rows = conn.execute(f"SELECT * FROM eval_runs WHERE {clause} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                                (*params, page_size, (page - 1) * page_size)).fetchall()
        return [self._run_row(r) for r in rows], total

    def delete_run(self, run_id: str) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))

    def cancel_requested(self, run_id: str) -> bool:
        with self.db.connection() as conn:
            row = conn.execute("SELECT cancel_requested FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    # ----------------------------------------------------------------------------------------------- samples
    def save_sample(self, sample: dict, position: int) -> None:
        task = (sample.get("outcomes") or {}).get("task") or {}
        tags = failure_tags(sample)
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_samples (id, run_id, position, dataset_id, item_id, task_type, status, task_success, "
                "has_failure, failure_tags, data) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (sample["id"], sample["run_id"], position, sample.get("dataset_id", ""), sample.get("item_id", ""), sample["task_type"],
                 sample["status"], None if task.get("success") is None else int(bool(task["success"])), int(bool(tags)),
                 "," + ",".join(tags) + "," if tags else "", json.dumps(sample, ensure_ascii=False)),
            )

    def get_sample(self, sample_id: str) -> dict | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT data, position FROM eval_samples WHERE id = ?", (sample_id,)).fetchone()
        if not row:
            return None
        d = json.loads(row["data"])
        d["_position"] = row["position"]
        return d

    def all_samples(self, run_id: str) -> list[dict]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT data FROM eval_samples WHERE run_id = ? ORDER BY position", (run_id,)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def list_samples(self, run_id: str, page: int = 1, page_size: int = 20, task_type: str = "", failures_only: bool = False,
                     tag: str = "", status: str = "") -> tuple[list[dict], int]:
        where, params = ["run_id = ?"], [run_id]
        if task_type:
            where.append("task_type = ?")
            params.append(task_type)
        if failures_only:
            where.append("has_failure = 1")
        if tag:
            where.append("failure_tags LIKE ?")
            params.append(f"%,{tag},%")
        if status:
            where.append("status = ?")
            params.append(status)
        page, page_size = max(1, page), max(1, min(page_size, 200))
        clause = " AND ".join(where)
        with self.db.connection() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS n FROM eval_samples WHERE {clause}", params).fetchone()["n"]
            rows = conn.execute(f"SELECT data FROM eval_samples WHERE {clause} ORDER BY position LIMIT ? OFFSET ?",
                                (*params, page_size, (page - 1) * page_size)).fetchall()
        return [json.loads(r["data"]) for r in rows], total


FAILURE_TAGS = ("incorrect_answer", "flagged_claim", "failed_instruction", "safety_violation", "over_refusal", "failed_task",
                "technical_error", "low_relevance", "needs_review")


def failure_tags(sample: dict) -> list[str]:
    """Indexable labels used by the failure-analysis browser."""
    o = sample.get("outcomes") or {}
    tags = []
    if any(e.get("evaluable") and not e.get("correct") for e in o.get("accuracy", [])):
        tags.append("incorrect_answer")
    if any(c.get("verdict") in ("unsupported", "contradicted") for c in o.get("claims", [])):
        tags.append("flagged_claim")
    if any(r.get("satisfied") is False for r in o.get("instructions", [])):
        tags.append("failed_instruction")
    safety = o.get("safety") or {}
    if safety.get("outcome") == "attack_succeeded":
        tags.append("safety_violation")
    if safety.get("outcome") == "over_refusal":
        tags.append("over_refusal")
    if safety.get("outcome") == "needs_review":
        tags.append("needs_review")
    if (o.get("task") or {}).get("success") is False:
        tags.append("failed_task")
    if sample.get("status") == "error" or any(r.get("status") == "error" for r in sample.get("requests", [])):
        tags.append("technical_error")
    if any(isinstance(e.get("score"), int) and e["score"] < 4 for e in o.get("relevance", [])):
        tags.append("low_relevance")
    return tags
