"""SQLite connection helper and schema initialisation."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS quizzes (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    subject     TEXT NOT NULL DEFAULT '',
    source_name TEXT NOT NULL DEFAULT '',
    source_text TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
    id         TEXT NOT NULL,
    quiz_id    TEXT NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    topic      TEXT NOT NULL,
    data       TEXT NOT NULL,
    PRIMARY KEY (quiz_id, id)
);
CREATE INDEX IF NOT EXISTS idx_questions_quiz ON questions(quiz_id, position);
CREATE TABLE IF NOT EXISTS attempts (
    id           TEXT PRIMARY KEY,
    quiz_id      TEXT NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    quiz_title   TEXT NOT NULL,
    correct      INTEGER NOT NULL,
    incorrect    INTEGER NOT NULL,
    unanswered   INTEGER NOT NULL,
    total        INTEGER NOT NULL,
    percent      REAL NOT NULL,
    duration_s   REAL,
    submitted_at TEXT NOT NULL,
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_quiz ON attempts(quiz_id, submitted_at);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activity (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    """Creates a short-lived connection per operation (safe with Streamlit reruns)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        self.init()

    def _connect(self) -> sqlite3.Connection:
        if str(self.path) == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn = self._memory_conn
        else:
            conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Transaction scope: commit on success, roll back on error."""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            if conn is not self._memory_conn:
                conn.close()

    def init(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)
