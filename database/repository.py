"""All SQL lives here. Every query is parameterised."""
from __future__ import annotations

import json

from database.db import Database
from models.schemas import Attempt, Question, Quiz, new_id, utcnow_iso


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ quizzes
    def save_quiz(self, quiz: Quiz) -> Quiz:
        """Insert or replace a quiz and all of its questions in one transaction."""
        quiz = quiz.model_copy(update={"updated_at": utcnow_iso()})
        with self.db.connection() as conn:
            conn.execute(
                """INSERT INTO quizzes (id, title, subject, source_name, source_text, model, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET title=excluded.title, subject=excluded.subject,
                     source_name=excluded.source_name, source_text=excluded.source_text,
                     model=excluded.model, updated_at=excluded.updated_at""",
                (quiz.id, quiz.title, quiz.subject, quiz.source_name, quiz.source_text,
                 quiz.model, quiz.created_at, quiz.updated_at),
            )
            conn.execute("DELETE FROM questions WHERE quiz_id = ?", (quiz.id,))
            conn.executemany(
                "INSERT INTO questions (id, quiz_id, position, text, difficulty, topic, data) VALUES (?,?,?,?,?,?,?)",
                [
                    (q.id, quiz.id, i, q.question, q.difficulty, q.topic, q.model_dump_json())
                    for i, q in enumerate(quiz.questions)
                ],
            )
        return quiz

    def get_quiz(self, quiz_id: str) -> Quiz | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
            if row is None:
                return None
            qrows = conn.execute(
                "SELECT data FROM questions WHERE quiz_id = ? ORDER BY position", (quiz_id,)
            ).fetchall()
        return Quiz(
            id=row["id"], title=row["title"], subject=row["subject"], source_name=row["source_name"],
            source_text=row["source_text"], model=row["model"], created_at=row["created_at"],
            updated_at=row["updated_at"],
            questions=[Question.model_validate_json(r["data"]) for r in qrows],
        )

    def list_quizzes(
        self, search: str = "", subject: str = "", difficulty: str = ""
    ) -> list[dict]:
        """Summary rows for the question bank (filterable)."""
        sql = [
            """SELECT q.id, q.title, q.subject, q.source_name, q.model, q.created_at, q.updated_at,
                      (SELECT COUNT(*) FROM questions x WHERE x.quiz_id = q.id) AS n_questions,
                      (SELECT COUNT(*) FROM attempts a WHERE a.quiz_id = q.id) AS n_attempts,
                      (SELECT MAX(a.percent) FROM attempts a WHERE a.quiz_id = q.id) AS best_percent,
                      (SELECT GROUP_CONCAT(DISTINCT x.difficulty) FROM questions x WHERE x.quiz_id = q.id) AS difficulties
               FROM quizzes q WHERE 1=1"""
        ]
        params: list = []
        if search.strip():
            like = f"%{_escape_like(search.strip())}%"
            sql.append(
                """AND (q.title LIKE ? ESCAPE '\\' OR q.subject LIKE ? ESCAPE '\\'
                        OR EXISTS (SELECT 1 FROM questions x WHERE x.quiz_id = q.id
                                   AND (x.text LIKE ? ESCAPE '\\' OR x.topic LIKE ? ESCAPE '\\')))"""
            )
            params += [like, like, like, like]
        if subject:
            sql.append("AND q.subject = ?")
            params.append(subject)
        if difficulty:
            sql.append("AND EXISTS (SELECT 1 FROM questions x WHERE x.quiz_id = q.id AND x.difficulty = ?)")
            params.append(difficulty)
        sql.append("ORDER BY q.updated_at DESC")
        with self.db.connection() as conn:
            return [dict(r) for r in conn.execute(" ".join(sql), params).fetchall()]

    def list_subjects(self) -> list[str]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT subject FROM quizzes WHERE subject <> '' ORDER BY subject"
            ).fetchall()
        return [r["subject"] for r in rows]

    def rename_quiz(self, quiz_id: str, title: str) -> None:
        title = title.strip()
        if not title:
            raise ValueError("Title must not be empty.")
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE quizzes SET title = ?, updated_at = ? WHERE id = ?", (title, utcnow_iso(), quiz_id)
            )

    def delete_quiz(self, quiz_id: str) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))

    def duplicate_quiz(self, quiz_id: str) -> Quiz:
        original = self.get_quiz(quiz_id)
        if original is None:
            raise KeyError(quiz_id)
        now = utcnow_iso()
        copy = original.model_copy(
            update={
                "id": new_id(),
                "title": f"{original.title} (copy)",
                "created_at": now,
                "updated_at": now,
                "questions": [q.model_copy(update={"id": new_id()}) for q in original.questions],
            }
        )
        return self.save_quiz(copy)

    # ----------------------------------------------------------------- attempts
    def save_attempt(self, attempt: Attempt) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """INSERT INTO attempts (id, quiz_id, quiz_title, correct, incorrect, unanswered,
                                         total, percent, duration_s, submitted_at, data)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (attempt.id, attempt.quiz_id, attempt.quiz_title, attempt.correct, attempt.incorrect,
                 attempt.unanswered, attempt.total, attempt.percent, attempt.duration_s,
                 attempt.submitted_at, attempt.model_dump_json()),
            )

    def list_attempts(self, quiz_id: str | None = None) -> list[Attempt]:
        sql, params = "SELECT data FROM attempts", []
        if quiz_id:
            sql += " WHERE quiz_id = ?"
            params.append(quiz_id)
        sql += " ORDER BY submitted_at DESC"
        with self.db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Attempt.model_validate_json(r["data"]) for r in rows]

    def clear_attempts(self) -> int:
        with self.db.connection() as conn:
            n = conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"]
            conn.execute("DELETE FROM attempts")
        return n

    # ------------------------------------------------------------ settings/stats
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    def load_setting(self, key: str, default=None):
        raw = self.get_setting(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def delete_setting(self, key: str) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))

    # unsaved generated quiz ("draft") - survives page reloads and restarts
    def save_draft(self, data: dict) -> None:
        self.set_setting("draft_current", data)

    def load_draft(self) -> dict | None:
        return self.load_setting("draft_current")

    def delete_draft(self) -> None:
        self.delete_setting("draft_current")

    # quiz in progress - lets the user resume after leaving the page
    def save_active_session(self, state: dict) -> None:
        self.set_setting("active_session", state)

    def load_active_session(self) -> dict | None:
        return self.load_setting("active_session")

    def clear_active_session(self) -> None:
        self.delete_setting("active_session")

    def bump_counter(self, name: str, by: int = 1) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO counters (name, value) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
                (name, by),
            )

    def counter(self, name: str) -> int:
        with self.db.connection() as conn:
            row = conn.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
        return row["value"] if row else 0

    def log_activity(self, kind: str, message: str) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO activity (ts, kind, message) VALUES (?, ?, ?)", (utcnow_iso(), kind, message)
            )

    def recent_activity(self, limit: int = 8) -> list[dict]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT ts, kind, message FROM activity ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def dashboard_stats(self) -> dict:
        with self.db.connection() as conn:
            quizzes = conn.execute("SELECT COUNT(*) AS n FROM quizzes").fetchone()["n"]
            questions = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
            a = conn.execute(
                "SELECT COUNT(*) AS n, AVG(percent) AS avg, MAX(percent) AS best FROM attempts"
            ).fetchone()
        return {
            "quizzes": quizzes,
            "questions_in_bank": questions,
            "questions_generated": self.counter("questions_generated"),
            "attempts": a["n"],
            "avg_percent": a["avg"],
            "best_percent": a["best"],
        }


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
