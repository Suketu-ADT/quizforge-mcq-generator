"""Background jobs (question generation, answer-key checks) that the browser polls for progress.

A job runs in a worker thread and appends what it produces to its own lists; the browser asks for
`/api/jobs/<id>?since=<n>` every half second and gets only the questions it has not seen yet.
Stopping sets an event that the job's progress callback turns into StopRequested, which also closes
the streaming connection to Ollama, so the model stops writing.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable

from models.schemas import Question

MAX_KEPT_JOBS = 20


class StopRequested(Exception):
    """Raised inside a job when the user pressed Stop."""


@dataclass
class Job:
    kind: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "running"  # running | done | stopped | error
    done: int = 0
    total: int = 0
    message: str = "Starting..."
    questions: list[Question] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rejected: int = 0
    error: str = ""
    refused: bool = False
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # -------------------------------------------------------------- called by the worker
    def check_stop(self) -> None:
        if self.stop_event.is_set():
            raise StopRequested()

    def progress(self, done: int, total: int, message: str) -> None:
        self.check_stop()
        with self.lock:
            self.done, self.total, self.message = done, total, message

    def add_question(self, q: Question) -> None:
        with self.lock:
            self.questions.append(q)
        self.check_stop()

    def replace_question(self, i: int, q: Question) -> None:
        with self.lock:
            self.questions[i] = q

    def finish(self, status: str = "done", error: str = "") -> None:
        with self.lock:
            if self.status == "running":
                self.status, self.error = status, error
                self.finished = time.monotonic()

    # ------------------------------------------------------------- called by the API
    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            since = max(0, min(since, len(self.questions)))
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "progress": {"done": self.done, "total": self.total, "message": self.message},
                "count": len(self.questions),
                "new_questions": [q.model_dump() for q in self.questions[since:]],
                # an answer-key check rewrites earlier questions, so the whole list is sent while it runs and at the end
                "questions": ([q.model_dump() for q in self.questions]
                              if self.kind == "check" or self.status != "running" else None),
                "warnings": list(self.warnings),
                "rejected": self.rejected,
                "error": self.error,
                "refused": self.refused,
                "elapsed_s": round((self.finished or time.monotonic()) - self.started, 1),
            }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, work: Callable[[Job], None]) -> Job:
        job = Job(kind)
        with self._lock:
            self._jobs[job.id] = job
            for old in list(self._jobs.values())[:-MAX_KEPT_JOBS]:
                if old.status != "running":
                    self._jobs.pop(old.id, None)

        def run() -> None:
            try:
                work(job)
            except StopRequested:
                job.finish("stopped")
            except Exception as exc:  # last resort: a job must never die silently
                traceback.print_exc()
                job.finish("error", f"Unexpected error: {exc}")
            job.finish("done")

        threading.Thread(target=run, name=f"quizforge-{kind}-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def running(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "running"]
