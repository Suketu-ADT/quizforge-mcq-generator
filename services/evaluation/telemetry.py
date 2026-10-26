"""Request instrumentation: an adapter around the provider client that records one trace per model call.

Every call is STREAMED so the time to the first content token is genuinely observed (monotonic clock). Ollama's own
counters (model load, prompt processing, generation time, token counts) are attached to the trace, which separates
model work from queueing / loading overhead.
"""
from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from services.ollama_client import (
    ModelNotFoundError,
    OllamaError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class EvaluationCancelled(Exception):
    """Raised before a model call when the user cancelled the run."""


_SECRET_RE = re.compile(r"(?i)\b(sk|api[_-]?key|token|bearer|password|secret)\b[\s:=\"']*[A-Za-z0-9._\-]{6,}")


def redact(message: str, limit: int = 300) -> str:
    """Error text without anything that looks like a credential, truncated."""
    return _SECRET_RE.sub(r"\1=[redacted]", message or "")[:limit]


def error_category(exc: Exception) -> str:
    if isinstance(exc, OllamaTimeoutError):
        return "timeout"
    if isinstance(exc, OllamaUnavailableError):
        return "connection"
    if isinstance(exc, ModelNotFoundError):
        return "model_not_found"
    if isinstance(exc, EvaluationCancelled):
        return "cancelled"
    return "provider_error"


def _ms(ns) -> float | None:
    return None if ns is None else ns / 1e6


class InstrumentedClient:
    """Drop-in replacement for OllamaClient inside the evaluation engine (same `chat` / `model` / usage interface)."""

    provider = "ollama"

    def __init__(
        self,
        base,
        role: str = "candidate",
        purpose: str = "generate",
        output_validator: Callable[[str], bool] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        self.base = base
        self.role = role
        self.purpose = purpose
        self.output_validator = output_validator
        self.cancel_check = cancel_check
        self.traces: list[dict] = []
        self.calls: list[dict] = []  # raw prompt + reply of every successful call (instruction-following analysis)

    # --- interface used by the project's QuestionGenerator / answer checker -------------------------------------------
    @property
    def model(self) -> str:
        return self.base.model

    @property
    def usage(self) -> dict:
        return self.base.usage

    def reset_usage(self) -> dict:
        return self.base.reset_usage()

    def chat(self, messages, schema=None, temperature=None, num_ctx=8192, on_progress=None) -> str:
        def call(progress):
            return self.base.chat(messages, schema=schema, temperature=temperature, num_ctx=num_ctx, on_progress=progress)

        out = self._run(call, messages, on_progress)
        return out

    def chat_with_logprobs(self, messages, top_logprobs: int = 20, num_predict: int = 4):
        return self._run(lambda progress: self.base.chat_with_logprobs(messages, top_logprobs, num_predict, on_progress=progress),
                         messages, None, purpose="answer")

    # --- recording ----------------------------------------------------------------------------------------------------
    def _run(self, call, messages, on_progress, purpose: str | None = None):
        if self.cancel_check and self.cancel_check():
            raise EvaluationCancelled("cancelled by the user")
        purpose = purpose or self.purpose
        attempt = 1 + sum(1 for t in self.traces if t["purpose"] == purpose)
        trace = {
            "request_id": uuid.uuid4().hex[:12], "role": self.role, "purpose": purpose, "model": self.model,
            "provider": self.provider, "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "attempt": attempt, "streamed": True, "ttft_ms": None, "latency_ms": None, "status": "ok",
            "error_category": None, "error_message": None, "prompt_tokens": None, "completion_tokens": None,
            "load_ms": None, "prompt_eval_ms": None, "eval_ms": None, "schema_valid": None,
        }
        start = time.monotonic()
        first: list[float] = []

        def progress(n: int) -> None:
            if not first:
                first.append(time.monotonic())
            if on_progress:
                on_progress(n)

        self.base.last_stats = {}
        try:
            result = call(progress)
        except (OllamaError, EvaluationCancelled) as exc:
            trace.update(status="error", latency_ms=(time.monotonic() - start) * 1000, error_category=error_category(exc),
                         error_message=redact(str(exc)), ttft_ms=(first[0] - start) * 1000 if first else None)
            self.traces.append(trace)
            raise
        end = time.monotonic()
        stats = getattr(self.base, "last_stats", {}) or {}
        text = result[0] if isinstance(result, tuple) else result
        trace.update(
            latency_ms=(end - start) * 1000, ttft_ms=(first[0] - start) * 1000 if first else None,
            prompt_tokens=stats.get("prompt_eval_count"), completion_tokens=stats.get("eval_count"),
            load_ms=_ms(stats.get("load_duration")), prompt_eval_ms=_ms(stats.get("prompt_eval_duration")),
            eval_ms=_ms(stats.get("eval_duration")),
        )
        if not first:  # no content token observed (empty reply): TTFT stays unavailable rather than guessed
            trace["streamed"] = True
        if self.output_validator is not None:
            try:
                trace["schema_valid"] = bool(self.output_validator(text))
            except Exception:
                trace["schema_valid"] = False
        self.traces.append(trace)
        self.calls.append({"messages": messages, "output": text})
        return result
