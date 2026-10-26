"""Model-pricing registry. Prices live in a JSON file (evaluation/pricing.json by default, or EVAL_PRICING_FILE), so a
price change never touches the calculation code. Unknown usage or an unlisted model gives an UNKNOWN cost - never zero."""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PRICING_FILE = Path(__file__).resolve().parents[2] / "evaluation" / "pricing.json"


def load_registry(path: str | Path | None = None) -> dict:
    path = Path(path or os.getenv("EVAL_PRICING_FILE") or DEFAULT_PRICING_FILE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": "missing", "currency": "USD", "models": {}}
    data.setdefault("models", {})
    return data


def task_cost(model: str, requests: list[dict], registry: dict) -> dict:
    """Cost of all candidate-model calls of one task (every call in the trace is included, retries too)."""
    entry = (registry.get("models") or {}).get(model)
    base = {"currency": registry.get("currency", "USD"), "pricing_version": registry.get("version", ""), "model": model}
    if entry is None:
        return {**base, "amount": None, "status": "unknown", "note": f"'{model}' is not in the pricing registry"}
    calls = [r for r in requests if r.get("role") == "candidate"]
    if any(r.get("prompt_tokens") is None or r.get("completion_tokens") is None for r in calls if r.get("status") == "ok"):
        return {**base, "amount": None, "status": "unknown", "note": "token usage missing for at least one call"}
    tin = sum(r.get("prompt_tokens") or 0 for r in calls)
    tout = sum(r.get("completion_tokens") or 0 for r in calls)
    amount = tin * float(entry["input_per_million"]) / 1e6 + tout * float(entry["output_per_million"]) / 1e6
    status = "measured"
    note = entry.get("note", "")
    if entry.get("compute_cost_per_hour") is not None:
        seconds = sum((r.get("latency_ms") or 0) for r in calls) / 1000
        amount += seconds * float(entry["compute_cost_per_hour"]) / 3600
        status = "estimated"
    return {**base, "amount": amount, "status": status, "note": note, "input_tokens": tin, "output_tokens": tout,
            "effective_date": entry.get("effective_date", "")}
