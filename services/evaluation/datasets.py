"""Evaluation datasets: schemas, validation, import (JSON / JSONL / CSV) and the built-in synthetic sets.

Task types and their fields (* = required):
  mcq_answer      id*, question*, answer*, options {A: text, ...} (multiple choice) - OR no options for a free-text answer,
                  context, category, difficulty, match ("exact" | "normalized" | "judge", free-text only; default "normalized")
  mcq_generation  id*, text* (source material), title, count (default 3), num_options (4), difficulty (Medium),
                  include_explanations (true), instructions (subjective, judged only if a judge is configured),
                  constraints {required_terms: [], forbidden_terms: [], max_question_words: int}, expected {min_questions: int}
  safety          id*, category*, kind* ("attack" | "benign"), text* (material containing the attack), title,
                  expected_behavior*, grader* {type: canary|disclosure|forbidden_terms|judge|should_comply, ...}

CSV uses flat columns: option_A..option_F; required_terms / forbidden_terms / markers / terms as ';'-separated lists;
grader_type, canary, min_hits, policy for safety.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from services.evaluation.contracts import MCQ_ANSWER, MCQ_GENERATION, SAFETY, TASK_TYPES

ROOT = Path(__file__).resolve().parents[2]
BUILTIN = [
    {"name": "QuizForge MCQ answering v1", "task_type": MCQ_ANSWER, "path": ROOT / "evaluation" / "dataset.json", "version": "1"},
    {"name": "QuizForge generation passages v1", "task_type": MCQ_GENERATION, "path": ROOT / "evaluation" / "passages.json", "version": "1"},
    {"name": "QuizForge safety suite v1", "task_type": SAFETY, "path": ROOT / "evaluation" / "safety_cases.json", "version": "1"},
]
MAX_ITEMS = 5000
LETTERS = "ABCDEF"
GRADER_TYPES = ("canary", "disclosure", "forbidden_terms", "judge", "should_comply")
SAFETY_CATEGORIES = ("prompt_injection", "jailbreak", "system_prompt_disclosure", "unsafe_compliance", "benign_control")


class DatasetError(ValueError):
    pass


def _list(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(";") if x.strip()]


def _bool(v, default: bool) -> bool:
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def _int(v, default: int | None) -> int | None:
    if v is None or v == "":
        return default
    return int(float(v))


# --------------------------------------------------------------------------------------------- per-type validation
def _validate_mcq_answer(raw: dict) -> dict:
    q = str(raw.get("question") or "").strip()
    ans = str(raw.get("answer") or "").strip()
    if not q:
        raise DatasetError("missing 'question'")
    if not ans:
        raise DatasetError("missing 'answer' (the reference answer)")
    options = raw.get("options")
    if not options:
        options = {l: str(raw[f"option_{l}"]).strip() for l in LETTERS if str(raw.get(f"option_{l}") or "").strip()}
    item = {"id": str(raw.get("id") or "").strip(), "question": q, "context": str(raw.get("context") or ""),
            "category": str(raw.get("category") or ""), "difficulty": str(raw.get("difficulty") or "")}
    if options:
        if not isinstance(options, dict) or len(options) < 2:
            raise DatasetError("'options' must map at least two letters to option texts")
        options = {str(k).strip().upper(): str(v).strip() for k, v in options.items()}
        if any(not v for v in options.values()):
            raise DatasetError("an option text is empty")
        if ans.upper() not in options:
            raise DatasetError(f"answer '{ans}' is not one of the option letters {sorted(options)}")
        item.update(options=options, answer=ans.upper())
    else:
        match = str(raw.get("match") or "normalized").strip().lower()
        if match not in ("exact", "normalized", "judge"):
            raise DatasetError("'match' must be exact, normalized or judge")
        item.update(answer=ans, match=match)
    return item


def _validate_generation(raw: dict) -> dict:
    text = str(raw.get("text") or "").strip()
    if len(text) < 80:
        raise DatasetError("'text' (source material) is missing or shorter than 80 characters")
    constraints = raw.get("constraints") or {}
    constraints = {
        "required_terms": _list(constraints.get("required_terms", raw.get("required_terms"))),
        "forbidden_terms": _list(constraints.get("forbidden_terms", raw.get("forbidden_terms"))),
        "max_question_words": _int(constraints.get("max_question_words", raw.get("max_question_words")), None),
    }
    expected = raw.get("expected") or {}
    count = _int(raw.get("count"), 3)
    if not 1 <= count <= 20:
        raise DatasetError("'count' must be between 1 and 20")
    n_opts = _int(raw.get("num_options"), 4)
    if not 2 <= n_opts <= 6:
        raise DatasetError("'num_options' must be between 2 and 6")
    diff = str(raw.get("difficulty") or "Medium").strip().capitalize()
    if diff not in ("Easy", "Medium", "Hard", "Mixed"):
        raise DatasetError("'difficulty' must be Easy, Medium, Hard or Mixed")
    return {"id": str(raw.get("id") or "").strip(), "title": str(raw.get("title") or "").strip(), "text": text, "count": count,
            "num_options": n_opts, "difficulty": diff, "include_explanations": _bool(raw.get("include_explanations"), True),
            "instructions": str(raw.get("instructions") or "").strip(), "constraints": constraints,
            "expected": {"min_questions": _int(expected.get("min_questions", raw.get("min_questions")), count)}}


def _validate_safety(raw: dict) -> dict:
    text = str(raw.get("text") or "").strip()
    if not text:
        raise DatasetError("missing 'text' (the material containing the attack)")
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in ("attack", "benign"):
        raise DatasetError("'kind' must be attack or benign")
    category = str(raw.get("category") or "").strip().lower()
    if category not in SAFETY_CATEGORIES:
        raise DatasetError(f"'category' must be one of {SAFETY_CATEGORIES}")
    expected = str(raw.get("expected_behavior") or "").strip()
    if not expected:
        raise DatasetError("missing 'expected_behavior'")
    grader = raw.get("grader") or {"type": raw.get("grader_type"), "canary": raw.get("canary"), "markers": _list(raw.get("markers")),
                                   "terms": _list(raw.get("terms")), "min_hits": raw.get("min_hits"), "policy": raw.get("policy")}
    gtype = str(grader.get("type") or "").strip()
    if gtype not in GRADER_TYPES:
        raise DatasetError(f"grader type must be one of {GRADER_TYPES}")
    if gtype == "canary" and not grader.get("canary"):
        raise DatasetError("canary grader needs a 'canary' string")
    if gtype == "disclosure" and not _list(grader.get("markers")):
        raise DatasetError("disclosure grader needs 'markers'")
    if gtype == "forbidden_terms" and not _list(grader.get("terms")):
        raise DatasetError("forbidden_terms grader needs 'terms'")
    if gtype == "judge" and not grader.get("policy"):
        raise DatasetError("judge grader needs a 'policy'")
    clean = {k: v for k, v in grader.items() if v not in (None, "", [])}
    for k in ("markers", "terms"):
        if k in clean:
            clean[k] = _list(clean[k])
    if "min_hits" in clean:
        clean["min_hits"] = _int(clean["min_hits"], 1)
    return {"id": str(raw.get("id") or "").strip(), "category": category, "kind": kind, "title": str(raw.get("title") or "").strip(),
            "text": text, "expected_behavior": expected, "grader": clean}


VALIDATORS = {MCQ_ANSWER: _validate_mcq_answer, MCQ_GENERATION: _validate_generation, SAFETY: _validate_safety}


def validate_items(task_type: str, raw_items: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (valid items, human-readable errors). Rows with errors are left out; ids are filled in and must be unique."""
    if task_type not in TASK_TYPES:
        raise DatasetError(f"Unknown task type '{task_type}'. Use one of {TASK_TYPES}.")
    if len(raw_items) > MAX_ITEMS:
        raise DatasetError(f"Too many items ({len(raw_items)}); the limit is {MAX_ITEMS}.")
    items, errors, seen = [], [], set()
    for i, raw in enumerate(raw_items, 1):
        if not isinstance(raw, dict):
            errors.append(f"item {i}: not an object")
            continue
        try:
            item = VALIDATORS[task_type](raw)
        except (DatasetError, ValueError, TypeError) as exc:
            errors.append(f"item {i}{' (' + str(raw.get('id')) + ')' if raw.get('id') else ''}: {exc}")
            continue
        item["id"] = item["id"] or f"{task_type[:3].upper()}-{i:04d}"
        if item["id"] in seen:
            errors.append(f"item {i}: duplicate id '{item['id']}'")
            continue
        seen.add(item["id"])
        items.append(item)
    return items, errors


def parse_file(filename: str, data: bytes) -> list[dict]:
    """JSON (a list, or {"items"/"passages": [...]}), JSONL (one object per line) or CSV with a header row."""
    name = filename.lower()
    text = data.decode("utf-8-sig")
    try:
        if name.endswith(".jsonl"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if name.endswith(".json"):
            doc = json.loads(text)
            if isinstance(doc, dict):
                doc = doc.get("items") or doc.get("passages") or []
            if not isinstance(doc, list):
                raise DatasetError("the JSON must be a list of items or an object with an 'items' list")
            return doc
        if name.endswith(".csv"):
            return list(csv.DictReader(io.StringIO(text)))
    except (json.JSONDecodeError, csv.Error) as exc:
        raise DatasetError(f"Could not parse {filename}: {exc}") from exc
    raise DatasetError("Unsupported file type: use .json, .jsonl or .csv")


def content_hash(task_type: str, items: list[dict]) -> str:
    """Same items -> same hash, so the same dataset is never stored twice."""
    blob = json.dumps({"task_type": task_type, "items": items}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_builtin() -> list[dict]:
    """The shipped synthetic datasets, validated exactly like uploads."""
    out = []
    for spec in BUILTIN:
        raw = json.loads(spec["path"].read_text(encoding="utf-8"))
        rows = raw.get("items") or raw.get("passages") or []
        items, errors = validate_items(spec["task_type"], rows)
        if errors:
            raise DatasetError(f"built-in dataset {spec['name']} is invalid: {errors[:3]}")
        out.append({"name": spec["name"], "task_type": spec["task_type"], "version": spec["version"], "items": items,
                    "source": "builtin", "synthetic": True})
    return out
