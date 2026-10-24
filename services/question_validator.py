"""Validation of model output: safe JSON parsing, normalisation, schema checks,
source traceability and duplicate detection.

The LLM is never trusted: every question must pass `validate_question` before
it can be shown, stored or scored.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from pydantic import ValidationError

from models.schemas import OPTION_LETTERS, Question


class MalformedOutputError(ValueError):
    """The model reply could not be parsed as the expected JSON."""


class QuestionValidationError(ValueError):
    """A single question failed validation (message explains why)."""


# --------------------------------------------------------------- JSON parsing
def parse_model_json(text: str) -> dict:
    """Parse the model reply safely (json.loads only - never eval)."""
    if not text or not text.strip():
        raise MalformedOutputError("The model returned an empty reply.")
    raw = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            data = {"questions": data}
        if isinstance(data, dict):
            return data
    raise MalformedOutputError("The model reply was not valid JSON.")


# ---------------------------------------------------------------- normalising
_PREFIX_RE = re.compile(r"^\s*(?:\(?[A-Ha-h]\)|[A-Ha-h][.):])\s+")


def _strip_prefix(text: str) -> str:
    return _PREFIX_RE.sub("", text).strip()


_TAG_RE = re.compile(r"</?\s*(?:material|source_text)\s*>", re.IGNORECASE)
_INLINE_OPTIONS_RE = re.compile(r"(?:^|\s)\(?A[.)]\s.+?\s\(?B[.)]\s", re.DOTALL)


def strip_prompt_tags(text: str) -> str:
    """Small models sometimes echo our <material> delimiters; remove them."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()


def normalize_raw_question(raw: dict, num_options: int | None = None) -> dict:
    """Map common model variations onto our schema without inventing content."""
    if not isinstance(raw, dict):
        raise QuestionValidationError("Question entry is not an object.")
    q = dict(raw)
    q["question"] = strip_prompt_tags(str(q.get("question") or q.get("text") or ""))

    opts = q.get("options")
    norm_opts: list[dict] = []
    if isinstance(opts, dict):  # {"A": "...", "B": "..."}
        opts = [{"id": k, "text": v} for k, v in opts.items()]
    if not isinstance(opts, list):
        raise QuestionValidationError("Options are missing.")
    for i, o in enumerate(opts):
        if isinstance(o, str):
            norm_opts.append({"id": OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else str(i), "text": _strip_prefix(strip_prompt_tags(o))})
        elif isinstance(o, dict):
            oid = str(o.get("id") or (OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else i)).strip().upper()
            norm_opts.append({"id": oid, "text": _strip_prefix(strip_prompt_tags(str(o.get("text", ""))))})
        else:
            raise QuestionValidationError("An option has an unexpected format.")
    q["options"] = norm_opts

    ans = str(q.get("correct_answer", "")).strip()
    ids = [o["id"] for o in norm_opts]
    if ans.upper() not in ids:
        # the model returned the option text itself (optionally with a "B." prefix)
        hits = [o["id"] for o in norm_opts if o["text"].strip().lower() == _strip_prefix(ans).lower()]
        m = re.match(r"^\(?([A-Ha-h])(?:[).:]|\)|$)", ans)  # "B." / "B) text" / "(B)"
        if len(hits) == 1:
            ans = hits[0]
        elif m and m.group(1).upper() in ids:
            ans = m.group(1)
    q["correct_answer"] = ans
    for key in ("explanation", "topic", "source_excerpt"):
        q[key] = strip_prompt_tags(str(q.get(key) or ""))
    diff = str(q.get("difficulty") or "").strip().capitalize()
    q["difficulty"] = diff if diff in ("Easy", "Medium", "Hard") else "Medium"
    if not q.get("id"):
        q.pop("id", None)  # let the schema assign a fresh id
    return q


# ------------------------------------------------------------- traceability
def _norm_for_match(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9%$.\-' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def excerpt_in_source(excerpt: str, source: str, threshold: float = 0.8) -> bool:
    """True when *excerpt* is (nearly) contiguous text from *source*.

    Exact substring after whitespace/case/quote normalisation, otherwise the
    longest matching block must cover >= threshold of the excerpt.
    """
    ex, src = _norm_for_match(excerpt), _norm_for_match(source)
    if not ex:
        return False
    if ex in src:
        return True
    if len(ex) < 15:
        return False
    match = SequenceMatcher(None, src, ex, autojunk=False).find_longest_match(0, len(src), 0, len(ex))
    return match.size / len(ex) >= threshold


# ---------------------------------------------------------------- validation
def validate_question(
    raw: dict | Question,
    num_options: int | None = None,
    source_text: str | None = None,
    require_excerpt: bool = False,
) -> Question:
    """Return a validated Question or raise QuestionValidationError."""
    try:
        if isinstance(raw, Question):
            question = Question.model_validate(raw.model_dump())
        else:
            question = Question.model_validate(normalize_raw_question(raw, num_options))
    except ValidationError as exc:
        msgs = "; ".join(e["msg"].replace("Value error, ", "") for e in exc.errors())
        raise QuestionValidationError(msgs) from exc

    if _INLINE_OPTIONS_RE.search(question.question):
        raise QuestionValidationError("the options were written inside the question text")
    if num_options is not None and len(question.options) != num_options:
        raise QuestionValidationError(
            f"expected {num_options} options but got {len(question.options)}"
        )
    if source_text:
        if question.source_excerpt:
            if not excerpt_in_source(question.source_excerpt, source_text):
                raise QuestionValidationError("source excerpt was not found in the supplied material")
        elif require_excerpt:
            raise QuestionValidationError("no supporting source excerpt was given")
    return question


# ---------------------------------------------------------------- duplicates
def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def questions_similar(a: str, b: str, threshold: float = 0.82) -> bool:
    na, nb = " ".join(sorted(_tokens(a))), " ".join(sorted(_tokens(b)))
    if not na or not nb:
        return False
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb)
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return jaccard >= threshold or ratio >= threshold


def is_duplicate(question: Question, existing: list[Question]) -> bool:
    return any(questions_similar(question.question, e.question) for e in existing)


def find_duplicate_pairs(questions: list[Question]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if questions_similar(questions[i].question, questions[j].question):
                pairs.append((i, j))
    return pairs
