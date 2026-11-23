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


# ------------------------------------------------------------------- wording
_ASK_START_RE = re.compile(
    r"^(?:which|what|who|whom|whose|when|where|why|how|identify|select|choose|name|complete|determine|find|pick)\b",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"^(?:a|an|the) ")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _answer_core(text: str) -> str:
    return _ARTICLE_RE.sub("", _norm_for_match(text).rstrip(". ")).strip()


def _mentions(phrase: str, text: str) -> bool:
    return len(phrase) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


_OPENER_RE = re.compile(
    r"^(?:which|what|who|whom|whose|when|where|why|how|is|are|was|were|does|do|did|can|could|would|should|will|has|"
    r"have|had|identify|select|choose|name|complete|determine|find|pick|according|based|given|if|suppose|"
    r"(?:in|on|at|by|for|from|to|with|during|under|of|after|before|between|among|through|into|within)\s+"
    r"(?:which|what|whose|whom|how))\b",
    re.IGNORECASE,
)
_INTRO_RE = re.compile(r"^[^,:?]{1,60}[,:]\s+")  # "In plants, ..." / "During photosynthesis, ..." / "Case 2: ..."
_WH_INSIDE_RE = re.compile(r"\s(?:what|which|who|whom|whose|where|when|how)\b", re.IGNORECASE)


def is_echo_question(text: str) -> bool:
    """A statement with the question word moved into the middle ("ReAct combines what with acting?"): it reads like a
    sentence copied from the material with a gap, not like an exam question."""
    text = _SENTENCE_END_RE.split(text.strip())[-1]  # a scenario may come first: "A doctor sees ... Which kind ...?"
    if "__" in text or _OPENER_RE.match(text):
        return False
    rest = _INTRO_RE.sub("", text, count=1)
    return not _OPENER_RE.match(rest) and bool(_WH_INSIDE_RE.search(rest))


def check_wording(question: Question, source_text: str | None = None) -> None:
    """Reject model-written questions that do not ask anything, give the answer away, are a statement with "what"
    in the middle, or put a sentence copied from the material in front of the question. The messages go back to
    the model as retry feedback."""
    text = question.question.strip()
    if "?" not in text and "__" not in text and not _ASK_START_RE.match(text):
        raise QuestionValidationError("the question text does not ask anything: write one question that ends with '?'")

    asked = _norm_for_match(text)
    key = next((o.text for o in question.options if o.id == question.correct_answer), "")
    others = [_answer_core(o.text) for o in question.options if o.id != question.correct_answer]
    if _mentions(_answer_core(key), asked) and not any(_mentions(o, asked) for o in others):  # "X or Y?" names all
        raise QuestionValidationError("the question gives away the answer: the correct option's words appear in the question")

    if source_text:
        for lead in _SENTENCE_END_RE.split(text)[:-1]:
            if len(lead.split()) >= 6 and excerpt_in_source(lead, source_text, threshold=0.9):
                raise QuestionValidationError(
                    "the question starts with a sentence copied from the material: ask the question on its own"
                )
    if is_echo_question(text):
        raise QuestionValidationError(
            "the question is a statement with 'what' in the middle: start it with a question word (What, Which, How...)"
        )


# ------------------------------------------------------------------ casing
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*\*?")
_AFTER_BREAK_RE = re.compile(r"(?:[.!?:]\s+|\n\s*(?:#+\s*|[-*]\s+)?)$")
_TEXT_START_RE = re.compile(r"\s*(?:#+\s*|[-*]\s+)?")


def _casing_unknown(text: str, start: int) -> bool:
    """A capital here may only be grammar: the word starts a sentence, heading or bullet, or is the second part of a
    hyphenated name ("Tree-of-Thought")."""
    before = text[max(0, start - 12):start]
    if before.endswith("-"):
        return True
    return bool(_AFTER_BREAK_RE.search(before)) or (start <= 12 and _TEXT_START_RE.fullmatch(text[:start]) is not None)


def _upgrades(word: str, form: str) -> bool:
    """*form* is *word* with more capitals (never fewer): "react" -> "ReAct", "React" -> "ReAct", "bfs" -> "BFS"."""
    return (form != word and form.lower() == word.lower()
            and all(f.isupper() for w, f in zip(word, form) if w.isupper())
            and any(f.isupper() and not w.isupper() for w, f in zip(word, form)))


_OPENERS = frozenset(
    "in on at as so but if then because however although though while since after before during this there each every "
    "all some many most both thus also finally first second third next unlike like".split()
)


def _is_acronym_like(word: str) -> bool:
    """ReAct, CoT, BFS, ATP, A*: capitals that are part of the name, not grammar."""
    return any(c.isupper() for c in word[1:]) or (len(word) >= 2 and word.isupper())


def _title_phrases(source_text: str) -> set[str]:
    """Multi-word names written in title case: "Breadth First Search", "Chemical Weapons Convention"."""
    lower_words = set(re.findall(r"(?<![A-Za-z])[a-z][a-z0-9]*", source_text))
    phrases = set()
    for run in re.finditer(r"[A-Z][A-Za-z0-9]*(?:[ ][A-Z][A-Za-z0-9]*)+", source_text):
        words = run.group(0).split(" ")
        # a sentence opener is not part of the name: "The Calvin", "Because ReAct"
        while words and (words[0].lower() in _STOPWORDS | _OPENERS or words[0].lower() in lower_words):
            words.pop(0)
        if len(words) >= 2:
            phrases.add(" ".join(words))
    return phrases


def source_casing(source_text: str) -> dict[str, str]:
    """{lower-case name: how the material writes it} for names the material always capitalises the same way:
    acronym-like words (ReAct, CoT, BFS, A*), multi-word title-case names (Breadth First Search), and single names
    capitalised in the middle of a sentence that the material never writes in lower case ("the Calvin cycle").
    A capital that may be grammar (sentence start, after a hyphen) or one word of a longer name ("Search" in
    "Breadth First Search") does not count."""
    out: dict[str, str] = {}
    variants: dict[str, set[str]] = {}
    for m in _WORD_RE.finditer(source_text):
        word = m.group(0)
        if _is_acronym_like(word) or not _casing_unknown(source_text, m.start()):
            variants.setdefault(word.lower(), set()).add(word)
    phrases = _title_phrases(source_text)
    in_phrases = {w.lower() for p in phrases for w in p.split(" ")}
    lower_words = set(re.findall(r"(?<![A-Za-z])[a-z][a-z0-9]*", source_text))
    for low, forms in variants.items():
        form = next(iter(forms))
        if len(forms) != 1:
            continue
        if _is_acronym_like(form):
            out[low] = form
        elif (form[0].isupper() and len(form) >= 3 and low not in lower_words and low not in in_phrases
              and low not in _STOPWORDS and low not in _OPENERS):
            out[low] = form
    for phrase in phrases:
        seen = {m.group(0) for m in re.finditer(rf"(?i)(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", source_text)}
        if all(s[1:] == phrase[1:] for s in seen):  # only a sentence-start capital may differ
            out[phrase.lower()] = phrase
    return out


def restore_casing(text: str, casing: dict[str, str]) -> str:
    """Write names the way the material does: models (fine-tuned ones in particular) often lower-case them. Only ever
    adds capitals ("breadth first search (bfs)" -> "Breadth First Search (BFS)")."""
    if not casing or not text:
        return text
    for low in sorted((k for k in casing if " " in k), key=len, reverse=True):
        text = re.sub(rf"(?i)(?<![A-Za-z0-9]){re.escape(low)}(?![A-Za-z0-9])",
                      lambda m, f=casing[low]: f if _upgrades(m.group(0), f) else m.group(0), text)
    return _WORD_RE.sub(lambda m: casing[m.group(0).lower()] if _upgrades(m.group(0), casing.get(m.group(0).lower(), ""))
                        else m.group(0), text)


# ---------------------------------------------------------------- validation
def validate_question(
    raw: dict | Question,
    num_options: int | None = None,
    source_text: str | None = None,
    require_excerpt: bool = False,
    check_wording_rules: bool = False,
) -> Question:
    """Return a validated Question or raise QuestionValidationError.

    *check_wording_rules* adds the checks for model-written questions (see check_wording); questions a teacher typed
    or edited are not held to them."""
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
    if check_wording_rules:
        check_wording(question, source_text)
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


_STOPWORDS = frozenset(
    "a an the of to in on at by for with and or is are was were be been it its this that these those what which who "
    "whom whose when where why how do does did can could would should will has have had not".split()
)


def content_tokens(s: str) -> set[str]:
    """The words that carry meaning: lower case, no hyphens, no plural "s", no stop words."""
    words = re.findall(r"[a-z0-9]+", s.lower().replace("-", ""))  # "Re-Act" and "ReAct" are the same word
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOPWORDS}


def answer_text(q: Question) -> str:
    return next((o.text for o in q.options if o.id == q.correct_answer), "")


def same_fact(a: Question, b: Question) -> bool:
    """Two differently worded questions that test the same fact: their questions are about the same things, and they
    share the answer (one key contains the other) or each names what the other asks for (asked in reverse).
    Also: the same answer taken from the same sentence of the material, however differently the questions are put."""
    ka, kb = _answer_core(answer_text(a)), _answer_core(answer_text(b))
    ea, eb = _norm_for_match(a.source_excerpt), _norm_for_match(b.source_excerpt)
    if ka and ka == kb and ea and eb and (ea in eb or eb in ea):
        return True
    ta, tb = content_tokens(a.question), content_tokens(b.question)
    if not ta or not tb or len(ta & tb) / min(len(ta), len(tb)) < 0.6:
        return False
    ka, kb = _answer_core(answer_text(a)), _answer_core(answer_text(b))
    if ka and kb and (ka == kb or _mentions(ka, kb) or _mentions(kb, ka)):
        return True
    return _mentions(ka, _norm_for_match(b.question)) or _mentions(kb, _norm_for_match(a.question))


def _duplicates(a: Question, b: Question) -> bool:
    return questions_similar(a.question, b.question) or same_fact(a, b)


def is_duplicate(question: Question, existing: list[Question]) -> bool:
    return any(_duplicates(question, e) for e in existing)


def find_duplicate_pairs(questions: list[Question]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if _duplicates(questions[i], questions[j]):
                pairs.append((i, j))
    return pairs
