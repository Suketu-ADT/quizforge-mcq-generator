"""Turn SciQ rows and teacher-written questions into chat-format training examples for the app's exact task.

Every example is a conversation the app itself would have with the model:
    system    = services.question_generator.SYSTEM_PROMPT            (unchanged)
    user      = build_user_prompt(request, n, material, section, [])  (the app's own prompt builder)
    assistant = {"questions": [...]}                                  (JSON in the app's schema)
and every target question must pass the app's validator (services.question_validator.validate_question) against the
material, including the verbatim source-excerpt check. Anything that fails is dropped, so the model never learns from
output the app would reject.

SciQ (Welbl et al., 2017; allenai/sciq, CC BY-NC 3.0) rows look like
    {"question", "correct_answer", "distractor1", "distractor2", "distractor3", "support"}
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from models.schemas import OPTION_LETTERS, GenerationRequest, Question
from services.question_generator import SYSTEM_PROMPT, build_user_prompt
from services.question_validator import QuestionValidationError, is_duplicate, validate_question

MAX_EXCERPT = 250  # rule 10 of the system prompt
MIN_SUPPORT_CHARS = 80
SCHEMA_FIELDS = ("question", "options", "correct_answer", "explanation", "difficulty", "topic", "source_excerpt")
BANNED_OPTIONS = ("all of the above", "none of the above", "both a and b")
STYLES = ["Conceptual", "Factual", "Application-based", "Mixed"]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def pick_excerpt(support: str, question: str, answer: str) -> str | None:
    """The support sentence that best backs the answer: it must contain the answer text when any sentence does,
    and among those the one sharing most words with the question. Must be copied verbatim and <= 250 characters."""
    cands = [s for s in sentences(support) if len(s) <= MAX_EXCERPT and len(s.split()) >= 4]
    if not cands:
        return None
    with_answer = [s for s in cands if answer.lower() in s.lower()]
    pool = with_answer or cands
    q_words = _words(question) | _words(answer)
    best = max(pool, key=lambda s: (len(_words(s) & q_words), -len(s)))
    if not with_answer and not (_words(best) & _words(answer)):
        return None  # nothing in the passage visibly supports the answer: not a trustworthy target
    return best


def explanation_from(excerpt: str) -> str:
    first = excerpt[0].lower() + excerpt[1:] if len(excerpt) > 1 and excerpt[1].islower() else excerpt
    return f"The material states that {first.rstrip()}"


def clean_row(row: dict) -> dict | None:
    q = str(row.get("question") or "").strip()
    ans = str(row.get("correct_answer") or "").strip()
    dis = [str(row.get(f"distractor{i}") or "").strip() for i in (1, 2, 3)]
    support = str(row.get("support") or "").strip()
    if not q or not ans or len(support) < MIN_SUPPORT_CHARS or any(not d for d in dis):
        return None
    texts = [ans.lower()] + [d.lower() for d in dis]
    if len(set(texts)) != 4 or any(b in t for t in texts for b in BANNED_OPTIONS):
        return None
    return {"question": q, "answer": ans, "distractors": dis, "support": support}


def sciq_question(row: dict, rng: random.Random, num_options: int, difficulty: str, topic: str,
                  explanations: bool) -> dict | None:
    excerpt = pick_excerpt(row["support"], row["question"], row["answer"])
    if excerpt is None:
        return None
    texts = [row["answer"]] + rng.sample(row["distractors"], num_options - 1)
    rng.shuffle(texts)
    letters = OPTION_LETTERS[:num_options]
    return {
        "question": row["question"],
        "options": [{"id": letter, "text": t} for letter, t in zip(letters, texts)],
        "correct_answer": letters[texts.index(row["answer"])],
        "explanation": explanation_from(excerpt) if explanations else "",
        "difficulty": difficulty,
        "topic": topic,
        "source_excerpt": excerpt,
    }


def make_example(material: str, questions: list[dict], req: GenerationRequest, section: str = "Part 1",
                 source: str = "") -> dict | None:
    """Validate every target question exactly like the app does; drop the example if any question fails."""
    checked: list[Question] = []
    for raw in questions:
        try:
            q = validate_question(raw, num_options=req.num_options, source_text=material, require_excerpt=True)
        except QuestionValidationError:
            return None
        if is_duplicate(q, checked):
            return None
        checked.append(q)
    if not checked:
        return None
    target = {"questions": [{k: q.model_dump()[k] for k in SCHEMA_FIELDS} for q in checked]}
    if not req.include_explanations:
        for q in target["questions"]:
            q["explanation"] = ""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(req, len(checked), material, section, [])},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "source": source,
    }


def build_sciq_examples(rows: list[dict], n_examples: int, seed: int = 7, max_per_example: int = 3) -> list[dict]:
    """Group 1-3 SciQ rows into one passage (their support paragraphs) and ask for one question per row.

    Variety mirrors the app's settings: difficulty Easy/Medium, every question style, 4 options (sometimes 3),
    explanations on (sometimes off, which teaches the 'leave explanation empty' instruction)."""
    rng = random.Random(seed)
    pool = [r for r in (clean_row(r) for r in rows) if r]
    rng.shuffle(pool)
    out, i = [], 0
    while len(out) < n_examples and i < len(pool):
        k = rng.randint(1, max_per_example)
        group = pool[i:i + k]
        i += k
        if len({g["support"] for g in group}) != len(group):
            continue
        num_options = 3 if rng.random() < 0.1 else 4
        difficulty = rng.choice(["Easy", "Medium"])
        explanations = rng.random() >= 0.1
        # SciQ has no topic labels: no subject line in the prompt and the app's own default label ("General", which
        # the app replaces with the section name); specific topics are learnt from the teacher's examples instead
        req = GenerationRequest(count=len(group), difficulty=difficulty, style=rng.choice(STYLES), num_options=num_options,
                                include_explanations=explanations)
        qs = [sciq_question(g, rng, num_options, difficulty, "General", explanations) for g in group]
        if any(q is None for q in qs):
            continue
        material = "\n\n".join(g["support"] for g in group)
        ex = make_example(material, qs, req, source="sciq")
        if ex:
            out.append(ex)
    return out


def distilled_example(material: str, questions: list[Question], req: GenerationRequest, section: str = "Part 1") -> dict | None:
    """A teacher-written set (already accepted by the app's generator) as a training example."""
    raws = [{k: q.model_dump()[k] for k in SCHEMA_FIELDS} for q in questions]
    return make_example(material, raws, req.model_copy(update={"count": len(raws)}), section, source="distilled")


def split(examples: list[dict], val_fraction: float = 0.05, seed: int = 7) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    items = list(examples)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_fraction)) if len(items) > 1 else 0
    return items[n_val:], items[:n_val]


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(examples: list[dict]) -> dict:
    by_source: dict[str, int] = {}
    questions = 0
    for ex in examples:
        by_source[ex.get("source", "")] = by_source.get(ex.get("source", ""), 0) + 1
        questions += len(json.loads(ex["messages"][-1]["content"])["questions"])
    return {"examples": len(examples), "questions": questions, "by_source": by_source}
