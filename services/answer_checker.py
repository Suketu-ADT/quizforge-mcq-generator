"""Optional second opinion on answer keys.

The model is shown each question WITHOUT the stored key (plus the supporting excerpt, when there is one) and
answers it blind. If it picks a different option than the key, the question is flagged for human review.

Agreement does not prove a key is correct (the same model wrote it), and disagreement does not prove it is
wrong - it is only a cheap signal for where to look first.
"""
from __future__ import annotations

from typing import Callable

from models.schemas import Question
from services.ollama_client import OllamaClient
from services.question_validator import MalformedOutputError, parse_model_json

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

SYSTEM = (
    "You answer multiple-choice questions. Use the supplied context when it is given. "
    'Reply ONLY with JSON like {"answer": "B"} where the value is the id of the single best option.'
)


def build_check_prompt(question: str, options: list[tuple[str, str]], excerpt: str = "") -> str:
    """The user message of the blind answer check (also used to build fine-tuning examples in the same format)."""
    lines = "\n".join(f"{oid}. {text}" for oid, text in options)
    context = f"Context: {excerpt}\n\n" if excerpt else ""
    return f"{context}Question: {question}\n\nOptions:\n{lines}"


def check_question(client: OllamaClient, q: Question) -> str:
    """Return "agrees", "disagrees" or "unchecked" (when the reply could not be used)."""
    prompt = build_check_prompt(q.question, [(o.id, o.text) for o in q.options], q.source_excerpt)
    reply = client.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        schema=SCHEMA,
        temperature=0.0,
        num_ctx=4096,
    )
    try:
        picked = str(parse_model_json(reply).get("answer", "")).strip().upper()[:1]
    except MalformedOutputError:
        return "unchecked"
    if picked not in {o.id for o in q.options}:
        return "unchecked"
    return "agrees" if picked == q.correct_answer else "disagrees"


def check_questions(
    client: OllamaClient,
    questions: list[Question],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Question]:
    """Return copies of *questions* with `key_check` filled in. Connection errors propagate."""
    out: list[Question] = []
    for i, q in enumerate(questions):
        out.append(q.model_copy(update={"key_check": check_question(client, q)}))
        if on_progress:
            on_progress(i + 1, len(questions))
    return out
