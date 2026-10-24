"""Shared test helpers (no Ollama needed)."""
from __future__ import annotations

import json

from models.schemas import Option, Question, Quiz


def make_question(n: int = 1, correct: str = "B", difficulty: str = "Medium", topic: str = "Biology") -> Question:
    return Question(
        question=f"What is the answer to sample question number {n}?",
        options=[Option(id=l, text=f"Choice {l} for question {n}") for l in "ABCD"],
        correct_answer=correct,
        explanation=f"Because choice {correct} is right ({n}).",
        difficulty=difficulty,
        topic=topic,
        source_excerpt="Photosynthesis converts light energy into chemical energy.",
    )


def make_quiz(n: int = 4) -> Quiz:
    diffs = ["Easy", "Medium", "Hard", "Medium"]
    topics = ["Cells", "Energy", "Cells", "Genetics"]
    qs = [make_question(i + 1, "ABCD"[i % 4], diffs[i % 4], topics[i % 4]) for i in range(n)]
    return Quiz(title="Biology basics", subject="Biology", questions=qs, source_name="notes.txt", model="qwen2.5:3b")


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def iter_lines(self):
        for line in self._lines:
            yield line.encode() if isinstance(line, str) else line

    def close(self):
        pass

    _lines: list = []

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Mimics requests.Session for OllamaClient tests."""

    def __init__(self, get=None, post=None):
        self._get, self._post = get, post
        self.posts: list[dict] = []

    def get(self, url, timeout=None):
        return self._resolve(self._get, url)

    def post(self, url, json=None, timeout=None, stream=False):
        self.posts.append(json)
        return self._resolve(self._post, url)

    @staticmethod
    def _resolve(handler, url):
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            out = handler(url)
            if isinstance(out, Exception):
                raise out
            return out
        return handler


def chat_reply(content: str) -> FakeResponse:
    return FakeResponse(200, {"message": {"role": "assistant", "content": content}})


QUESTION_BANK = [
    "What does photosynthesis convert light energy into?",
    "Where inside a plant cell does photosynthesis take place?",
    "Which pigment captures sunlight in chloroplasts?",
    "What gas is released as a by-product when water is split?",
    "Which cycle fixes carbon dioxide into glucose?",
    "Why do mitochondria break down glucose?",
    "Which organelle stores genetic information in eukaryotes?",
    "How do enzymes change the rate of chemical reactions?",
    "What distinguishes prokaryotic cells from eukaryotic cells?",
    "Which molecule carries amino acids to the ribosome?",
    "How does osmosis move water across a membrane?",
    "What is the function of the Golgi apparatus?",
]


def raw_question_json(n: int, excerpt: str, num_options: int = 4) -> dict:
    return {
        "question": QUESTION_BANK[(n - 1) % len(QUESTION_BANK)],
        "options": [{"id": l, "text": f"Candidate statement {l} about topic {n}"} for l in "ABCDEF"[:num_options]],
        "correct_answer": "A",
        "explanation": "The material says so.",
        "difficulty": "Medium",
        "topic": "Photosynthesis",
        "source_excerpt": excerpt,
    }


def stream_reply(pieces: list[str]) -> FakeResponse:
    import json as _json

    resp = FakeResponse(200, {})
    lines = [_json.dumps({"message": {"role": "assistant", "content": p}, "done": False}) for p in pieces]
    lines.append(_json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
    resp._lines = lines
    return resp
