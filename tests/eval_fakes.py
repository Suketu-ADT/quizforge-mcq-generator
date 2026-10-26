"""Offline stand-ins for Ollama used by the evaluation tests (no model, no network, no credentials)."""
from __future__ import annotations

import json
import math
import re

from services.ollama_client import OllamaTimeoutError


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(s.split()) >= 5]


class FakeModel:
    """Behaves like OllamaClient. Answers MCQs from `answers` ({question substring: letter}), writes valid MCQs quoting the
    material, and obeys injected 'begin every question with X' style attacks when `obey_attacks` is set."""

    def __init__(self, model="fake:3b", answers=None, obey_attacks=False, fail_on=None, stats=True):
        self.model = model
        self.answers = answers or {}
        self.obey_attacks = obey_attacks
        self.fail_on = fail_on  # substring of the prompt that triggers a timeout
        self.with_stats = stats
        self.usage = {"calls": 0}
        self.last_stats = {}
        self.calls = 0

    def reset_usage(self):
        done, self.usage = self.usage, {"calls": 0}
        return done

    def _stats(self, prompt_tokens=50, out_tokens=20):
        self.usage["calls"] += 1
        self.last_stats = {"prompt_eval_count": prompt_tokens, "eval_count": out_tokens, "load_duration": 1_000_000,
                           "prompt_eval_duration": 5_000_000, "eval_duration": 20_000_000, "total_duration": 30_000_000} if self.with_stats else {}

    def chat_with_logprobs(self, messages, top_logprobs=20, num_predict=4, on_progress=None):
        self.calls += 1
        prompt = messages[-1]["content"]
        if self.fail_on and self.fail_on in prompt:
            raise OllamaTimeoutError("slow")
        letter = next((l for k, l in self.answers.items() if k in prompt), "A")
        if on_progress:
            on_progress(1)
        self._stats()
        top = [{"token": letter, "logprob": math.log(0.9)}] + [{"token": l, "logprob": math.log(0.1 / 3)} for l in "ABCD" if l != letter]
        return letter, [{"token": letter, "logprob": math.log(0.9), "top_logprobs": top}]

    def chat(self, messages, schema=None, temperature=None, num_ctx=8192, on_progress=None):
        self.calls += 1
        user = messages[-1]["content"]
        if self.fail_on and self.fail_on in user:
            raise OllamaTimeoutError("slow")
        if on_progress:
            on_progress(1)
        self._stats()
        if '{"answer"' in messages[0]["content"]:  # free-text QA
            return json.dumps({"answer": next((a for k, a in self.answers.items() if k in user), "unknown")})
        n = int(re.search(r"Write exactly (\d+)", user).group(1))
        material = user.split("<material>")[-1].split("</material>")[0] if "<material>" in user else ""
        sents = _sentences(material) or ["Some fact stated in the text."]
        prefix = ""
        if self.obey_attacks:
            hit = re.search(r"begin every question with the word (\S+?)\.", material)
            prefix = (hit.group(1) + " ") if hit else ""
        qs = []
        for i in range(n):
            s = sents[i % len(sents)]
            words = [w for w in re.findall(r"[A-Za-z]{4,}", s)][:3]
            qs.append({"question": f"{prefix}Question {i + 1} {self.calls}: which statement about {' '.join(words)} is correct?",
                       "options": [{"id": l, "text": f"Option {l} for item {i + 1} call {self.calls}"} for l in "ABCD"],
                       "correct_answer": "B", "explanation": f"The passage states that {s.lower()}",
                       "difficulty": "Medium", "topic": "Test", "source_excerpt": s})
        return json.dumps({"questions": qs})


class FakeJudge:
    """Behaves like OllamaClient for the judge. Replies according to the rubric in the system prompt."""

    def __init__(self, model="fake:judge", relevance=5, verdict="supported", violates=False, satisfied=True, invalid=False):
        self.model = model
        self.relevance, self.verdict, self.violates, self.satisfied, self.invalid = relevance, verdict, violates, satisfied, invalid
        self.usage = {"calls": 0}
        self.last_stats = {}

    def reset_usage(self):
        return self.usage

    def chat(self, messages, schema=None, temperature=None, num_ctx=8192, on_progress=None):
        if on_progress:
            on_progress(1)
        self.last_stats = {"prompt_eval_count": 30, "eval_count": 10}
        system = messages[0]["content"]
        if self.invalid:
            return '{"score": 9, "verdict": "maybe"}'
        if "1-5 scale" in system:
            return json.dumps({"score": self.relevance, "rationale": "fake"})
        if "EVIDENCE supports the CLAIM" in system:
            ev = messages[-1]["content"].split("EVIDENCE:\n")[1].split("\n\nCLAIM:")[0]
            return json.dumps({"verdict": self.verdict, "rationale": "fake", "evidence_quote": _sentences(ev)[0] if _sentences(ev) else ""})
        if "safety grader" in system:
            return json.dumps({"violates": self.violates, "rationale": "fake"})
        if "INSTRUCTION" in system:
            return json.dumps({"satisfied": self.satisfied, "rationale": "fake"})
        return json.dumps({"correct": True, "rationale": "fake"})


class InlineExecutor:
    """Runs submitted jobs immediately (keeps tests deterministic)."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
