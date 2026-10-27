"""Teacher data: the app's real generator, driven by the Quality model, writes questions; only filtered ones are kept.

For every passage (2-4 SciQ support paragraphs from a split the SciQ examples do not use):
  1. QuestionGenerator(teacher).generate(...)  - the app's own pipeline: schema-constrained JSON, validator
     (verbatim excerpt, one correct option, no duplicates, ...), bounded repair.
  2. answer_checker.check_question(checker, q) - a second model answers each question blind; only questions whose key
     it "agrees" with are kept (a cheap guard against wrong answer keys).
  3. The surviving questions become one training example (prepare_data.distilled_example, re-validated).

Run where Ollama has both models (the Colab notebook installs it):
    python -m finetune.distill --rows data/sciq_validation.jsonl --out data/distilled.jsonl --passages 80
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from finetune.prepare_data import clean_row, distilled_example, read_jsonl, summary, write_jsonl
from models.schemas import GenerationRequest
from services.answer_checker import check_question
from services.ollama_client import OllamaClient, OllamaError
from services.question_generator import QuestionGenerator

STYLES = ["Conceptual", "Factual", "Application-based", "Mixed"]
DIFFICULTIES = ["Easy", "Medium", "Hard", "Mixed"]


def make_passages(rows: list[dict], n: int, seed: int = 11) -> list[str]:
    rng = random.Random(seed)
    supports = list(dict.fromkeys(r["support"] for r in (clean_row(r) for r in rows) if r))
    rng.shuffle(supports)
    out, i = [], 0
    while len(out) < n and i < len(supports):
        k = rng.randint(2, 4)
        out.append("\n\n".join(supports[i:i + k]))
        i += k
    return out


def distill(passages: list[str], teacher, checker, seed: int = 11, log=print) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    gen = QuestionGenerator(teacher)
    examples, stats = [], {"passages": 0, "generated": 0, "agreed": 0, "disagreed": 0, "unchecked": 0, "failed_passages": 0}
    for i, material in enumerate(passages, 1):
        req = GenerationRequest(count=rng.randint(2, 3), difficulty=rng.choice(DIFFICULTIES), style=rng.choice(STYLES))
        t0 = time.monotonic()
        stats["passages"] += 1
        try:
            result = gen.generate(material, req)
        except (OllamaError, ValueError) as exc:
            stats["failed_passages"] += 1
            log(f"  [{i}/{len(passages)}] generation failed: {exc}")
            continue
        kept = []
        for q in result.questions:
            stats["generated"] += 1
            try:
                verdict = check_question(checker, q)
            except OllamaError:
                verdict = "unchecked"
            stats[{"agrees": "agreed", "disagrees": "disagreed"}.get(verdict, "unchecked")] += 1
            if verdict == "agrees":
                kept.append(q)
        ex = distilled_example(material, kept, req) if kept else None
        if ex:
            examples.append(ex)
        log(f"  [{i}/{len(passages)}] {len(result.questions)} generated, {len(kept)} kept, {time.monotonic() - t0:.0f}s")
    return examples, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True, help="SciQ rows (JSON lines) to take passages from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passages", type=int, default=80)
    ap.add_argument("--teacher", default="qwen2.5:7b")
    ap.add_argument("--checker", default="qwen2.5:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args(argv)

    teacher = OllamaClient(args.base_url, args.teacher, temperature=0.4, timeout_s=args.timeout)
    checker = OllamaClient(args.base_url, args.checker, temperature=0.0, timeout_s=args.timeout)
    for client in (teacher, checker):
        status = client.status()
        if not status.connected or not status.model_installed:
            raise SystemExit(f"Ollama at {args.base_url} must be running with '{client.model}' installed.")
    passages = make_passages(read_jsonl(args.rows), args.passages)
    print(f"Teacher {args.teacher}, checker {args.checker}, {len(passages)} passages")
    examples, stats = distill(passages, teacher, checker)
    write_jsonl(args.out, examples)
    print(json.dumps({"stats": stats, "dataset": summary(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
