"""USE-CASE TEST: how well does the model do the job this project exists for - turning source material into
validated multiple-choice questions?

For every passage the project's REAL generator (`QuestionGenerator`: prompts, JSON schema, validator, repair loop)
is asked for N questions. We then measure:

  delivery rate, acceptance rate, valid-JSON rate, LLM calls per accepted question, rejection breakdown
  (ungrounded / duplicate / bad format), seconds per accepted question, generation tokens/s,
  and answer-key quality:  blind-answer agreement by the same model and by a second "judge" model,
  plus - once a human has labelled the keys - key accuracy and how well the checks catch wrong keys.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

from evaluation.common import Metric, f, ratio, wilson_interval
from models.schemas import GenerationRequest
from services.answer_checker import check_question
from services.evaluation.evaluators import classify_rejection  # noqa: F401  (shared with the app's engine)
from services.ollama_client import OllamaError
from services.question_generator import QuestionGenerator

ROOT = Path(__file__).resolve().parent
DEFAULT_PASSAGES = ROOT / "passages.json"
LABEL_COLUMN = "human_key_correct"


def load_passages(path: Path = DEFAULT_PASSAGES) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    passages = data["passages"] if isinstance(data, dict) else data
    if not passages:
        raise ValueError("No passages found.")
    return passages


# ------------------------------------------------------------------------------------------ classifying rejections
# ------------------------------------------------------------------------------------------------------- running
def run_generation_suite(
    client, passages: list[dict], count: int = 3, judge_client=None, num_options: int = 4, log=print
) -> dict:
    """Generate `count` questions from each passage with the real pipeline, then blind-check every accepted key."""
    out_passages = []
    for i, passage in enumerate(passages, 1):
        client.reset_usage()
        log(f"  [{i}/{len(passages)}] {passage['id']} '{passage['title']}': generating {count} questions...")
        req = GenerationRequest(topic=passage["title"], count=count, difficulty="Medium", num_options=num_options)
        t0 = time.monotonic()
        error = ""
        try:
            result = QuestionGenerator(client).generate(passage["text"], req)
            questions, rejected, warnings = result.questions, result.rejected, result.warnings
        except OllamaError as exc:  # keep going: one failed passage should not lose the others
            questions, rejected, warnings, error = [], [], [], str(exc)
            log(f"      generation failed: {exc}")
        seconds = time.monotonic() - t0
        usage = client.reset_usage()
        log(f"      -> {len(questions)}/{count} accepted, {len(rejected)} rejected, {usage['calls']} model call(s), {seconds:.0f}s")

        checked = []
        for q in questions:
            row = q.model_dump()
            row["self_check"] = _safe_check(client, q, log)
            row["judge_check"] = _safe_check(judge_client, q, log) if judge_client is not None else "not run"
            checked.append(row)
        out_passages.append({
            "id": passage["id"], "title": passage["title"], "requested": count, "accepted": len(questions),
            "rejected": list(rejected), "warnings": list(warnings), "seconds": seconds, "usage": usage,
            "error": error, "questions": checked,
        })
    return {
        "model": client.model, "judge": getattr(judge_client, "model", None), "count_per_passage": count,
        "num_options": num_options, "timestamp": datetime.now().isoformat(timespec="seconds"), "passages": out_passages,
    }


def _safe_check(client, question, log) -> str:
    try:
        return check_question(client, question)
    except OllamaError as exc:
        log(f"      (answer check skipped: {exc})")
        return "unchecked"


# ------------------------------------------------------------------------------------------------------- metrics
def generation_metrics(gen: dict) -> list[Metric]:
    ps = gen["passages"]
    requested = sum(p["requested"] for p in ps)
    accepted = sum(p["accepted"] for p in ps)
    reasons = [r for p in ps for r in p["rejected"]]
    kinds = {k: sum(classify_rejection(r) == k for r in reasons) for k in ("json", "grounding", "duplicate", "format")}
    question_rejections = kinds["grounding"] + kinds["duplicate"] + kinds["format"]
    candidates = accepted + question_rejections
    calls = sum(p["usage"]["calls"] for p in ps)
    seconds = sum(p["seconds"] for p in ps)
    gen_tokens = sum(p["usage"]["gen_tokens"] for p in ps)
    gen_seconds = sum(p["usage"]["gen_seconds"] for p in ps)
    m: list[Metric] = []

    m.append(Metric(
        "Delivery rate", ratio(accepted, requested), "higher", "accepted questions / questions requested",
        [f"requested = {len(ps)} passages x {gen['count_per_passage']} = {requested}, accepted = {accepted}",
         f"Delivery rate = {accepted} / {requested} = {f(ratio(accepted, requested))}"],
        note="Below 1.0 means the app had to report a shortfall to the user."))
    m.append(Metric(
        "Acceptance rate", ratio(accepted, candidates), "higher",
        "accepted / (accepted + rejected candidate questions)",
        [f"accepted = {accepted}; rejected candidates = grounding {kinds['grounding']} + duplicate {kinds['duplicate']} + format {kinds['format']} = {question_rejections}",
         f"Acceptance rate = {accepted} / ({accepted} + {question_rejections}) = {f(ratio(accepted, candidates))}"],
        note="How many of the questions the model wrote survived validation."))
    ok_calls = max(calls - kinds["json"], 0)
    m.append(Metric(
        "Valid-JSON rate", ratio(ok_calls, calls), "higher", "replies that parsed as the required JSON / total replies",
        [f"model replies = {calls}, unusable JSON replies = {kinds['json']}",
         f"Valid-JSON rate = ({calls} - {kinds['json']}) / {calls} = {f(ratio(ok_calls, calls))}"]))
    m.append(Metric(
        "Model calls per accepted question", ratio(calls, accepted), "lower", "total model replies / accepted questions",
        [f"Calls per accepted question = {calls} / {accepted} = {f(ratio(calls, accepted))}"],
        note="1 batch of up to 3 questions is one call, so the ideal is about 0.33; higher means repairs were needed."))
    m.append(Metric(
        "Ungrounded-candidate rate", ratio(kinds["grounding"], candidates), "lower",
        "candidates whose quoted source sentence was not found in the text / candidates",
        [f"ungrounded = {kinds['grounding']}, candidates = {accepted} + {question_rejections} = {candidates}",
         f"Rate = {kinds['grounding']} / {candidates} = {f(ratio(kinds['grounding'], candidates))}"],
        note="Every ACCEPTED question is grounded by construction; this shows how often the model invented support."))
    m.append(Metric(
        "Duplicate-candidate rate", ratio(kinds["duplicate"], candidates), "lower", "duplicate candidates / candidates",
        [f"Rate = {kinds['duplicate']} / {candidates} = {f(ratio(kinds['duplicate'], candidates))}"]))
    m.append(Metric(
        "Format-defect rate", ratio(kinds["format"], candidates), "lower",
        "candidates with a broken structure (wrong option count, invalid answer, options inside the question...) / candidates",
        [f"Rate = {kinds['format']} / {candidates} = {f(ratio(kinds['format'], candidates))}"]))
    m.append(Metric(
        "Seconds per accepted question", ratio(seconds, accepted), "lower", "total generation time / accepted questions",
        [f"total generation time = {f(seconds, 1)}s (all passages, repairs included)",
         f"Seconds per question = {f(seconds, 1)} / {accepted} = {f(ratio(seconds, accepted), 2)}"]))
    m.append(Metric(
        "Generation speed (tokens/s)", ratio(gen_tokens, gen_seconds), "higher", "eval_count / (eval_duration / 1e9)",
        [f"tokens generated = {gen_tokens}, generation time = {f(gen_seconds, 1)}s (from Ollama's own counters)",
         f"Tokens per second = {gen_tokens} / {f(gen_seconds, 1)} = {f(ratio(gen_tokens, gen_seconds), 2)}"]))

    qs = [q for p in ps for q in p["questions"]]
    for key, label, who in (("self_check", "Blind-answer agreement (same model)", "the same model"),
                            ("judge_check", "Blind-answer agreement (judge model)", f"judge model {gen.get('judge')}")):
        decided = [q for q in qs if q[key] in ("agrees", "disagrees")]
        if not decided:
            continue
        agree = sum(q[key] == "agrees" for q in decided)
        lo, hi = wilson_interval(agree, len(decided))
        m.append(Metric(
            label, ratio(agree, len(decided)), "higher", "questions where the checker's blind answer equals the stored key / questions checked",
            [f"checker = {who}; it answers each question WITHOUT seeing the key (only the supporting sentence as context)",
             f"agree = {agree}, checked = {len(decided)}",
             f"Agreement = {agree} / {len(decided)} = {f(ratio(agree, len(decided)))}   (95% interval [{f(lo, 3)}, {f(hi, 3)}])"],
            note="A hint, not proof: disagreement flags a key for human review; agreement does not prove it is right."))
    return m


# ------------------------------------------------------------------------------------------------------- reporting
def rejection_breakdown(gen: dict) -> list[str]:
    reasons = [r for p in gen["passages"] for r in p["rejected"]]
    if not reasons:
        return ["no candidates were rejected"]
    from collections import Counter

    return [f"{n:>3} x {reason}" for reason, n in Counter(reasons).most_common()]


def passage_table(gen: dict) -> list[str]:
    lines = [f"{'passage':<34}{'accepted':>9}{'rejected':>9}{'calls':>7}{'seconds':>9}"]
    for p in gen["passages"]:
        lines.append(f"{p['id'] + ' ' + p['title']:<34}{str(p['accepted']) + '/' + str(p['requested']):>9}{len(p['rejected']):>9}{p['usage']['calls']:>7}{p['seconds']:>9.0f}")
    return lines


def questions_text(gen: dict) -> list[str]:
    out = []
    for p in gen["passages"]:
        out.append(f"{p['id']} - {p['title']}")
        for i, q in enumerate(p["questions"], 1):
            out.append(f"  {i}. {q['question']}")
            for o in q["options"]:
                mark = "*" if o["id"] == q["correct_answer"] else " "
                out.append(f"     {mark} {o['id']}. {o['text']}")
            out.append(f"     key={q['correct_answer']}  same-model check: {q['self_check']}  judge check: {q['judge_check']}")
    return out


# --------------------------------------------------------------------------------- human labels: export and scoring
def labels_csv(gen: dict) -> str:
    """CSV for a human reviewer: fill the 'human_key_correct' column with yes / no."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    n_opts = gen.get("num_options", 4)
    letters = list("ABCDEF"[:n_opts])
    w.writerow(["model", "passage_id", "no", "question", *[f"option_{l}" for l in letters], "key", "source_excerpt",
                "self_check", "judge_check", LABEL_COLUMN, "notes"])
    for p in gen["passages"]:
        for i, q in enumerate(p["questions"], 1):
            opts = {o["id"]: o["text"] for o in q["options"]}
            w.writerow([gen["model"], p["id"], i, q["question"], *[opts.get(l, "") for l in letters], q["correct_answer"],
                        q["source_excerpt"], q["self_check"], q["judge_check"], "", ""])
    return buf.getvalue()


_YES = {"yes", "y", "true", "1", "correct", "right"}
_NO = {"no", "n", "false", "0", "wrong", "incorrect"}


def score_labels(path: Path) -> tuple[list[Metric], list[str]]:
    """Key accuracy from the human labels, and how well each automatic check catches the wrong keys."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    labelled = []
    for r in rows:
        v = (r.get(LABEL_COLUMN) or "").strip().lower()
        if v in _YES or v in _NO:
            labelled.append((r, v in _YES))
    info = [f"{len(rows)} questions in the file, {len(labelled)} labelled by a human"]
    metrics: list[Metric] = []
    if not labelled:
        return metrics, info + [f"Fill the '{LABEL_COLUMN}' column with yes / no and run again."]
    n = len(labelled)
    right = sum(ok for _, ok in labelled)
    lo, hi = wilson_interval(right, n)
    metrics.append(Metric(
        "Key accuracy (human-labelled)", right / n, "higher", "answer keys a human judged correct / keys reviewed",
        [f"correct keys = {right}, reviewed = {n}", f"Key accuracy = {right} / {n} = {f(right / n)}   (95% interval [{f(lo, 3)}, {f(hi, 3)}])"],
        note="This is the real accuracy of the generated answer keys."))
    wrong = n - right
    for col, name in (("self_check", "same-model check"), ("judge_check", "judge-model check")):
        pairs = [(r[col].strip().lower(), ok) for r, ok in labelled if r.get(col, "").strip().lower() in ("agrees", "disagrees")]
        if not pairs:
            continue
        # classifier question: "is this key WRONG?"  predicted wrong = the check disagreed with the key
        tp = sum(1 for c, ok in pairs if c == "disagrees" and not ok)
        fp = sum(1 for c, ok in pairs if c == "disagrees" and ok)
        fn = sum(1 for c, ok in pairs if c == "agrees" and not ok)
        tn = sum(1 for c, ok in pairs if c == "agrees" and ok)
        m = len(pairs)
        prec, rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
        f1 = ratio(2 * prec * rec, prec + rec)
        acc = (tp + tn) / m
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / den if den else 0.0
        po = acc
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (m * m)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        head = f"{name}: positives = 'key is wrong'; TP={tp} (flagged & truly wrong), FP={fp}, FN={fn} (missed wrong key), TN={tn}"
        metrics += [
            Metric(f"Flag precision ({name})", prec, "higher", "TP / (TP + FP)", [head, f"Precision = {tp} / ({tp} + {fp}) = {f(prec)}"],
                   note="Of the keys it flagged, how many really were wrong."),
            Metric(f"Wrong-key recall ({name})", rec, "higher", "TP / (TP + FN)", [f"Recall = {tp} / ({tp} + {fn}) = {f(rec)}"],
                   note=f"Of the {wrong} wrong key(s), how many it caught." if wrong else "No wrong keys in the labelled set, so recall is not informative."),
            Metric(f"Flag F1 ({name})", f1, "higher", "2PR / (P + R)", [f"F1 = 2*{f(prec, 3)}*{f(rec, 3)} / ({f(prec, 3)}+{f(rec, 3)}) = {f(f1)}"]),
            Metric(f"Check accuracy ({name})", acc, "higher", "(TP + TN) / total", [f"Accuracy = ({tp} + {tn}) / {m} = {f(acc)}"]),
            Metric(f"MCC ({name})", mcc, "higher", "(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
                   [f"numerator = {tp}*{tn} - {fp}*{fn} = {tp * tn - fp * fn}", f"denominator = {f(den, 3)}", f"MCC = {f(mcc)}"]),
            Metric(f"Cohen's kappa ({name})", kappa, "higher", "(p_o - p_e) / (1 - p_e)",
                   [f"p_o = {f(po)}, p_e = {f(pe)}", f"kappa = ({f(po)} - {f(pe)}) / (1 - {f(pe)}) = {f(kappa)}"]),
        ]
    return metrics, info
