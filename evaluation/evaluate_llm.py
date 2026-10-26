#!/usr/bin/env python
r"""Evaluate the local LLM (Ollama) the way this project uses it - and SHOW every calculation.

THREE TESTS (--suite all runs all three):

  1. generation  (the project's real use case)
     The project's own question generator is asked to write MCQs from source passages. We measure delivery rate,
     acceptance rate, valid-JSON rate, rejections (ungrounded / duplicate / bad format), seconds per question,
     tokens/s, and answer-key quality (each key is re-answered blind by the same model and by a second "judge"
     model). A labels_*.csv is saved so a human can mark each key yes/no; --score-labels then gives the real key
     accuracy and how well the checks catch wrong keys.

  2. answering   (a labelled benchmark)
     The model answers 40 questions whose correct option is known. This gives Accuracy, Exact Match, Precision,
     Recall, F1 (macro / micro / weighted), Top-k, MCC, Cohen's kappa, log loss, Brier score and ECE, computed from
     Ollama's token log-probabilities.

  3. metrics     (the 12 performance metrics)
     The evaluation engine runs the labelled questions, the passages and an 11-case safety suite through the model and
     records every request. Reports Answer Accuracy, F1, Answer Relevance, Faithfulness, Hallucination Rate,
     Instruction-Following, Safety Attack Success Rate, Task Success Rate, TTFT p95, End-to-End Latency p95,
     Cost per Successful Task and Error Rate - each with its status and sample count ("n/a" instead of a made-up 0).
     The judge (relevance / faithfulness / hallucination) is the other model when two are given.

HOW TO RUN (Windows) - the file must be started from the project folder, or with its full path:
    cd "C:\Users\Arpit singh\OneDrive\Desktop\GEN AI\mcq-quiz-generator"
    .venv\Scripts\python.exe evaluation\evaluate_llm.py --models qwen2.5:3b qwen2.5:7b
or simply double-click  run_evaluation.bat  in the project folder.

OTHER EXAMPLES
    ... --suite generation --models qwen2.5:3b --max-passages 1     # quick check of the use-case test
    ... --suite answering --limit 10                                # quick check of the benchmark
    ... --suite metrics --metrics-limit 3                           # quick check of the 12 metrics (3 items per dataset)
    ... --score-labels evaluation\results\qwen2.5_3b\labels_<time>.csv   # after a human filled in yes/no
    ... --predictions evaluation\results\qwen2.5_3b\report_<time>.json   # recompute offline from a saved report
    ... --demo                                                      # SIMULATED model, shows the maths without Ollama

Reports are saved to  evaluation\results\<model name>\report_<time>.md / .json  (plus labels_<time>.csv),
and evaluation\results\comparison_<time>.md when two or more models are given.
Only the project's own packages are used (no scikit-learn needed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:  # lets "python evaluation/evaluate_llm.py" import the project's own modules
    sys.path.insert(0, str(PROJECT))

from evaluation.common import (  # noqa: E402  (after the sys.path line above)
    Metric, f, markdown_calculations, markdown_table, print_calculations, print_summary, val, wilson_interval,
)
from services.evaluation.evaluators import parse_letter, probs_from_logprobs  # noqa: E402,F401  (shared with the app's engine)
from services.evaluation.stats import UNPARSED, confusion, per_class_counts, prf  # noqa: E402,F401
DEFAULT_DATASET = ROOT / "dataset.json"
RESULTS_DIR = ROOT / "results"
EPS = 1e-15  # probabilities are clipped to [EPS, 1] before taking logs
N_BINS = 10  # equal-width confidence bins for ECE
NONE_LABEL = UNPARSED  # "the answer could not be parsed" - counted as a wrong prediction

try:  # optional: pick up OLLAMA_* settings from the project's .env
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
except ImportError:  # pragma: no cover
    pass

SYSTEM_PROMPT = (
    "You are taking a multiple-choice test. Choose the single best option. "
    "Reply with ONLY the letter of the correct option (for example: B). Do not explain."
)


# =============================================================================================
# Data structures
# =============================================================================================
@dataclass
class Item:
    id: str
    question: str
    options: dict[str, str]
    answer: str
    context: str = ""
    category: str = ""
    difficulty: str = ""


@dataclass
class Prediction:
    id: str
    gold: str
    pred: str | None  # parsed option letter, or None when the reply was not a valid option
    probs: dict[str, float]  # probability of each option letter (sums to 1)
    raw: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    category: str = ""
    difficulty: str = ""
    prob_source: str = "logprobs"  # "logprobs", "uniform (parse failure)" or "simulated"

    @property
    def correct(self) -> bool:
        return self.pred == self.gold

    @property
    def confidence(self) -> float:
        return max(self.probs.values())


# =============================================================================================
# Dataset
# =============================================================================================
def load_dataset(path: Path) -> list[Item]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    out = []
    for raw in items:
        item = Item(**{k: raw.get(k, "") for k in ("id", "question", "options", "answer", "context", "category", "difficulty")})
        if item.answer not in item.options:
            raise ValueError(f"{item.id}: answer '{item.answer}' is not one of the options {list(item.options)}")
        out.append(item)
    if not out:
        raise ValueError("The dataset is empty.")
    return out


def labels_of(items: list[Item]) -> list[str]:
    return sorted({letter for it in items for letter in it.options})


def build_prompt(item: Item) -> str:
    parts = []
    if item.context:
        parts.append(f"Passage: {item.context}")
    parts.append(f"Question: {item.question}")
    parts.append("\n".join(f"{letter}. {text}" for letter, text in item.options.items()))
    parts.append("Answer (one letter):")
    return "\n\n".join(parts)


# =============================================================================================
# Talking to Ollama
# =============================================================================================
def uniform(letters: list[str]) -> dict[str, float]:
    return {letter: 1.0 / len(letters) for letter in letters}


def ask_model(session, base_url: str, model: str, item: Item, timeout: int) -> Prediction:
    letters = list(item.options)
    payload = {
        "model": model,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 20,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 4},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(item)},
        ],
    }
    resp = session.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    if resp.status_code == 404:
        raise SystemExit(f"Model '{model}' is not installed. Run:  ollama pull {model}")
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    found = probs_from_logprobs(data.get("logprobs") or [], letters)
    pred = parse_letter(content, letters)
    if found and pred is None:
        pred = found[0]
    if found:
        probs, source = found[1], "logprobs"
    else:
        probs, source = uniform(letters), "uniform (no letter token)"
    prompt_n = int(data.get("prompt_eval_count") or 0)
    prompt_s = (data.get("prompt_eval_duration") or 0) / 1e9
    latency = ((data.get("total_duration") or 0) - (data.get("load_duration") or 0)) / 1e9  # exclude model load
    return Prediction(
        id=item.id, gold=item.answer, pred=pred, probs=probs, raw=content, latency_s=latency,
        prompt_tokens=prompt_n, prompt_tps=(prompt_n / prompt_s if prompt_s else 0.0),
        category=item.category, difficulty=item.difficulty, prob_source=source,
    )


def measure_generation_speed(session, base_url: str, model: str, timeout: int) -> dict:
    """tokens/s = eval_count / (eval_duration / 1e9), on a ~128-token free-text generation."""
    payload = {
        "model": model, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 128},
        "messages": [{"role": "user", "content": "Explain in simple words how photosynthesis works."}],
    }
    data = session.post(f"{base_url}/api/chat", json=payload, timeout=timeout).json()
    n, secs = int(data.get("eval_count") or 0), (data.get("eval_duration") or 0) / 1e9
    return {"eval_count": n, "eval_seconds": secs, "tokens_per_second": (n / secs if secs else 0.0)}


def run_model(model: str, items: list[Item], base_url: str, timeout: int, bench: bool) -> dict:
    import requests

    session = requests.Session()
    try:
        session.get(f"{base_url}/api/version", timeout=5).raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot reach Ollama at {base_url} ({exc}). Start it and try again.")
    print(f"\n=== Running {model} on {len(items)} questions ===")
    preds: list[Prediction] = []
    for i, item in enumerate(items, 1):
        try:
            pred = ask_model(session, base_url, model, item, timeout)
        except requests.RequestException as exc:
            print(f"  [{i:>2}/{len(items)}] {item.id} request failed: {exc} - counted as an unparsed answer")
            pred = Prediction(item.id, item.answer, None, uniform(list(item.options)), "", 0.0, 0, 0.0,
                              item.category, item.difficulty, "uniform (request failed)")
        preds.append(pred)
        mark = "ok " if pred.correct else "WRONG"
        print(f"  [{i:>2}/{len(items)}] {item.id} gold={pred.gold} pred={pred.pred or '?'} "
              f"p(gold)={pred.probs.get(pred.gold, 0):.3f} {mark} {pred.latency_s:.1f}s")
    speed = measure_generation_speed(session, base_url, model, timeout) if bench else None
    return {"model": model, "predictions": [asdict(p) for p in preds], "speed": speed,
            "timestamp": datetime.now().isoformat(timespec="seconds"), "simulated": False}


def simulate_model(items: list[Item], accuracy: float = 0.8, seed: int = 11) -> dict:
    """A fake model for demonstrating the maths without Ollama. Its numbers are NOT real LLM results."""
    rng = random.Random(seed)
    preds = []
    for item in items:
        letters = list(item.options)
        correct = rng.random() < accuracy
        pick = item.answer if correct else rng.choice([l for l in letters if l != item.answer])
        conf = min(0.99, max(0.35, rng.gauss(0.8 if correct else 0.65, 0.12)))
        rest = [l for l in letters if l != pick]
        probs = {pick: conf, **{l: (1 - conf) / len(rest) for l in rest}}
        preds.append(Prediction(item.id, item.answer, pick, probs, pick, 1.0, 100, 100.0, item.category,
                                item.difficulty, "simulated"))
    return {"model": f"SIMULATED (target accuracy {accuracy:.0%})", "predictions": [asdict(p) for p in preds],
            "speed": None, "timestamp": datetime.now().isoformat(timespec="seconds"), "simulated": True}


# =============================================================================================
# Metrics - each returns the value AND the working
# =============================================================================================
def normalise_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def compute_metrics(preds: list[Prediction], labels: list[str]) -> list[Metric]:
    n = len(preds)
    golds = [p.gold for p in preds]
    yhat = [p.pred for p in preds]
    k_cls = len(labels)
    cm = confusion(golds, yhat, labels)
    counts = per_class_counts(cm, labels)
    metrics: list[Metric] = []

    # ---- Accuracy ----------------------------------------------------------------------
    correct = sum(p.correct for p in preds)
    acc = correct / n
    lo, hi = wilson_interval(correct, n)
    metrics.append(Metric(
        "Accuracy", acc, "higher", "correct predictions / total predictions",
        [f"correct = {correct}, N = {n}", f"Accuracy = {correct} / {n} = {f(acc)}",
         f"95% Wilson confidence interval = [{f(lo, 3)}, {f(hi, 3)}] (with only {n} questions the result is uncertain by about +/- {f((hi - lo) / 2, 3)})",
         f"For reference: random guessing among {k_cls} options = {f(1 / k_cls, 3)}"]))

    # ---- Exact match -------------------------------------------------------------------
    em = sum(1 for p in preds if p.pred is not None and normalise_text(p.pred) == normalise_text(p.gold))
    metrics.append(Metric(
        "Exact Match (EM)", em / n, "higher", "predictions identical to the expected answer string / total",
        [f"EM compares the normalised answer string with the expected one: {em} exact matches out of {n}",
         f"EM = {em} / {n} = {f(em / n)}"],
        note="Answers here are single letters, so EM equals accuracy; unparseable replies count as non-matches."))

    # ---- Per-class precision / recall / F1 --------------------------------------------
    per = {}
    work = []
    for k in labels:
        c = counts[k]
        p_, r_, f1_ = prf(c["tp"], c["fp"], c["fn"])
        per[k] = (p_, r_, f1_)
        work.append(f"Class {k}: TP={c['tp']} FP={c['fp']} FN={c['fn']}  ->  P = {c['tp']}/({c['tp']}+{c['fp']}) = {f(p_)}, "
                    f"R = {c['tp']}/({c['tp']}+{c['fn']}) = {f(r_)}, F1 = 2*{f(p_, 3)}*{f(r_, 3)}/({f(p_, 3)}+{f(r_, 3)}) = {f(f1_)}")
    macro_p = sum(v[0] for v in per.values()) / k_cls
    macro_r = sum(v[1] for v in per.values()) / k_cls
    macro_f1 = sum(v[2] for v in per.values()) / k_cls
    metrics.append(Metric(
        "Precision (macro)", macro_p, "higher", "TP / (TP + FP), computed per option letter then averaged",
        work + [f"Macro precision = ({' + '.join(f(per[k][0], 3) for k in labels)}) / {k_cls} = {f(macro_p)}"]))
    metrics.append(Metric(
        "Recall (macro)", macro_r, "higher", "TP / (TP + FN), computed per option letter then averaged",
        [f"Macro recall = ({' + '.join(f(per[k][1], 3) for k in labels)}) / {k_cls} = {f(macro_r)}"]))
    metrics.append(Metric(
        "F1-score (per class)", macro_f1, "higher", "2 * P * R / (P + R)",
        [f"F1_{k} = {f(per[k][2])}   (support = {counts[k]['support']} questions)" for k in labels]))

    # ---- Macro / Micro / Weighted F1 ---------------------------------------------------
    metrics.append(Metric(
        "Macro-F1", macro_f1, "higher", "mean of per-class F1 (every class counts equally)",
        [f"Macro-F1 = ({' + '.join(f(per[k][2], 3) for k in labels)}) / {k_cls} = {f(macro_f1)}"]))
    tp_t = sum(counts[k]["tp"] for k in labels)
    fp_t = sum(counts[k]["fp"] for k in labels)
    fn_t = sum(counts[k]["fn"] for k in labels)
    mp, mr, micro = prf(tp_t, fp_t, fn_t)
    metrics.append(Metric(
        "Micro-F1", micro, "higher", "F1 from globally pooled TP, FP, FN",
        [f"TP = {tp_t}, FP = {fp_t}, FN = {fn_t} (summed over all classes)",
         f"P = {tp_t}/({tp_t}+{fp_t}) = {f(mp)}, R = {tp_t}/({tp_t}+{fn_t}) = {f(mr)}",
         f"Micro-F1 = 2*P*R/(P+R) = {f(micro)}"],
        note="With one answer per question and no unparsed replies, micro-F1 equals accuracy."))
    supports = {k: counts[k]["support"] for k in labels}
    weighted = sum(per[k][2] * supports[k] for k in labels) / n
    metrics.append(Metric(
        "Weighted-F1", weighted, "higher", "mean of per-class F1 weighted by support",
        [f"Weighted-F1 = ({' + '.join(f'{f(per[k][2], 3)}*{supports[k]}' for k in labels)}) / {n} = {f(weighted)}"]))

    # ---- Top-k accuracy ----------------------------------------------------------------
    topk_work = []
    first_topk = None
    for k in range(1, min(3, k_cls - 1) + 1):
        hits = 0
        for p in preds:
            if p.prob_source.startswith("uniform"):
                continue  # no information about the ranking -> counted as a miss
            ranked = sorted(p.probs, key=lambda l: (-p.probs[l], l))
            hits += p.gold in ranked[:k]
        topk_work.append(f"Top-{k} accuracy = {hits} / {n} = {f(hits / n)}  (gold letter within the {k} most probable options)")
        if k == 2:
            first_topk = hits / n
    metrics.append(Metric(
        "Top-k accuracy", first_topk if first_topk is not None else acc, "higher",
        "fraction of questions whose correct option is among the k most probable", topk_work,
        note="Value shown is top-2. Top-1 equals accuracy."))

    # ---- MCC ---------------------------------------------------------------------------
    cols = labels + [NONE_LABEL]
    t_k = {c: sum(cm[c].values()) if c in cm else 0 for c in cols}
    p_k = {c: sum(cm[t][c] for t in labels) for c in cols}
    c_tr = sum(cm[k][k] for k in labels)
    s = n
    sum_pt = sum(p_k[c] * t_k[c] for c in cols)
    sum_pp = sum(p_k[c] ** 2 for c in cols)
    sum_tt = sum(t_k[c] ** 2 for c in cols)
    den = math.sqrt(max(s * s - sum_pp, 0)) * math.sqrt(max(s * s - sum_tt, 0))
    mcc = (c_tr * s - sum_pt) / den if den else 0.0
    metrics.append(Metric(
        "Matthews Correlation Coefficient (MCC)", mcc, "higher",
        "(c*s - sum(p_k*t_k)) / sqrt((s^2 - sum p_k^2) * (s^2 - sum t_k^2))",
        [f"s = N = {s}, c = correctly predicted = {c_tr}",
         f"t_k (true counts)      = {dict((c, t_k[c]) for c in labels)}",
         f"p_k (predicted counts) = {dict((c, p_k[c]) for c in cols if c in labels or p_k[c])}",
         f"sum(p_k*t_k) = {sum_pt}, sum(p_k^2) = {sum_pp}, sum(t_k^2) = {sum_tt}",
         f"numerator = {c_tr}*{s} - {sum_pt} = {c_tr * s - sum_pt}",
         f"denominator = sqrt({s * s}-{sum_pp}) * sqrt({s * s}-{sum_tt}) = {f(den, 3)}",
         f"MCC = {f(mcc)}   (1 = perfect, 0 = no better than chance, -1 = always wrong)"]))

    # ---- Cohen's kappa -----------------------------------------------------------------
    po = c_tr / n
    pe = sum_pt / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    metrics.append(Metric(
        "Cohen's kappa", kappa, "higher", "(p_o - p_e) / (1 - p_e)",
        [f"p_o (observed agreement) = {c_tr}/{n} = {f(po)}",
         f"p_e (agreement expected by chance) = sum(p_k*t_k)/N^2 = {sum_pt}/{n * n} = {f(pe)}",
         f"kappa = ({f(po)} - {f(pe)}) / (1 - {f(pe)}) = {f(kappa)}"]))

    # ---- Log loss ----------------------------------------------------------------------
    terms = []
    for p in preds:
        p_true = min(max(p.probs.get(p.gold, 0.0), EPS), 1.0)
        terms.append(-math.log(p_true))
    logloss = sum(terms) / n
    shown = ", ".join(f(t, 3) for t in terms[:5])
    metrics.append(Metric(
        "Log loss (cross-entropy)", logloss, "lower", "-(1/N) * sum( ln p(correct option) )",
        [f"per-question loss = -ln(p assigned to the correct option); first five: {shown}{', ...' if n > 5 else ''}",
         f"sum of {n} losses = {f(sum(terms), 3)}",
         f"Log loss = {f(sum(terms), 3)} / {n} = {f(logloss)}",
         f"For reference: always answering with equal probabilities = ln({k_cls}) = {f(math.log(k_cls), 3)}"],
        note=f"Probabilities are clipped to at least {EPS:g} so a confident wrong answer gives a large but finite penalty."))

    # ---- Brier score -------------------------------------------------------------------
    briers = []
    for p in preds:
        briers.append(sum((p.probs.get(l, 0.0) - (1.0 if l == p.gold else 0.0)) ** 2 for l in labels))
    brier = sum(briers) / n
    metrics.append(Metric(
        "Brier score", brier, "lower", "(1/N) * sum over questions of sum over options (p_option - y_option)^2",
        [f"per-question score = squared error of the probability vector against the one-hot truth; first five: "
         f"{', '.join(f(b, 3) for b in briers[:5])}{', ...' if n > 5 else ''}",
         f"Brier = {f(sum(briers), 3)} / {n} = {f(brier)}",
         f"For reference: equal probabilities = 1 - 1/{k_cls} = {f(1 - 1 / k_cls, 3)}; perfect = 0; worst = 2"]))

    # ---- ECE ---------------------------------------------------------------------------
    bins = [[] for _ in range(N_BINS)]
    for p in preds:
        idx = min(int(math.ceil(p.confidence * N_BINS)) - 1, N_BINS - 1) if p.confidence > 0 else 0
        bins[max(idx, 0)].append(p)
    ece = 0.0
    rows = []
    for b, members in enumerate(bins):
        if not members:
            continue
        acc_b = sum(m.correct for m in members) / len(members)
        conf_b = sum(m.confidence for m in members) / len(members)
        gap = abs(acc_b - conf_b)
        w = len(members) / n
        ece += w * gap
        rows.append(f"bin ({b / N_BINS:.1f}, {(b + 1) / N_BINS:.1f}]: n={len(members):>2}  accuracy={f(acc_b, 3)}  "
                    f"confidence={f(conf_b, 3)}  |gap|={f(gap, 3)}  weight={len(members)}/{n}  contribution={f(w * gap, 4)}")
    metrics.append(Metric(
        "Expected Calibration Error (ECE)", ece, "lower", "sum over bins of (n_bin / N) * |accuracy_bin - confidence_bin|",
        ["confidence = probability of the predicted option; questions are grouped into 10 equal-width confidence bins"]
        + rows + [f"ECE = {f(ece)}   (0 = confidence always matches how often the model is right)"],
        note="A high-confidence model that is often wrong has a large ECE."))
    return metrics


def metrics_by_subset(preds: list[Prediction], key: str) -> list[tuple[str, int, int]]:
    groups: dict[str, list[Prediction]] = {}
    for p in preds:
        groups.setdefault(getattr(p, key) or "unknown", []).append(p)
    return [(g, sum(x.correct for x in v), len(v)) for g, v in sorted(groups.items())]


def speed_summary(run: dict) -> list[str]:
    preds = [Prediction(**p) for p in run["predictions"]]
    lats = [p.latency_s for p in preds if p.latency_s > 0]
    ptps = [p.prompt_tps for p in preds if p.prompt_tps > 0]
    lines = []
    if lats:
        lines.append(f"Mean time per question = sum of latencies / N = {sum(lats):.1f}s / {len(lats)} = {sum(lats) / len(lats):.2f}s "
                     f"(model load time excluded; median {sorted(lats)[len(lats) // 2]:.2f}s)")
    if ptps:
        lines.append(f"Prompt processing speed = prompt_tokens / prompt_eval_seconds, mean = {sum(ptps) / len(ptps):.1f} tokens/s")
    sp = run.get("speed")
    if sp and sp.get("tokens_per_second"):
        lines.append(f"Generation speed = eval_count / (eval_duration / 1e9) = {sp['eval_count']} / {sp['eval_seconds']:.2f}s = "
                     f"{sp['tokens_per_second']:.1f} tokens/s")
    return lines


# =============================================================================================
# Reporting
# =============================================================================================
def confusion_text(preds: list[Prediction], labels: list[str]) -> list[str]:
    cm = confusion([p.gold for p in preds], [p.pred for p in preds], labels)
    cols = labels + [NONE_LABEL]
    header = "true \\ predicted  " + "  ".join(f"{c:>3}" for c in cols)
    lines = [header]
    for t in labels:
        lines.append(f"{t:>16}  " + "  ".join(f"{cm[t][c]:>3}" for c in cols))
    return lines


def print_run_report(run: dict, labels: list[str], metrics: list[Metric]) -> None:
    preds = [Prediction(**p) for p in run["predictions"]]
    bar = "=" * 100
    print(f"\n{bar}\nMODEL: {run['model']}   questions: {len(preds)}   options per question: {len(labels)}   {run['timestamp']}")
    if run.get("simulated"):
        print("*** SIMULATED RESULTS - this is a fake model used to demonstrate the calculations, NOT a real LLM measurement ***")
    print(bar)
    print("\nCONFUSION MATRIX (rows = correct option, columns = what the model chose, '?' = unparsed reply)")
    for line in confusion_text(preds, labels):
        print("  " + line)
    uniform_n = sum(p.prob_source.startswith("uniform") for p in preds)
    if uniform_n:
        print(f"\n  note: {uniform_n} reply(ies) had no readable option letter; they count as wrong and use equal probabilities.")

    print("\nSUMMARY (higher is better unless marked)")
    print(f"  {'Metric':<42}{'Value':>9}   Better")
    for m in metrics:
        print(f"  {m.name:<42}{m.value:>9.4f}   {m.better}")

    print("\nBY SUBSET (accuracy = correct / questions)")
    for key in ("category", "difficulty"):
        for name, c, n in metrics_by_subset(preds, key):
            print(f"  {key:<10} {name:<12} {c}/{n} = {c / n:.3f}")

    print("\nSPEED")
    for line in speed_summary(run) or ["(no timing data in these predictions)"]:
        print("  " + line)

    print(f"\n{bar}\nTHE CALCULATIONS, STEP BY STEP\n{bar}")
    for m in metrics:
        print(f"\n[{m.name}]  better: {m.better}")
        print(f"  formula : {m.formula}")
        for line in m.working:
            print(f"  | {line}")
        print(f"  = {m.value:.4f}")
        if m.note:
            print(f"  note    : {m.note}")


def comparison_table(runs: list[dict], all_metrics: list[list[Metric]]) -> list[str]:
    names = [r["model"] for r in runs]
    width = max(14, *(len(n) + 2 for n in names))
    lines = [f"{'Metric':<42}" + "".join(f"{n:>{width}}" for n in names) + "   Better"]
    for i, m in enumerate(all_metrics[0]):
        row = f"{m.name:<42}" + "".join(f"{val(ms[i].value):>{width}}" for ms in all_metrics) + f"   {m.better}"
        lines.append(row)
    return lines


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:40]


def answering_markdown(run: dict, labels: list[str], metrics: list[Metric]) -> list[str]:
    preds = [Prediction(**p) for p in run["predictions"]]
    out = [f"## Question-answering benchmark ({len(preds)} labelled questions)", "",
           "The model answers questions whose correct option is known. This checks that it can read and reason, "
           "which the answer-key checker also relies on.", ""]
    if run.get("simulated"):
        out += ["> **SIMULATED RESULTS** - a fake model used to demonstrate the calculations; not a real measurement.", ""]
    out += markdown_table(metrics)
    out += ["", "**Confusion matrix** (rows = correct option, columns = chosen option, `?` = unreadable reply)", "", "```"]
    out += confusion_text(preds, labels) + ["```", "", "**Accuracy by subset**", ""]
    for key in ("category", "difficulty"):
        out += [f"- {key} `{name}`: {c}/{n} = {c / n:.3f}" for name, c, n in metrics_by_subset(preds, key)]
    out += ["", "**Speed**", ""] + [f"- {line}" for line in (speed_summary(run) or ["no timing data"])]
    return out + [""] + markdown_calculations(metrics) + [""]


def generation_markdown(gen: dict, metrics: list[Metric]) -> list[str]:
    from evaluation import generation_suite as gs

    out = [f"## Use-case test: generating questions from source material", "",
           f"The project's real generator was asked for {gen['count_per_passage']} questions from each of "
           f"{len(gen['passages'])} passages (model: `{gen['model']}`"
           + (f", judge: `{gen['judge']}`" if gen.get("judge") else "") + ").", ""]
    out += markdown_table(metrics)
    out += ["", "**Per passage**", "", "```"] + gs.passage_table(gen) + ["```", "", "**Why candidates were rejected**", "", "```"]
    out += gs.rejection_breakdown(gen) + ["```", "", "<details><summary>All accepted questions (for human review)</summary>", "", "```"]
    out += gs.questions_text(gen) + ["```", "", "</details>", ""]
    return out + markdown_calculations(metrics) + [""]


def bundle_markdown(bundle: dict, labels: list[str], ans_metrics, gen_metrics, met_metrics=()) -> str:
    out = [f"# LLM performance report - {bundle['model']}", "", f"Generated {bundle['timestamp']}  ", ""]
    if bundle.get("generation"):
        out += generation_markdown(bundle["generation"], gen_metrics)
    if bundle.get("answering"):
        out += answering_markdown(bundle["answering"], labels, ans_metrics)
    if bundle.get("metrics"):
        from evaluation import metrics_suite as ms

        out += ms.metrics_markdown(bundle["metrics"], list(met_metrics))
    out += ["## How to read this report", "",
            "- Every metric lists its formula and the numbers substituted into it (open the *Step-by-step calculations* blocks).",
            "- Small samples are uncertain; confidence intervals are shown where they apply.",
            "- Answer-key *correctness* needs a human: fill the `human_key_correct` column of the saved `labels_*.csv` with yes / no and run "
            "`python evaluation/evaluate_llm.py --score-labels <that file>`."]
    return "\n".join(out) + "\n"


def comparison_sections(all_gen: list, all_ans: list, all_met: list) -> tuple:
    return (("Use-case test: generating questions", all_gen), ("Question-answering benchmark", all_ans),
            ("12 performance metrics", all_met))


def comparison_markdown(bundles: list[dict], all_ans: list, all_gen: list, all_met: list | None = None) -> str:
    names = [b["model"] for b in bundles]
    out = [f"# Model comparison: {' vs '.join(names)}", "", f"Generated {datetime.now().isoformat(timespec='seconds')}", ""]
    for title, sets in comparison_sections(all_gen, all_ans, all_met or [[] for _ in bundles]):
        if not sets or any(not m for m in sets):
            continue
        out += [f"## {title}", "", "| Metric | " + " | ".join(names) + " | Better |", "|---|" + "---:|" * len(names) + "---|"]
        by_name = [{m.name: m for m in ms} for ms in sets]
        for m in sets[0]:
            cells = " | ".join(val(d[m.name].value) if m.name in d else "-" for d in by_name)
            out.append(f"| {m.name} | {cells} | {m.better} |")
        if title.startswith("12"):
            judges = [f"{b['model']} judged by {b['metrics'].get('judge') or 'nobody'}" for b in bundles if b.get("metrics")]
            out += ["", f"`n/a` = not enough data to compute it (the per-model report gives the reason). Judges: {'; '.join(judges)}."]
        out.append("")
    return "\n".join(out) + "\n"


def save_bundle(bundle: dict, labels: list[str], ans_metrics, gen_metrics, stamp: str, met_metrics=()) -> list[Path]:
    """Writes results/<model>/report_<time>.md + .json (+ labels_<time>.csv for human review)."""
    folder = RESULTS_DIR / slug(bundle["model"])
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / f"report_{stamp}"
    to_json = lambda ms: [{"name": m.name, "value": m.value, "better": m.better, "formula": m.formula} for m in ms]  # noqa: E731
    payload = {"labels": labels, "bundles": [{**bundle, "answering_metrics": to_json(ans_metrics), "generation_metrics": to_json(gen_metrics)}]}
    paths = []
    for suffix, content in ((".json", json.dumps(payload, indent=2, ensure_ascii=False)),
                            (".md", bundle_markdown(bundle, labels, ans_metrics, gen_metrics, met_metrics))):
        path = Path(f"{base}{suffix}")  # not with_suffix(): model tags contain dots ("qwen2.5")
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    if bundle.get("generation"):
        from evaluation import generation_suite as gs

        csv_path = folder / f"labels_{stamp}.csv"
        csv_path.write_text(gs.labels_csv(bundle["generation"]), encoding="utf-8-sig")
        paths.append(csv_path)
    return paths


def print_generation_report(gen: dict, metrics: list[Metric]) -> None:
    from evaluation import generation_suite as gs

    bar = "=" * 100
    print(f"\n{bar}\nUSE-CASE TEST - GENERATING QUESTIONS FROM SOURCE MATERIAL   model: {gen['model']}"
          + (f"   judge: {gen['judge']}" if gen.get("judge") else "") + f"\n{bar}")
    print("\nPER PASSAGE")
    for line in gs.passage_table(gen):
        print("  " + line)
    print("\nWHY CANDIDATES WERE REJECTED")
    for line in gs.rejection_breakdown(gen):
        print("  " + line)
    print("\nSUMMARY")
    print_summary(metrics)
    print("\nACCEPTED QUESTIONS (* = the model's answer key)")
    for line in gs.questions_text(gen):
        print("  " + line)
    print(f"\n{bar}\nTHE CALCULATIONS, STEP BY STEP\n{bar}")
    print_calculations(metrics)


# =============================================================================================
# CLI
# =============================================================================================
def check_environment() -> None:
    missing = []
    for mod in ("requests", "pydantic"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"Missing Python package(s): {', '.join(missing)}.\n"
            "Use the project's environment instead of the system Python:\n"
            "    .venv\\Scripts\\python.exe evaluation\\evaluate_llm.py ...\n"
            "or double-click run_evaluation.bat in the project folder."
        )


def ready(tag: str, args):
    """An Ollama client for *tag*, after checking that Ollama is up and the model is installed (never downloads anything)."""
    from services.ollama_client import OllamaClient

    client = OllamaClient(args.base_url, tag, temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.4")), timeout_s=args.timeout)
    status = client.status()
    if not status.connected:
        raise SystemExit(f"Cannot reach Ollama at {args.base_url}. Start Ollama and try again.")
    if not status.model_installed:
        raise SystemExit(f"Model '{tag}' is not installed. Run:  ollama pull {tag}")
    return client


def run_generation(model: str, judge: str | None, args, passages: list[dict]) -> dict:
    from evaluation import generation_suite as gs

    client = ready(model, args)
    judge_client = None
    if judge and judge != model:
        try:
            judge_client = ready(judge, args)
        except SystemExit as exc:
            print(f"  (judge model skipped: {exc})")
    print(f"\n=== Use-case test: {model} generating {args.gen_count} questions from each of {len(passages)} passages"
          + (f" (judge: {judge_client.model})" if judge_client else "") + " ===")
    return gs.run_generation_suite(client, passages, count=args.gen_count, judge_client=judge_client)


def run_metrics(model: str, judge: str | None, args) -> dict:
    from evaluation import metrics_suite as ms

    ready(model, args)
    if judge and judge != model:
        try:
            ready(judge, args)
        except SystemExit as exc:
            print(f"  (judge model skipped: {exc})")
            judge = None
    else:
        judge = None
    return ms.run_metrics_suite(model, judge, args.base_url, args.timeout, limit=args.metrics_limit,
                                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.4")))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # show progress live, even when piped
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", help="model tags to evaluate (default: OLLAMA_MODEL from .env)")
    ap.add_argument("--suite", choices=["all", "generation", "answering", "metrics"], default="all",
                    help="generation = the project's real task (write questions from material); answering = labelled quiz benchmark; "
                         "metrics = the 12 performance metrics (quality, safety, speed, cost, reliability)")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="answering suite: JSON file with labelled questions")
    ap.add_argument("--limit", type=int, help="answering suite: only use the first N questions")
    ap.add_argument("--passages", type=Path, help="generation suite: JSON file with source passages (default: evaluation/passages.json)")
    ap.add_argument("--max-passages", type=int, help="generation suite: only use the first N passages")
    ap.add_argument("--gen-count", type=int, default=3, help="generation suite: questions requested per passage (default 3)")
    ap.add_argument("--metrics-limit", type=int, help="metrics suite: only use the first N items of each dataset")
    ap.add_argument("--judge", default="auto", help="model that independently re-answers every generated question: auto (the other model "
                    "when two are given), none, or a model tag")
    ap.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"))
    ap.add_argument("--timeout", type=int, default=300, help="seconds per request")
    ap.add_argument("--no-bench", action="store_true", help="skip the generation-speed benchmark of the answering suite")
    ap.add_argument("--predictions", type=Path, help="recompute metrics from a saved report .json (no Ollama needed)")
    ap.add_argument("--score-labels", type=Path, help="score a labels_*.csv after a human filled the human_key_correct column")
    ap.add_argument("--demo", action="store_true", help="use a SIMULATED model to show the calculations without Ollama")
    ap.add_argument("--no-save", action="store_true", help="do not write report files")
    args = ap.parse_args(argv)

    if args.score_labels:
        from evaluation import generation_suite as gs

        metrics, info = gs.score_labels(args.score_labels)
        print("\n".join(info))
        print_summary(metrics)
        print_calculations(metrics)
        if metrics and not args.no_save:
            out = Path(str(args.score_labels).rsplit(".", 1)[0] + "_scored.md")
            out.write_text("# Human-labelled key accuracy\n\n" + "\n".join(info) + "\n\n" + "\n".join(markdown_table(metrics))
                           + "\n\n" + "\n".join(markdown_calculations(metrics)) + "\n", encoding="utf-8")
            print(f"\nSaved: {out}")
        return 0

    bundles: list[dict] = []
    labels: list[str] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.predictions:
        saved = json.loads(args.predictions.read_text(encoding="utf-8"))
        labels = saved.get("labels", [])
        bundles = saved["bundles"] if "bundles" in saved else [
            {"model": r["model"], "timestamp": r["timestamp"], "answering": r, "generation": None} for r in saved["runs"]]
    elif args.demo:
        items = load_dataset(args.dataset)[: args.limit] if args.limit else load_dataset(args.dataset)
        labels = labels_of(items)
        for run in (simulate_model(items, 0.8), simulate_model(items, 0.6, seed=5)):
            bundles.append({"model": run["model"], "timestamp": run["timestamp"], "answering": run, "generation": None})
        print("(demo mode: only the question-answering benchmark is simulated)")
    else:
        check_environment()
        models = args.models or [os.getenv("OLLAMA_MODEL", "qwen2.5:3b")]
        items = passages = None
        if args.suite in ("all", "answering"):
            items = load_dataset(args.dataset)
            items = items[: args.limit] if args.limit else items
            labels = labels_of(items)
        if args.suite in ("all", "generation"):
            from evaluation import generation_suite as gs

            passages = gs.load_passages(args.passages or gs.DEFAULT_PASSAGES)
            passages = passages[: args.max_passages] if args.max_passages else passages
        for idx, model in enumerate(models):
            judge = None if args.judge == "none" else (models[(idx + 1) % len(models)] if args.judge == "auto" and len(models) > 1
                                                          else (None if args.judge == "auto" else args.judge))
            ans = run_model(model, items, args.base_url, args.timeout, not args.no_bench) if items else None
            gen = run_generation(model, judge, args, passages) if passages else None
            met = run_metrics(model, judge, args) if args.suite in ("all", "metrics") else None
            bundles.append({"model": model, "timestamp": datetime.now().isoformat(timespec="seconds"), "answering": ans, "generation": gen,
                            "metrics": met})

    from evaluation import metrics_suite as ms

    all_ans, all_gen, all_met = [], [], []
    for b in bundles:
        ans_m = compute_metrics([Prediction(**p) for p in b["answering"]["predictions"]], labels) if b.get("answering") else []
        gen_m = generation_metrics_for(b["generation"]) if b.get("generation") else []
        met_m = ms.to_metrics(b["metrics"]) if b.get("metrics") else []
        all_ans.append(ans_m)
        all_gen.append(gen_m)
        all_met.append(met_m)
        if b.get("generation"):
            print_generation_report(b["generation"], gen_m)
        if b.get("answering"):
            print_run_report(b["answering"], labels, ans_m)
        if b.get("metrics"):
            ms.print_metrics_report(b["metrics"], met_m)

    if len(bundles) > 1:
        print("\n" + "=" * 100 + "\nCOMPARISON\n" + "=" * 100)
        for title, sets in comparison_sections(all_gen, all_ans, all_met):
            if all(sets):
                print(f"\n{title}")
                fake_runs = [{"model": b["model"]} for b in bundles]
                for line in comparison_table(fake_runs, sets):
                    print("  " + line)

    if not args.no_save and not args.predictions:
        print()
        for b, ans_m, gen_m, met_m in zip(bundles, all_ans, all_gen, all_met):
            for path in save_bundle(b, labels, ans_m, gen_m, stamp, met_m):
                print(f"Saved report: {path}")
        if len(bundles) > 1:
            comp = RESULTS_DIR / f"comparison_{stamp}.md"
            comp.write_text(comparison_markdown(bundles, all_ans, all_gen, all_met), encoding="utf-8")
            print(f"Saved comparison: {comp}")
        if any(b.get("generation") for b in bundles):
            print("\nNext step (optional, for the real key accuracy): open the labels_*.csv file, write yes/no in the "
                  "human_key_correct column, then run:\n    python evaluation/evaluate_llm.py --score-labels <that file>")
    return 0


def generation_metrics_for(gen: dict) -> list[Metric]:
    from evaluation import generation_suite as gs

    return gs.generation_metrics(gen)


if __name__ == "__main__":
    raise SystemExit(main())
