#!/usr/bin/env python
"""Evaluate the local LLM (Ollama) on a labelled multiple-choice test set - and SHOW every calculation.

The model answers each question (A/B/C/D). Because the correct answer is known, we can measure:

    Accuracy, Exact Match, Precision, Recall, F1, Macro-F1, Micro-F1, Weighted-F1,
    Top-k accuracy, MCC, Cohen's kappa, Log loss, Brier score, Expected Calibration Error

plus speed (latency, prompt tokens/s, generation tokens/s).

Probabilities come from Ollama's token log-probabilities: for the first generated token we read the
probability of each option letter and renormalise over the letters. That gives a real probability
distribution for every question, which log loss, Brier, ECE and top-k need.

Usage (from the project folder):
    python evaluation/evaluate_llm.py                          # default model from .env
    python evaluation/evaluate_llm.py --models qwen2.5:3b qwen2.5:7b
    python evaluation/evaluate_llm.py --limit 10               # quick run
    python evaluation/evaluate_llm.py --predictions evaluation/results/<file>.json   # recompute offline
    python evaluation/evaluate_llm.py --demo                   # SIMULATED model, no Ollama needed

Only the Python standard library and `requests` are used (no scikit-learn needed).
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
DEFAULT_DATASET = ROOT / "dataset.json"
RESULTS_DIR = ROOT / "results"
EPS = 1e-15  # probabilities are clipped to [EPS, 1] before taking logs
N_BINS = 10  # equal-width confidence bins for ECE
NONE_LABEL = "?"  # "the answer could not be parsed" - counted as a wrong prediction

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


@dataclass
class Metric:
    name: str
    value: float
    better: str  # "higher" / "lower"
    formula: str
    working: list[str] = field(default_factory=list)  # the calculation, with the real numbers substituted
    note: str = ""


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
def probs_from_logprobs(logprobs: list[dict], letters: list[str]) -> tuple[str | None, dict[str, float]] | None:
    """Find the first generated token that is an option letter and turn its top-logprobs into probabilities.

    Returns (letter chosen, {letter: probability}) or None if no generated token is an option letter.
    For every letter we take the largest probability among tokens such as "B", " B", "b" and then
    renormalise over the option letters only (so the numbers form a proper distribution).
    """
    for entry in logprobs or []:
        token = (entry.get("token") or "").strip().upper()
        if token not in letters:
            continue
        mass = {letter: 0.0 for letter in letters}
        for cand in entry.get("top_logprobs") or [entry]:
            letter = (cand.get("token") or "").strip().upper()
            if letter in mass:
                mass[letter] = max(mass[letter], math.exp(cand["logprob"]))
        total = sum(mass.values())
        if total <= 0:
            return None
        return token, {letter: m / total for letter, m in mass.items()}
    return None


def parse_letter(text: str, letters: list[str]) -> str | None:
    """Accept 'B', 'b.', '(B)', 'Answer: B' - anything that is clearly a single option letter."""
    match = re.match(r"^\W*(?:answer\W*)?([A-Za-z])\b", text.strip(), flags=re.IGNORECASE)
    if match and match.group(1).upper() in letters:
        return match.group(1).upper()
    return None


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
def f(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}"


def confusion(golds: list[str], preds: list[str | None], labels: list[str]) -> dict[str, dict[str, int]]:
    """matrix[true][predicted]; the extra column '?' collects unparsed answers."""
    cols = labels + [NONE_LABEL]
    m = {t: {p: 0 for p in cols} for t in labels}
    for g, p in zip(golds, preds):
        m[g][p if p in labels else NONE_LABEL] += 1
    return m


def per_class_counts(m: dict, labels: list[str]) -> dict[str, dict[str, int]]:
    n = sum(sum(row.values()) for row in m.values())
    out = {}
    for k in labels:
        tp = m[k][k]
        fp = sum(m[t][k] for t in labels if t != k)  # predicted k, truly something else
        fn = sum(m[k][p] for p in m[k] if p != k)  # truly k, predicted something else (incl. '?')
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": n - tp - fp - fn, "support": tp + fn}
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


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
        row = f"{m.name:<42}" + "".join(f"{ms[i].value:>{width}.4f}" for ms in all_metrics) + f"   {m.better}"
        lines.append(row)
    return lines


def markdown_report(runs: list[dict], labels: list[str], all_metrics: list[list[Metric]]) -> str:
    out = ["# LLM performance report", "", f"Generated {datetime.now().isoformat(timespec='seconds')}  ", f"Options per question: {len(labels)}  "]
    if any(r.get("simulated") for r in runs):
        out += ["", "> **SIMULATED RESULTS** - produced by a fake model to demonstrate the calculations; not a real LLM measurement."]
    for run, metrics in zip(runs, all_metrics):
        preds = [Prediction(**p) for p in run["predictions"]]
        out += ["", f"## {run['model']}", "", f"Questions: {len(preds)}", "", "| Metric | Value | Better |", "|---|---:|---|"]
        out += [f"| {m.name} | {m.value:.4f} | {m.better} |" for m in metrics]
        out += ["", "**Confusion matrix** (rows = correct option, columns = chosen option)", "", "```"] + confusion_text(preds, labels) + ["```", "", "**Speed**", ""]
        out += [f"- {line}" for line in (speed_summary(run) or ["no timing data"])]
        out += ["", "<details><summary>Step-by-step calculations</summary>", ""]
        for m in metrics:
            out += [f"### {m.name}", f"*Formula:* `{m.formula}`  ", "", "```"] + m.working + [f"= {m.value:.4f}", "```"]
            if m.note:
                out += [f"*{m.note}*"]
            out += [""]
        out += ["</details>"]
    if len(runs) > 1:
        out += ["", "## Comparison", "", "```"] + comparison_table(runs, all_metrics) + ["```"]
    return "\n".join(out) + "\n"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:40]


def save_results(runs: list[dict], labels: list[str], all_metrics: list[list[Metric]]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "_vs_".join(slug(r["model"]) for r in runs)
    base = RESULTS_DIR / f"{'demo_' if any(r.get('simulated') for r in runs) else ''}{name}_{stamp}"
    payload = {"labels": labels, "runs": [
        {**run, "metrics": [{"name": m.name, "value": m.value, "better": m.better, "formula": m.formula} for m in ms]}
        for run, ms in zip(runs, all_metrics)]}
    json_path = Path(f"{base}.json")  # not with_suffix(): model tags contain dots ("qwen2.5") and would be cut off
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = Path(f"{base}.md")
    md_path.write_text(markdown_report(runs, labels, all_metrics), encoding="utf-8")
    return json_path, md_path


# =============================================================================================
# CLI
# =============================================================================================
def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # show progress live, even when piped
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", help="model tags to evaluate (default: OLLAMA_MODEL from .env)")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSON file with the labelled questions")
    ap.add_argument("--limit", type=int, help="only use the first N questions")
    ap.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"))
    ap.add_argument("--timeout", type=int, default=300, help="seconds per request")
    ap.add_argument("--no-bench", action="store_true", help="skip the generation-speed benchmark")
    ap.add_argument("--predictions", type=Path, help="recompute metrics from a saved results JSON (no Ollama needed)")
    ap.add_argument("--demo", action="store_true", help="use a SIMULATED model to show the calculations without Ollama")
    ap.add_argument("--no-save", action="store_true", help="do not write the .json/.md report files")
    args = ap.parse_args(argv)

    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    labels = labels_of(items)

    if args.predictions:
        saved = json.loads(args.predictions.read_text(encoding="utf-8"))
        runs = saved["runs"]
        labels = saved.get("labels", labels)
    elif args.demo:
        runs = [simulate_model(items, 0.8), simulate_model(items, 0.6, seed=5)]
    else:
        models = args.models or [os.getenv("OLLAMA_MODEL", "qwen2.5:3b")]
        runs = [run_model(m, items, args.base_url, args.timeout, not args.no_bench) for m in models]

    all_metrics = []
    for run in runs:
        preds = [Prediction(**p) for p in run["predictions"]]
        metrics = compute_metrics(preds, labels)
        all_metrics.append(metrics)
        print_run_report(run, labels, metrics)
    if len(runs) > 1:
        print("\n" + "=" * 100 + "\nCOMPARISON\n" + "=" * 100)
        for line in comparison_table(runs, all_metrics):
            print("  " + line)
    if not args.no_save and not args.predictions:
        json_path, md_path = save_results(runs, labels, all_metrics)
        print(f"\nSaved: {json_path}\n       {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
