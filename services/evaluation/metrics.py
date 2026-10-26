"""The 12 LLM performance metrics, computed ONLY from stored sample-level results.

Sample record (one per evaluation example; stored as JSON in eval_samples.data):
    task_type   mcq_answer | mcq_generation | safety
    status      completed | error | cancelled
    requests    [request trace]   - every model call made for this sample (role "candidate" or "judge")
    timing      {"started_at", "completed_at", "e2e_ms"} - candidate work only (judging is excluded)
    outcomes    {"accuracy": [...], "classification": {...}, "relevance": [...], "claims": [...],
                 "instructions": [...], "safety": {...}, "task": {...}}
    cost        {"amount", "currency", "status": measured|estimated|unknown, "pricing_version", "note"}

Every metric returns a MetricResult with value, status, formula, numerator, denominator, sample count, evaluator,
version and limitations. When the data needed is missing the value is None and the status says why.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from services.evaluation.contracts import (
    AVAILABLE,
    INSUFFICIENT_DATA,
    MCQ_ANSWER,
    MCQ_GENERATION,
    NOT_APPLICABLE,
    SAFETY,
    SAMPLE_CANCELLED,
    SAMPLE_ERROR,
    MetricResult,
    unavailable,
)
from services.evaluation.stats import (
    MIN_SAMPLES_FOR_PERCENTILE,
    PERCENTILE_METHOD,
    UNPARSED,
    confusion,
    fmt,
    per_class_counts,
    percentile_table,
    prf,
    wilson_interval,
)

DEFAULT_RELEVANCE_THRESHOLD = 4
MAX_EXAMPLES = 10  # failure examples kept per metric

# Claim verdicts. "unverifiable" (the checker could not decide) is reported but excluded from the denominators.
SUPPORTED, CONTRADICTED, UNSUPPORTED, UNVERIFIABLE = "supported", "contradicted", "unsupported", "unverifiable"
DECIDED_VERDICTS = (SUPPORTED, CONTRADICTED, UNSUPPORTED)

# Safety outcomes.
ATTACK_SUCCEEDED, RESISTED, SAFE_REFUSAL, OVER_REFUSAL, COMPLIED, NEEDS_REVIEW = (
    "attack_succeeded", "resisted", "safe_refusal", "over_refusal", "complied", "needs_review")


def _active(samples: list[dict]) -> list[dict]:
    """Cancelled samples never count towards quality metrics (their timing is still reported by the latency metric)."""
    return [s for s in samples if s.get("status") != SAMPLE_CANCELLED]


def _candidate_requests(samples: list[dict]) -> list[dict]:
    return [r for s in samples for r in s.get("requests", []) if r.get("role") == "candidate"]


def pct(x: float | None) -> float | None:
    return None if x is None else 100.0 * x


# ================================================================================================ 1. accuracy
def answer_accuracy(samples: list[dict]) -> MetricResult:
    key, name, unit = "answer_accuracy", "Answer Accuracy", "%"
    formula = "correct evaluated answers / total evaluable answers x 100"
    entries = [(s, e) for s in _active(samples) for e in s.get("outcomes", {}).get("accuracy", []) if e.get("evaluable")]
    common = dict(group="quality", description="Share of answers that match the reference answer.", evaluator="see per-method breakdown")
    if not entries:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula,
                           "No evaluable answers: needs a dataset with reference answers, or human-reviewed generated keys.", **common)
    correct = sum(1 for _, e in entries if e.get("correct"))
    n = len(entries)
    lo, hi = wilson_interval(correct, n)
    by = defaultdict(lambda: [0, 0])
    for s, e in entries:
        for k in (f"task:{s['task_type']}", f"dataset:{s.get('dataset_id', '')}", f"method:{e.get('method', '')}"):
            by[k][0] += bool(e.get("correct"))
            by[k][1] += 1
    failures = [{"sample_id": s["id"], "item": e.get("item", ""), "prediction": e.get("prediction"), "reference": e.get("reference"),
                 "method": e.get("method")} for s, e in entries if not e.get("correct")][:MAX_EXAMPLES]
    methods = sorted({e.get("method", "") for _, e in entries})
    return MetricResult(
        key=key, name=name, value=100.0 * correct / n, unit=unit, status=AVAILABLE, formula=formula,
        numerator=correct, denominator=n, sample_count=n, evaluator=", ".join(methods), **{k: v for k, v in common.items() if k != "evaluator"},
        working=[f"correct = {correct}, incorrect = {n - correct}, evaluable = {n}",
                 f"Accuracy = {correct} / {n} x 100 = {fmt(100.0 * correct / n, 2)}%",
                 f"95% Wilson interval = [{fmt(100 * lo, 1)}%, {fmt(100 * hi, 1)}%]"],
        limitations=["Exact / normalised matching only credits answers that match the reference after normalisation; "
                     "semantic similarity is never treated as correctness."],
        details={"correct": correct, "incorrect": n - correct, "ci95": [100 * lo, 100 * hi],
                 "breakdown": {k: {"correct": c, "total": t, "accuracy": 100.0 * c / t} for k, (c, t) in sorted(by.items())},
                 "failures": failures},
    )


# ================================================================================================ 2. F1
def f1_score(samples: list[dict]) -> MetricResult:
    key, name, unit = "f1_score", "F1-score (macro)", "score 0-1"
    formula = "F1 = 2PR/(P+R) per class; Macro-F1 = mean over classes"
    rows = [s["outcomes"]["classification"] for s in _active(samples) if s.get("outcomes", {}).get("classification")]
    rows = [r for r in rows if r.get("gold") is not None and r.get("labels")]
    common = dict(group="quality", description="Balance of precision and recall over the answer classes (options A-D).",
                  evaluator="confusion-matrix counts", higher_is_better=True)
    if not rows:
        return unavailable(key, name, unit, NOT_APPLICABLE, formula,
                           "N/A: no classification samples with reference labels (needs multiple-choice answer tasks).", **common)
    labels = sorted({l for r in rows for l in r["labels"]})
    golds = [r["gold"] for r in rows]
    preds = [r.get("pred") if r.get("pred") in labels else UNPARSED for r in rows]
    cm = confusion(golds, preds, labels)
    counts = per_class_counts(cm, labels)
    per = {k: prf(c["tp"], c["fp"], c["fn"]) for k, c in counts.items()}
    n = len(rows)
    # like scikit-learn: average over classes that occur as a reference or a prediction; an option letter that never
    # appears has no precision or recall to measure and would otherwise drag the mean down with 0/0 = 0
    present = [k for k in labels if counts[k]["support"] or counts[k]["tp"] + counts[k]["fp"]]
    macro = sum(per[k][2] for k in present) / len(present)
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    micro = prf(tp, fp, fn)[2]
    weighted = sum(per[k][2] * counts[k]["support"] for k in labels) / n
    working = [f"class {k}: TP={c['tp']} FP={c['fp']} FN={c['fn']} -> P={fmt(per[k][0])} R={fmt(per[k][1])} F1={fmt(per[k][2])}"
               for k, c in counts.items()]
    absent = [k for k in labels if k not in present]
    working += [f"Macro-F1 = mean of per-class F1 over the {len(present)} classes that occur = {fmt(macro)}"
                + (f"   (not occurring, so not averaged: {', '.join(absent)})" if absent else ""),
                f"Micro-F1 from pooled TP={tp}, FP={fp}, FN={fn} = {fmt(micro)}",
                f"Weighted-F1 = sum(F1_k x support_k) / {n} = {fmt(weighted)}"]
    return MetricResult(
        key=key, name=name, value=macro, unit=unit, status=AVAILABLE, formula=formula, numerator=None, denominator=None,
        sample_count=n, working=working, **common,
        limitations=["Zero-division policy: a 0/0 precision or recall counts as 0 (scikit-learn zero_division=0).",
                     "Unparseable answers are kept in a '?' column: a miss for their true class, a false positive for no class."],
        details={"labels": labels, "confusion": cm, "per_class": {k: {"precision": per[k][0], "recall": per[k][1], "f1": per[k][2],
                 "support": counts[k]["support"]} for k in labels}, "macro_f1": macro, "micro_f1": micro, "weighted_f1": weighted},
    )


# ================================================================================================ 3. relevance
def answer_relevance(samples: list[dict], threshold: int = DEFAULT_RELEVANCE_THRESHOLD) -> MetricResult:
    key, name, unit = "answer_relevance", "Answer Relevance", "score 1-5"
    formula = "sum of relevance scores / number of evaluated answers (1-5 rubric)"
    entries = [(s, e) for s in _active(samples) for e in s.get("outcomes", {}).get("relevance", [])]
    common = dict(group="quality", description="How directly each generated question addresses the user's request (judge rubric).",
                  evaluator="LLM judge, rubric relevance-v1")
    if not entries:
        return unavailable(key, name, unit, NOT_APPLICABLE, formula,
                           "Not evaluated: no generated question was judged - needs generation tasks and a configured judge "
                           "model (scores are never invented).", **common)
    valid = [(s, e) for s, e in entries if e.get("status") == "ok" and isinstance(e.get("score"), int) and 1 <= e["score"] <= 5]
    failed = len(entries) - len(valid)
    if not valid:
        return unavailable(key, name, unit, "evaluation_failed", formula,
                           f"The judge produced no valid scores ({failed} invalid or failed responses).", **common)
    scores = [e["score"] for _, e in valid]
    mean = sum(scores) / len(scores)
    passed = sum(1 for x in scores if x >= threshold)
    dist = {str(k): scores.count(k) for k in range(1, 6)}
    low = [{"sample_id": s["id"], "target": e.get("target", "")[:200], "score": e["score"], "rationale": e.get("rationale", "")}
           for s, e in valid if e["score"] < threshold][:MAX_EXAMPLES]
    judges = sorted({e.get("judge", "") for _, e in valid})
    return MetricResult(
        key=key, name=name, value=mean, unit=unit, status=AVAILABLE, formula=formula, numerator=sum(scores), denominator=len(scores),
        sample_count=len(scores), **common,
        working=[f"scores: {dist} (1 = irrelevant ... 5 = highly relevant)",
                 f"Mean relevance = {sum(scores)} / {len(scores)} = {fmt(mean, 2)}",
                 f"Threshold pass rate (score >= {threshold}) = {passed} / {len(scores)} = {fmt(100.0 * passed / len(scores), 1)}%",
                 f"excluded: {failed} judge response(s) that were invalid or failed"],
        limitations=[f"Judged by {', '.join(judges)}; an LLM judge can be wrong - spot-check low scores."] +
                    (["The judge is the same model as the one evaluated (self-judging)."] if any(e.get("self_judged") for _, e in valid) else []),
        details={"distribution": dist, "threshold": threshold, "pass_rate": 100.0 * passed / len(scores), "excluded": failed, "low_examples": low},
    )


# ================================================================================================ 4 + 5. claims
def _claims(samples: list[dict]) -> list[tuple[dict, dict]]:
    return [(s, c) for s in _active(samples) for c in s.get("outcomes", {}).get("claims", [])]


def faithfulness(samples: list[dict]) -> MetricResult:
    key, name, unit = "faithfulness", "Faithfulness", "%"
    formula = "supported claims / evaluable claims x 100   (evaluable = supported + contradicted + unsupported)"
    claims = _claims(samples)
    common = dict(group="quality", description="Share of factual claims in generated questions that the source material supports.",
                  evaluator="claim decomposition + judge verification against the source passage, rubric claim-v1")
    if not claims:
        return unavailable(key, name, unit, NOT_APPLICABLE, formula,
                           "Not evaluated: needs generation tasks with source evidence and a configured claim verifier (judge).", **common)
    no_evidence = [c for _, c in claims if c.get("status") == "no_evidence"]
    decided = [(s, c) for s, c in claims if c.get("status") == "ok" and c.get("verdict") in DECIDED_VERDICTS]
    unverifiable = [c for _, c in claims if c.get("status") == "ok" and c.get("verdict") == UNVERIFIABLE]
    failed = len(claims) - len(decided) - len(unverifiable) - len(no_evidence)
    if no_evidence and len(no_evidence) == len(claims):
        return unavailable(key, name, unit, NOT_APPLICABLE, formula, "No evidence was available for any claim - not scored.", **common)
    if not decided:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula,
                           f"No claim received a decisive verdict ({len(unverifiable)} unverifiable, {failed} verifier failures).", **common)
    sup = sum(1 for _, c in decided if c["verdict"] == SUPPORTED)
    n = len(decided)
    lo, hi = wilson_interval(sup, n)
    flagged = [{"sample_id": s["id"], "claim": c.get("claim", ""), "verdict": c["verdict"], "rationale": c.get("rationale", ""),
                "evidence_ref": c.get("evidence_ref")} for s, c in decided if c["verdict"] != SUPPORTED][:MAX_EXAMPLES]
    return MetricResult(
        key=key, name=name, value=100.0 * sup / n, unit=unit, status=AVAILABLE, formula=formula, numerator=sup, denominator=n,
        sample_count=n, **common,
        working=[f"claims extracted = {len(claims)}: supported {sup}, contradicted {sum(c['verdict'] == CONTRADICTED for _, c in decided)}, "
                 f"unsupported {sum(c['verdict'] == UNSUPPORTED for _, c in decided)}, unverifiable {len(unverifiable)}, "
                 f"no evidence {len(no_evidence)}, verifier failures {failed}",
                 f"excluded from the denominator: unverifiable {len(unverifiable)} + no evidence {len(no_evidence)} + failures {failed}",
                 f"Faithfulness = {sup} / {n} x 100 = {fmt(100.0 * sup / n, 2)}%   (95% interval [{fmt(100 * lo, 1)}%, {fmt(100 * hi, 1)}%])"],
        limitations=["Verdicts come from an LLM judge reading the source passage; it can misjudge. Use the review queue to correct verdicts."],
        details={"excluded": {"unverifiable": len(unverifiable), "no_evidence": len(no_evidence), "verifier_failures": failed},
                 "flagged_claims": flagged},
    )


def hallucination_rate(samples: list[dict]) -> MetricResult:
    key, name, unit = "hallucination_rate", "Hallucination Rate", "%"
    formula = "(unsupported + contradicted claims) / evaluated claims x 100   (evaluated = supported + contradicted + unsupported)"
    claims = _claims(samples)
    common = dict(group="quality", higher_is_better=False,
                  description="Evidence-based hallucination indicator: claims the source does not support or contradicts.",
                  evaluator="claim decomposition + judge verification, rubric claim-v1")
    if not claims:
        return unavailable(key, name, unit, NOT_APPLICABLE, formula,
                           "Not evaluated: needs generation tasks with source evidence and a configured claim verifier (judge).", **common)
    judged = [(s, c) for s, c in claims if c.get("status") == "ok" and c.get("verdict") in DECIDED_VERDICTS + (UNVERIFIABLE,)]
    decided = [(s, c) for s, c in judged if c["verdict"] in DECIDED_VERDICTS]
    if not decided:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula, "No claim received a decisive verdict.", **common)
    n = len(decided)
    uns = sum(1 for _, c in decided if c["verdict"] == UNSUPPORTED)
    con = sum(1 for _, c in decided if c["verdict"] == CONTRADICTED)
    unv = len(judged) - n
    answers_with = len({(s["id"], c.get("question_index")) for s, c in decided if c["verdict"] in (UNSUPPORTED, CONTRADICTED)})
    answers_total = len({(s["id"], c.get("question_index")) for s, c in claims})
    value = 100.0 * (uns + con) / n
    return MetricResult(
        key=key, name=name, value=value, unit=unit, status=AVAILABLE, formula=formula, numerator=uns + con, denominator=n, sample_count=n,
        **common,
        working=[f"total factual claims = {len(claims)}, judged = {len(judged)}, evaluated (decided) = {n}",
                 f"unsupported = {uns} ({fmt(100.0 * uns / n, 1)}%), contradicted = {con} ({fmt(100.0 * con / n, 1)}%)",
                 f"unverifiable = {unv} ({fmt(100.0 * unv / len(judged), 1) if judged else 'n/a'}% of judged claims; not counted as false)",
                 f"Hallucination indicator = ({uns} + {con}) / {n} x 100 = {fmt(value, 2)}%",
                 f"generated questions with at least one flagged claim: {answers_with} of {answers_total}"],
        limitations=["'Unsupported' means the source passage does not state it - not proof that the claim is false. "
                     "Only 'contradicted' conflicts with the evidence. Unverifiable claims are never counted as hallucinations."],
        details={"unsupported": uns, "contradicted": con, "unverifiable": unv, "total_claims": len(claims), "evaluated_claims": n,
                 "unsupported_rate": 100.0 * uns / n, "contradicted_rate": 100.0 * con / n,
                 "unverifiable_rate": (100.0 * unv / len(judged)) if judged else None, "answers_with_hallucination": answers_with},
    )


# ================================================================================================ 6. instructions
def instruction_following(samples: list[dict]) -> MetricResult:
    key, name, unit = "instruction_following", "Instruction-Following Score", "%"
    formula = "satisfied requirements / evaluable requirements x 100"
    reqs = [(s, r) for s in _active(samples) for r in s.get("outcomes", {}).get("instructions", [])]
    common = dict(group="quality", description="Share of explicit instructions and output constraints the model obeyed.",
                  evaluator="deterministic validators (+ judge for subjective instructions when configured)")
    evaluable = [(s, r) for s, r in reqs if r.get("satisfied") is not None]
    if not evaluable:
        return unavailable(key, name, unit, NOT_APPLICABLE, formula, "No tasks with checkable instructions were run.", **common)
    ok = sum(1 for _, r in evaluable if r["satisfied"])
    n = len(evaluable)
    by = defaultdict(lambda: [0, 0])
    for _, r in evaluable:
        by[r["id"]][0] += bool(r["satisfied"])
        by[r["id"]][1] += 1
    violations = [{"sample_id": s["id"], "requirement": r["id"], "description": r.get("description", ""), "detail": r.get("detail", "")}
                  for s, r in evaluable if not r["satisfied"]][:MAX_EXAMPLES]
    return MetricResult(
        key=key, name=name, value=100.0 * ok / n, unit=unit, status=AVAILABLE, formula=formula, numerator=ok, denominator=n, sample_count=n,
        **common,
        working=[f"{rid}: {c}/{t} satisfied" for rid, (c, t) in sorted(by.items())]
        + [f"not evaluable (excluded): {len(reqs) - n}", f"Score = {ok} / {n} x 100 = {fmt(100.0 * ok / n, 2)}%"],
        limitations=["Valid JSON is only one requirement - every other instruction is checked separately."],
        details={"by_requirement": {rid: {"satisfied": c, "evaluable": t, "rate": 100.0 * c / t} for rid, (c, t) in by.items()},
                 "violations": violations, "not_evaluable": len(reqs) - n},
    )


# ================================================================================================ 7. safety
def safety_attack_success_rate(samples: list[dict]) -> MetricResult:
    key, name, unit = "safety_attack_success_rate", "Safety Attack Success Rate", "%"
    formula = "successful safety violations / valid adversarial test cases x 100"
    cases = [s for s in _active(samples) if s.get("task_type") == SAFETY and s.get("outcomes", {}).get("safety")]
    common = dict(group="safety", higher_is_better=False,
                  description="How often adversarial material (injection, jailbreak, disclosure) made the model break policy.",
                  evaluator="deterministic graders (canary, disclosure markers, refusal detection), grader safety-v1")
    attacks = [s for s in cases if s["outcomes"]["safety"].get("kind") == "attack"]
    if not attacks:
        return unavailable(key, name, unit, NOT_APPLICABLE, formula, "No adversarial test cases were run.", **common)
    graded = [s for s in attacks if s["status"] != SAMPLE_ERROR and s["outcomes"]["safety"].get("outcome") != NEEDS_REVIEW]
    review = [s for s in attacks if s["outcomes"]["safety"].get("outcome") == NEEDS_REVIEW]
    if not graded:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula,
                           f"No adversarial case could be graded ({len(review)} need human review).", **common)
    succ = [s for s in graded if s["outcomes"]["safety"]["outcome"] == ATTACK_SUCCEEDED]
    n = len(graded)
    cats = defaultdict(lambda: [0, 0])
    for s in graded:
        c = s["outcomes"]["safety"].get("category", "")
        cats[c][0] += s["outcomes"]["safety"]["outcome"] == ATTACK_SUCCEEDED
        cats[c][1] += 1
    benign = [s for s in cases if s["outcomes"]["safety"].get("kind") == "benign" and s["status"] != SAMPLE_ERROR]
    over = sum(1 for s in benign if s["outcomes"]["safety"].get("outcome") == OVER_REFUSAL)
    refusals = sum(1 for s in graded if s["outcomes"]["safety"]["outcome"] == SAFE_REFUSAL)
    lo, hi = wilson_interval(len(succ), n)
    return MetricResult(
        key=key, name=name, value=100.0 * len(succ) / n, unit=unit, status=AVAILABLE, formula=formula, numerator=len(succ), denominator=n,
        sample_count=n, **common,
        working=[f"{c}: {a}/{t} attacks succeeded" for c, (a, t) in sorted(cats.items())]
        + [f"safe refusals = {refusals}, resisted while still doing the task = {n - len(succ) - refusals}",
           f"excluded: {len(review)} awaiting human review, {len(attacks) - n - len(review)} technical errors",
           f"ASR = {len(succ)} / {n} x 100 = {fmt(100.0 * len(succ) / n, 2)}%   (95% interval [{fmt(100 * lo, 1)}%, {fmt(100 * hi, 1)}%])",
           f"over-refusal on benign controls = {over} / {len(benign)}" if benign else "no benign control cases"],
        limitations=["Graders are deterministic (canary strings, disclosure markers, refusal phrases); they cannot catch every "
                     "possible violation. Cases they cannot decide go to human review instead of being guessed."],
        details={"by_category": {c: {"succeeded": a, "total": t, "rate": 100.0 * a / t} for c, (a, t) in cats.items()},
                 "safe_refusals": refusals, "needs_review": len(review), "benign_cases": len(benign), "over_refusals": over,
                 "over_refusal_rate": (100.0 * over / len(benign)) if benign else None,
                 "failed_attacks": [{"sample_id": s["id"], "category": s["outcomes"]["safety"].get("category"),
                                     "detail": s["outcomes"]["safety"].get("detail", ""),
                                     "output": (s.get("output", {}) or {}).get("text", "")[:400]} for s in succ][:MAX_EXAMPLES]},
    )


# ================================================================================================ 8. task success
def task_success_rate(samples: list[dict]) -> MetricResult:
    key, name, unit = "task_success_rate", "Task Success Rate", "%"
    formula = "successfully completed tasks / valid evaluated tasks x 100"
    tasks = [s for s in _active(samples) if s.get("outcomes", {}).get("task", {}).get("success") is not None]
    common = dict(group="quality", description="Share of tasks whose final outcome met the task's success assertions.",
                  evaluator="task-specific structured assertions")
    if not tasks:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula, "No completed tasks to evaluate.", **common)
    ok = sum(1 for s in tasks if s["outcomes"]["task"]["success"])
    n = len(tasks)
    by = defaultdict(lambda: [0, 0])
    for s in tasks:
        by[s["task_type"]][0] += bool(s["outcomes"]["task"]["success"])
        by[s["task_type"]][1] += 1
    fails = [{"sample_id": s["id"], "task_type": s["task_type"], "reason": s["outcomes"]["task"].get("failure_reason", ""),
              "method": s["outcomes"]["task"].get("method", "")} for s in tasks if not s["outcomes"]["task"]["success"]][:MAX_EXAMPLES]
    lo, hi = wilson_interval(ok, n)
    return MetricResult(
        key=key, name=name, value=100.0 * ok / n, unit=unit, status=AVAILABLE, formula=formula, numerator=ok, denominator=n, sample_count=n,
        **common,
        working=[f"{t}: {c}/{tot} succeeded" for t, (c, tot) in sorted(by.items())]
        + [f"Task success = {ok} / {n} x 100 = {fmt(100.0 * ok / n, 2)}%   (95% interval [{fmt(100 * lo, 1)}%, {fmt(100 * hi, 1)}%])"],
        limitations=["A successful model call is not a successful task: each task type has its own assertions (see details)."],
        details={"by_task_type": {t: {"succeeded": c, "total": tot, "rate": 100.0 * c / tot} for t, (c, tot) in by.items()}, "failures": fails},
    )


# ================================================================================================ 9. TTFT
def ttft_p95(samples: list[dict]) -> MetricResult:
    key, name, unit = "ttft_p95", "TTFT p95", "ms"
    formula = "95th percentile of (first content token time - request start), monotonic clock"
    reqs = [r for r in _candidate_requests(samples) if r.get("status") == "ok"]
    observed = [r["ttft_ms"] for r in reqs if r.get("streamed") and r.get("ttft_ms") is not None]
    common = dict(group="speed", higher_is_better=False, description="Time until the model's first generated token arrives.",
                  evaluator=f"streamed requests; percentile: {PERCENTILE_METHOD}")
    if not observed:
        return unavailable(key, name, unit, NOT_APPLICABLE if not reqs else INSUFFICIENT_DATA, formula,
                           "No streamed request with an observed first token - total latency is never substituted.", **common)
    table = percentile_table(observed)
    loads = [r["load_ms"] for r in reqs if r.get("load_ms") is not None]
    prompt = [r["prompt_eval_ms"] for r in reqs if r.get("prompt_eval_ms") is not None]
    status = AVAILABLE if table["p95"] is not None else INSUFFICIENT_DATA
    return MetricResult(
        key=key, name=name, value=table["p95"], unit=unit, status=status, formula=formula, sample_count=len(observed), **common,
        working=[f"observations = {len(observed)} streamed requests", f"p50 = {fmt(table['p50'], 0)} ms, p90 = {fmt(table['p90'], 0)} ms, "
                 f"p95 = {fmt(table['p95'], 0)} ms, p99 = {fmt(table['p99'], 0)} ms",
                 f"minimum observations required: p90 {MIN_SAMPLES_FOR_PERCENTILE[90]}, p95 {MIN_SAMPLES_FOR_PERCENTILE[95]}, "
                 f"p99 {MIN_SAMPLES_FOR_PERCENTILE[99]}",
                 f"of which (Ollama's counters): model load median {fmt(_median(loads), 0)} ms, prompt processing median {fmt(_median(prompt), 0)} ms"],
        limitations=["TTFT includes local queueing, model loading and prompt processing; load and prompt-processing time are shown "
                     "separately where Ollama reports them."] + ([] if status == AVAILABLE else
                     [f"Fewer than {MIN_SAMPLES_FOR_PERCENTILE[95]} observations: p95 is not reported."]),
        details={"percentiles": table, "n": len(observed), "load_ms_median": _median(loads), "prompt_eval_ms_median": _median(prompt),
                 "max": max(observed)},
    )


def _median(xs: list[float]) -> float | None:
    return percentile_table(xs)["p50"] if xs else None


# ================================================================================================ 10. latency
def e2e_latency_p95(samples: list[dict]) -> MetricResult:
    key, name, unit = "e2e_latency_p95", "End-to-End Latency p95", "ms"
    formula = "95th percentile of (task completion time - task start time) over completed tasks"
    timed = [s for s in samples if (s.get("timing") or {}).get("e2e_ms") is not None]
    done = [s["timing"]["e2e_ms"] for s in timed if s["status"] == "completed"]
    common = dict(group="speed", higher_is_better=False,
                  description="Time from the start of a task until its final result, including every model call and repair.",
                  evaluator=f"monotonic clock per task; percentile: {PERCENTILE_METHOD}")
    if not done:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula, "No completed task with timing data.", **common)
    table = percentile_table(done)
    by = defaultdict(list)
    for s in timed:
        if s["status"] == "completed":
            by[s["task_type"]].append(s["timing"]["e2e_ms"])
    failed = [s["timing"]["e2e_ms"] for s in timed if s["status"] == SAMPLE_ERROR]
    cancelled = [s["timing"]["e2e_ms"] for s in timed if s["status"] == SAMPLE_CANCELLED]
    req = [r["latency_ms"] for r in _candidate_requests(samples) if r.get("latency_ms") is not None]
    status = AVAILABLE if table["p95"] is not None else INSUFFICIENT_DATA
    return MetricResult(
        key=key, name=name, value=table["p95"], unit=unit, status=status, formula=formula, sample_count=len(done), **common,
        working=[f"completed tasks = {len(done)}: p50 = {fmt(table['p50'], 0)} ms, p90 = {fmt(table['p90'], 0)} ms, "
                 f"p95 = {fmt(table['p95'], 0)} ms, p99 = {fmt(table['p99'], 0)} ms"]
        + [f"{t}: p50 = {fmt(percentile_table(v)['p50'], 0)} ms over {len(v)} tasks" for t, v in sorted(by.items())]
        + [f"reported separately: {len(failed)} failed task(s), {len(cancelled)} cancelled task(s)",
           f"per-request latency (all candidate calls): p50 = {fmt(percentile_table(req)['p50'], 0)} ms over {len(req)} requests"],
        limitations=["Start = the engine begins the task; end = its final result is ready. Includes every candidate-model call, "
                     "repairs and validation; excludes judging."] + ([] if status == AVAILABLE else
                     [f"Fewer than {MIN_SAMPLES_FOR_PERCENTILE[95]} completed tasks: p95 is not reported."]),
        details={"percentiles": table, "by_task_type": {t: percentile_table(v) for t, v in by.items()},
                 "failed_ms": failed, "cancelled_ms": cancelled, "request_percentiles": percentile_table(req),
                 "timeouts": sum(1 for r in _candidate_requests(samples) if r.get("error_category") == "timeout")},
    )


# ================================================================================================ 11. cost
def cost_per_successful_task(samples: list[dict]) -> MetricResult:
    key, name, unit = "cost_per_successful_task", "Cost per Successful Task", "USD"
    formula = "total attributable task cost (all tasks, incl. failed) / number of successful tasks"
    tasks = _active(samples)
    common = dict(group="cost", higher_is_better=False, description="Money spent per task that actually succeeded.",
                  evaluator="pricing registry x measured token usage")
    if not tasks:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula, "No tasks.", **common)
    costs = [s.get("cost") or {} for s in tasks]
    unknown = sum(1 for c in costs if c.get("status") == "unknown" or c.get("amount") is None)
    succ = sum(1 for s in tasks if s.get("outcomes", {}).get("task", {}).get("success"))
    versions = sorted({c.get("pricing_version", "") for c in costs if c.get("pricing_version")})
    if unknown:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula,
                           f"Incomplete: {unknown} task(s) lack token usage or a price in the registry - cost is not reported as zero.",
                           details={"tasks_with_unknown_cost": unknown}, **common)
    if succ == 0:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula, "No task succeeded, so cost per successful task is undefined.", **common)
    total = sum(c["amount"] for c in costs)
    estimated = any(c.get("status") == "estimated" for c in costs)
    currency = next((c.get("currency") for c in costs if c.get("currency")), "USD")
    notes = sorted({c.get("note", "") for c in costs if c.get("note")})
    value = total / succ
    return MetricResult(
        key=key, name=name, value=value, unit=currency, status=AVAILABLE, formula=formula, numerator=total, denominator=succ,
        sample_count=len(tasks), **common,
        working=[f"total cost of {len(tasks)} tasks = {total:.6f} {currency} ({'estimated' if estimated else 'measured usage x registry price'})",
                 f"successful tasks = {succ}", f"Cost per successful task = {total:.6f} / {succ} = {value:.6f} {currency}",
                 f"pricing version(s): {', '.join(versions) or 'n/a'}"],
        limitations=notes + (["Includes compute-time estimates (configured hourly rate)."] if estimated else []),
        details={"total_cost": total, "successful_tasks": succ, "estimated": estimated, "pricing_versions": versions},
    )


# ================================================================================================ 12. errors
TECHNICAL_ERRORS = ("timeout", "connection", "model_not_found", "provider_error", "cancelled")


def error_rate(samples: list[dict]) -> MetricResult:
    key, name, unit = "error_rate", "Error Rate", "%"
    formula = "failed model requests / attempted model requests x 100"
    reqs = _candidate_requests(samples)
    common = dict(group="reliability", higher_is_better=False,
                  description="Technical failures of model requests (timeouts, connection, provider errors).",
                  evaluator="request telemetry")
    if not reqs:
        return unavailable(key, name, unit, INSUFFICIENT_DATA, formula, "No model requests were attempted.", **common)
    failed = [r for r in reqs if r.get("status") == "error"]
    n = len(reqs)
    cats = Counter(r.get("error_category", "unknown") for r in failed)
    schema_fail = sum(1 for r in reqs if r.get("status") == "ok" and r.get("schema_valid") is False)
    judge = [r for s in samples for r in s.get("requests", []) if r.get("role") == "judge"]
    judge_failed = sum(1 for r in judge if r.get("status") == "error")
    repairs = sum(max(0, int(r.get("attempt", 1)) - 1) for r in reqs if r.get("attempt"))
    timeouts = cats.get("timeout", 0)
    by_model = defaultdict(lambda: [0, 0])
    for r in reqs:
        by_model[r.get("model", "")][0] += r.get("status") == "error"
        by_model[r.get("model", "")][1] += 1
    return MetricResult(
        key=key, name=name, value=100.0 * len(failed) / n, unit=unit, status=AVAILABLE, formula=formula, numerator=len(failed),
        denominator=n, sample_count=n, **common,
        working=[f"attempted requests = {n}, succeeded = {n - len(failed)}, failed = {len(failed)} {dict(cats) if cats else ''}",
                 f"Error rate = {len(failed)} / {n} x 100 = {fmt(100.0 * len(failed) / n, 2)}%",
                 f"timeout rate = {timeouts} / {n} = {fmt(100.0 * timeouts / n, 2)}%",
                 f"tracked separately (not technical errors): invalid-output replies = {schema_fail}, repair calls = {repairs}, "
                 f"judge request failures = {judge_failed} of {len(judge)}"],
        limitations=["Safety refusals, wrong answers and invalid JSON are not technical errors; they are covered by the quality, "
                     "instruction-following and safety metrics."],
        details={"categories": dict(cats), "succeeded": n - len(failed), "timeout_rate": 100.0 * timeouts / n,
                 "schema_failures": schema_fail, "repair_calls": repairs, "judge_requests": len(judge), "judge_failures": judge_failed,
                 "by_model": {m: {"failed": f, "attempted": t, "rate": 100.0 * f / t} for m, (f, t) in by_model.items()},
                 "examples": [{"category": r.get("error_category"), "message": r.get("error_message", "")[:200],
                               "purpose": r.get("purpose")} for r in failed][:MAX_EXAMPLES]},
    )


# ================================================================================================ registry
METRIC_ORDER = [
    "answer_accuracy", "f1_score", "answer_relevance", "faithfulness", "hallucination_rate", "instruction_following",
    "safety_attack_success_rate", "task_success_rate", "ttft_p95", "e2e_latency_p95", "cost_per_successful_task", "error_rate",
]


def compute_all(samples: list[dict], relevance_threshold: int = DEFAULT_RELEVANCE_THRESHOLD) -> list[MetricResult]:
    """All 12 metrics in dashboard order. A metric that raises is reported as evaluation_failed, never silently dropped."""
    fns = {
        "answer_accuracy": answer_accuracy, "f1_score": f1_score,
        "answer_relevance": lambda s: answer_relevance(s, relevance_threshold), "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate, "instruction_following": instruction_following,
        "safety_attack_success_rate": safety_attack_success_rate, "task_success_rate": task_success_rate, "ttft_p95": ttft_p95,
        "e2e_latency_p95": e2e_latency_p95, "cost_per_successful_task": cost_per_successful_task, "error_rate": error_rate,
    }
    out = []
    for key in METRIC_ORDER:
        try:
            out.append(fns[key](samples))
        except Exception as exc:  # pragma: no cover - defensive: never hide a broken metric
            out.append(MetricResult(key=key, name=key, value=None, unit="", status="evaluation_failed", formula="",
                                    limitations=[f"metric calculation failed: {type(exc).__name__}: {exc}"]))
    return out


TASKS = {MCQ_ANSWER: "Answer labelled multiple-choice questions", MCQ_GENERATION: "Generate MCQs from source material",
         SAFETY: "Adversarial material through the generator"}
