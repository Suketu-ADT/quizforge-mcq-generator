"""The 12 evaluation metrics, checked against hand-worked values and every edge case (empty data, zero denominators,
missing evidence / usage / prices, failed and cancelled requests, invalid judge output)."""
import math

import pytest

from services.evaluation import metrics as M
from services.evaluation.contracts import AVAILABLE, INSUFFICIENT_DATA, NOT_APPLICABLE
from services.evaluation.stats import percentile, percentile_table, prf, wilson_interval


def sample(task_type="mcq_answer", status="completed", **outcomes):
    s = {"id": f"s{id(outcomes)}", "task_type": task_type, "status": status, "dataset_id": "d1", "requests": outcomes.pop("requests", []),
         "timing": outcomes.pop("timing", {"e2e_ms": None}), "cost": outcomes.pop("cost", {}), "output": outcomes.pop("output", {}),
         "outcomes": outcomes}
    return s


def acc(correct, evaluable=True, method="mcq_letter_exact_match"):
    return {"evaluable": evaluable, "correct": correct, "method": method, "prediction": "B", "reference": "A", "item": "q"}


def req(status="ok", ttft=100.0, latency=500.0, cat=None, role="candidate", tokens=(10, 5), schema_valid=None, attempt=1, streamed=True):
    return {"role": role, "status": status, "ttft_ms": ttft, "latency_ms": latency, "error_category": cat, "streamed": streamed,
            "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "schema_valid": schema_valid, "attempt": attempt, "model": "m",
            "purpose": "generate", "load_ms": 1.0, "prompt_eval_ms": 2.0}


# ------------------------------------------------------------------------------------------------- statistics
def test_percentile_linear_interpolation_matches_hand_calculation():
    # sorted 10,20,30,40,50 -> p95: h = 4*0.95 = 3.8 -> 40 + 0.8*(50-40) = 48
    assert percentile([30, 10, 50, 20, 40], 95) == pytest.approx(48)
    assert percentile([30, 10, 50, 20, 40], 50) == 30
    assert percentile([7], 99) == 7 and percentile([], 50) is None


def test_percentiles_need_enough_observations():
    t = percentile_table([float(i) for i in range(1, 20)])  # 19 values
    assert t["p50"] is not None and t["p90"] is not None and t["p95"] is None and t["p99"] is None
    assert percentile_table([float(i) for i in range(20)])["p95"] is not None


def test_prf_zero_division_is_zero_and_wilson_is_sane():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(2, 2, 0) == (0.5, 1.0, pytest.approx(2 / 3))
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi and wilson_interval(0, 0) == (0.0, 0.0)


# ----------------------------------------------------------------------------------------------------- 1. accuracy
def test_accuracy_counts_only_evaluable_answers():
    s = [sample(accuracy=[acc(True)]), sample(accuracy=[acc(False)]), sample(accuracy=[acc(None, evaluable=False, method="human_review_pending")]),
         sample(accuracy=[acc(True)])]
    m = M.answer_accuracy(s)
    assert m.status == AVAILABLE and m.value == pytest.approx(100 * 2 / 3) and (m.numerator, m.denominator) == (2, 3)
    assert m.details["incorrect"] == 1 and m.details["failures"][0]["reference"] == "A"
    assert "method:mcq_letter_exact_match" in m.details["breakdown"]


def test_accuracy_without_evaluable_data_is_not_zero():
    m = M.answer_accuracy([sample(accuracy=[acc(None, evaluable=False)])])
    assert m.value is None and m.status == INSUFFICIENT_DATA
    assert M.answer_accuracy([]).status == INSUFFICIENT_DATA


def test_cancelled_samples_never_count():
    m = M.answer_accuracy([sample(accuracy=[acc(True)]), sample(status="cancelled", accuracy=[acc(False)])])
    assert m.value == 100.0 and m.denominator == 1


# ----------------------------------------------------------------------------------------------------------- 2. F1
def cls(gold, pred, labels="ABC"):
    return sample(classification={"gold": gold, "pred": pred, "labels": list(labels)})


def test_f1_variants_match_the_documented_example():
    # y_true = ABCABC, y_pred = ACBAAB -> macro P 0.2222 R 0.3333 F1 0.2667, micro 0.3333, weighted 0.2667
    m = M.f1_score([cls(g, p) for g, p in zip("ABCABC", "ACBAAB")])
    assert m.value == pytest.approx(0.2667, abs=1e-4)
    assert m.details["micro_f1"] == pytest.approx(1 / 3) and m.details["weighted_f1"] == pytest.approx(0.2667, abs=1e-4)
    assert m.details["per_class"]["A"]["precision"] == pytest.approx(2 / 3)
    assert m.details["confusion"]["B"]["C"] == 1


def test_f1_binary_and_unparsed_predictions():
    m = M.f1_score([cls("Y", "Y", "YN"), cls("N", "Y", "YN"), cls("Y", None, "YN")])
    # class Y: TP1 FP1 FN1 -> F1 0.5 ; class N: TP0 FP0 FN1 -> 0
    assert m.details["per_class"]["Y"]["f1"] == pytest.approx(0.5) and m.value == pytest.approx(0.25)
    assert m.details["confusion"]["Y"]["?"] == 1


def test_f1_macro_ignores_option_letters_that_never_occur():
    # two questions, both answered correctly, only A and B ever appear: C and D have nothing to measure
    m = M.f1_score([cls("A", "A", "ABCD"), cls("B", "B", "ABCD")])
    assert m.value == pytest.approx(1.0)  # sklearn f1_score(["A","B"], ["A","B"], average="macro") == 1.0
    assert "not averaged: C, D" in m.working[-3]


def test_f1_is_na_without_labelled_classification():
    m = M.f1_score([sample(accuracy=[acc(True)])])
    assert m.status == NOT_APPLICABLE and m.value is None


# -------------------------------------------------------------------------------------------------- 3. relevance
def rel(score, status="ok"):
    return {"score": score, "status": status, "rationale": "r", "target": "t", "judge": "j"}


def test_relevance_mean_distribution_and_threshold():
    m = M.answer_relevance([sample(task_type="mcq_generation", relevance=[rel(5), rel(4), rel(2), rel(None, "invalid"), rel(9, "ok")])], threshold=4)
    # valid scores 5,4,2 (the 9 fails schema validation) -> mean 11/3, pass 2/3
    assert m.value == pytest.approx(11 / 3) and m.denominator == 3
    assert m.details["pass_rate"] == pytest.approx(200 / 3) and m.details["excluded"] == 2
    assert m.details["distribution"]["2"] == 1 and m.details["low_examples"][0]["score"] == 2


def test_relevance_without_judge_or_valid_scores():
    assert M.answer_relevance([sample()]).status == NOT_APPLICABLE
    m = M.answer_relevance([sample(relevance=[rel(None, "judge_error")])])
    assert m.status == "evaluation_failed" and m.value is None


# ------------------------------------------------------------------------------------- 4 + 5. faithfulness, hallucination
def claim(verdict, status="ok", q=0):
    return {"claim": "c", "verdict": verdict, "status": status, "question_index": q, "rationale": "r", "evidence_ref": "e"}


def claims_sample():
    return sample(task_type="mcq_generation", claims=[claim("supported"), claim("supported"), claim("unsupported", q=1),
                                                 claim("contradicted", q=1), claim("unverifiable"), claim(None, "invalid"),
                                                 claim(None, "no_evidence")])


def test_faithfulness_excludes_unverifiable_and_failures():
    m = M.faithfulness([claims_sample()])
    # decided = 2 supported + 1 unsupported + 1 contradicted = 4 -> 2/4
    assert m.value == pytest.approx(50.0) and (m.numerator, m.denominator) == (2, 4)
    assert m.details["excluded"] == {"unverifiable": 1, "no_evidence": 1, "verifier_failures": 1}


def test_hallucination_rate_tracks_each_category_separately():
    m = M.hallucination_rate([claims_sample()])
    assert m.value == pytest.approx(50.0) and m.numerator == 2
    d = m.details
    assert d["unsupported"] == 1 and d["contradicted"] == 1 and d["unverifiable"] == 1
    assert d["unverifiable_rate"] == pytest.approx(100 * 1 / 5)  # of the 5 judged claims
    assert d["answers_with_hallucination"] == 1  # both flagged claims belong to question 1
    assert any("not proof" in lim for lim in m.limitations)


def test_missing_evidence_is_reported_as_unavailable_not_perfect():
    m = M.faithfulness([sample(claims=[claim(None, "no_evidence"), claim(None, "no_evidence")])])
    assert m.value is None and m.status == NOT_APPLICABLE
    assert M.faithfulness([sample()]).status == NOT_APPLICABLE
    assert M.hallucination_rate([sample(claims=[claim("unverifiable")])]).status == INSUFFICIENT_DATA


# ------------------------------------------------------------------------------------------ 6. instruction following
def ins(rid, ok):
    return {"id": rid, "description": rid, "satisfied": ok, "detail": "", "method": "deterministic"}


def test_instruction_following_scores_each_requirement():
    m = M.instruction_following([sample(task_type="mcq_generation", instructions=[ins("json_schema", True), ins("option_count", False),
                                                                             ins("option_count", True), ins("user_instructions", None)])])
    assert m.value == pytest.approx(100 * 2 / 3) and m.details["not_evaluable"] == 1
    assert m.details["violations"][0]["requirement"] == "option_count"
    assert m.details["by_requirement"]["option_count"] == {"satisfied": 1, "evaluable": 2, "rate": 50.0}


def test_instruction_following_without_requirements():
    assert M.instruction_following([sample()]).status == NOT_APPLICABLE


# ------------------------------------------------------------------------------------------------------- 7. safety
def safety(outcome, kind="attack", cat="prompt_injection", status="completed"):
    return sample(task_type="safety", status=status, safety={"outcome": outcome, "kind": kind, "category": cat, "detail": "d"})


def test_attack_success_rate_by_category_with_refusals_and_over_refusals():
    s = [safety("attack_succeeded"), safety("resisted"), safety("safe_refusal", cat="jailbreak"), safety("needs_review", cat="jailbreak"),
         safety(None, status="error"), safety("complied", kind="benign", cat="benign_control"),
         safety("over_refusal", kind="benign", cat="benign_control")]
    m = M.safety_attack_success_rate(s)
    assert m.value == pytest.approx(100 / 3) and m.denominator == 3  # needs_review + error excluded
    assert m.details["by_category"]["prompt_injection"] == {"succeeded": 1, "total": 2, "rate": 50.0}
    assert m.details["safe_refusals"] == 1 and m.details["needs_review"] == 1
    assert m.details["over_refusals"] == 1 and m.details["over_refusal_rate"] == 50.0


def test_safety_without_a_grader_result_is_not_fabricated():
    assert M.safety_attack_success_rate([safety("needs_review")]).status == INSUFFICIENT_DATA
    assert M.safety_attack_success_rate([sample()]).status == NOT_APPLICABLE


# -------------------------------------------------------------------------------------------------- 8. task success
def test_task_success_rate_and_breakdown():
    s = [sample(task={"success": True}), sample(task={"success": False, "failure_reason": "delivered 1 of 3"}),
         sample(task_type="mcq_generation"), sample(status="cancelled", task={"success": True})]
    s[2]["outcomes"]["task"] = {"success": True}
    m = M.task_success_rate(s)
    assert m.value == pytest.approx(100 * 2 / 3) and m.denominator == 3
    assert m.details["failures"][0]["reason"] == "delivered 1 of 3"
    assert M.task_success_rate([sample(task={"success": None})]).status == INSUFFICIENT_DATA


# --------------------------------------------------------------------------------------------------------- 9. TTFT
def test_ttft_uses_only_streamed_first_token_observations():
    reqs = [req(ttft=float(i)) for i in range(1, 21)] + [req(ttft=None, streamed=False), req(status="error", ttft=None), req(role="judge", ttft=9999)]
    m = M.ttft_p95([sample(requests=reqs)])
    assert m.sample_count == 20 and m.value == pytest.approx(percentile([float(i) for i in range(1, 21)], 95))
    assert m.status == AVAILABLE


def test_ttft_never_substitutes_total_latency():
    m = M.ttft_p95([sample(requests=[req(ttft=None, streamed=False, latency=800)])])
    assert m.value is None and m.status == INSUFFICIENT_DATA
    few = M.ttft_p95([sample(requests=[req(ttft=100.0)] * 5)])
    assert few.value is None and few.status == INSUFFICIENT_DATA and few.details["percentiles"]["p50"] == 100.0


# ------------------------------------------------------------------------------------------------------ 10. latency
def test_e2e_latency_percentiles_and_separate_failed_and_cancelled():
    s = [sample(timing={"e2e_ms": float(v)}) for v in range(100, 2100, 100)]  # 20 completed: 100..2000
    s += [sample(status="error", timing={"e2e_ms": 5000.0}), sample(status="cancelled", timing={"e2e_ms": 50.0})]
    m = M.e2e_latency_p95(s)
    assert m.sample_count == 20 and m.value == pytest.approx(percentile([float(v) for v in range(100, 2100, 100)], 95))
    assert m.details["failed_ms"] == [5000.0] and m.details["cancelled_ms"] == [50.0]
    assert M.e2e_latency_p95([sample()]).status == INSUFFICIENT_DATA


# --------------------------------------------------------------------------------------------------------- 11. cost
def cost(amount, status="measured"):
    return {"amount": amount, "status": status, "currency": "USD", "pricing_version": "v1", "note": ""}


def test_cost_per_successful_task_includes_failed_tasks_cost():
    s = [sample(task={"success": True}, cost=cost(0.02)), sample(task={"success": False}, cost=cost(0.01)),
         sample(task={"success": True}, cost=cost(0.03))]
    m = M.cost_per_successful_task(s)
    assert m.value == pytest.approx(0.06 / 2) and (m.numerator, m.denominator) == (pytest.approx(0.06), 2)


def test_unknown_costs_are_never_zero():
    m = M.cost_per_successful_task([sample(task={"success": True}, cost=cost(None, "unknown")), sample(task={"success": True}, cost=cost(0.0))])
    assert m.value is None and m.status == INSUFFICIENT_DATA
    assert "not reported as zero" in " ".join(m.limitations)


def test_no_success_means_cost_per_success_is_unavailable():
    m = M.cost_per_successful_task([sample(task={"success": False}, cost=cost(0.01))])
    assert m.value is None and m.status == INSUFFICIENT_DATA


def test_estimated_costs_are_labelled():
    m = M.cost_per_successful_task([sample(task={"success": True}, cost=cost(0.5, "estimated"))])
    assert m.details["estimated"] is True and any("estimate" in lim for lim in m.limitations)


# --------------------------------------------------------------------------------------------------------- 12. errors
def test_error_rate_counts_technical_failures_only():
    reqs = [req(), req(), req(status="error", cat="timeout"), req(status="error", cat="connection"), req(schema_valid=False),
            req(attempt=2), req(role="judge", status="error", cat="timeout")]
    m = M.error_rate([sample(requests=reqs)])
    assert m.value == pytest.approx(100 * 2 / 6) and (m.numerator, m.denominator) == (2, 6)
    d = m.details
    assert d["categories"] == {"timeout": 1, "connection": 1} and d["timeout_rate"] == pytest.approx(100 / 6)
    assert d["schema_failures"] == 1 and d["repair_calls"] == 1 and d["judge_failures"] == 1
    assert M.error_rate([sample()]).status == INSUFFICIENT_DATA


# ------------------------------------------------------------------------------------------------------- registry
def test_compute_all_returns_all_12_in_dashboard_order_even_for_empty_input():
    out = M.compute_all([])
    assert [m.key for m in out] == M.METRIC_ORDER and len(out) == 12
    assert all(m.value is None for m in out) and all(m.status != AVAILABLE for m in out)
    assert all(not (isinstance(m.value, float) and math.isnan(m.value)) for m in out)
