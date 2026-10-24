"""The evaluation metrics, checked against values worked out by hand (and textbook / library-documentation examples)."""
import math

import pytest

from evaluation.evaluate_llm import (
    NONE_LABEL,
    Prediction,
    build_prompt,
    compute_metrics,
    load_dataset,
    parse_letter,
    probs_from_logprobs,
    wilson_interval,
    DEFAULT_DATASET,
    Item,
    simulate_model,
)

L3 = ["A", "B", "C"]
L4 = list("ABCD")


def pred(gold, chosen, probs=None, labels=L4, source="logprobs"):
    if probs is None:  # confident in the chosen letter
        probs = {l: (0.7 if l == chosen else 0.3 / (len(labels) - 1)) for l in labels}
    return Prediction("q", gold, chosen, probs, "", 1.0, 10, 10.0, "c", "easy", source)


def metrics(preds, labels):
    return {m.name: m.value for m in compute_metrics(preds, labels)}


# ---------- the classic 3-class example used in scikit-learn's documentation -------------------------------------
# y_true = [0,1,2,0,1,2], y_pred = [0,2,1,0,0,1]  ->  macro P=0.2222 R=0.3333 F1=0.2667, micro all 0.3333, weighted F1=0.2667
def doc_example():
    gold, chosen = "ABCABC", "ACBAAB"
    return [pred(g, c, labels=L3) for g, c in zip(gold, chosen)]


def test_precision_recall_f1_variants_match_the_documented_example():
    m = metrics(doc_example(), L3)
    assert m["Precision (macro)"] == pytest.approx(0.2222, abs=1e-4)
    assert m["Recall (macro)"] == pytest.approx(0.3333, abs=1e-4)
    assert m["Macro-F1"] == pytest.approx(0.2667, abs=1e-4)
    assert m["Micro-F1"] == pytest.approx(1 / 3)
    assert m["Weighted-F1"] == pytest.approx(0.2667, abs=1e-4)
    assert m["Accuracy"] == pytest.approx(1 / 3)


def test_the_worked_example_in_the_evaluation_readme():
    # correct A,B,C; model says A,C,C -> accuracy 2/3, macro-F1 (1 + 0 + 2/3)/3, micro-F1 2/3
    m = metrics([pred(g, c, labels=L3) for g, c in zip("ABC", "ACC")], L3)
    assert m["Accuracy"] == pytest.approx(2 / 3)
    assert m["Macro-F1"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert m["Weighted-F1"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert m["Micro-F1"] == pytest.approx(2 / 3)


def test_kappa_worked_by_hand():
    # y_true=[2,0,2,2,0,1], y_pred=[0,0,2,2,0,2]: p_o = 4/6, p_e = (2*3 + 1*0 + 3*3)/36 -> kappa = 0.428571...
    gold, chosen = "CACCAB", "AACCAC"
    m = metrics([pred(g, c, labels=L3) for g, c in zip(gold, chosen)], L3)
    assert m["Cohen's kappa"] == pytest.approx(0.4285714, abs=1e-6)


def test_mcc_binary_example_from_the_library_docs():
    # y_true=[+1,+1,+1,-1], y_pred=[+1,-1,+1,+1] -> MCC = -1/3
    labels = ["P", "N"]
    m = metrics([pred(g, c, labels=labels) for g, c in zip("PPPN", "PNPP")], labels)
    assert m["Matthews Correlation Coefficient (MCC)"] == pytest.approx(-1 / 3)


def test_perfect_and_worst_predictions():
    perfect = [pred(g, g) for g in "ABCDABCD"]
    m = metrics(perfect, L4)
    assert m["Accuracy"] == 1 and m["Macro-F1"] == 1 and m["Matthews Correlation Coefficient (MCC)"] == pytest.approx(1)
    assert m["Cohen's kappa"] == pytest.approx(1)
    worst = [pred(g, c) for g, c in zip("ABCDABCD", "BCDABCDA")]
    mw = metrics(worst, L4)
    assert mw["Accuracy"] == 0 and mw["Macro-F1"] == 0 and mw["Matthews Correlation Coefficient (MCC)"] < 0


# ---------- probability metrics ----------------------------------------------------------------------------------
def test_log_loss_matches_a_hand_calculation():
    # two questions, correct option given p=0.8 and p=0.5 -> -(ln 0.8 + ln 0.5)/2
    preds = [pred("A", "A", {"A": 0.8, "B": 0.1, "C": 0.1}, L3), pred("B", "B", {"A": 0.3, "B": 0.5, "C": 0.2}, L3)]
    expected = -(math.log(0.8) + math.log(0.5)) / 2
    assert metrics(preds, L3)["Log loss (cross-entropy)"] == pytest.approx(expected)


def test_log_loss_of_equal_probabilities_is_ln_k_and_confident_wrong_answers_are_finite():
    equal = [pred("A", "A", {l: 0.25 for l in L4}) for _ in range(4)]
    assert metrics(equal, L4)["Log loss (cross-entropy)"] == pytest.approx(math.log(4))
    certain_wrong = [pred("A", "B", {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0})]
    loss = metrics(certain_wrong, L4)["Log loss (cross-entropy)"]
    assert loss == pytest.approx(-math.log(1e-15)) and math.isfinite(loss)


def test_brier_matches_a_hand_calculation_and_the_equal_probability_reference():
    p = {"A": 0.7, "B": 0.2, "C": 0.1}
    # one question, truth A: (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09 + 0.04 + 0.01 = 0.14
    assert metrics([pred("A", "A", p, L3)], L3)["Brier score"] == pytest.approx(0.14)
    equal = [pred("A", "A", {l: 0.25 for l in L4})]
    assert metrics(equal, L4)["Brier score"] == pytest.approx(1 - 1 / 4)  # 0.75


def test_ece_perfectly_calibrated_is_zero_and_overconfident_is_large():
    # confidence 0.8, correct 4 of 5 times -> accuracy equals confidence in the only bin -> ECE 0
    calibrated = [pred("A", "A", {"A": 0.8, "B": 0.1, "C": 0.1}, L3) for _ in range(4)] + [pred("A", "B", {"A": 0.1, "B": 0.8, "C": 0.1}, L3)]
    assert metrics(calibrated, L3)["Expected Calibration Error (ECE)"] == pytest.approx(0, abs=1e-12)
    # 99% confident but only 50% right -> |0.5 - 0.99| = 0.49
    over = [pred("A", "A", {"A": 0.99, "B": 0.005, "C": 0.005}, L3), pred("A", "B", {"A": 0.005, "B": 0.99, "C": 0.005}, L3)]
    assert metrics(over, L3)["Expected Calibration Error (ECE)"] == pytest.approx(0.49)


def test_top_k_accuracy():
    # gold C is the 2nd most probable in the first question, the 3rd in the second
    preds = [pred("C", "A", {"A": 0.5, "C": 0.3, "B": 0.2}, L3), pred("C", "A", {"A": 0.5, "B": 0.3, "C": 0.2}, L3)]
    m = metrics(preds, L3)
    assert m["Top-k accuracy"] == pytest.approx(0.5)  # top-2: 1 of 2


# ---------- unparsed replies ---------------------------------------------------------------------------------------
def test_unparsed_answers_count_as_wrong_and_use_equal_probabilities():
    preds = [pred("A", "A"), Prediction("q", "B", None, {l: 0.25 for l in L4}, "??", 1.0, 10, 10.0, "c", "easy", "uniform (no letter token)")]
    m = metrics(preds, L4)
    assert m["Accuracy"] == 0.5
    assert m["Exact Match (EM)"] == 0.5
    # TP=1, FP=0, FN=1 (the unparsed reply is a miss for its true class but a false positive for no class):
    # precision = 1/1, recall = 1/2, micro-F1 = 2*1*0.5/1.5 = 2/3 - so micro-F1 differs from accuracy here
    assert m["Micro-F1"] == pytest.approx(2 / 3)


def test_confusion_has_a_column_for_unparsed_replies():
    from evaluation.evaluate_llm import confusion

    cm = confusion(["A", "B"], ["A", None], L4)
    assert cm["B"][NONE_LABEL] == 1 and cm["A"]["A"] == 1


# ---------- helpers --------------------------------------------------------------------------------------------------
def test_wilson_interval_is_wide_for_small_samples():
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi and (hi - lo) > 0.3
    assert wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize("text,expected", [("B", "B"), (" b.", "B"), ("(C)", "C"), ("Answer: D", "D"), ("The answer is B", None), ("E", None), ("", None)])
def test_parse_letter(text, expected):
    assert parse_letter(text, L4) == expected


def test_probabilities_come_from_logprobs_and_are_renormalised_over_the_letters():
    entry = {"token": "B", "logprob": math.log(0.6), "top_logprobs": [
        {"token": "B", "logprob": math.log(0.6)}, {"token": " C", "logprob": math.log(0.2)},
        {"token": "b", "logprob": math.log(0.1)}, {"token": "The", "logprob": math.log(0.1)}]}
    letter, probs = probs_from_logprobs([entry], L4)
    assert letter == "B"
    assert sum(probs.values()) == pytest.approx(1)
    assert probs["B"] == pytest.approx(0.6 / 0.8) and probs["C"] == pytest.approx(0.2 / 0.8) and probs["A"] == 0


def test_a_reply_with_no_letter_token_gives_no_probabilities():
    assert probs_from_logprobs([{"token": "The", "logprob": -0.1, "top_logprobs": []}], L4) is None


def test_prompt_contains_passage_question_and_all_options():
    item = Item("Q1", "Pick one?", {"A": "x", "B": "y"}, "A", context="Some passage.")
    text = build_prompt(item)
    assert "Passage: Some passage." in text and "Question: Pick one?" in text and "A. x" in text and "B. y" in text
    assert "Passage" not in build_prompt(Item("Q2", "Pick?", {"A": "x", "B": "y"}, "B"))


# ---------- the shipped dataset ----------------------------------------------------------------------------------------
def test_dataset_is_valid_and_balanced():
    items = load_dataset(DEFAULT_DATASET)
    assert len(items) == 40 and len({i.id for i in items}) == 40
    assert all(i.answer in i.options and len(i.options) == 4 for i in items)
    assert all(len(set(i.options.values())) == 4 for i in items)  # no duplicate option texts
    counts = [sum(i.answer == l for i in items) for l in L4]
    assert max(counts) - min(counts) <= 3  # the correct letter is not concentrated on one position


def test_demo_runs_end_to_end_without_ollama():
    items = load_dataset(DEFAULT_DATASET)
    run = simulate_model(items, 0.8)
    preds = [Prediction(**p) for p in run["predictions"]]
    m = metrics(preds, L4)
    assert run["simulated"] and 0.5 < m["Accuracy"] <= 1
    assert all(0 <= v <= 2 for v in (m["Brier score"],))
