"""The use-case evaluation (generate questions from source material) - offline, with a scripted fake model."""
import csv
import json

import pytest

from evaluation import generation_suite as gs
from evaluation.evaluate_llm import main as evaluate_main
from tests.test_question_generator import MATERIAL, q_json


class FakeClient:
    """Stands in for OllamaClient: replays scripted replies and keeps usage counters like the real client."""

    def __init__(self, replies, model="fake:1b"):
        self.replies, self.model = list(replies), model
        self.usage = self._new()

    @staticmethod
    def _new():
        return {"calls": 0, "prompt_tokens": 0, "prompt_seconds": 0.0, "gen_tokens": 0, "gen_seconds": 0.0}

    def reset_usage(self):
        done, self.usage = self.usage, self._new()
        return done

    def chat(self, messages, schema=None, **kw):
        self.usage["calls"] += 1
        self.usage["gen_tokens"] += 100
        self.usage["gen_seconds"] += 5.0
        return self.replies.pop(0) if self.replies else '{"questions": []}'


def answer(letter):
    return json.dumps({"answer": letter})


PASSAGE = {"id": "P1", "title": "Systems", "text": MATERIAL}


# ---------------------------------------------------------------------------------------------- classification
@pytest.mark.parametrize("reason,kind", [
    ("The model reply was not valid JSON.", "json"),
    ("The model returned an empty reply.", "json"),
    ("missing questions array", "json"),
    ("source excerpt was not found in the supplied material", "grounding"),
    ("no supporting source excerpt was given", "grounding"),
    ("duplicate question", "duplicate"),
    ("expected 4 options but got 3", "format"),
    ("correct_answer 'Q' does not match any option id", "format"),
    ("the options were written inside the question text", "format"),
])
def test_rejection_reasons_are_classified(reason, kind):
    assert gs.classify_rejection(reason) == kind


# -------------------------------------------------------------------------------------------------- running
def test_suite_runs_the_real_generator_and_checks_every_key():
    # generation reply (2 valid questions), then one blind check per accepted question
    client = FakeClient([q_json([1, 2]), answer("A"), answer("B")])
    judge = FakeClient([answer("A"), answer("A")], model="judge:7b")
    gen = gs.run_generation_suite(client, [PASSAGE], count=2, judge_client=judge, log=lambda *_: None)
    p = gen["passages"][0]
    assert (p["requested"], p["accepted"]) == (2, 2) and p["usage"]["calls"] == 1
    assert [q["self_check"] for q in p["questions"]] == ["agrees", "disagrees"]  # key is A for both: A agrees, B disagrees
    assert [q["judge_check"] for q in p["questions"]] == ["agrees", "agrees"]
    assert gen["judge"] == "judge:7b" and gen["model"] == "fake:1b"


def test_a_failing_passage_does_not_lose_the_others():
    from services.ollama_client import OllamaTimeoutError

    class Boom(FakeClient):
        def chat(self, *a, **k):
            if not self.replies:
                raise OllamaTimeoutError("slow")
            return super().chat(*a, **k)

    gen = gs.run_generation_suite(Boom([]), [PASSAGE, PASSAGE], count=2, log=lambda *_: None)
    assert [p["accepted"] for p in gen["passages"]] == [0, 0] and all(p["error"] for p in gen["passages"])


# ---------------------------------------------------------------------------------------------- the maths
def gen_fixture():
    """2 passages x 3 requested = 6; accepted 4; rejections: 1 json, 1 grounding, 1 duplicate, 1 format."""
    def q(check, judge):
        return {"question": "q", "options": [{"id": "A", "text": "x"}, {"id": "B", "text": "y"}], "correct_answer": "A",
                "source_excerpt": "s", "self_check": check, "judge_check": judge}
    return {"model": "m", "judge": "j", "count_per_passage": 3, "num_options": 2, "timestamp": "t", "passages": [
        {"id": "P1", "title": "a", "requested": 3, "accepted": 3, "seconds": 60.0, "error": "", "warnings": [],
         "rejected": ["The model reply was not valid JSON.", "source excerpt was not found in the supplied material"],
         "usage": {"calls": 3, "prompt_tokens": 0, "prompt_seconds": 0, "gen_tokens": 600, "gen_seconds": 60.0},
         "questions": [q("agrees", "agrees"), q("agrees", "disagrees"), q("disagrees", "agrees")]},
        {"id": "P2", "title": "b", "requested": 3, "accepted": 1, "seconds": 40.0, "error": "", "warnings": [],
         "rejected": ["duplicate question", "expected 4 options but got 3"],
         "usage": {"calls": 3, "prompt_tokens": 0, "prompt_seconds": 0, "gen_tokens": 400, "gen_seconds": 40.0},
         "questions": [q("agrees", "agrees")]},
    ]}


def by_name(gen):
    return {m.name: m for m in gs.generation_metrics(gen)}


def test_generation_metrics_match_hand_calculations():
    m = by_name(gen_fixture())
    assert m["Delivery rate"].value == pytest.approx(4 / 6)
    # candidates = accepted 4 + question-level rejections (grounding 1 + duplicate 1 + format 1) = 7
    assert m["Acceptance rate"].value == pytest.approx(4 / 7)
    assert m["Valid-JSON rate"].value == pytest.approx((6 - 1) / 6)
    assert m["Model calls per accepted question"].value == pytest.approx(6 / 4)
    assert m["Ungrounded-candidate rate"].value == pytest.approx(1 / 7)
    assert m["Duplicate-candidate rate"].value == pytest.approx(1 / 7)
    assert m["Format-defect rate"].value == pytest.approx(1 / 7)
    assert m["Seconds per accepted question"].value == pytest.approx(100 / 4)
    assert m["Generation speed (tokens/s)"].value == pytest.approx(1000 / 100)
    assert m["Blind-answer agreement (same model)"].value == pytest.approx(3 / 4)
    assert m["Blind-answer agreement (judge model)"].value == pytest.approx(3 / 4)


def test_every_metric_shows_its_working():
    for metric in gs.generation_metrics(gen_fixture()):
        assert metric.formula and metric.working, metric.name
        assert any("=" in line for line in metric.working)


def test_empty_run_does_not_divide_by_zero():
    gen = gen_fixture()
    for p in gen["passages"]:
        p.update(accepted=0, rejected=[], questions=[], seconds=0.0, usage={"calls": 0, "prompt_tokens": 0, "prompt_seconds": 0, "gen_tokens": 0, "gen_seconds": 0})
    assert all(m.value == 0 for m in gs.generation_metrics(gen))


# -------------------------------------------------------------------------------------- human labels and scoring
def test_labels_csv_has_one_row_per_question_and_an_empty_label_column():
    rows = list(csv.DictReader(gs.labels_csv(gen_fixture()).splitlines()))
    assert len(rows) == 4 and all(r[gs.LABEL_COLUMN] == "" for r in rows)
    assert {"question", "key", "source_excerpt", "self_check", "judge_check"} <= set(rows[0])


def write_labelled(tmp_path, labels):
    rows = list(csv.DictReader(gs.labels_csv(gen_fixture()).splitlines()))
    path = tmp_path / "labels.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r, label in zip(rows, labels):
            r[gs.LABEL_COLUMN] = label
            w.writerow(r)
    return path


def test_score_labels_computes_key_accuracy_and_flag_quality(tmp_path):
    # rows (same-model check): 1 agrees, 2 agrees, 3 DISAGREES, 4 agrees
    # human: 1 yes, 2 no (a wrong key the check MISSED), 3 no (a wrong key the check CAUGHT), 4 yes
    metrics, info = gs.score_labels(write_labelled(tmp_path, ["yes", "no", "no", "yes"]))
    m = {x.name: x.value for x in metrics}
    assert m["Key accuracy (human-labelled)"] == pytest.approx(2 / 4)
    # positives = "key is wrong": TP=1 (row 3), FP=0, FN=1 (row 2), TN=2 (rows 1, 4)
    assert m["Flag precision (same-model check)"] == pytest.approx(1.0)
    assert m["Wrong-key recall (same-model check)"] == pytest.approx(0.5)
    assert m["Flag F1 (same-model check)"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert m["Check accuracy (same-model check)"] == pytest.approx(3 / 4)
    assert m["MCC (same-model check)"] == pytest.approx((1 * 2 - 0 * 1) / (1 * 2 * 3 * 2) ** 0.5)
    assert "4 labelled" in info[0]


def test_score_labels_accepts_various_spellings_and_ignores_blanks(tmp_path):
    metrics, info = gs.score_labels(write_labelled(tmp_path, ["Y", "wrong", "", "TRUE"]))
    assert "3 labelled" in info[0]
    assert {x.name: x.value for x in metrics}["Key accuracy (human-labelled)"] == pytest.approx(2 / 3)


def test_score_labels_without_any_label_explains_what_to_do(tmp_path):
    metrics, info = gs.score_labels(write_labelled(tmp_path, ["", "", "", ""]))
    assert metrics == [] and "human_key_correct" in info[-1]


def test_cli_scores_a_labels_file(tmp_path, capsys):
    path = write_labelled(tmp_path, ["yes", "no", "no", "yes"])
    assert evaluate_main(["--score-labels", str(path), "--no-save"]) == 0
    assert "Key accuracy (human-labelled)" in capsys.readouterr().out
