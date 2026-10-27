"""Fine-tuning data pipeline: SciQ conversion, teacher filtering, dataset split, Ollama Modelfile, Colab bundle."""
import json

import pytest

from finetune import build_dataset, distill, make_colab_bundle
from finetune import prepare_data as P
from finetune.import_to_ollama import build_modelfile
from services.question_generator import SYSTEM_PROMPT
from services.question_validator import validate_question
from tests.eval_fakes import FakeModel

SUPPORTS = [
    "Plants are producers. Through photosynthesis, plants convert light energy into chemical energy stored in glucose. "
    "This happens in the chloroplasts of leaf cells.",
    "The mitochondria are known as the powerhouse of the cell. Cellular respiration takes place in the mitochondria and "
    "releases energy from glucose.",
    "Water boils at 100 degrees Celsius at sea level. At higher altitudes the air pressure is lower, so water boils at a "
    "lower temperature.",
    "Igneous rock forms when magma or lava cools and hardens. Granite is an igneous rock that cools slowly underground.",
]
ROWS = [
    {"question": "What process do plants use to convert light energy into chemical energy?", "correct_answer": "photosynthesis",
     "distractor1": "respiration", "distractor2": "fermentation", "distractor3": "transpiration", "support": SUPPORTS[0]},
    {"question": "Which organelle is the site of cellular respiration?", "correct_answer": "mitochondria",
     "distractor1": "ribosome", "distractor2": "nucleus", "distractor3": "vacuole", "support": SUPPORTS[1]},
    {"question": "At sea level, water boils at what temperature?", "correct_answer": "100 degrees Celsius",
     "distractor1": "90 degrees Celsius", "distractor2": "120 degrees Celsius", "distractor3": "80 degrees Celsius", "support": SUPPORTS[2]},
    {"question": "What type of rock forms when magma cools and hardens?", "correct_answer": "igneous",
     "distractor1": "sedimentary", "distractor2": "metamorphic", "distractor3": "fossil", "support": SUPPORTS[3]},
]


def rows(n=40):
    return [dict(r, question=f"{r['question'][:-1]} (variant {i})?", support=r["support"] + f" Note {i}.") for i in range(n // 4) for r in ROWS]


def target(ex):
    return json.loads(ex["messages"][2]["content"])["questions"]


def material(ex):
    return ex["messages"][1]["content"].split("<material>\n")[1].split("\n</material>")[0]


# --------------------------------------------------------------------------------------------- SciQ conversion
def test_sciq_examples_are_exactly_what_the_app_would_accept():
    examples = P.build_sciq_examples(rows(), 10, seed=3)
    assert examples
    for ex in examples:
        system, user, assistant = (m["role"] for m in ex["messages"])
        assert (system, user, assistant) == ("system", "user", "assistant")
        assert ex["messages"][0]["content"] == SYSTEM_PROMPT
        qs = target(ex)
        assert f"Write exactly {len(qs)} multiple-choice question(s)" in ex["messages"][1]["content"]
        for q in qs:
            validate_question(q, num_options=len(q["options"]), source_text=material(ex), require_excerpt=True)
            assert q["source_excerpt"] in material(ex)  # verbatim, as rule 10 demands
            assert len(q["source_excerpt"]) <= 250


def test_variety_of_counts_options_and_answer_positions():
    examples = P.build_sciq_examples(rows(200), 60, seed=5)
    counts = {len(target(e)) for e in examples}
    options = {len(q["options"]) for e in examples for q in target(e)}
    keys = {q["correct_answer"] for e in examples for q in target(e)}
    assert counts == {1, 2, 3} and options == {3, 4} and keys >= {"A", "B", "C"}
    no_expl = [e for e in examples if 'Leave "explanation" as an empty string.' in e["messages"][1]["content"]]
    assert no_expl and all(q["explanation"] == "" for e in no_expl for q in target(e))


def test_bad_rows_are_dropped():
    good = ROWS[0]
    assert P.clean_row(good)
    assert P.clean_row(dict(good, support="")) is None
    assert P.clean_row(dict(good, distractor1="Photosynthesis")) is None  # duplicate option
    assert P.clean_row(dict(good, distractor2="all of the above")) is None


def test_excerpt_backs_the_answer():
    ex = P.pick_excerpt(SUPPORTS[1], ROWS[1]["question"], "mitochondria")
    assert ex == "Cellular respiration takes place in the mitochondria and releases energy from glucose."
    assert P.pick_excerpt("Short. Tiny.", "What is x?", "x") is None


# ---------------------------------------------------------------------------------------------- teacher step
class Checker:
    def __init__(self, letter):
        self.letter = letter

    def chat(self, messages, schema=None, temperature=None, num_ctx=8192, on_progress=None):
        return json.dumps({"answer": self.letter})


def test_teacher_questions_are_kept_only_when_the_checker_agrees():
    passages = distill.make_passages(rows(), 3)
    assert len(passages) == 3 and all("\n\n" in p for p in passages)
    kept, stats = distill.distill(passages, FakeModel(), Checker("B"), log=lambda *_: None)  # FakeModel's keys are "B"
    assert len(kept) == 3 and stats["agreed"] == stats["generated"] > 0
    for ex in kept:
        assert ex["source"] == "distilled"
        for q in target(ex):
            validate_question(q, num_options=4, source_text=material(ex), require_excerpt=True)
    none, stats = distill.distill(passages, FakeModel(), Checker("A"), log=lambda *_: None)
    assert none == [] and stats["disagreed"] == stats["generated"]


# ------------------------------------------------------------------------------------------------ dataset split
def test_build_weights_teacher_examples_without_leaking_into_validation():
    distilled, _ = distill.distill(distill.make_passages(rows(), 6), FakeModel(), Checker("B"), log=lambda *_: None)
    train, val, info = build_dataset.build(rows(200), distilled, 40, distilled_weight=2, val_fraction=0.2)
    key = lambda e: e["messages"][1]["content"]  # noqa: E731
    assert not {key(e) for e in train} & {key(e) for e in val}
    train_dist = [e for e in train if e["source"] == "distilled"]
    assert len(train_dist) == 2 * len({key(e) for e in train_dist})
    assert info["train"]["by_source"]["distilled"] == len(train_dist)


def test_jsonl_roundtrip(tmp_path):
    ex = P.build_sciq_examples(rows(), 3, seed=1)
    path = P.write_jsonl(tmp_path / "d" / "x.jsonl", ex)
    assert P.read_jsonl(path) == ex


# ----------------------------------------------------------------------------------------------- Ollama import
QWEN_TEMPLATE = "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}..."


def test_modelfile_reuses_the_base_template_and_stop_tokens():
    mf = build_modelfile("quizforge-3b.Q4_K_M.gguf", QWEN_TEMPLATE, ['stop "<|im_start|>"', 'stop "<|im_end|>"'])
    assert mf.startswith("FROM ./quizforge-3b.Q4_K_M.gguf\n")
    assert f'TEMPLATE """{QWEN_TEMPLATE}"""' in mf
    assert 'PARAMETER stop "<|im_end|>"' in mf


def test_modelfile_adds_qwen_stop_tokens_when_the_base_has_none():
    mf = build_modelfile("x.gguf", QWEN_TEMPLATE, [])
    assert 'PARAMETER stop "<|im_end|>"' in mf and 'PARAMETER stop "<|endoftext|>"' in mf
    assert mf.count("PARAMETER stop") == 2


def test_modelfile_refuses_a_missing_template():
    with pytest.raises(ValueError):
        build_modelfile("x.gguf", "", [])


def test_colab_bundle_contains_code_only():
    names = [p.relative_to(make_colab_bundle.ROOT).as_posix() for p in make_colab_bundle.bundle_files()]
    assert "finetune/train.py" in names and "services/question_generator.py" in names and "config.py" in names
    assert not [n for n in names if n.endswith((".db", ".env", ".gguf", ".jsonl", ".pyc")) or "/exports/" in n or "/data/" in n]


# ------------------------------------------------------------------------------------------- export clean-up
def test_export_cleanup_keeps_the_final_model_and_adapter(tmp_path):
    import time

    from finetune.train import remove_intermediates

    out = tmp_path / "exports"
    merged = out / "quizforge-7b-gguf"
    merged.mkdir(parents=True)
    (merged / "model-00001.safetensors").write_bytes(b"x" * 1000)
    (merged / "unsloth.BF16.gguf").write_bytes(b"x" * 500)
    adapter = out / "quizforge-7b-lora"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x" * 10)
    final = out / "quizforge-7b.Q4_K_M.gguf"
    final.write_bytes(b"x" * 100)
    started = time.time() - 1
    freed = remove_intermediates([str(merged), str(out)], started, keep=final, dirs=[merged, out / "quizforge-7b-checkpoints"])
    assert freed == 1500
    assert final.exists() and adapter.exists() and not merged.exists()
