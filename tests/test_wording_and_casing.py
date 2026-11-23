"""Echo questions, names written as the material writes them, the different-fact rule and answering examples for
fine-tuning (no model needed)."""
import json
from pathlib import Path

import pytest

from models.schemas import GenerationRequest, Question
from services.document_processor import Chunk
from services.question_generator import QuestionGenerator, build_user_prompt
from services.question_validator import (
    QuestionValidationError,
    is_echo_question,
    restore_casing,
    source_casing,
    validate_question,
)

ROOT = Path(__file__).resolve().parents[1]
_P = json.loads((ROOT / "evaluation" / "passages.json").read_text(encoding="utf-8"))
PASSAGES = {p["id"]: p["text"] for p in (_P["passages"] if isinstance(_P, dict) else _P)}


# ------------------------------------------------------------------------------------------------ echo questions
@pytest.mark.parametrize("text", [  # written by quizforge-3b / quizforge-7b
    "React combines what with acting?",
    "Greedy best-first search uses only what?",
    "The a* algorithm evaluates each node with what formula?",
    "Depth first search uses what data structure?",
    "Users should never enter their password after following what?",
])
def test_statements_with_what_in_the_middle_are_echo_questions(text):
    assert is_echo_question(text)


@pytest.mark.parametrize("text", [
    "What does ReAct combine with acting?",
    "In which scenario would abductive reasoning be most appropriate?",
    "According to the material, which algorithm uses a stack?",
    "In plants, where does photosynthesis take place?",
    "Which of the following best describes ReAct?",
    "How many levels does the tree have?",
    "Is BFS guaranteed to find the shortest path?",
    "Name the pigment that absorbs red and blue light.",
    "The Calvin cycle takes place in the ____.",
    "Given f(n) = g(n) + h(n), what does h(n) estimate?",
])
def test_real_questions_are_not_echo_questions(text):
    assert not is_echo_question(text)


def test_an_echo_question_goes_back_to_the_model_with_a_reason():
    raw = {"question": "React combines what with acting?", "correct_answer": "A",
           "options": [{"id": l, "text": t} for l, t in zip("ABCD", ["reasoning", "planning", "searching", "memory"])],
           "explanation": "", "source_excerpt": "ReAct combines reasoning with acting."}
    with pytest.raises(QuestionValidationError, match="start it with a question word"):
        validate_question(raw, check_wording_rules=True)
    validate_question(raw)  # a teacher's own wording is not held to it


# ------------------------------------------------------------------------------------------------------ casing
@pytest.mark.parametrize("pid,model_text,expected", [
    ("P3", "What does react combine?", "What does ReAct combine?"),
    ("P3", "chain-of-thought (cot) and tree-of-thought (tot)", "chain-of-thought (CoT) and tree-of-thought (ToT)"),
    ("P2", "breadth first search (bfs)", "Breadth First Search (BFS)"),
    ("P2", "What does the a* algorithm evaluate?", "What does the A* algorithm evaluate?"),
    ("P2", "greedy best-first search comes first", "greedy best-first search comes first"),  # no stray "First"
    ("P4", "the calvin cycle uses atp and nadph", "the Calvin cycle uses ATP and NADPH"),  # a name mid-sentence
    ("P2", "depth first search goes deep", "Depth First Search goes deep"),
    ("P1", "deductive", "deductive"),  # capital only at the start of a sentence: grammar, not a name
])
def test_names_are_written_the_way_the_material_writes_them(pid, model_text, expected):
    assert restore_casing(model_text, source_casing(PASSAGES[pid])) == expected


def test_casing_only_ever_adds_capitals():
    casing = source_casing("The model is called ReAct. We use BFS here.")
    assert restore_casing("REACT and Bfs", casing) == "REACT and BFS"
    assert "because react" not in source_casing("Because ReAct uses tools, it is useful. ReAct is popular.")


def test_generated_questions_get_the_names_from_the_material():
    q = Question(question="What does react combine with acting?", correct_answer="A",
                 options=[{"id": l, "text": t} for l, t in zip("ABCD", ["reasoning", "cot", "tot", "planning"])],
                 explanation="react combines reasoning with acting.", source_excerpt="ReAct combines reasoning with acting.")
    out = QuestionGenerator._finalise(q, GenerationRequest(count=1), Chunk(0, PASSAGES["P3"], "Prompting"))
    assert out.question == "What does ReAct combine with acting?" and out.explanation.startswith("ReAct")
    assert {o.text for o in out.options} == {"reasoning", "CoT", "ToT", "planning"}
    assert next(o.text for o in out.options if o.id == out.correct_answer) == "reasoning"


# ---------------------------------------------------------------------------------------------- different facts
def test_the_prompt_asks_for_a_different_fact_per_question():
    rule = "Every question must test a different fact"
    assert rule in build_user_prompt(GenerationRequest(count=3), 3, "Some material.", "Part 1", [])
    assert rule not in build_user_prompt(GenerationRequest(count=1), 1, "Some material.", "Part 1", [])


# ------------------------------------------------------------------------------------- answering examples (replay)
SCIQ = [{"question": f"Which organelle is number {i}?", "correct_answer": f"organelle {i}",
         "distractor1": f"wrong a{i}", "distractor2": f"wrong b{i}", "distractor3": f"wrong c{i}",
         "support": f"Cells contain many parts. The answer is organelle {i}, which is found in every cell of the body."}
        for i in range(40)]


def test_answer_examples_use_the_apps_answer_check_format():
    from finetune.prepare_data import build_answer_examples, is_generation_example
    from services.answer_checker import SYSTEM

    examples = build_answer_examples(SCIQ, 20)
    assert len(examples) == 20 and not any(is_generation_example(e) for e in examples)
    for ex in examples:
        system, user, assistant = (m["content"] for m in ex["messages"])
        letter = json.loads(assistant)["answer"]
        assert system == SYSTEM and "Question: " in user and "\nOptions:\n" in user
        assert f"{letter}. organelle" in user  # the target letter is the correct option
    assert any(m["messages"][1]["content"].startswith("Context: ") for m in examples)
    assert any(not m["messages"][1]["content"].startswith("Context: ") for m in examples)  # some closed-book


def test_the_dataset_mixes_answering_examples_into_train_and_val():
    from finetune.build_dataset import build

    train, val, info = build(SCIQ, [], sciq_examples=10, answer_examples=20)
    assert info["answer_examples"] == 20
    assert info["train"]["by_source"].get("answer", 0) + info["val"]["by_source"].get("answer", 0) == 20
