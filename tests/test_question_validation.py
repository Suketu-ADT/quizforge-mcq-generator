import pytest

from models.schemas import Option, Question
from services.question_validator import (
    MalformedOutputError,
    QuestionValidationError,
    excerpt_in_source,
    find_duplicate_pairs,
    is_duplicate,
    parse_model_json,
    validate_question,
)
from tests.helpers import make_question, raw_question_json

SOURCE = "Photosynthesis converts light energy into chemical energy. It happens in the chloroplast."


def valid_raw(**over):
    raw = raw_question_json(1, "Photosynthesis converts light energy into chemical energy.")
    raw.update(over)
    return raw


def test_valid_question_passes():
    q = validate_question(valid_raw(), num_options=4, source_text=SOURCE)
    assert isinstance(q, Question)
    assert q.correct_answer == "A" and len(q.options) == 4


def test_missing_options_rejected():
    with pytest.raises(QuestionValidationError):
        validate_question(valid_raw(options=None), num_options=4)


def test_too_few_options_rejected():
    raw = valid_raw(options=[{"id": "A", "text": "only one"}])
    with pytest.raises(QuestionValidationError):
        validate_question(raw)


def test_wrong_option_count_rejected():
    raw = raw_question_json(1, "x", num_options=3)
    with pytest.raises(QuestionValidationError, match="expected 4 options"):
        validate_question(raw, num_options=4)


def test_duplicate_option_ids_rejected():
    raw = valid_raw(options=[{"id": "A", "text": "one"}, {"id": "A", "text": "two"},
                             {"id": "C", "text": "three"}, {"id": "D", "text": "four"}])
    with pytest.raises(QuestionValidationError, match="unique"):
        validate_question(raw, num_options=4)


def test_duplicate_option_text_rejected():
    raw = valid_raw(options=[{"id": l, "text": "same"} for l in "ABCD"])
    with pytest.raises(QuestionValidationError):
        validate_question(raw, num_options=4)


def test_invalid_correct_answer_reference():
    with pytest.raises(QuestionValidationError, match="correct_answer"):
        validate_question(valid_raw(correct_answer="Z"), num_options=4)


def test_empty_question_and_option_text_rejected():
    with pytest.raises(QuestionValidationError):
        validate_question(valid_raw(question="   "))
    raw = valid_raw()
    raw["options"][1]["text"] = " "
    with pytest.raises(QuestionValidationError):
        validate_question(raw)


def test_answer_given_as_option_text_is_mapped():
    raw = valid_raw(correct_answer="Candidate statement C about topic 1")
    assert validate_question(raw, num_options=4).correct_answer == "C"


def test_answer_text_starting_with_a_is_not_misread_as_letter():
    raw = valid_raw()
    raw["options"][2]["text"] = "A tree grows"
    raw["correct_answer"] = "A tree grows"
    assert validate_question(raw, num_options=4).correct_answer == "C"


def test_options_as_plain_strings_get_letters():
    raw = valid_raw(options=["A. red", "B. green", "C. blue", "D. black"], correct_answer="b")
    q = validate_question(raw, num_options=4)
    assert [o.id for o in q.options] == ["A", "B", "C", "D"]
    assert q.options[1].text == "green" and q.correct_answer == "B"


def test_bad_difficulty_is_normalised_not_trusted():
    assert validate_question(valid_raw(difficulty="hard")).difficulty == "Hard"
    assert validate_question(valid_raw(difficulty="impossible")).difficulty == "Medium"


def test_untraceable_excerpt_rejected():
    raw = valid_raw(source_excerpt="Mitochondria are the powerhouse of the cell and make ATP.")
    with pytest.raises(QuestionValidationError, match="not found"):
        validate_question(raw, source_text=SOURCE)


def test_missing_excerpt_rejected_when_required():
    with pytest.raises(QuestionValidationError, match="no supporting"):
        validate_question(valid_raw(source_excerpt=""), source_text=SOURCE, require_excerpt=True)


def test_excerpt_matching_is_tolerant_to_case_and_whitespace():
    assert excerpt_in_source("PHOTOSYNTHESIS  converts light\nenergy", SOURCE)
    assert not excerpt_in_source("", SOURCE)


def test_duplicate_questions_detected():
    a = make_question(1)
    b = a.model_copy(update={"question": "What is the answer to sample question number 1 ?"})
    c = Question(question="Name the organelle where photosynthesis takes place.",
                 options=[Option(id="A", text="x1"), Option(id="B", text="y2")], correct_answer="A")
    assert is_duplicate(b, [a])
    assert not is_duplicate(c, [a])
    assert find_duplicate_pairs([a, c, b]) == [(0, 2)]


# ---- JSON parsing -------------------------------------------------------
def test_parse_plain_and_fenced_json():
    assert parse_model_json('{"questions": []}') == {"questions": []}
    assert parse_model_json('```json\n{"questions": []}\n```') == {"questions": []}
    assert parse_model_json('Here you go: {"questions": [1]} thanks') == {"questions": [1]}
    assert parse_model_json("[{\"a\": 1}]") == {"questions": [{"a": 1}]}


@pytest.mark.parametrize("bad", ["", "   ", "not json at all", '{"questions": [', "__import__('os').system('x')"])
def test_malformed_json_raises(bad):
    with pytest.raises(MalformedOutputError):
        parse_model_json(bad)


def test_echoed_prompt_tags_are_stripped_before_checks():
    raw = valid_raw(question="<Material> Which process stores light energy? </Material>",
                    source_excerpt="<Material>Photosynthesis converts light energy into chemical energy.")
    q = validate_question(raw, num_options=4, source_text=SOURCE)
    assert q.question == "Which process stores light energy?"
    assert "<" not in q.source_excerpt


def test_options_written_inside_question_text_are_rejected():
    raw = valid_raw(question="Which stage fixes CO2? A) Light reactions B) Calvin cycle C) Glycolysis D) Krebs cycle")
    with pytest.raises(QuestionValidationError, match="inside the question"):
        validate_question(raw, num_options=4)
    # a normal question mentioning a letter is fine
    assert validate_question(valid_raw(question="What does vitamin A do for the eye?"), num_options=4)


# --------------------------------------------------------------------- wording of model-written questions
def passage(pid):
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent.parent / "evaluation" / "passages.json").read_text(encoding="utf-8"))
    return next(p["text"] for p in data["passages"] if p["id"] == pid)


def worded(question, options, key="A", excerpt=""):
    return {"question": question, "options": [{"id": l, "text": t} for l, t in zip("ABCD", options)],
            "correct_answer": key, "explanation": "", "source_excerpt": excerpt}


def check(raw, source):
    return validate_question(raw, num_options=4, source_text=source, require_excerpt=False, check_wording_rules=True)


def test_a_copied_paragraph_is_not_a_question():  # real reply from quizforge-3b, passage P2
    text = passage("P2")
    raw = worded("The A* algorithm evaluates each node with the formula f(n) = g(n) + h(n). A* always expands the node "
                 "with the lowest f(n).", ["A* algorithm", "breadth-first search", "depth-first search", "greedy search"])
    with pytest.raises(QuestionValidationError, match="does not ask anything"):
        check(raw, text)


def test_a_question_that_contains_its_answer_is_rejected():  # real reply from quizforge-3b, passage P3
    raw = worded("Chain-of-thought (CoT) prompting makes what type of language model reason step by step before giving "
                 "the final answer?", ["a language model", "a vision model", "a speech model", "a search engine"])
    with pytest.raises(QuestionValidationError, match="gives away the answer"):
        check(raw, passage("P3"))


def test_a_question_led_by_a_copied_sentence_is_rejected():  # real reply from quizforge-3b, passage P1
    raw = worded("Abductive reasoning looks at an observation and infers the most plausible explanation for it. If the "
                 "road is wet, what is the best explanation?", ["it probably rained", "a pipe burst", "it snowed",
                                                                 "someone washed a car"])
    with pytest.raises(QuestionValidationError, match="copied from the material"):
        check(raw, passage("P1"))


@pytest.mark.parametrize("question,options", [
    ("What type of reasoning infers the most plausible explanation for an observation?",
     ["abductive", "deductive", "inductive", "analogical"]),
    ("A doctor sees a fever and a rash and names the most likely illness. Which kind of reasoning is this?",
     ["abductive", "deductive", "inductive", "analogical"]),  # a scenario written by the model is fine
    ("Is abductive reasoning about the best explanation or about a general rule?",
     ["the best explanation", "a general rule", "a past situation", "a proof"]),  # names every choice it offers
    ("Identify the kind of reasoning that starts from a general rule.", ["deductive", "inductive", "abductive", "analogical"]),
    ("Reasoning from a general rule to a particular case is called ____.", ["deduction", "induction", "abduction", "analogy"]),
])
def test_well_worded_questions_still_pass(question, options):
    assert check(worded(question, options), passage("P1")).question == question


def test_questions_typed_by_a_teacher_are_not_held_to_the_wording_rules():
    raw = worded("Photosynthesis happens in the chloroplast.", ["true", "false", "only at night", "never"])
    assert validate_question(raw, num_options=4, source_text=SOURCE).question.startswith("Photosynthesis")


# ------------------------------------------------------------------------------ same fact, different words
def mcq(question, options, key="A"):
    return Question(question=question, options=[Option(id=l, text=t) for l, t in zip("ABCD", options)], correct_answer=key)


def test_the_same_fact_asked_two_ways_is_a_duplicate():  # real pairs from quizforge-7b and quizforge-3b
    react1 = mcq("What does Re-Act combine?", ["reasoning with acting", "search with planning", "rules with data", "text"])
    react2 = mcq("ReAct combines reasoning with what?", ["acting", "searching", "planning", "voting"])
    astar1 = mcq("Which algorithm evaluates each node with the formula f(n) = g(n) + h(n)?",
                 ["A*", "breadth-first search", "depth-first search", "greedy best-first search"])
    astar2 = mcq("A* algorithm evaluates each node with what formula?",
                 ["f(n) = g(n) + h(n)", "f(n) = g(n)", "f(n) = h(n)", "f(n) = g(n) - h(n)"])
    assert is_duplicate(react2, [react1]) and is_duplicate(astar2, [astar1])
    assert find_duplicate_pairs([react1, astar1, react2, astar2]) == [(0, 2), (1, 3)]


def test_different_facts_are_not_duplicates():
    bfs1 = mcq("Which algorithm explores a graph level by level?", ["breadth-first search", "A*", "depth-first search", "greedy"])
    bfs2 = mcq("Which algorithm guarantees the shortest path in unweighted graphs?",
               ["breadth-first search", "A*", "depth-first search", "greedy"])  # same answer, different fact
    dfs = mcq("Which algorithm explores one path deeply before backtracking?",
              ["depth-first search", "A*", "breadth-first search", "greedy"])  # same subject, different answer
    assert not is_duplicate(bfs2, [bfs1]) and not is_duplicate(dfs, [bfs1, bfs2])
