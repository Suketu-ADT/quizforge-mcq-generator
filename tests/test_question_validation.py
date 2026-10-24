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
