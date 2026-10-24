import csv
import io
import json

from models.schemas import Option, Question, Quiz
from services import export_service as ex
from services.quiz_service import score_quiz
from tests.helpers import make_quiz


def test_json_export_roundtrip_and_unicode():
    quiz = make_quiz(3)
    quiz.questions[0].question = "Qu'est-ce que la photosynthèse ? – 日本語"
    data = json.loads(ex.quiz_to_json(quiz))
    assert data["title"] == quiz.title and len(data["questions"]) == 3
    assert data["questions"][0]["question"].startswith("Qu'est-ce")
    assert "correct_answer" in data["questions"][0]


def test_question_paper_json_has_no_answers():
    text = ex.quiz_to_json(make_quiz(3), include_answers=False)
    assert "correct_answer" not in text and "explanation" not in text


def test_csv_export_is_valid_and_complete():
    quiz = make_quiz(3)
    quiz.questions[1].question = 'Contains, a comma and "quotes"\nand a newline'
    rows = list(csv.reader(io.StringIO(ex.quiz_to_csv(quiz))))
    assert rows[0][:2] == ["no", "question"]
    assert len(rows) == 4
    assert rows[2][1] == 'Contains, a comma and "quotes"\nand a newline'
    assert "correct_answer" in rows[0]


def test_csv_without_answers_leaks_nothing():
    quiz = make_quiz(2)
    text = ex.quiz_to_csv(quiz, include_answers=False)
    header = text.splitlines()[0]
    assert "correct_answer" not in header and "explanation" not in header
    assert quiz.questions[0].explanation not in text


def test_csv_formula_injection_neutralised():
    quiz = Quiz(title="t", questions=[Question(
        question="=cmd|' /C calc'!A0",
        options=[Option(id="A", text="+1"), Option(id="B", text="@x")], correct_answer="A")])
    rows = list(csv.reader(io.StringIO(ex.quiz_to_csv(quiz))))
    assert rows[1][1].startswith("'=") and rows[1][2].startswith("'+")


def test_answer_key_csv():
    quiz = make_quiz(4)
    rows = list(csv.reader(io.StringIO(ex.answer_key_csv(quiz))))
    assert [r[1] for r in rows[1:]] == ["A", "B", "C", "D"]


def test_results_exports():
    quiz = make_quiz(4)
    attempt = score_quiz(quiz, {quiz.questions[0].id: "A", quiz.questions[1].id: "A"}, 42)
    doc = json.loads(ex.results_to_json(quiz, attempt))
    assert doc["score"]["correct"] == 1 and doc["score"]["unanswered"] == 2
    rows = list(csv.reader(io.StringIO(ex.results_to_csv(quiz, attempt))))
    assert [r[4] for r in rows[1:]] == ["correct", "incorrect", "unanswered", "unanswered"]


def test_question_paper_html_hides_answers_and_escapes():
    quiz = make_quiz(2)
    quiz.questions[0].question = "<script>alert(1)</script> What is 2 & 2?"
    page = ex.question_paper_html(quiz)
    assert page.startswith("<!doctype html>") and 'charset="utf-8"' in page
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert quiz.questions[0].explanation not in page and "Answer:" not in page


def test_answer_key_html_has_answers():
    quiz = make_quiz(2)
    page = ex.answer_key_html(quiz)
    assert "Answer: A." in page and quiz.questions[0].explanation in page


def test_download_bytes_encoding():
    assert ex.to_download_bytes("é", bom=True).startswith(b"\xef\xbb\xbf")
    assert ex.to_download_bytes("é") == "é".encode("utf-8")
