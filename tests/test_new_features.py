"""Streaming callbacks, warm-up, answer double-check, drafts, resumable sessions, PDF export."""
import json

import pymupdf
import pytest
import requests

from database.db import Database
from database.repository import Repository
from models.schemas import GenerationRequest
from services import export_service as ex
from services.answer_checker import check_question, check_questions
from services.ollama_client import OllamaClient, OllamaUnavailableError
from services.question_generator import PER_CALL_MAX, QuestionGenerator
from services.quiz_service import QuizSession
from tests.helpers import FakeResponse, FakeSession, chat_reply, make_question, make_quiz, raw_question_json
from tests.test_question_generator import MATERIAL, SENTENCES, ScriptedClient, q_json


@pytest.fixture()
def repo(tmp_path):
    return Repository(Database(tmp_path / "t.db"))


# ------------------------------------------------------------- ollama client
def test_requests_keep_the_model_loaded():
    s = FakeSession(post=chat_reply("{}"))
    OllamaClient(session=s).chat([])
    assert s.posts[0]["keep_alive"] == "30m"


def test_warm_up_never_raises():
    ok = FakeSession(post=FakeResponse(200, {}))
    assert OllamaClient(session=ok).warm_up() is True
    assert ok.posts[0] == {"model": "qwen2.5:7b", "keep_alive": "30m"}
    assert OllamaClient(session=FakeSession(post=requests.ConnectionError("down"))).warm_up() is False


# --------------------------------------------------------------- generator
def test_on_question_fires_for_each_accepted_question_only():
    seen = []
    bad = raw_question_json(2, "This sentence is not in the material at all.")
    reply = json.dumps({"questions": [raw_question_json(1, SENTENCES[0]), bad, raw_question_json(3, SENTENCES[2])]})
    res = QuestionGenerator(ScriptedClient([reply, "{}", "{}"]), max_retries=0).generate(
        MATERIAL, GenerationRequest(count=3, topic="x"), on_question=seen.append)
    assert [q.id for q in seen] == [q.id for q in res.questions] and len(seen) == 2


def test_batches_are_small():
    assert PER_CALL_MAX == 3


# ------------------------------------------------------------ answer checker
def blind_client(answer):
    return ScriptedClient([json.dumps({"answer": answer})])


def test_checker_agrees_disagrees_and_unchecked():
    q = make_question(1, "B")
    assert check_question(blind_client("B"), q) == "agrees"
    assert check_question(blind_client("c"), q) == "disagrees"
    assert check_question(blind_client("Z"), q) == "unchecked"
    assert check_question(ScriptedClient(["not json"]), q) == "unchecked"


def test_checker_never_shows_the_key_to_the_model():
    client = blind_client("B")
    check_question(client, make_question(1, "B"))
    prompt = client.last_messages[1]["content"]
    assert "correct_answer" not in prompt and "Because choice" not in prompt  # explanation (which names the key) is withheld
    assert "Photosynthesis converts light energy" in prompt  # supporting excerpt is given as context


def test_check_questions_returns_copies_and_reports_progress():
    qs = [make_question(1, "A"), make_question(2, "B")]
    client = ScriptedClient([json.dumps({"answer": "A"}), json.dumps({"answer": "D"})])
    steps = []
    out = check_questions(client, qs, on_progress=lambda i, n: steps.append((i, n)))
    assert [q.key_check for q in out] == ["agrees", "disagrees"]
    assert [q.key_check for q in qs] == ["unchecked", "unchecked"]  # originals untouched
    assert steps == [(1, 2), (2, 2)]


def test_checker_connection_errors_propagate():
    with pytest.raises(OllamaUnavailableError):
        check_questions(ScriptedClient([OllamaUnavailableError("down")]), [make_question(1)])


# ------------------------------------------------------------------ drafts
def test_draft_roundtrip_and_delete(repo):
    assert repo.load_draft() is None
    repo.save_draft({"quiz": {"title": "x"}, "target": 5})
    assert repo.load_draft()["target"] == 5
    repo.save_draft({"quiz": {"title": "y"}, "target": 6})
    assert repo.load_draft()["quiz"]["title"] == "y"
    repo.delete_draft()
    assert repo.load_draft() is None


# ---------------------------------------------------------- resumable session
def test_session_state_roundtrip_restores_progress():
    quiz = make_quiz(4)
    s = QuizSession(quiz, time_limit_s=600, instant_feedback=True)
    s.select(quiz.questions[0].id, "A")
    s.select(quiz.questions[2].id, "C")
    s.toggle_flag(quiz.questions[1].id)
    s.goto(2)
    s.started_at -= 90
    state = json.loads(json.dumps(s.to_state(mode="One question at a time")))
    t = QuizSession.from_state(quiz, state)
    assert t.answers == s.answers and t.flagged == s.flagged and t.current == 2
    assert t.time_limit_s == 600 and t.instant_feedback and t.timed
    assert 89 <= t.elapsed() <= 95  # elapsed time carries over


def test_from_state_ignores_unknown_questions_and_options():
    quiz = make_quiz(2)
    bad = {"answers": {quiz.questions[0].id: "Z", "ghost": "A", quiz.questions[1].id: "B"}, "flagged": ["ghost"], "current": 99}
    t = QuizSession.from_state(quiz, bad)
    assert t.answers == {quiz.questions[1].id: "B"} and t.flagged == set() and t.current == 1


def test_active_session_persistence(repo):
    repo.save_active_session({"quiz_id": "q1", "answers": {}})
    assert repo.load_active_session()["quiz_id"] == "q1"
    repo.clear_active_session()
    assert repo.load_active_session() is None


def test_flags_blocked_after_submit_and_cleared_on_reset():
    quiz = make_quiz(2)
    s = QuizSession(quiz)
    s.toggle_flag(quiz.questions[0].id)
    s.toggle_flag(quiz.questions[0].id)
    assert s.flagged == set()
    s.toggle_flag(quiz.questions[0].id)
    s.submit()
    s.toggle_flag(quiz.questions[1].id)
    assert s.flagged == {quiz.questions[0].id}
    s.reset()
    assert s.flagged == set()


def test_missed_questions_include_wrong_and_unanswered():
    quiz = make_quiz(4)
    s = QuizSession(quiz)
    s.select(quiz.questions[0].id, quiz.questions[0].correct_answer)  # right
    s.select(quiz.questions[1].id, "D")  # wrong (key is B)
    s.submit()
    assert [q.id for q in s.missed_questions()] == [q.id for q in quiz.questions[1:]]
    assert QuizSession(quiz).missed_questions() == []  # not submitted


# --------------------------------------------------------------------- PDF
def test_pdf_exports_are_valid_and_complete():
    quiz = make_quiz(25)
    paper = ex.question_paper_pdf(quiz)
    doc = pymupdf.open(stream=paper, filetype="pdf")
    text = "".join(p.get_text() for p in doc)
    assert paper.startswith(b"%PDF") and doc.page_count >= 2
    assert "sample question number 25" in text and "Because choice" not in text  # no answers in the paper
    key = pymupdf.open(stream=ex.answer_key_pdf(quiz), filetype="pdf")
    assert "Answer:" in "".join(p.get_text() for p in key)


def test_pdf_handles_special_characters():
    quiz = make_quiz(1)
    quiz.questions[0].question = "Is 5 < 6 & ₹60,000 > ₹50,000? <b>bold</b>"
    text = "".join(p.get_text() for p in pymupdf.open(stream=ex.question_paper_pdf(quiz), filetype="pdf"))
    assert "<b>bold</b>" in text and "5 < 6 & " in text


def test_json_export_has_no_model_field():
    assert "model" not in json.loads(ex.quiz_to_json(make_quiz(1)))
