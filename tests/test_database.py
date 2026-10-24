import pytest

from database.db import Database
from database.repository import Repository
from services.quiz_service import score_quiz
from tests.helpers import make_quiz


@pytest.fixture()
def repo(tmp_path):
    return Repository(Database(tmp_path / "t.db"))


def test_create_and_read_roundtrip(repo):
    quiz = make_quiz(4)
    repo.save_quiz(quiz)
    got = repo.get_quiz(quiz.id)
    assert got is not None and got.title == quiz.title
    assert [q.id for q in got.questions] == [q.id for q in quiz.questions]
    assert got.questions[0].options[0].text == quiz.questions[0].options[0].text


def test_missing_quiz_returns_none(repo):
    assert repo.get_quiz("nope") is None


def test_update_replaces_questions(repo):
    quiz = make_quiz(4)
    repo.save_quiz(quiz)
    edited = quiz.model_copy(update={"questions": quiz.questions[:2]})
    repo.save_quiz(edited)
    assert len(repo.get_quiz(quiz.id).questions) == 2


def test_rename_and_empty_title(repo):
    quiz = make_quiz(2)
    repo.save_quiz(quiz)
    repo.rename_quiz(quiz.id, "New name")
    assert repo.get_quiz(quiz.id).title == "New name"
    with pytest.raises(ValueError):
        repo.rename_quiz(quiz.id, "  ")


def test_duplicate_gets_new_ids(repo):
    quiz = make_quiz(3)
    repo.save_quiz(quiz)
    copy = repo.duplicate_quiz(quiz.id)
    assert copy.id != quiz.id and copy.title.endswith("(copy)")
    assert {q.id for q in copy.questions}.isdisjoint({q.id for q in quiz.questions})
    assert len(repo.list_quizzes()) == 2


def test_delete_cascades_to_questions_and_attempts(repo):
    quiz = make_quiz(2)
    repo.save_quiz(quiz)
    repo.save_attempt(score_quiz(quiz, {}))
    repo.delete_quiz(quiz.id)
    assert repo.get_quiz(quiz.id) is None
    assert repo.list_attempts() == []


def test_search_and_filters(repo):
    a = make_quiz(4)
    b = make_quiz(2).model_copy(update={"id": "bq", "title": "World history", "subject": "History"})
    for q in b.questions:
        q.difficulty = "Easy"
        q.topic = "Rome"
        q.question = f"Who ruled Rome in era {q.id}?"
    repo.save_quiz(a)
    repo.save_quiz(b)
    assert {r["title"] for r in repo.list_quizzes(search="history")} == {"World history"}
    assert {r["title"] for r in repo.list_quizzes(search="Rome")} == {"World history"}
    assert {r["title"] for r in repo.list_quizzes(subject="Biology")} == {"Biology basics"}
    assert {r["title"] for r in repo.list_quizzes(difficulty="Hard")} == {"Biology basics"}
    assert repo.list_quizzes(search="100%") == []  # LIKE wildcards are escaped
    assert repo.list_subjects() == ["Biology", "History"]


def test_sql_injection_text_is_inert(repo):
    repo.save_quiz(make_quiz(1))
    assert repo.list_quizzes(search="'; DROP TABLE quizzes; --") == []
    assert len(repo.list_quizzes()) == 1


def test_attempts_stats_and_clear(repo):
    quiz = make_quiz(4)
    repo.save_quiz(quiz)
    repo.save_attempt(score_quiz(quiz, {q.id: q.correct_answer for q in quiz.questions}, 30))
    repo.save_attempt(score_quiz(quiz, {}, 10))
    stats = repo.dashboard_stats()
    assert stats["attempts"] == 2 and stats["best_percent"] == 100.0 and stats["avg_percent"] == 50.0
    assert len(repo.list_attempts(quiz.id)) == 2
    assert repo.clear_attempts() == 2
    assert repo.dashboard_stats()["attempts"] == 0
    assert repo.get_quiz(quiz.id) is not None  # quizzes survive clearing history


def test_empty_stats_have_no_fake_numbers(repo):
    s = repo.dashboard_stats()
    assert s["attempts"] == 0 and s["avg_percent"] is None and s["best_percent"] is None


def test_settings_counters_activity(repo):
    assert repo.load_setting("x", 5) == 5
    repo.set_setting("x", {"a": 1})
    repo.set_setting("x", {"a": 2})
    assert repo.load_setting("x") == {"a": 2}
    repo.bump_counter("questions_generated", 3)
    repo.bump_counter("questions_generated", 2)
    assert repo.counter("questions_generated") == 5
    repo.log_activity("generate", "one")
    repo.log_activity("quiz", "two")
    assert [a["message"] for a in repo.recent_activity(5)] == ["two", "one"]


def test_failed_transaction_rolls_back(repo):
    quiz = make_quiz(2)
    bad = quiz.model_copy(update={"questions": [quiz.questions[0], quiz.questions[0]]})  # duplicate PK
    with pytest.raises(Exception):
        repo.save_quiz(bad)
    assert repo.get_quiz(quiz.id) is None
