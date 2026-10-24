"""End-to-end UI flows driven headlessly with Streamlit's AppTest (Ollama is faked)."""
import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from models.schemas import Question
from services.ollama_client import OllamaStatus
from services.question_generator import GenerationResult
from tests.helpers import make_quiz

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MCQ_DB_PATH", str(tmp_path / "flow.db"))
    import config
    importlib.reload(config)
    import components.context as ctx
    importlib.reload(ctx)
    ctx.get_repo.clear()
    ctx._status_cached.clear()
    monkeypatch.setattr(
        ctx, "_status_cached",
        lambda url, model: OllamaStatus(connected=True, model_installed=True, installed_models=[model], version="test"),
    )
    ctx._status_cached.clear = lambda: None
    return ctx


def app(page: str) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["_nav_request"] = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at: AppTest, label: str) -> None:
    btn = next(b for b in at.button if b.label.strip().startswith(label))
    btn.click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def start_quiz(at: AppTest, instant: bool) -> None:
    at.toggle[0].set_value(instant)  # "Instant feedback" is the first toggle on the setup screen
    at.run()
    click(at, "Start quiz")


def pick(at: AppTest, letter: str) -> None:
    click(at, f"{letter}.")


def page_html(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if not m.value.lstrip().startswith("<style>"))


def test_exam_mode_hides_answers_until_submit_and_scores_from_key(env):
    repo = env.get_repo()
    quiz = make_quiz(4)
    repo.save_quiz(quiz)
    at = app("Take a Quiz")
    start_quiz(at, instant=False)
    pick(at, "A")  # Q1 key is A
    # exam mode: no reveal, no verdict, and the answer can be changed
    assert "qf-verdict" not in page_html(at) and "Explanation" not in page_html(at)
    click(at, "Next")
    pick(at, "D")  # Q2 key is B -> wrong
    pick(at, "B")  # change the answer to the right one
    click(at, "Next")
    pick(at, "A")  # Q3 key is C -> wrong
    click(at, "Submit quiz")
    assert any("unanswered" in w.value for w in at.warning)
    click(at, "Yes, submit")
    assert "Results" in page_html(at) and "Explanation" in page_html(at)
    attempts = repo.list_attempts(quiz.id)
    assert len(attempts) == 1
    assert (attempts[0].correct, attempts[0].incorrect, attempts[0].unanswered) == (2, 1, 1)
    assert attempts[0].duration_s is None  # timer was off


def test_instant_feedback_reveals_correct_and_locks_first_click(env):
    repo = env.get_repo()
    repo.save_quiz(make_quiz(3))
    at = app("Take a Quiz")
    start_quiz(at, instant=True)
    html = page_html(at)
    assert "qf-verdict" not in html and "qf-opt correct" not in html  # nothing revealed before clicking
    pick(at, "D")  # Q1 key is A -> wrong
    html = page_html(at)
    assert "qf-verdict bad" in html and "qf-anim" in html
    assert "qf-opt correct" in html and "qf-opt wrong" in html  # right answer shown, wrong pick marked
    assert not any(b.label.startswith("A.") for b in at.button), "option cards are replaced by the revealed card"
    s = at.session_state["quiz_session"]
    assert s.answers[s.quiz.questions[0].id] == "D"
    click(at, "Next")
    pick(at, "B")  # Q2 key is B -> correct
    assert "qf-verdict ok" in page_html(at)


def test_question_bank_duplicate_and_filter(env):
    repo = env.get_repo()
    repo.save_quiz(make_quiz(3))
    at = app("My Question Bank")
    assert any("Biology basics" in m.value for m in at.markdown)
    click(at, "Copy")
    assert len(repo.list_quizzes()) == 2
    assert any("(copy)" in m.value for m in at.markdown)


def test_dashboard_shows_real_numbers(env):
    repo = env.get_repo()
    from services.quiz_service import score_quiz

    quiz = make_quiz(4)
    repo.save_quiz(quiz)
    repo.save_attempt(score_quiz(quiz, {q.id: q.correct_answer for q in quiz.questions[:3]}))
    at = app("Dashboard")
    html = " ".join(m.value for m in at.markdown)
    assert "75%" in html and ">1<" in html  # average/best score and attempt count come from the DB


def test_generate_review_edit_and_save(env, monkeypatch):
    import pages.generate_quiz as gq

    produced = make_quiz(3).questions

    class FakeGenerator:
        def __init__(self, client):
            pass

        def generate(self, text, req, existing=None, progress=None, on_question=None):
            qs = [q.model_copy(deep=True) for q in produced]
            for q in qs:
                if on_question:
                    on_question(q)
            return GenerationResult(questions=qs, requested=req.count)

    monkeypatch.setattr(gq, "QuestionGenerator", FakeGenerator)
    at = app("Generate Quiz")
    at.text_area(key="material_text").set_value("Photosynthesis converts light into chemical energy. " * 5)
    at.text_input(key="cfg_topic").set_value("Biology")
    at.run()
    click(at, "Generate")
    assert any("Review and edit" in m.value for m in at.markdown)
    assert at.session_state["draft"].title == "Biology quiz"
    assert len(at.session_state["draft"].questions) == 3
    click(at, "Save to question bank")
    repo = env.get_repo()
    saved = repo.list_quizzes()
    assert len(saved) == 1 and saved[0]["n_questions"] == 3
    assert repo.dashboard_stats()["questions_generated"] == 3


def test_generate_blocked_when_ollama_down(env, monkeypatch):
    monkeypatch.setattr(env, "_status_cached", lambda url, model: OllamaStatus(connected=False, error="Cannot reach Ollama"))
    at = app("Generate Quiz")
    assert any("Cannot reach Ollama" in w.value for w in at.warning)
    gen = next(b for b in at.button if b.label.startswith("Generate"))
    assert gen.disabled


def test_settings_save_persists(env):
    at = app("Settings")
    at.text_input[0].set_value("http://localhost:11434")
    next(b for b in at.button if b.label == "Save settings").click()
    at.run()
    assert not at.exception
    assert env.get_repo().load_setting("settings")["model"]
