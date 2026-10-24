"""UI flows for: Stop/partial results, draft restore, resume, retry-missed, flags (headless AppTest)."""
import pytest

from models.schemas import GenerationRequest, Quiz
from tests.helpers import make_quiz
from tests.test_app_flows import app, click, env, page_html, pick, start_quiz  # noqa: F401  (env is a fixture)


def test_stop_keeps_questions_produced_so_far(env):
    produced = make_quiz(3).questions[:2]
    from streamlit.testing.v1 import AppTest
    from tests.test_app_flows import APP

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["_nav_request"] = "Generate Quiz"
    # a job that was "running" when the user pressed Stop (the run was interrupted) with 2 validated questions
    at.session_state["gen_job"] = {
        "status": "running", "req": GenerationRequest(count=5, topic="Biology"), "material": "text",
        "append": False, "double_check": False, "partial": produced, "warnings": [], "rejected": 1,
    }
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    draft = at.session_state["draft"]
    assert len(draft.questions) == 2
    assert any("stopped" in w.value for w in at.warning)
    assert any("remaining 3" in b.label for b in at.button)  # offers to generate the rest
    assert env.get_repo().load_draft() is not None  # and it was auto-saved


def test_stop_with_nothing_produced_shows_a_message_not_an_empty_draft(env):
    from streamlit.testing.v1 import AppTest
    from tests.test_app_flows import APP

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["_nav_request"] = "Generate Quiz"
    at.session_state["gen_job"] = {
        "status": "running", "req": GenerationRequest(count=5, topic="x"), "material": "", "append": False,
        "double_check": False, "partial": [], "warnings": [], "rejected": 0,
    }
    at.run()
    assert not at.exception
    assert "draft" not in at.session_state or at.session_state["draft"] is None
    assert any("No valid questions" in e.value for e in at.error)


def test_unsaved_draft_is_restored_after_a_restart(env):
    quiz = make_quiz(3)
    quiz.title = "Unsaved biology quiz"
    env.get_repo().save_draft({
        "quiz": quiz.model_dump(), "req": GenerationRequest(count=5).model_dump(), "target": 5,
        "warnings": [], "rejected": 0, "title": quiz.title, "subject": "Biology",
    })
    at = app("Generate Quiz")  # brand-new session = what a browser refresh gives
    assert any("Restored your unsaved draft" in i.value for i in at.info)
    assert len(at.session_state["draft"].questions) == 3
    assert at.text_input(key="draft_title").value == "Unsaved biology quiz"
    # dashboard advertises it too
    at2 = app("Dashboard")
    assert any("Unsaved draft" in m.value for m in at2.markdown)


def test_saving_the_draft_removes_the_stored_copy(env):
    quiz = make_quiz(2)
    env.get_repo().save_draft({"quiz": quiz.model_dump(), "req": GenerationRequest(count=2).model_dump(), "target": 2,
                               "warnings": [], "rejected": 0, "title": quiz.title, "subject": ""})
    at = app("Generate Quiz")
    click(at, "Save to question bank")
    assert env.get_repo().load_draft() is None
    assert len(env.get_repo().list_quizzes()) == 1


def test_unfinished_quiz_can_be_resumed_with_answers_intact(env):
    repo = env.get_repo()
    quiz = make_quiz(4)
    repo.save_quiz(quiz)
    at = app("Take a Quiz")
    start_quiz(at, instant=False)
    pick(at, "A")
    click(at, "Next")
    pick(at, "C")
    saved = repo.load_active_session()
    assert saved and len(saved["answers"]) == 2 and saved["current"] == 1
    # leave the app entirely: a fresh session must offer to resume
    at2 = app("Take a Quiz")
    assert any("unfinished quiz" in m.value for m in at2.markdown)
    click(at2, "Resume")
    s = at2.session_state["quiz_session"]
    assert len(s.answers) == 2 and s.current == 1


def test_dashboard_continue_card_resumes_directly(env):
    repo = env.get_repo()
    quiz = make_quiz(3)
    repo.save_quiz(quiz)
    repo.save_active_session({"quiz_id": quiz.id, "answers": {quiz.questions[0].id: "A"}, "current": 0,
                              "instant_feedback": False, "mode": "One question at a time"})
    at = app("Dashboard")
    assert any("Continue your quiz" in m.value for m in at.markdown)
    click(at, "Resume")
    assert at.session_state["quiz_session"].answers == {quiz.questions[0].id: "A"}


def test_finishing_clears_the_saved_progress(env):
    repo = env.get_repo()
    repo.save_quiz(make_quiz(2))
    at = app("Take a Quiz")
    start_quiz(at, instant=False)
    assert repo.load_active_session() is not None
    click(at, "Submit quiz")
    click(at, "Yes, submit")
    assert repo.load_active_session() is None


def test_retry_missed_runs_a_short_unsaved_quiz(env):
    repo = env.get_repo()
    quiz = make_quiz(4)
    repo.save_quiz(quiz)
    at = app("Take a Quiz")
    start_quiz(at, instant=False)
    pick(at, "A")  # Q1 right
    click(at, "Submit quiz")
    click(at, "Yes, submit")
    assert any(b.label == "Retry 3 missed" for b in at.button)
    click(at, "Retry 3 missed")
    s = at.session_state["quiz_session"]
    assert len(s.quiz.questions) == 3 and s.quiz.title.endswith("(retry missed)")
    assert repo.load_active_session() is None  # derived quizzes are not resumable
    assert len(repo.list_quizzes()) == 1  # and not added to the bank


def test_flag_marks_the_palette_and_shows_in_confirm(env):
    repo = env.get_repo()
    repo.save_quiz(make_quiz(3))
    at = app("Take a Quiz")
    start_quiz(at, instant=False)
    click(at, "⚑ Flag")
    assert any("⚑" in b.label for b in at.button if b.label.startswith("1"))
    click(at, "Submit quiz")
    assert any("Flagged for review" in i.value for i in at.info)
