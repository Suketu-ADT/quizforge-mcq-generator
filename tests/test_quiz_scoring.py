import pytest

from services.quiz_service import QuizSession, performance_by, recommendations, score_quiz
from tests.helpers import make_quiz


def correct_answers(quiz):
    return {q.id: q.correct_answer for q in quiz.questions}


def test_perfect_score():
    quiz = make_quiz(4)
    a = score_quiz(quiz, correct_answers(quiz))
    assert (a.correct, a.incorrect, a.unanswered, a.total, a.percent) == (4, 0, 0, 4, 100.0)


def test_mixed_score_and_unanswered():
    quiz = make_quiz(4)
    ans = correct_answers(quiz)
    ans[quiz.questions[0].id] = "D" if quiz.questions[0].correct_answer != "D" else "A"  # wrong
    del ans[quiz.questions[1].id]  # unanswered
    a = score_quiz(quiz, ans)
    assert (a.correct, a.incorrect, a.unanswered) == (2, 1, 1)
    assert a.percent == 50.0
    assert a.details[1].selected is None


def test_all_unanswered_and_empty_quiz():
    quiz = make_quiz(3)
    a = score_quiz(quiz, {})
    assert a.correct == 0 and a.unanswered == 3 and a.percent == 0.0
    assert score_quiz(make_quiz(0), {}).percent == 0.0


def test_invalid_option_ids_are_treated_as_unanswered():
    quiz = make_quiz(2)
    a = score_quiz(quiz, {quiz.questions[0].id: "Z", "ghost-id": "A"})
    assert a.unanswered == 2 and "ghost-id" not in a.answers


def test_session_select_validates_and_submit_is_idempotent():
    quiz = make_quiz(3)
    s = QuizSession(quiz)
    with pytest.raises(ValueError):
        s.select(quiz.questions[0].id, "Z")
    with pytest.raises(ValueError):
        s.select("nope", "A")
    for q in quiz.questions:
        s.select(q.id, q.correct_answer)
    first = s.submit()
    second = s.submit()
    assert first is second and first.percent == 100.0
    with pytest.raises(RuntimeError):
        s.select(quiz.questions[0].id, "A")


def test_unanswered_tracking():
    quiz = make_quiz(3)
    s = QuizSession(quiz)
    s.select(quiz.questions[0].id, "A")
    assert s.answered_count == 1 and len(s.unanswered_ids) == 2


def test_reset_clears_answers_and_reopens_quiz():
    quiz = make_quiz(3)
    s = QuizSession(quiz)
    s.select(quiz.questions[0].id, "A")
    s.goto(2)
    s.submit()
    s.reset()
    assert s.answers == {} and s.current == 0 and not s.submitted and s.attempt is None
    s.select(quiz.questions[0].id, "B")  # allowed again


def test_retake_creates_independent_session():
    quiz = make_quiz(3)
    s = QuizSession(quiz, time_limit_s=60, instant_feedback=True)
    s.select(quiz.questions[0].id, "A")
    s.submit()
    t = s.retake()
    assert t is not s and t.answers == {} and not t.submitted
    assert t.time_limit_s == 60 and t.instant_feedback is True
    assert s.attempt is not None  # original result untouched


def test_remaining_time_and_duration_cap():
    quiz = make_quiz(2)
    s = QuizSession(quiz, time_limit_s=10)
    s.started_at -= 100
    assert s.remaining() == 0.0
    assert s.submit().duration_s == 10.0


def test_goto_is_clamped():
    s = QuizSession(make_quiz(3))
    s.goto(99)
    assert s.current == 2
    s.goto(-5)
    assert s.current == 0


def test_performance_breakdowns_and_recommendations():
    quiz = make_quiz(4)
    ans = {q.id: ("A" if q.correct_answer != "A" else "B") for q in quiz.questions}  # all wrong
    a = score_quiz(quiz, ans)
    by_topic = performance_by(a.details, "topic")
    assert set(by_topic["topic"]) == {"Cells", "Energy", "Genetics"}
    assert by_topic["accuracy"].eq(0).all()
    by_diff = performance_by(a.details, "difficulty")
    assert list(by_diff["difficulty"]) == ["Easy", "Medium", "Hard"]
    tips = recommendations(a)
    assert any("Revisit" in t for t in tips)
    assert recommendations([])[0].startswith("Take a quiz")
