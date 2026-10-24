"""Quiz engine: session state, scoring from the stored answer key, analytics helpers.

Scores are computed ONLY from the saved answer key - never from a new LLM call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from models.schemas import AnswerDetail, Attempt, Question, Quiz, utcnow_iso


def score_quiz(quiz: Quiz, answers: dict[str, str], duration_s: float | None = None) -> Attempt:
    """Grade `answers` ({question_id: option_id}) against the quiz's answer key."""
    details: list[AnswerDetail] = []
    correct = incorrect = unanswered = 0
    for q in quiz.questions:
        selected = answers.get(q.id)
        if selected not in {o.id for o in q.options}:
            selected = None
        if selected is None:
            unanswered += 1
        elif selected == q.correct_answer:
            correct += 1
        else:
            incorrect += 1
        details.append(
            AnswerDetail(
                question_id=q.id,
                selected=selected,
                correct=q.correct_answer,
                is_correct=selected == q.correct_answer,
                topic=q.topic,
                difficulty=q.difficulty,
            )
        )
    total = len(quiz.questions)
    percent = round(100.0 * correct / total, 1) if total else 0.0
    return Attempt(
        quiz_id=quiz.id,
        quiz_title=quiz.title,
        answers={k: v for k, v in answers.items() if k in {q.id for q in quiz.questions}},
        details=details,
        correct=correct,
        incorrect=incorrect,
        unanswered=unanswered,
        total=total,
        percent=percent,
        duration_s=round(duration_s, 1) if duration_s is not None else None,
        submitted_at=utcnow_iso(),
    )


@dataclass
class QuizSession:
    """State of one quiz run. Pure Python so it is easy to unit-test."""

    quiz: Quiz
    time_limit_s: int | None = None
    instant_feedback: bool = False
    timed: bool = False  # record time taken (stopwatch); implied by a time limit
    answers: dict[str, str] = field(default_factory=dict)
    current: int = 0
    started_at: float = field(default_factory=time.time)
    submitted: bool = False
    attempt: Attempt | None = None
    persisted: bool = False  # set once the attempt has been written to the database
    flagged: set[str] = field(default_factory=set)  # question ids marked "review later"

    def __post_init__(self) -> None:
        if self.time_limit_s:
            self.timed = True

    # navigation / answering ------------------------------------------------
    def select(self, question_id: str, option_id: str) -> None:
        if self.submitted:
            raise RuntimeError("Quiz already submitted.")
        q = next((x for x in self.quiz.questions if x.id == question_id), None)
        if q is None or option_id not in {o.id for o in q.options}:
            raise ValueError("Unknown question or option.")
        self.answers[question_id] = option_id

    def clear(self, question_id: str) -> None:
        if not self.submitted:
            self.answers.pop(question_id, None)

    def toggle_flag(self, question_id: str) -> None:
        if self.submitted:
            return
        self.flagged.symmetric_difference_update({question_id})

    def goto(self, index: int) -> None:
        self.current = max(0, min(index, len(self.quiz.questions) - 1))

    @property
    def answered_count(self) -> int:
        return len([q for q in self.quiz.questions if q.id in self.answers])

    @property
    def unanswered_ids(self) -> list[str]:
        return [q.id for q in self.quiz.questions if q.id not in self.answers]

    def elapsed(self) -> float:
        if self.attempt and self.attempt.duration_s is not None:
            return self.attempt.duration_s
        return time.time() - self.started_at

    def remaining(self) -> float | None:
        return None if not self.time_limit_s else max(0.0, self.time_limit_s - self.elapsed())

    # lifecycle -------------------------------------------------------------
    def submit(self) -> Attempt:
        """Idempotent: a second call returns the first result (no duplicate scoring)."""
        if self.submitted and self.attempt is not None:
            return self.attempt
        duration: float | None = None
        if self.timed:
            duration = self.elapsed()
            if self.time_limit_s:
                duration = min(duration, float(self.time_limit_s))
        self.attempt = score_quiz(self.quiz, self.answers, duration)
        self.submitted = True
        return self.attempt

    def reset(self) -> None:
        """Restart with all answers cleared (same quiz)."""
        self.answers = {}
        self.flagged = set()
        self.current = 0
        self.started_at = time.time()
        self.submitted = False
        self.attempt = None
        self.persisted = False

    # persistence ----------------------------------------------------------
    def to_state(self, mode: str = "") -> dict:
        """Small JSON-able snapshot so an unfinished quiz can be resumed later."""
        return {
            "quiz_id": self.quiz.id,
            "answers": dict(self.answers),
            "flagged": sorted(self.flagged),
            "current": self.current,
            "elapsed_s": round(self.elapsed(), 1),
            "time_limit_s": self.time_limit_s,
            "instant_feedback": self.instant_feedback,
            "timed": self.timed,
            "mode": mode,
        }

    @classmethod
    def from_state(cls, quiz: Quiz, state: dict) -> "QuizSession":
        valid = {q.id: {o.id for o in q.options} for q in quiz.questions}
        answers = {k: v for k, v in (state.get("answers") or {}).items() if v in valid.get(k, set())}
        s = cls(
            quiz,
            time_limit_s=state.get("time_limit_s"),
            instant_feedback=bool(state.get("instant_feedback")),
            timed=bool(state.get("timed")),
            answers=answers,
            flagged={f for f in state.get("flagged", []) if f in valid},
        )
        s.goto(int(state.get("current", 0)))
        s.started_at = time.time() - float(state.get("elapsed_s", 0))
        return s

    def missed_questions(self) -> list[Question]:
        """Questions answered wrongly or skipped (only meaningful after submit)."""
        if not self.attempt:
            return []
        ok = {d.question_id for d in self.attempt.details if d.is_correct}
        return [q for q in self.quiz.questions if q.id not in ok]

    def retake(self) -> "QuizSession":
        """A brand-new session for the same quiz."""
        return QuizSession(self.quiz, self.time_limit_s, self.instant_feedback, self.timed)


# ---------------------------------------------------------------- analytics
def performance_by(details: list[AnswerDetail], key: str) -> pd.DataFrame:
    """Aggregate correct / total / accuracy by 'topic' or 'difficulty'."""
    if not details:
        return pd.DataFrame(columns=[key, "correct", "total", "accuracy"])
    df = pd.DataFrame([d.model_dump() for d in details])
    grouped = df.groupby(key).agg(correct=("is_correct", "sum"), total=("is_correct", "count")).reset_index()
    grouped["accuracy"] = (100.0 * grouped["correct"] / grouped["total"]).round(1)
    if key == "difficulty":
        order = {"Easy": 0, "Medium": 1, "Hard": 2}
        grouped = grouped.sort_values(key, key=lambda s: s.map(order))
    return grouped.reset_index(drop=True)


def aggregate_details(attempts: list[Attempt]) -> list[AnswerDetail]:
    out: list[AnswerDetail] = []
    for a in attempts:
        out.extend(a.details)
    return out


def recommendations(attempt_or_details: Attempt | list[AnswerDetail]) -> list[str]:
    """Plain-language study advice derived only from observed results."""
    details = attempt_or_details.details if isinstance(attempt_or_details, Attempt) else attempt_or_details
    if not details:
        return ["Take a quiz first - recommendations are based on your actual answers."]
    tips: list[str] = []
    by_topic = performance_by(details, "topic")
    by_diff = performance_by(details, "difficulty")
    weak = by_topic[(by_topic["accuracy"] < 60)].sort_values("accuracy")
    for _, r in weak.head(3).iterrows():
        tips.append(f"Revisit '{r['topic']}': {int(r['correct'])}/{int(r['total'])} correct ({r['accuracy']:.0f}%).")
    strong = by_topic[(by_topic["accuracy"] >= 80) & (by_topic["total"] >= 2)]
    if not strong.empty:
        tips.append("Strong topics: " + ", ".join(strong["topic"].head(3)) + ".")
    easy = by_diff[by_diff["difficulty"] == "Easy"]
    hard = by_diff[by_diff["difficulty"] == "Hard"]
    if not easy.empty and easy.iloc[0]["accuracy"] < 70:
        tips.append("Easy questions were missed - review the core definitions before moving on.")
    if not hard.empty and hard.iloc[0]["accuracy"] < 50:
        tips.append("Hard questions are the gap - practise applying concepts rather than recalling them.")
    skipped = sum(1 for d in details if d.selected is None)
    if skipped:
        tips.append(f"{skipped} question(s) were left unanswered - attempt every question next time.")
    if not tips:
        tips.append("Great work - try a harder difficulty or a new topic to keep progressing.")
    return tips


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    return f"{s // 60}m {s % 60:02d}s"
