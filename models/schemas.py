"""Pydantic data models shared by every layer of the application."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Difficulty = Literal["Easy", "Medium", "Hard"]
OPTION_LETTERS = "ABCDEFGH"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Option(BaseModel):
    id: str
    text: str

    @field_validator("id")
    @classmethod
    def _id_ok(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("option id must not be empty")
        return v

    @field_validator("text")
    @classmethod
    def _text_ok(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("option text must not be empty")
        return v


class Question(BaseModel):
    """One multiple-choice question with exactly one correct option."""

    id: str = Field(default_factory=new_id)
    question: str
    options: list[Option]
    correct_answer: str
    explanation: str = ""
    difficulty: Difficulty = "Medium"
    topic: str = "General"
    source_excerpt: str = ""
    source_section: str = ""
    # result of the optional "double-check" pass: the model answered the question blind and either matched
    # the stored key ("agrees") or not ("disagrees"). It is a warning signal, never proof of correctness.
    key_check: Literal["unchecked", "agrees", "disagrees"] = "unchecked"

    @field_validator("question")
    @classmethod
    def _question_ok(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question text must not be empty")
        return v

    @field_validator("correct_answer")
    @classmethod
    def _answer_ok(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("topic")
    @classmethod
    def _topic_ok(cls, v: str) -> str:
        return v.strip() or "General"

    @model_validator(mode="after")
    def _check_options(self) -> "Question":
        if len(self.options) < 2:
            raise ValueError("a question needs at least 2 options")
        if len(self.options) > len(OPTION_LETTERS):
            raise ValueError(f"at most {len(OPTION_LETTERS)} options are supported")
        ids = [o.id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError("option ids must be unique")
        texts = [o.text.lower() for o in self.options]
        if len(set(texts)) != len(texts):
            raise ValueError("option texts must be distinct")
        if self.correct_answer not in ids:
            raise ValueError(
                f"correct_answer '{self.correct_answer}' does not match any option id {ids}"
            )
        return self

    def correct_text(self) -> str:
        return next(o.text for o in self.options if o.id == self.correct_answer)


class QuestionSet(BaseModel):
    """The structure we ask the LLM to return."""

    questions: list[dict]


class GenerationRequest(BaseModel):
    topic: str = ""
    count: int = Field(default=10, ge=1, le=50)
    difficulty: Literal["Easy", "Medium", "Hard", "Mixed"] = "Medium"
    style: Literal["Conceptual", "Factual", "Application-based", "Mixed"] = "Mixed"
    num_options: int = Field(default=4, ge=2, le=6)
    instructions: str = ""
    include_explanations: bool = True
    include_sources: bool = True


class Quiz(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    subject: str = ""
    questions: list[Question] = Field(default_factory=list)
    source_name: str = ""
    source_text: str = ""  # kept locally so single questions can be regenerated later
    model: str = ""
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)


class AnswerDetail(BaseModel):
    question_id: str
    selected: str | None
    correct: str
    is_correct: bool
    topic: str
    difficulty: str


class Attempt(BaseModel):
    id: str = Field(default_factory=new_id)
    quiz_id: str
    quiz_title: str = ""
    answers: dict[str, str] = Field(default_factory=dict)
    details: list[AnswerDetail] = Field(default_factory=list)
    correct: int = 0
    incorrect: int = 0
    unanswered: int = 0
    total: int = 0
    percent: float = 0.0
    duration_s: float | None = None
    submitted_at: str = Field(default_factory=utcnow_iso)
