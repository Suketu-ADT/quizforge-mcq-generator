"""Download buttons for a quiz (question paper, answer key, JSON, CSV)."""
from __future__ import annotations

import re

import streamlit as st

from models.schemas import Attempt, Quiz
from services import export_service as ex


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:50] or "quiz"


def render_quiz_downloads(quiz: Quiz, key: str) -> None:
    if not quiz.questions:
        st.info("Add at least one question to enable exports.")
        return
    slug = _slug(quiz.title)
    st.caption("Practice pack: give students the question paper, keep the answer key separate.")
    c = st.columns(4)
    c[0].download_button("Question paper (HTML)", ex.question_paper_html(quiz).encode("utf-8"),
                         f"{slug}_question_paper.html", "text/html", key=f"{key}_dl_paper", use_container_width=True,
                         help="Open in a browser and use Print -> Save as PDF")
    c[1].download_button("Answer key (HTML)", ex.answer_key_html(quiz).encode("utf-8"),
                         f"{slug}_answer_key.html", "text/html", key=f"{key}_dl_key", use_container_width=True)
    c[2].download_button("Answer key (CSV)", ex.to_download_bytes(ex.answer_key_csv(quiz), bom=True),
                         f"{slug}_answer_key.csv", "text/csv", key=f"{key}_dl_keycsv", use_container_width=True)
    c[3].download_button("Questions (JSON)", ex.quiz_to_json(quiz).encode("utf-8"),
                         f"{slug}.json", "application/json", key=f"{key}_dl_json", use_container_width=True)
    c = st.columns(4)
    c[0].download_button("Questions + answers (CSV)", ex.to_download_bytes(ex.quiz_to_csv(quiz), bom=True),
                         f"{slug}.csv", "text/csv", key=f"{key}_dl_csv", use_container_width=True)
    c[1].download_button("Questions only (CSV)", ex.to_download_bytes(ex.quiz_to_csv(quiz, include_answers=False), bom=True),
                         f"{slug}_questions_only.csv", "text/csv", key=f"{key}_dl_csvq", use_container_width=True)
    c[2].download_button("Questions only (JSON)", ex.quiz_to_json(quiz, include_answers=False).encode("utf-8"),
                         f"{slug}_questions_only.json", "application/json", key=f"{key}_dl_jsonq", use_container_width=True)


def render_result_downloads(quiz: Quiz, attempt: Attempt, key: str) -> None:
    slug = _slug(quiz.title)
    c = st.columns(2)
    c[0].download_button("Results (JSON)", ex.results_to_json(quiz, attempt).encode("utf-8"),
                         f"{slug}_results.json", "application/json", key=f"{key}_res_json", use_container_width=True)
    c[1].download_button("Results (CSV)", ex.to_download_bytes(ex.results_to_csv(quiz, attempt), bom=True),
                         f"{slug}_results.csv", "text/csv", key=f"{key}_res_csv", use_container_width=True)
