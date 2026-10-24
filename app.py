"""QuizForge - Automatic MCQ Quiz Generator (PS 97) using a local Qwen 2.5 model via Ollama.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="QuizForge - Local MCQ Generator", page_icon="🎓", layout="wide")

from components.sidebar import render_sidebar  # noqa: E402  (after set_page_config)
from components.styles import inject_css  # noqa: E402
from pages import about, analytics, dashboard, generate_quiz, question_bank, settings, take_quiz  # noqa: E402

PAGES = {
    "Dashboard": dashboard.render,
    "Generate Quiz": generate_quiz.render,
    "My Question Bank": question_bank.render,
    "Take a Quiz": take_quiz.render,
    "Analytics": analytics.render,
    "Settings": settings.render,
    "About Project": about.render,
}


def main() -> None:
    inject_css()
    page = render_sidebar()
    try:
        PAGES[page]()
    except Exception as exc:  # last-resort guard: never show a raw traceback to the user
        st.error(f"Something went wrong: {exc}")
        st.exception(exc)


main()
