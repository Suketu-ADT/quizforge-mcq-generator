"""My Question Bank: search, filter, rename, duplicate, delete, edit and export saved quizzes."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.context import get_client, get_repo, get_status, navigate
from components.downloads import render_quiz_downloads
from components.metrics import empty_state, fmt_percent, page_header
from components.question_card import render_add_question, render_quiz_editor, show_flash
from components.styles import badge, esc
from models.schemas import Quiz
from services.question_generator import QuestionGenerator
from services.question_validator import QuestionValidationError, validate_question

EDITOR_KEY = "bank_editor"


def _date(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%d %b %Y")
    except ValueError:
        return ts


def render() -> None:
    page_header("My Question Bank", "Every saved quiz set - search, edit, duplicate, export or practise.")
    repo = get_repo()

    editing_id = st.session_state.get("bank_edit_id")
    if editing_id:
        _render_edit(editing_id)
        return

    f1, f2, f3 = st.columns([2, 1, 1])
    search = f1.text_input("Search", placeholder="Search titles, subjects, topics or question text", label_visibility="collapsed")
    subject = f2.selectbox("Subject", ["All subjects", *repo.list_subjects()], label_visibility="collapsed")
    difficulty = f3.selectbox("Difficulty", ["Any difficulty", "Easy", "Medium", "Hard"], label_visibility="collapsed")
    rows = repo.list_quizzes(
        search=search,
        subject="" if subject.startswith("All") else subject,
        difficulty="" if difficulty.startswith("Any") else difficulty,
    )
    if not rows:
        empty_state(
            "No quizzes found" if (search or not subject.startswith("All") or not difficulty.startswith("Any")) else "Your question bank is empty",
            "Try different filters." if (search or not subject.startswith("All") or not difficulty.startswith("Any"))
            else "Generate a quiz and save it to see it here.",
        )
        if st.button("Generate a quiz", type="primary"):
            navigate("Generate Quiz")
        return
    st.caption(f"{len(rows)} quiz set(s)")
    for row in rows:
        with st.container(border=True):
            info, actions = st.columns([3, 2])
            diffs = [d for d in (row["difficulties"] or "").split(",") if d]
            info.markdown(
                f'<div style="font-weight:700;font-size:1.05rem">{esc(row["title"])}</div>'
                f'<div class="qf-muted">{esc(row["subject"] or "No subject")} · {row["n_questions"]} questions · '
                f'created {esc(_date(row["created_at"]))} · {row["n_attempts"]} attempt(s) · best {esc(fmt_percent(row["best_percent"]))}</div>'
                f'<div style="margin-top:.4rem">{" ".join(badge(d) for d in sorted(diffs, key=["Easy", "Medium", "Hard"].index))}</div>',
                unsafe_allow_html=True,
            )
            b = actions.columns(4)
            if b[0].button("Take", key=f"take_{row['id']}", use_container_width=True, type="primary"):
                st.session_state["take_quiz_id"] = row["id"]
                navigate("Take a Quiz")
            if b[1].button("Edit", key=f"edit_{row['id']}", use_container_width=True):
                st.session_state["bank_edit_id"] = row["id"]
                st.rerun()
            if b[2].button("Copy", key=f"dup_{row['id']}", use_container_width=True, help="Duplicate this quiz"):
                copy = repo.duplicate_quiz(row["id"])
                repo.log_activity("duplicate", f"Duplicated '{row['title']}'")
                st.toast(f"Created '{copy.title}'")
                st.rerun()
            with b[3].popover("More", use_container_width=True):
                new_title = st.text_input("Rename", row["title"], key=f"ren_{row['id']}")
                if st.button("Save name", key=f"rensave_{row['id']}"):
                    try:
                        repo.rename_quiz(row["id"], new_title)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                st.divider()
                confirm = st.checkbox("Yes, delete this quiz and its attempts", key=f"delconf_{row['id']}")
                if st.button("Delete quiz", key=f"del_{row['id']}", disabled=not confirm):
                    repo.delete_quiz(row["id"])
                    repo.log_activity("delete", f"Deleted '{row['title']}'")
                    st.rerun()


def _render_edit(quiz_id: str) -> None:
    repo = get_repo()
    key = f"bank_quiz_{quiz_id}"
    if key not in st.session_state:
        loaded = repo.get_quiz(quiz_id)
        if loaded is None:
            st.session_state.pop("bank_edit_id", None)
            st.error("That quiz no longer exists.")
            st.rerun()
        st.session_state[key] = loaded
    quiz: Quiz = st.session_state[key]

    def close() -> None:
        st.session_state.pop("bank_edit_id", None)
        st.session_state.pop(key, None)

    top1, top2 = st.columns([1, 4])
    top1.button("← Back to bank", on_click=close)
    show_flash(EDITOR_KEY)
    c1, c2 = st.columns(2)
    quiz.title = c1.text_input("Quiz title", quiz.title, key=f"{key}_title") or quiz.title
    quiz.subject = c2.text_input("Subject", quiz.subject, key=f"{key}_subject")
    st.caption(f"{len(quiz.questions)} question(s) · source: {quiz.source_name or 'n/a'}")
    generator = QuestionGenerator(get_client()) if get_status().ready and quiz.source_text else None
    if generator is None:
        st.caption("Regeneration needs the local model running and the original source text.")
    render_quiz_editor(quiz, EDITOR_KEY, generator=generator, show_source=True)
    render_add_question(quiz, EDITOR_KEY, len(quiz.questions[0].options) if quiz.questions else 4)
    if st.button("Save changes", type="primary"):
        try:
            quiz.questions = [validate_question(q, num_options=len(q.options)) for q in quiz.questions]
        except QuestionValidationError as exc:
            st.error(f"Cannot save: {exc}")
        else:
            repo.save_quiz(quiz)
            repo.log_activity("edit", f"Updated '{quiz.title}'")
            st.success("Saved.")
    with st.expander("Export"):
        render_quiz_downloads(quiz, "bank")
