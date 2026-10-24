"""Dashboard: real statistics from SQLite + live local-model status."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.context import get_repo, get_settings, get_status, navigate
from components.metrics import fmt_percent, metric_row, page_header, status_pill
from components.styles import esc


def _when(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%d %b, %H:%M")
    except ValueError:
        return ts


def _continue_cards(repo) -> None:
    """Unfinished quiz and unsaved draft: the two things that must never be lost."""
    session = repo.load_active_session()
    quiz = repo.get_quiz(session["quiz_id"]) if session else None
    if quiz is not None:
        done = len([a for a in session.get("answers", {}) if a in {q.id for q in quiz.questions}])
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**Continue your quiz** - {esc(quiz.title)} · {done} of {len(quiz.questions)} answered")
            if right.button("Resume", key="dash_resume", type="primary", use_container_width=True):
                st.session_state["resume_quiz"] = True
                navigate("Take a Quiz")
    draft = repo.load_draft()
    if draft:
        title = esc((draft.get("quiz") or {}).get("title", "Untitled"))
        n = len((draft.get("quiz") or {}).get("questions", []))
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**Unsaved draft** - {title} · {n} question(s) waiting for review")
            if right.button("Open draft", key="dash_draft", use_container_width=True):
                navigate("Generate Quiz")


def _onboarding(repo, status, stats) -> None:
    if stats["quizzes"] or stats["attempts"] or repo.load_setting("onboarding_done"):
        return
    with st.container(border=True):
        st.markdown("#### Get started in three steps")
        steps = [
            (status.connected, "Start Ollama", "Open the Ollama app (or run `ollama serve`)."),
            (status.model_installed, "Install an AI model", "Settings shows the exact `ollama pull` command for your choice."),
            (False, "Generate your first quiz", "Paste notes or upload a PDF, DOCX or TXT file."),
        ]
        for i, (done, title, hint) in enumerate(steps, 1):
            mark = "&#10003;" if done else str(i)
            cls = "done" if done else "todo"
            st.markdown(f'<div class="qf-step {cls}" style="margin:.25rem 0;border-radius:12px"><span>{mark}</span>'
                        f'<b>{esc(title)}</b>&nbsp;<span style="background:none;width:auto" class="qf-muted">- {esc(hint)}</span></div>',
                        unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.3, 1.3, 3])
        if c1.button("Open Settings", key="onb_settings"):
            navigate("Settings")
        if c2.button("Hide this guide", key="onb_hide"):
            repo.set_setting("onboarding_done", True)
            st.rerun()


def render() -> None:
    repo = get_repo()
    settings = get_settings()
    status = get_status()
    stats = repo.dashboard_stats()

    st.markdown(
        '<div class="qf-hero"><h1>Turn any study material into a quiz</h1>'
        "<p>Paste notes or upload a PDF, DOCX or TXT file. A local AI model running on your own "
        "machine writes the multiple-choice questions - you review, edit and practise.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 3])
    if c1.button("Generate a quiz", type="primary", use_container_width=True):
        navigate("Generate Quiz")

    _continue_cards(repo)
    _onboarding(repo, status, stats)

    metric_row(
        [
            ("Questions generated", str(stats["questions_generated"]), f"{stats['questions_in_bank']} currently saved"),
            ("Quizzes taken", str(stats["attempts"]), f"{stats['quizzes']} quiz sets saved"),
            ("Average score", fmt_percent(stats["avg_percent"]), "across all attempts" if stats["attempts"] else "no attempts yet"),
            ("Best score", fmt_percent(stats["best_percent"]), "personal best" if stats["attempts"] else "no attempts yet"),
        ]
    )
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Recent activity")
        items = repo.recent_activity(8)
        if not items:
            st.markdown('<div class="qf-card qf-muted">Nothing yet - generate your first quiz to get started.</div>', unsafe_allow_html=True)
        else:
            rows = "".join(
                f'<div class="qf-activity"><span>{esc(a["message"])}</span><span class="qf-muted">{esc(_when(a["ts"]))}</span></div>'
                for a in items
            )
            st.markdown(f'<div class="qf-card">{rows}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Local model")
        detail = (
            f"Endpoint: <b>{esc(settings.base_url)}</b>"
            + (f"<br>Ollama version: {esc(status.version)}" if status.version else "")
        )
        st.markdown(f'<div class="qf-card">{status_pill(status)}<p class="qf-muted" style="margin-top:.7rem">{detail}</p></div>', unsafe_allow_html=True)
        if not status.connected:
            st.warning(status.error or "Ollama is not running.")
        elif not status.model_installed:
            st.warning(status.error)
        if st.button("Open settings", use_container_width=True):
            navigate("Settings")
