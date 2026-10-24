"""Read-only question card and the editable review list."""
from __future__ import annotations

import streamlit as st

from components.styles import badge, esc
from models.schemas import OPTION_LETTERS, Option, Question, Quiz
from services.ollama_client import OllamaError
from services.question_generator import QuestionGenerator
from services.question_validator import QuestionValidationError, is_duplicate, validate_question


def key_badge(q: Question) -> str:
    if q.key_check == "agrees":
        return '<span class="qf-badge qf-easy" title="The model answered this blind and picked the same option">&#10003; key double-checked</span>'
    if q.key_check == "disagrees":
        return '<span class="qf-badge qf-medium" title="The model answered this blind and picked a different option">&#9888; review the key</span>'
    return ""


def question_html(
    q: Question,
    number: int,
    reveal: bool = False,
    selected: str | None = None,
    show_source: bool = True,
    show_explanation: bool = True,
    show_key_check: bool = False,
) -> str:
    """HTML for a question. Correct answer is only marked when `reveal` is True."""
    opts = []
    for o in q.options:
        cls = ""
        if reveal and o.id == q.correct_answer:
            cls = " correct"
        elif reveal and selected == o.id:
            cls = " wrong"
        mark = ""
        if selected == o.id:
            mark = " &nbsp;<span class='qf-muted'>(your answer)</span>"
        opts.append(f'<div class="qf-opt{cls}"><b>{esc(o.id)}</b><span>{esc(o.text)}{mark}</span></div>')
    parts = [
        f'<div class="qf-qhead"><span class="qf-qnum">Q{number}</span>{badge(q.difficulty)}'
        f'<span class="qf-badge qf-topic">{esc(q.topic)}</span>{key_badge(q) if show_key_check else ""}</div>',
        f'<div class="qf-qtext">{esc(q.question)}</div>',
        "".join(opts),
    ]
    if reveal and show_explanation and q.explanation:
        parts.append(f'<div class="qf-expl"><b>Explanation:</b> {esc(q.explanation)}</div>')
    if show_source and q.source_excerpt:
        sec = f" ({esc(q.source_section)})" if q.source_section else ""
        parts.append(f'<div class="qf-src">Source{sec}: &ldquo;{esc(q.source_excerpt)}&rdquo;</div>')
    return "".join(parts)


def render_quiz_editor(
    quiz: Quiz,
    key: str,
    generator: QuestionGenerator | None = None,
    show_source: bool = True,
    num_options: int = 4,
) -> bool:
    """Editable cards. Mutates `quiz.questions` in place; returns True if anything changed."""
    changed = False
    for idx, q in enumerate(list(quiz.questions)):
        with st.container(border=True):
            reveal_key = f"{key}_reveal_{q.id}"
            st.markdown(question_html(q, idx + 1, reveal=False, show_source=show_source, show_key_check=True), unsafe_allow_html=True)
            if st.toggle("Reveal correct answer and explanation", key=reveal_key):
                st.markdown(
                    f'<div class="qf-opt correct"><b>{esc(q.correct_answer)}</b><span>{esc(q.correct_text())}</span></div>'
                    + (f'<div class="qf-expl"><b>Explanation:</b> {esc(q.explanation)}</div>' if q.explanation else ""),
                    unsafe_allow_html=True,
                )
            c1, c2, c3, _ = st.columns([1.1, 1.7, 1.1, 2.2])
            with c1:
                edit_open = st.toggle("Edit", key=f"{key}_edit_{q.id}")
            if c2.button("Regenerate", key=f"{key}_regen_{q.id}", disabled=generator is None, help="Ask the local model for a replacement question"):
                with st.spinner("Regenerating this question with the local model..."):
                    try:
                        new_q = generator.regenerate_one(
                            q, [x for x in quiz.questions if x.id != q.id], quiz.source_text
                        )
                        quiz.questions[idx] = new_q
                        changed = True
                        st.session_state[f"{key}_flash"] = f"Question {idx + 1} regenerated."
                        st.rerun()
                    except (OllamaError, QuestionValidationError) as exc:
                        st.error(str(exc))
            if c3.button("Delete", key=f"{key}_del_{q.id}"):
                quiz.questions.pop(idx)
                st.session_state[f"{key}_flash"] = f"Question {idx + 1} deleted."
                st.rerun()
            if edit_open:
                _edit_form(quiz, idx, q, key)
    return changed


def _edit_form(quiz: Quiz, idx: int, q: Question, key: str) -> None:
    # the content signature in each key resets the inputs after a regenerate / external change
    sig = f"{q.id}_{abs(hash(q.model_dump_json())) % 10**8}"
    with st.form(f"{key}_form_{sig}"):
        text = st.text_area("Question", q.question, height=90, key=f"{key}_t_{sig}")
        opt_texts = [
            st.text_input(f"Option {o.id}", o.text, key=f"{key}_o_{sig}_{o.id}") for o in q.options
        ]
        ids = [o.id for o in q.options]
        correct = st.selectbox("Correct answer", ids, index=ids.index(q.correct_answer), key=f"{key}_c_{sig}")
        c1, c2 = st.columns(2)
        diff = c1.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=["Easy", "Medium", "Hard"].index(q.difficulty), key=f"{key}_d_{sig}")
        topic = c2.text_input("Topic", q.topic, key=f"{key}_p_{sig}")
        expl = st.text_area("Explanation", q.explanation, height=80, key=f"{key}_e_{sig}")
        if st.form_submit_button("Validate and apply", type="primary"):
            try:
                updated = validate_question(
                    Question(
                        id=q.id, question=text,
                        options=[Option(id=o.id, text=t) for o, t in zip(q.options, opt_texts)],
                        correct_answer=correct, explanation=expl, difficulty=diff, topic=topic,
                        source_excerpt=q.source_excerpt, source_section=q.source_section,
                    ),
                    num_options=len(q.options),
                )
            except (QuestionValidationError, ValueError) as exc:
                st.error(f"Not saved - {exc}")
                return
            others = [x for i, x in enumerate(quiz.questions) if i != idx]
            if is_duplicate(updated, others):
                st.warning("This looks very similar to another question in the set (applied anyway).")
            quiz.questions[idx] = updated
            st.session_state[f"{key}_flash"] = f"Question {idx + 1} updated."
            st.rerun()


def render_add_question(quiz: Quiz, key: str, num_options: int = 4) -> None:
    with st.expander("Add a question manually"):
        with st.form(f"{key}_add"):
            text = st.text_area("Question", height=80)
            letters = OPTION_LETTERS[:num_options]
            texts = [st.text_input(f"Option {l}") for l in letters]
            correct = st.selectbox("Correct answer", list(letters))
            c1, c2 = st.columns(2)
            diff = c1.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
            topic = c2.text_input("Topic", quiz.subject or "General")
            expl = st.text_area("Explanation (optional)", height=70)
            if st.form_submit_button("Add question"):
                try:
                    new_q = validate_question(
                        Question(
                            question=text,
                            options=[Option(id=l, text=t) for l, t in zip(letters, texts)],
                            correct_answer=correct, explanation=expl, difficulty=diff, topic=topic,
                        ),
                        num_options=num_options,
                    )
                except (QuestionValidationError, ValueError) as exc:
                    st.error(f"Not added - {exc}")
                    return
                quiz.questions.append(new_q)
                st.session_state[f"{key}_flash"] = "Question added."
                st.rerun()


def show_flash(key: str) -> None:
    msg = st.session_state.pop(f"{key}_flash", None)
    if msg:
        st.success(msg)
