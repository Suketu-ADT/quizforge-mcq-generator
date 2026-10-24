"""Take a Quiz: setup -> run (one-at-a-time or all-at-once) -> results."""
from __future__ import annotations

import streamlit as st

from components.charts import accuracy_bar, outcome_donut
from components.context import get_repo, navigate
from components.downloads import render_result_downloads
from components.metrics import empty_state, metric_row, page_header
from components.question_card import question_html
from components.styles import esc, md_escape
from models.schemas import Quiz
from services.quiz_service import QuizSession, format_duration, performance_by, recommendations

SESSION = "quiz_session"


# ----------------------------------------------------------------- helpers
def _session() -> QuizSession | None:
    return st.session_state.get(SESSION)


def _save_progress() -> None:
    """Persist an unfinished quiz so it can be resumed after leaving the page or restarting the app."""
    s = _session()
    if s is None or s.submitted or st.session_state.get("quiz_ephemeral"):
        return
    get_repo().save_active_session(s.to_state(st.session_state.get("quiz_mode", "")))


def _start(quiz: Quiz, mode: str, timer_on: bool, minutes: int, instant: bool, ephemeral: bool = False) -> None:
    st.session_state[SESSION] = QuizSession(
        quiz, time_limit_s=minutes * 60 if timer_on and minutes > 0 else None,
        instant_feedback=instant, timed=timer_on,
    )
    st.session_state["quiz_mode"] = mode
    st.session_state["quiz_ephemeral"] = ephemeral
    st.session_state["quiz_run"] = st.session_state.get("quiz_run", 0) + 1
    st.session_state.pop("confirm_submit", None)
    _save_progress()


def _resume(quiz: Quiz, state: dict) -> None:
    st.session_state[SESSION] = QuizSession.from_state(quiz, state)
    st.session_state["quiz_mode"] = state.get("mode") or "One question at a time"
    st.session_state["quiz_ephemeral"] = False
    st.session_state["quiz_run"] = st.session_state.get("quiz_run", 0) + 1


def _discard_saved() -> None:
    get_repo().clear_active_session()


def _finish() -> None:
    """Submit exactly once and persist the attempt exactly once."""
    s = _session()
    if s is None:
        return
    attempt = s.submit()
    if not s.persisted:
        repo = get_repo()
        repo.save_attempt(attempt)
        repo.log_activity("quiz", f"Completed '{s.quiz.title}': {attempt.correct}/{attempt.total} ({attempt.percent:.0f}%)")
        s.persisted = True
    if not st.session_state.get("quiz_ephemeral"):
        get_repo().clear_active_session()
    st.session_state.pop("confirm_submit", None)


def _pick(qid: str, option_id: str) -> None:
    """Click handler for an option card. In instant-feedback mode the first click is final."""
    s = _session()
    if s is None or s.submitted:
        return
    if s.instant_feedback and qid in s.answers:
        return
    s.select(qid, option_id)
    _save_progress()


def _goto(i: int) -> None:
    s = _session()
    if s:
        s.goto(i)
        _save_progress()


def _toggle_flag(qid: str) -> None:
    s = _session()
    if s:
        s.toggle_flag(qid)
        _save_progress()


def _restart() -> None:
    s = _session()
    if s:
        s.reset()
        st.session_state["quiz_run"] = st.session_state.get("quiz_run", 0) + 1
        st.session_state.pop("confirm_submit", None)
        _save_progress()


def _retake() -> None:
    s = _session()
    if s:
        st.session_state[SESSION] = s.retake()
        st.session_state["quiz_run"] = st.session_state.get("quiz_run", 0) + 1
        _save_progress()


def _retry_missed() -> None:
    """New session containing only the questions that were answered wrongly or skipped."""
    s = _session()
    if not s:
        return
    missed = s.missed_questions()
    if not missed:
        return
    quiz = s.quiz.model_copy(update={"title": f"{s.quiz.title} (retry missed)", "questions": missed})
    st.session_state[SESSION] = QuizSession(quiz, s.time_limit_s, s.instant_feedback, s.timed)
    st.session_state["quiz_ephemeral"] = True  # a derived quiz is not saved or resumable
    st.session_state["quiz_run"] = st.session_state.get("quiz_run", 0) + 1


def _leave() -> None:
    st.session_state.pop(SESSION, None)
    st.session_state.pop("quiz_ephemeral", None)
    st.session_state.pop("confirm_submit", None)


def _ask_confirm() -> None:
    st.session_state["confirm_submit"] = True


def _cancel_confirm() -> None:
    st.session_state.pop("confirm_submit", None)


# -------------------------------------------------------------------- page
def render() -> None:
    page_header("Take a Quiz", "Practise with your saved questions. Scores come from the stored answer key.")
    s = _session()
    if s is None:
        _render_setup()
    elif s.submitted:
        _render_results(s)
    else:
        if s.time_limit_s and s.remaining() == 0:
            st.warning("Time is up - your quiz was submitted automatically.")
            _finish()
            st.rerun()
        _render_run(s)


def _render_setup() -> None:
    repo = get_repo()
    rows = repo.list_quizzes()
    if not rows:
        empty_state("No quizzes yet", "Generate and save a quiz first.")
        if st.button("Generate a quiz", type="primary"):
            navigate("Generate Quiz")
        return
    saved = repo.load_active_session()
    saved_quiz = repo.get_quiz(saved["quiz_id"]) if saved else None
    if saved and saved_quiz is None:
        repo.clear_active_session()  # its quiz was deleted
        saved = None
    if saved and saved_quiz is not None and st.session_state.pop("resume_quiz", False):
        _resume(saved_quiz, saved)
        st.rerun()
    if saved and saved_quiz is not None:
        done = len([a for a in saved.get("answers", {}) if a in {q.id for q in saved_quiz.questions}])
        with st.container(border=True):
            st.markdown(f"**You have an unfinished quiz:** {esc(saved_quiz.title)} - {done} of {len(saved_quiz.questions)} answered.")
            r1, r2, _ = st.columns([1, 1.2, 3])
            if r1.button("Resume", type="primary"):
                _resume(saved_quiz, saved)
                st.rerun()
            r2.button("Discard it", on_click=_discard_saved)
    ids = [r["id"] for r in rows]
    labels = {r["id"]: f"{r['title']}  ({r['n_questions']} questions)" for r in rows}
    preselect = st.session_state.pop("take_quiz_id", None)
    index = ids.index(preselect) if preselect in ids else 0
    quiz_id = st.selectbox("Choose a quiz", ids, index=index, format_func=labels.get)
    c1, c2 = st.columns(2)
    mode = c1.radio("Display", ["One question at a time", "All questions at once"])
    instant = c2.toggle("Instant feedback after each answer", True,
                        help="On: clicking an option immediately shows whether it is right (the first click is final). "
                             "Off (exam mode): answers can be changed and are revealed only after you submit.")
    timer_on = c2.toggle("Timer", False)
    minutes = c2.number_input("Time limit in minutes (0 = stopwatch only)", 0, 240, 0, disabled=not timer_on)
    if st.button("Start quiz", type="primary"):
        quiz = repo.get_quiz(quiz_id)
        if quiz is None or not quiz.questions:
            st.error("This quiz has no questions.")
            return
        _start(quiz, mode, timer_on, int(minutes), instant)
        st.rerun()


@st.fragment(run_every=1)
def _timer_fragment() -> None:
    s = _session()
    if s is None or s.submitted or not s.timed:
        return
    elapsed = int(s.elapsed())
    if s.time_limit_s:
        rem = int(s.remaining() or 0)
        color = "#ef4444" if rem <= 30 else "var(--qf-text)"
        st.markdown(f'<div class="qf-metric"><div class="l">Time left</div><div class="v" style="color:{color}">{rem // 60:02d}:{rem % 60:02d}</div></div>', unsafe_allow_html=True)
        if rem <= 0:
            st.rerun()  # full rerun: page() auto-submits
    else:
        st.markdown(f'<div class="qf-metric"><div class="l">Elapsed</div><div class="v">{elapsed // 60:02d}:{elapsed % 60:02d}</div></div>', unsafe_allow_html=True)


def _answer_widget(s: QuizSession, i: int) -> None:
    q = s.quiz.questions[i]
    run = st.session_state.get("quiz_run", 0)
    current = s.answers.get(q.id)
    if s.instant_feedback and current is not None:
        correct = current == q.correct_answer
        verdict = (
            '<div class="qf-verdict ok"><span class="qf-ico">&#10003;</span>Correct! Well done.</div>' if correct
            else f'<div class="qf-verdict bad"><span class="qf-ico">&#10007;</span>Not quite - the correct answer is '
                 f'{esc(q.correct_answer)}: {esc(q.correct_text())}</div>'
        )
        st.markdown(
            f'<div class="qf-anim">{question_html(q, i + 1, reveal=True, selected=current, show_source=False)}{verdict}</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f'<div class="qf-qhead"><span class="qf-qnum">Question {i + 1} of {len(s.quiz.questions)}</span></div>'
        f'<div class="qf-qtext">{esc(q.question)}</div>',
        unsafe_allow_html=True,
    )
    hint = "Click an option to see right away whether it is correct." if s.instant_feedback else "Click an option to select it. You can change it until you submit."
    st.caption(hint)
    with st.container(key=f"qopts_{run}_{i}"):
        for o in q.options:
            st.button(
                f"{o.id}.  {md_escape(o.text)}", key=f"opt_{run}_{q.id}_{o.id}",
                type="primary" if current == o.id else "secondary",
                on_click=_pick, args=(q.id, o.id), use_container_width=True,
            )


def _palette(s: QuizSession) -> None:
    n = len(s.quiz.questions)
    per_row = 10
    for start in range(0, n, per_row):
        cols = st.columns(per_row)
        for j, col in enumerate(cols):
            i = start + j
            if i >= n:
                break
            answered = s.quiz.questions[i].id in s.answers
            flag = "⚑" if s.quiz.questions[i].id in s.flagged else ""
            col.button(
                f"{i + 1}{'✓' if answered else ''}{flag}", key=f"pal_{st.session_state.get('quiz_run', 0)}_{i}",
                type="primary" if i == s.current else "secondary", on_click=_goto, args=(i,),
                use_container_width=True, help="Answered" if answered else "Not answered yet",
            )


def _render_run(s: QuizSession) -> None:
    n = len(s.quiz.questions)
    mode = st.session_state.get("quiz_mode", "One question at a time")
    head, timer = st.columns([4, 1])
    head.markdown(
        f'<div style="font-size:1.25rem;font-weight:700">{esc(s.quiz.title)}</div>'
        f'<div class="qf-muted">{s.answered_count} of {n} answered</div>',
        unsafe_allow_html=True,
    )
    head.progress(s.answered_count / n)
    if s.timed:
        with timer:
            _timer_fragment()

    if mode.startswith("One"):
        _palette(s)
        with st.container(border=True):
            _answer_widget(s, s.current)
        qid = s.quiz.questions[s.current].id
        b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 1])
        b1.button("← Previous", disabled=s.current == 0, on_click=_goto, args=(s.current - 1,), use_container_width=True)
        b2.button("Next →", disabled=s.current >= n - 1, on_click=_goto, args=(s.current + 1,), use_container_width=True)
        b3.button("⚑ Unflag" if qid in s.flagged else "⚑ Flag", on_click=_toggle_flag, args=(qid,), use_container_width=True,
                  help="Mark this question to review before submitting")
        b4.button("Restart quiz", on_click=_restart, use_container_width=True, help="Clear all answers and start over")
        b5.button("Submit quiz", type="primary", on_click=_ask_confirm, use_container_width=True)
        st.caption("Keyboard: A-D pick an option · ← → move · M flag")
    else:
        for i in range(n):
            with st.container(border=True):
                _answer_widget(s, i)
                q_id = s.quiz.questions[i].id
                st.button("⚑ Unflag" if q_id in s.flagged else "⚑ Flag for review", key=f"flag_{st.session_state.get('quiz_run', 0)}_{q_id}",
                          on_click=_toggle_flag, args=(q_id,))
        b1, b2 = st.columns(2)
        b1.button("Restart quiz", on_click=_restart, use_container_width=True)
        b2.button("Submit quiz", type="primary", on_click=_ask_confirm, use_container_width=True)

    if st.session_state.get("confirm_submit"):
        with st.container(border=True):
            left = len(s.unanswered_ids)
            st.warning(
                f"{left} question(s) are unanswered and will count as incorrect." if left
                else "You have answered every question."
            )
            if s.flagged:
                nums = ", ".join(str(i + 1) for i, q in enumerate(s.quiz.questions) if q.id in s.flagged)
                st.info(f"Flagged for review: question(s) {nums}.")
            st.write("Submit now? Answers cannot be changed afterwards.")
            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("Yes, submit", type="primary"):
                _finish()
                st.rerun()
            c2.button("Keep working", on_click=_cancel_confirm)
    if mode.startswith("One"):
        _inject_shortcuts()


def _render_results(s: QuizSession) -> None:
    a = s.attempt
    assert a is not None
    quiz = s.quiz
    st.markdown(f"### Results · {esc(quiz.title)}")
    if a.percent >= 80:
        st.success("Excellent work!")
        if st.session_state.get("_balloons_for") != a.id:  # celebrate once per attempt
            st.session_state["_balloons_for"] = a.id
            st.balloons()
    elif a.percent >= 50:
        st.info("Good effort - review the explanations below to close the gaps.")
    else:
        st.warning("Keep practising - the explanations below show where to focus.")
    items = [
        ("Score", f"{a.correct} / {a.total}", "correct answers"),
        ("Accuracy", f"{a.percent:.0f}%", f"{a.incorrect} wrong · {a.unanswered} skipped"),
    ]
    if a.duration_s is not None:
        items.append(("Time taken", format_duration(a.duration_s), "timer enabled"))
    metric_row(items)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(outcome_donut(a.correct, a.incorrect, a.unanswered), use_container_width=True)
    by_topic = performance_by(a.details, "topic")
    by_diff = performance_by(a.details, "difficulty")
    c2.plotly_chart(accuracy_bar(by_topic, "topic", "Accuracy by topic"), use_container_width=True)
    c3.plotly_chart(accuracy_bar(by_diff, "difficulty", "Accuracy by difficulty"), use_container_width=True)

    st.markdown("#### Recommendations")
    for tip in recommendations(a):
        st.markdown(f"- {tip}")

    st.markdown("#### Review")
    views = ["All questions", "Incorrect only", "Unanswered only"] + (["Flagged only"] if s.flagged else [])
    view = st.radio("Show", views, horizontal=True)
    by_id = {d.question_id: d for d in a.details}
    shown = 0
    for i, q in enumerate(quiz.questions):
        d = by_id[q.id]
        if view == "Incorrect only" and (d.is_correct or d.selected is None):
            continue
        if view == "Unanswered only" and d.selected is not None:
            continue
        if view == "Flagged only" and q.id not in s.flagged:
            continue
        shown += 1
        with st.container(border=True):
            st.markdown(question_html(q, i + 1, reveal=True, selected=d.selected, show_source=True), unsafe_allow_html=True)
            if d.selected is None:
                st.caption("You did not answer this question.")
    if not shown:
        st.info("Nothing to show for this filter - nice!" if view != "All questions" else "No questions.")

    st.markdown("#### Next")
    missed = len(s.missed_questions())
    b1, b2, b3, b4 = st.columns(4)
    b1.button("Retake quiz", type="primary", on_click=_retake, use_container_width=True)
    b2.button(f"Retry {missed} missed" if missed else "Nothing missed", disabled=not missed, on_click=_retry_missed,
              use_container_width=True, help="A short quiz with only the questions you got wrong or skipped")
    b3.button("Choose another quiz", on_click=_leave, use_container_width=True)
    with b4.popover("Export results", use_container_width=True):
        render_result_downloads(quiz, a, "results")


_SHORTCUTS_JS = """
<script>
(function () {
  const P = window.parent;
  if (!P || P.__qfKeys) return;
  P.__qfKeys = true;
  P.document.addEventListener('keydown', function (e) {
    const t = e.target;
    if (t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    const click = (pred) => {
      const b = Array.from(P.document.querySelectorAll('button')).find(pred);
      if (b && !b.disabled) { b.click(); e.preventDefault(); }
    };
    if (k.length === 1 && 'abcdef'.includes(k)) {
      click((b) => b.closest('[class*="st-key-qopts"]') && b.innerText.trim().toLowerCase().startsWith(k + '.'));
    } else if (k === 'arrowright') { click((b) => b.innerText.trim().startsWith('Next'));
    } else if (k === 'arrowleft') { click((b) => b.innerText.includes('Previous'));
    } else if (k === 'm') { click((b) => /Flag|Unflag/.test(b.innerText)); }
  });
})();
</script>
"""


def _inject_shortcuts() -> None:
    """Keyboard shortcuts (A-D, arrows, F). The listener is installed once per browser tab."""
    import streamlit.components.v1 as components

    components.html(_SHORTCUTS_JS, height=0)
