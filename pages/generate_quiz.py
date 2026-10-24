"""Generate Quiz: material -> configure -> generate (streamed, stoppable) -> review/edit -> save.

Generation runs inside one script run. Questions are stored in `st.session_state["gen_job"]` the moment they
pass validation, so pressing Stop (or leaving the page) keeps everything produced so far. The unsaved draft is
also written to SQLite, so a browser refresh or restart does not lose it.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime

import streamlit as st

from components.context import get_client, get_repo, get_settings, get_status, navigate, switch_model, tier_of
from components.downloads import render_quiz_downloads
from components.metrics import page_header
from components.question_card import question_html, render_add_question, render_quiz_editor, show_flash
from components.styles import esc
from config import MAX_QUESTIONS, MAX_UPLOAD_MB, QUESTION_COUNT_PRESETS, QUESTION_STYLES, TIER_TAGS
from models.schemas import GenerationRequest, Question, Quiz
from services.answer_checker import check_question
from services.document_processor import DocumentError, chunk_text, extract_text
from services.ollama_client import OllamaError
from services.question_generator import CHUNK_CHARS, QuestionGenerator
from services.question_validator import QuestionValidationError, find_duplicate_pairs, validate_question

EDITOR_KEY = "draft_editor"


# ------------------------------------------------------------------ helpers
def _truncate_material(limit: int) -> None:
    st.session_state["material_text"] = st.session_state.get("material_text", "")[:limit]


def _handle_upload(uploaded) -> None:
    sig = (uploaded.name, uploaded.size)
    if st.session_state.get("_upload_sig") == sig:
        return
    st.session_state["_upload_sig"] = sig
    st.session_state.pop("_upload_error", None)
    st.session_state.pop("_upload_warnings", None)
    try:
        doc = extract_text(uploaded.name, uploaded.getvalue())
    except DocumentError as exc:
        st.session_state["_upload_error"] = str(exc)
        return
    st.session_state["material_text"] = doc.text
    st.session_state["material_name"] = uploaded.name
    st.session_state["_upload_warnings"] = doc.warnings + (
        [f"{doc.pages} page(s) read."] if doc.pages else []
    )


def _on_tier_change() -> None:
    """Fast / Quality toggle. Refuses (and snaps back) if the chosen model is not installed."""
    choice = st.session_state.get("cfg_tier")
    if not choice:  # clicking the active segment again clears it: keep the current model
        st.session_state["cfg_tier"] = tier_of(get_settings().model)
        return
    error = switch_model(TIER_TAGS[choice])
    if error:
        st.session_state["tier_error"] = error
        st.session_state["cfg_tier"] = tier_of(get_settings().model)


def _warm_up_once() -> None:
    """Load the model into memory in the background so the first generation starts sooner."""
    if st.session_state.get("_warmed"):
        return
    st.session_state["_warmed"] = True
    client = get_client()
    threading.Thread(target=client.warm_up, daemon=True).start()


def _steps(material_ok: bool, has_draft: bool, generating: bool) -> None:
    items = [
        ("Material", "done" if material_ok else "active"),
        ("Configure", "done" if material_ok else "todo"),
        ("Generate", "active" if generating or (material_ok and not has_draft) else ("done" if has_draft else "todo")),
        ("Review", "active" if has_draft else "todo"),
    ]
    html = "".join(
        f'<div class="qf-step {state}"><span>{"&#10003;" if state == "done" else i + 1}</span>{label}</div>'
        for i, (label, state) in enumerate(items)
    )
    st.markdown(f'<div class="qf-steps">{html}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------- draft persistence
def _draft_state() -> dict:
    draft: Quiz = st.session_state["draft"]
    return {
        "quiz": draft.model_dump(),
        "req": (st.session_state.get("draft_req") or GenerationRequest()).model_dump(),
        "target": st.session_state.get("draft_target", len(draft.questions)),
        "warnings": st.session_state.get("draft_warnings", []),
        "rejected": st.session_state.get("draft_rejected", 0),
        "title": st.session_state.get("draft_title", draft.title),
        "subject": st.session_state.get("draft_subject", draft.subject),
    }


def _persist_draft() -> None:
    """Write the unsaved draft to SQLite when it changed since the last write."""
    if st.session_state.get("draft") is None:
        return
    state = _draft_state()
    sig = json.dumps(state, sort_keys=True, default=str)
    if st.session_state.get("_draft_sig") == sig:
        return
    get_repo().save_draft(state)
    st.session_state["_draft_sig"] = sig


def _restore_draft() -> None:
    if "draft" in st.session_state or st.session_state.get("_draft_checked"):
        return
    st.session_state["_draft_checked"] = True
    data = get_repo().load_draft()
    if not data:
        return
    try:
        st.session_state["draft"] = Quiz.model_validate(data["quiz"])
        st.session_state["draft_req"] = GenerationRequest.model_validate(data.get("req") or {})
    except Exception:  # corrupt or from an older version: drop it rather than crash
        get_repo().delete_draft()
        return
    st.session_state["draft_target"] = data.get("target", len(st.session_state["draft"].questions))
    st.session_state["draft_warnings"] = data.get("warnings", [])
    st.session_state["draft_rejected"] = data.get("rejected", 0)
    st.session_state["draft_title"] = data.get("title") or st.session_state["draft"].title
    st.session_state["draft_subject"] = data.get("subject", "")
    st.session_state["_draft_restored"] = True


def _discard_draft() -> None:
    get_repo().delete_draft()
    for key in ("draft", "draft_req", "draft_target", "draft_warnings", "draft_rejected", "_draft_sig", "_draft_restored"):
        st.session_state.pop(key, None)


# ------------------------------------------------------------------- page
def render() -> None:
    page_header("Generate Quiz", "Add your material, choose the settings, and let your local AI model write the questions.")
    settings = get_settings()
    status = get_status()
    _restore_draft()
    if not status.ready:
        st.warning(
            (status.error or "The local model is not ready.")
            + "  You can still prepare your material; generation unlocks once Ollama is running with the model installed."
        )
    else:
        _warm_up_once()
    if err := st.session_state.pop("gen_error", None):
        st.error(err)

    job = st.session_state.get("gen_job")
    material_now = st.session_state.get("material_text", "")
    _steps(bool(material_now.strip() or st.session_state.get("cfg_topic", "").strip()),
           st.session_state.get("draft") is not None, bool(job))
    if job and job["status"] == "running":  # the previous run was interrupted (Stop button or navigation)
        _finalize_job(job, stopped=True)

    # ------------------------------------------------------------ A. material
    with st.container(border=True):
        st.markdown("#### 1 · Learning material")
        uploaded = st.file_uploader(
            f"Upload a PDF, DOCX or TXT file (max {MAX_UPLOAD_MB} MB) - or paste text below",
            type=["pdf", "docx", "txt"],
            help="Files are read on this computer only. The extracted text appears below so you can review it first.",
        )
        if uploaded is not None:
            _handle_upload(uploaded)
            if err := st.session_state.get("_upload_error"):
                st.error(err)
            for w in st.session_state.get("_upload_warnings", []):
                st.info(w)
        st.text_area(
            "Material (review and edit before generating)",
            key="material_text",
            height=240,
            placeholder="Paste lecture notes, an article or a textbook section here...",
        )
        material = st.session_state.get("material_text", "")
        chars, words = len(material), len(material.split())
        over = chars > settings.max_input_chars
        n_chunks = len(chunk_text(material, CHUNK_CHARS)) if material.strip() else 0
        st.caption(
            f"{chars:,} characters · {words:,} words · {n_chunks} section(s) of about {CHUNK_CHARS:,} characters "
            f"· input limit {settings.max_input_chars:,} characters (change in Settings)"
        )
        if over:
            st.error(
                f"The material is {chars - settings.max_input_chars:,} characters over the limit. Nothing is truncated "
                "silently - shorten the text yourself or use the button to keep only the first part."
            )
            st.button(f"Keep the first {settings.max_input_chars:,} characters", on_click=_truncate_material, args=(settings.max_input_chars,))

    # ----------------------------------------------------------- B. configure
    with st.container(border=True):
        st.markdown("#### 2 · Configure")
        c1, c2, c3 = st.columns(3)
        topic = c1.text_input("Topic / subject", placeholder="e.g. Photosynthesis", key="cfg_topic")
        choice = c2.selectbox("Number of questions", [*map(str, QUESTION_COUNT_PRESETS), "Custom"], index=1, key="cfg_count_choice")
        if choice == "Custom":
            count = c2.number_input("Custom number (max 50)", 1, MAX_QUESTIONS, 12, key="cfg_count_custom")
        else:
            count = int(choice)
        difficulty = c3.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"], index=1, key="cfg_diff")
        c4, c5, c6 = st.columns(3)
        style = c4.selectbox("Question style", QUESTION_STYLES, index=3, key="cfg_style")
        num_options = c5.number_input("Options per question", 2, 6, 4, key="cfg_nopt")
        explanations = c6.toggle("Include explanations", True, key="cfg_expl")
        sources = c6.toggle("Include source references", True, key="cfg_src")
        double_check = st.toggle(
            "Double-check answer keys after generating (slower)", False, key="cfg_check",
            help="The model re-answers every question without seeing its key. Questions where it disagrees get a "
                 "'review the key' badge. This is a hint, not proof either way.",
        )
        st.session_state["cfg_tier"] = tier_of(settings.model)  # stay in sync with Settings
        st.segmented_control(
            "Model", list(TIER_TAGS), key="cfg_tier", on_change=_on_tier_change, selection_mode="single",
            help="Fast is quicker; Quality is slower but more accurate. Manage models in Settings.",
        )
        if err := st.session_state.pop("tier_error", None):
            st.warning(err)
        instructions = st.text_area("Optional instructions", key="cfg_instr", height=70,
                                    placeholder="e.g. Focus on definitions. Avoid questions about dates.")

    # ------------------------------------------------------------ C. generate
    has_input = bool(material.strip() or topic.strip())
    blocked = (not status.ready) or over or not has_input or bool(job)
    reason = (
        "Start Ollama and install the model first." if not status.ready
        else "Shorten the material first." if over
        else "Add material or at least a topic." if not has_input else ""
    )
    with st.container(border=True):
        st.markdown("#### 3 · Generate")
        st.caption(f"{int(count)} questions · {difficulty} · {style} · {int(num_options)} options each")
        if job and job["status"] == "pending":
            _run_job(job)
        elif st.button(f"Generate {int(count)} questions", type="primary", disabled=blocked, help=reason or None):
            req = GenerationRequest(
                topic=topic, count=int(count), difficulty=difficulty, style=style, num_options=int(num_options),
                instructions=instructions, include_explanations=explanations, include_sources=sources,
            )
            _start_job(req, material, append=False, double_check=double_check)
        if blocked and reason:
            st.caption(reason)

    draft: Quiz | None = st.session_state.get("draft")
    if draft is not None:
        _render_draft(draft, double_check)


# -------------------------------------------------------------- generation job
def _start_job(req: GenerationRequest, material: str, append: bool, double_check: bool = False) -> None:
    st.session_state["gen_job"] = {
        "status": "pending", "req": req, "material": material, "append": append,
        "double_check": double_check, "partial": [], "warnings": [], "rejected": 0,
    }
    st.rerun()


def _run_job(job: dict) -> None:
    """Runs the generator inside this script run. Anything that interrupts the run is handled by _finalize_job."""
    job["status"] = "running"
    req: GenerationRequest = job["req"]
    existing = st.session_state["draft"].questions if job["append"] and st.session_state.get("draft") else []
    top = st.columns([4, 1])
    top[0].markdown("**Writing questions with your local model...** each one appears below as soon as it passes validation.")
    top[1].button("■ Stop", key="gen_stop", help="Keep the questions produced so far")
    bar = st.progress(0.0, text="Starting...")
    live = st.container()
    t0 = time.monotonic()
    last = [0.0]
    number = [len(existing)]

    def progress(done: int, total: int, msg: str) -> None:
        now = time.monotonic()
        if now - last[0] < 0.4 and done < total:  # throttle token-level updates
            return
        last[0] = now
        got = len(job["partial"])
        bar.progress(min(done / max(total, 1), 1.0), text=f"{msg}  ·  {got} of {req.count} ready  ·  {int(now - t0)}s")

    def on_question(q: Question) -> None:
        job["partial"].append(q)
        number[0] += 1
        with live:
            st.markdown(f'<div class="qf-card qf-live">{question_html(q, number[0], show_source=False)}</div>', unsafe_allow_html=True)

    try:
        result = QuestionGenerator(get_client()).generate(
            job["material"], req, existing=existing, progress=progress, on_question=on_question
        )
    except (OllamaError, ValueError) as exc:
        job["warnings"].append(str(exc))
        _finalize_job(job, error=str(exc))
        return
    job["warnings"] += result.warnings
    job["rejected"] = len(result.rejected)
    if job["double_check"] and job["partial"]:
        checked = list(job["partial"])
        try:
            for i, q in enumerate(checked):
                bar.progress(i / len(checked), text=f"Double-checking answer keys: {i + 1} of {len(checked)}...")
                checked[i] = q.model_copy(update={"key_check": check_question(get_client(), q)})
                job["partial"][i] = checked[i]  # keep progress if the run is interrupted
        except OllamaError as exc:
            job["warnings"].append(f"Answer-key double-check stopped early: {exc}")
    _finalize_job(job, result_questions=job["partial"])


def _finalize_job(job: dict, stopped: bool = False, error: str | None = None, result_questions: list[Question] | None = None) -> None:
    questions = list(result_questions if result_questions is not None else job["partial"])
    req: GenerationRequest = job["req"]
    repo = get_repo()
    warnings = list(job["warnings"])
    if stopped:
        warnings.append(f"Generation was stopped - {len(questions)} question(s) were kept.")
    st.session_state.pop("gen_job", None)

    if not questions and not st.session_state.get("draft"):
        st.session_state["gen_error"] = error or "No valid questions were produced. Try again, or use a larger model."
        st.rerun()

    if job["append"] and st.session_state.get("draft") is not None:
        quiz: Quiz = st.session_state["draft"]
        quiz.questions.extend(questions)
        target = st.session_state.get("draft_target", len(quiz.questions))
        original_req = st.session_state.get("draft_req") or req
    else:
        name = st.session_state.get("material_name", "Pasted text") if job["material"].strip() else "Topic only"
        quiz = Quiz(
            title=(req.topic.strip() or name.rsplit(".", 1)[0]) + " quiz",
            subject=req.topic.strip(), questions=questions, source_name=name,
            source_text=job["material"], model=get_settings().model,
        )
        target, original_req = req.count, req
        st.session_state["draft_title"] = quiz.title
        st.session_state["draft_subject"] = quiz.subject
        st.session_state.pop("_draft_restored", None)
    st.session_state["draft"] = quiz
    st.session_state["draft_req"] = original_req
    st.session_state["draft_target"] = target
    st.session_state["draft_warnings"] = warnings
    st.session_state["draft_rejected"] = job["rejected"]
    if questions:
        repo.bump_counter("questions_generated", len(questions))
        repo.log_activity("generate", f"Generated {len(questions)} question(s) for '{quiz.title}'")
    _persist_draft()
    st.rerun()


# ------------------------------------------------------------------ review
def _render_draft(draft: Quiz, double_check: bool) -> None:
    req: GenerationRequest = st.session_state.get("draft_req") or GenerationRequest()
    st.markdown("### 4 · Review and edit")
    show_flash(EDITOR_KEY)
    if st.session_state.pop("_draft_restored", False):
        st.info(f"Restored your unsaved draft '{draft.title}' ({len(draft.questions)} questions).")
    for w in st.session_state.get("draft_warnings", []):
        st.warning(w)
    if st.session_state.get("draft_rejected"):
        st.caption(f"{st.session_state['draft_rejected']} model candidate(s) were rejected by validation and are not shown.")
    target = st.session_state.get("draft_target", len(draft.questions))
    need = target - len(draft.questions)
    status = get_status()
    if need > 0 and not st.session_state.get("gen_job"):
        if st.button(f"Generate the remaining {need} question(s)", disabled=not status.ready):
            _start_job(req.model_copy(update={"count": need}), draft.source_text, append=True)

    t1, t2 = st.columns(2)
    draft.title = t1.text_input("Quiz title", key="draft_title") or draft.title
    draft.subject = t2.text_input("Subject", key="draft_subject")
    unchecked = sum(1 for q in draft.questions if q.key_check == "unchecked")
    cap = st.columns([2.3, 1.7])
    cap[0].caption(f"{len(draft.questions)} question(s) · answer keys come from the AI model - please review them.")
    if unchecked and cap[1].button("Double-check answer keys", disabled=not status.ready,
                                   help="The model re-answers each question without seeing the key and flags disagreements."):
        _double_check_draft(draft)

    generator = QuestionGenerator(get_client())
    render_quiz_editor(draft, EDITOR_KEY, generator=generator, show_source=req.include_sources, num_options=req.num_options)
    render_add_question(draft, EDITOR_KEY, req.num_options)

    st.markdown("#### Save")
    s1, s2, s3, _ = st.columns([1.3, 1.3, 1.1, 1.5])
    save = s1.button("Save to question bank", type="primary", disabled=not draft.questions)
    start = s2.button("Save and take quiz", disabled=not draft.questions)
    s3.button("Discard draft", on_click=_discard_draft)
    if save or start:
        try:
            validated = [validate_question(q, num_options=len(q.options)) for q in draft.questions]
        except QuestionValidationError as exc:
            st.error(f"Cannot save: {exc}")
            return
        dups = find_duplicate_pairs(validated)
        if dups:
            st.warning("Possible duplicates: " + ", ".join(f"Q{a + 1} & Q{b + 1}" for a, b in dups) + " (saved anyway).")
        draft.questions = validated
        saved = get_repo().save_quiz(draft)
        get_repo().log_activity("save", f"Saved quiz '{saved.title}' ({len(saved.questions)} questions)")
        get_repo().delete_draft()
        st.session_state["draft"] = saved
        st.session_state["_draft_sig"] = json.dumps(_draft_state(), sort_keys=True, default=str)
        if start:
            st.session_state["take_quiz_id"] = saved.id
            navigate("Take a Quiz")
        st.success(f"Saved '{saved.title}' to the question bank.")
    with st.expander("Export"):
        render_quiz_downloads(draft, "draft")
    _persist_draft()


def _double_check_draft(draft: Quiz) -> None:
    bar = st.progress(0.0, text="Double-checking answer keys...")
    client = get_client()
    try:
        for i, q in enumerate(draft.questions):
            bar.progress(i / len(draft.questions), text=f"Double-checking answer keys: {i + 1} of {len(draft.questions)}...")
            if q.key_check == "unchecked":
                draft.questions[i] = q.model_copy(update={"key_check": check_question(client, q)})
    except OllamaError as exc:
        bar.empty()
        st.error(str(exc))
        return
    flagged = sum(1 for q in draft.questions if q.key_check == "disagrees")
    st.session_state[f"{EDITOR_KEY}_flash"] = (
        f"Double-check finished: {flagged} question(s) need a closer look." if flagged
        else "Double-check finished: the model agreed with every key (this is a hint, not proof)."
    )
    st.rerun()
