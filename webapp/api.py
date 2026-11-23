"""JSON API used by the browser front end (webapp/static). Every route is under /api.

The model is treated as an untrusted writer: everything it returns goes through services/question_validator.py,
and scores are computed only from the stored answer key.
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict

from flask import Blueprint, Response, current_app, jsonify, request
from pydantic import ValidationError

from config import (
    ALLOWED_EXTENSIONS,
    DB_PATH,
    MAX_QUESTIONS,
    MAX_UPLOAD_MB,
    QUESTION_COUNT_PRESETS,
    QUESTION_STYLES,
)
from models.schemas import GenerationRequest, Question, Quiz
from services import export_service as ex
from services.answer_checker import check_question
from services.document_processor import DocumentError, chunk_text, extract_text, validate_upload
from services.material_guard import MaterialRefused, guard_material
from services.ollama_client import OllamaClient, OllamaError, same_model
from services.question_generator import CHUNK_CHARS, QuestionGenerator
from services.question_validator import QuestionValidationError, find_duplicate_pairs, is_duplicate, validate_question
from services.quiz_service import aggregate_details, performance_by, recommendations, score_quiz
from webapp.context import AppContext, is_local_url, model_name
from webapp.jobs import Job, StopRequested

bp = Blueprint("api", __name__, url_prefix="/api")


def ctx() -> AppContext:
    return current_app.extensions["quizforge"]


def body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def fail(message: str, status: int = 400, **extra) -> tuple[Response, int]:
    return jsonify({"error": message, **extra}), status


def pydantic_message(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "__root__")
    msg = str(first.get("msg", "invalid value")).removeprefix("Value error, ")
    return f"{where}: {msg}" if where else msg


def records(df) -> list[dict]:
    """DataFrame rows as plain JSON-able dicts (numpy numbers become Python numbers)."""
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in df.to_dict("records")]


def parse_questions(items) -> list[Question]:
    if not isinstance(items, list):
        return []
    return [Question.model_validate(q) for q in items]


# ------------------------------------------------------------------------------------------ status & settings
@bp.get("/status")
def status():
    return jsonify(ctx().status(force=request.args.get("force") == "1"))


@bp.get("/settings")
def get_settings():
    s = ctx().settings()
    return jsonify({
        **asdict(s),
        "model_name": model_name(s.model),
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB, "extensions": list(ALLOWED_EXTENSIONS), "max_questions": MAX_QUESTIONS,
            "count_presets": QUESTION_COUNT_PRESETS, "styles": QUESTION_STYLES, "chunk_chars": CHUNK_CHARS,
        },
        "db_path": str(DB_PATH),
    })


@bp.put("/settings")
def put_settings():
    data, c = body(), ctx()
    s = c.settings()
    url = str(data.get("base_url", s.base_url)).strip().rstrip("/")
    model = str(data.get("model", s.model)).strip()
    if not url.startswith(("http://", "https://")):
        return fail("The Ollama URL must start with http:// or https://")
    if not model:
        return fail("Choose or type a model tag.")
    try:
        temperature = float(data.get("temperature", s.temperature))
        timeout_s = int(data.get("timeout_s", s.timeout_s))
        max_chars = int(data.get("max_input_chars", s.max_input_chars))
    except (TypeError, ValueError):
        return fail("Temperature, timeout and input size must be numbers.")
    if not 0.0 <= temperature <= 1.2:
        return fail("Temperature must be between 0 and 1.2.")
    if not 30 <= timeout_s <= 3600:
        return fail("The timeout must be between 30 and 3600 seconds.")
    if not 2000 <= max_chars <= 500_000:
        return fail("The input limit must be between 2,000 and 500,000 characters.")
    s.base_url, s.model, s.temperature, s.timeout_s, s.max_input_chars = url, model, temperature, timeout_s, max_chars
    c.save_settings(s)
    c.repo.log_activity("settings", "Settings saved")
    out = {"ok": True}
    if not is_local_url(url):
        out["warning"] = "This Ollama server is not on this computer: your study material will be sent there."
    return jsonify(out)


@bp.post("/settings/test")
def test_connection():
    data, c = body(), ctx()
    s = c.settings()
    s.base_url = str(data.get("base_url") or s.base_url).strip().rstrip("/")
    s.model = str(data.get("model") or s.model).strip()
    st = c.client_factory(s).status()
    installed = st.connected and any(same_model(s.model, m) for m in st.installed_models)
    if installed:
        return jsonify({"ok": True, "message": f"Connected to {s.base_url}. The model is installed and ready."})
    if st.connected:
        return jsonify({"ok": False, "message": "Connected, but that model is not installed on this server."})
    return jsonify({"ok": False, "message": st.error or "Cannot reach Ollama."})


@bp.post("/model")
def switch_model():
    """Make a model active. Refused when Ollama is reachable and the model is not installed."""
    tag = str(body().get("tag", "")).strip()
    if not tag:
        return fail("No model given.")
    c = ctx()
    st = c.status(force=True)
    if st["connected"] and not any(same_model(tag, m) for m in st["installed_models"]):
        return fail("That model is not installed on this computer yet. See Settings for how to install it.", 409)
    s = c.settings()
    s.model = tag
    c.save_settings(s)
    c.repo.log_activity("settings", f"Switched to the {model_name(tag)} model")
    threading.Thread(target=c.client(tag).warm_up, daemon=True).start()
    return jsonify({"ok": True, "model_name": model_name(tag)})


@bp.post("/warmup")
def warmup():
    """Ask Ollama to load the model now, so the first generation does not wait for it."""
    threading.Thread(target=ctx().client().warm_up, daemon=True).start()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------------------------- material
@bp.post("/extract")
def extract():
    f = request.files.get("file")
    if f is None or not f.filename:
        return fail("Choose a PDF, DOCX or TXT file.")
    data = f.read()
    try:
        validate_upload(f.filename, len(data))
        doc = extract_text(f.filename, data)
    except DocumentError as exc:
        return fail(str(exc))
    limit = ctx().settings().max_input_chars
    return jsonify({"name": doc.name, "text": doc.text, "pages": doc.pages, "warnings": doc.warnings,
                    "chars": doc.chars, "words": doc.words, "over_limit": doc.chars > limit, "limit": limit})


@bp.post("/material/check")
def check_material():
    """Counts, sections and the prompt-injection preview for the text box (no model call)."""
    text = str(body().get("text", ""))
    limit = ctx().settings().max_input_chars
    guarded = guard_material(text) if text.strip() else None
    return jsonify({
        "chars": len(text), "words": len(text.split()),
        "sections": len(chunk_text(text, CHUNK_CHARS)) if text.strip() else 0,
        "limit": limit, "over_limit": len(text) > limit,
        "removed": guarded.removed if guarded else [],
        "refused": bool(guarded and guarded.refused),
    })


# -------------------------------------------------------------------------------------------- generation
@bp.post("/generate")
def generate():
    data, c = body(), ctx()
    try:
        req = GenerationRequest.model_validate(data.get("request") or {})
        existing = parse_questions(data.get("existing"))
    except ValidationError as exc:
        return fail(pydantic_message(exc))
    material = str(data.get("material", ""))
    limit = c.settings().max_input_chars
    if len(material) > limit:
        return fail(f"The material is {len(material) - limit:,} characters over the {limit:,}-character limit.")
    if not material.strip() and not req.topic.strip():
        return fail("Add some learning material or at least a topic.")
    st = c.status()
    if not st["ready"]:
        return fail(st["error"] or "The local model is not ready.", 409)
    client = c.client()
    job = c.jobs.start("generate", lambda j: _generation(j, c, client, material, req, existing,
                                                         bool(data.get("double_check"))))
    return jsonify({"job_id": job.id, "model_name": model_name(client.model)}), 202


def _generation(job: Job, c: AppContext, client: OllamaClient, material: str, req: GenerationRequest,
                existing: list[Question], double_check: bool) -> None:
    job.total = req.count
    try:
        result = QuestionGenerator(client).generate(material, req, existing=existing, progress=job.progress,
                                                    on_question=job.add_question)
        job.warnings += result.warnings
        job.rejected = len(result.rejected)
    except StopRequested:
        _record_generated(c, job)
        raise
    except MaterialRefused as exc:
        job.refused = True
        job.finish("error", str(exc))
        return
    except (OllamaError, ValueError) as exc:
        if not job.questions:
            job.finish("error", str(exc))
            return
        job.warnings.append(str(exc))
    _record_generated(c, job)
    if double_check and job.questions:
        try:
            _check_all(job, client, len(job.questions))
        except OllamaError as exc:
            job.warnings.append(f"The answer-key double-check stopped early: {exc}")
    job.progress(job.total, job.total, "Done")
    job.finish("done")


def _record_generated(c: AppContext, job: Job) -> None:
    n = len(job.questions)
    if n:
        c.repo.bump_counter("questions_generated", n)
        c.repo.log_activity("generate", f"Generated {n} question(s)")


def _check_all(job: Job, client: OllamaClient, n: int) -> None:
    for i in range(n):
        job.progress(i, n, f"Double-checking answer key {i + 1} of {n}...")
        q = job.questions[i]
        if q.key_check == "unchecked":
            job.replace_question(i, q.model_copy(update={"key_check": check_question(client, q)}))


@bp.post("/check-keys")
def check_keys():
    c = ctx()
    try:
        questions = parse_questions(body().get("questions"))
    except ValidationError as exc:
        return fail(pydantic_message(exc))
    if not questions:
        return fail("There are no questions to check.")
    st = c.status()
    if not st["ready"]:
        return fail(st["error"] or "The local model is not ready.", 409)
    client = c.client()

    def work(job: Job) -> None:
        job.questions = [q.model_copy(update={"key_check": "unchecked"}) for q in questions]
        try:
            _check_all(job, client, len(questions))
        except OllamaError as exc:
            job.finish("error", str(exc))
            return
        flagged = sum(q.key_check == "disagrees" for q in job.questions)
        job.progress(len(questions), len(questions), f"{flagged} question(s) need a closer look." if flagged
                     else "The model agreed with every key (a hint, not proof).")
        job.finish("done")

    return jsonify({"job_id": c.jobs.start("check", work).id}), 202


@bp.get("/jobs/<job_id>")
def job_status(job_id: str):
    job = ctx().jobs.get(job_id)
    if job is None:
        return fail("This job is no longer available.", 404)
    return jsonify(job.snapshot(request.args.get("since", 0, type=int)))


@bp.post("/jobs/<job_id>/stop")
def job_stop(job_id: str):
    job = ctx().jobs.get(job_id)
    if job is None:
        return fail("This job is no longer available.", 404)
    job.stop_event.set()
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------------------- one question
@bp.post("/questions/validate")
def validate_one():
    """Re-check a question a person edited or wrote by hand."""
    data = body()
    raw = data.get("question") or {}
    try:
        n = len(raw.get("options") or []) or None
        q = validate_question(raw, num_options=n)
        others = parse_questions(data.get("others"))
    except (QuestionValidationError, ValidationError, AttributeError) as exc:
        return fail(pydantic_message(exc) if isinstance(exc, ValidationError) else str(exc))
    keep = {"source_section": str(raw.get("source_section") or ""), "key_check": "unchecked"}  # an edited key is no longer checked
    if raw.get("id"):
        keep["id"] = str(raw["id"])
    q = q.model_copy(update=keep)
    warning = "This looks like a duplicate of another question." if is_duplicate(q, [o for o in others if o.id != q.id]) else ""
    return jsonify({"question": q.model_dump(), "warning": warning})


@bp.post("/questions/regenerate")
def regenerate_one():
    data, c = body(), ctx()
    try:
        old = Question.model_validate(data.get("question") or {})
        others = parse_questions(data.get("others"))
        req = GenerationRequest.model_validate(data.get("request") or {}) if data.get("request") else None
    except ValidationError as exc:
        return fail(pydantic_message(exc))
    st = c.status()
    if not st["ready"]:
        return fail(st["error"] or "The local model is not ready.", 409)
    try:
        new = QuestionGenerator(c.client()).regenerate_one(old, others, str(data.get("source_text", "")), req)
    except (QuestionValidationError, OllamaError, ValueError) as exc:
        return fail(str(exc), 422)
    c.repo.bump_counter("questions_generated", 1)
    return jsonify({"question": new.model_dump()})


# -------------------------------------------------------------------------------------------------- draft
@bp.get("/draft")
def get_draft():
    return jsonify({"draft": ctx().repo.load_draft()})


@bp.put("/draft")
def put_draft():
    draft = body().get("draft")
    if not isinstance(draft, dict):
        return fail("No draft given.")
    ctx().repo.save_draft(draft)
    return jsonify({"ok": True})


@bp.delete("/draft")
def delete_draft():
    ctx().repo.delete_draft()
    return jsonify({"ok": True})


# ------------------------------------------------------------------------------------------------ quizzes
@bp.get("/quizzes")
def list_quizzes():
    repo = ctx().repo
    rows = repo.list_quizzes(request.args.get("search", ""), request.args.get("subject", ""),
                             request.args.get("difficulty", ""))
    for r in rows:
        r["model_name"] = model_name(r.get("model") or "") if r.get("model") else ""
    return jsonify({"quizzes": rows, "subjects": repo.list_subjects()})


def _save(quiz_data: dict, quiz_id: str | None = None):
    c = ctx()
    try:
        quiz = Quiz.model_validate(quiz_data)
    except ValidationError as exc:
        return fail(pydantic_message(exc))
    if quiz_id:
        old = c.repo.get_quiz(quiz_id)
        if old is None:
            return fail("Quiz not found.", 404)
        quiz = quiz.model_copy(update={"id": quiz_id, "created_at": old.created_at,
                                       "source_text": quiz.source_text or old.source_text})
    if not quiz.title.strip():
        return fail("Give the quiz a title.")
    if not quiz.questions:
        return fail("A quiz needs at least one question.")
    checked = []
    for i, q in enumerate(quiz.questions, 1):
        try:
            v = validate_question(q, num_options=len(q.options))
        except QuestionValidationError as exc:
            return fail(f"Question {i}: {exc}")
        checked.append(v.model_copy(update={"id": q.id, "key_check": q.key_check, "source_section": q.source_section}))
    quiz = quiz.model_copy(update={"questions": checked, "title": quiz.title.strip()})
    if not quiz.model:
        quiz = quiz.model_copy(update={"model": c.settings().model})
    saved = c.repo.save_quiz(quiz)
    c.repo.log_activity("save", f"Saved quiz '{saved.title}' ({len(saved.questions)} questions)")
    dups = find_duplicate_pairs(saved.questions)
    warning = ("Possible duplicates: " + ", ".join(f"Q{a + 1} & Q{b + 1}" for a, b in dups) + " (saved anyway).") if dups else ""
    return jsonify({"quiz": saved.model_dump(), "warning": warning})


@bp.post("/quizzes")
def create_quiz():
    return _save(body().get("quiz") or {})


@bp.get("/quizzes/<quiz_id>")
def get_quiz(quiz_id: str):
    quiz = ctx().repo.get_quiz(quiz_id)
    if quiz is None:
        return fail("Quiz not found.", 404)
    out = quiz.model_dump()
    out["model_name"] = model_name(quiz.model) if quiz.model else ""
    return jsonify({"quiz": out})


@bp.put("/quizzes/<quiz_id>")
def update_quiz(quiz_id: str):
    return _save(body().get("quiz") or {}, quiz_id)


@bp.patch("/quizzes/<quiz_id>")
def rename_quiz(quiz_id: str):
    repo = ctx().repo
    if repo.get_quiz(quiz_id) is None:
        return fail("Quiz not found.", 404)
    try:
        repo.rename_quiz(quiz_id, str(body().get("title", "")))
    except ValueError as exc:
        return fail(str(exc))
    return jsonify({"ok": True})


@bp.delete("/quizzes/<quiz_id>")
def delete_quiz(quiz_id: str):
    repo = ctx().repo
    quiz = repo.get_quiz(quiz_id)
    if quiz is None:
        return fail("Quiz not found.", 404)
    repo.delete_quiz(quiz_id)
    session = repo.load_active_session()
    if session and session.get("quiz_id") == quiz_id:
        repo.clear_active_session()
    repo.log_activity("delete", f"Deleted quiz '{quiz.title}'")
    return jsonify({"ok": True})


@bp.post("/quizzes/<quiz_id>/duplicate")
def duplicate_quiz(quiz_id: str):
    repo = ctx().repo
    try:
        copy = repo.duplicate_quiz(quiz_id)
    except KeyError:
        return fail("Quiz not found.", 404)
    repo.log_activity("save", f"Duplicated quiz as '{copy.title}'")
    return jsonify({"quiz": copy.model_dump()})


EXPORTS = {
    "paper.pdf": ("application/pdf", lambda q: ex.question_paper_pdf(q), "question-paper.pdf"),
    "paper.html": ("text/html", lambda q: ex.question_paper_html(q).encode("utf-8"), "question-paper.html"),
    "key.pdf": ("application/pdf", lambda q: ex.answer_key_pdf(q), "answer-key.pdf"),
    "key.html": ("text/html", lambda q: ex.answer_key_html(q).encode("utf-8"), "answer-key.html"),
    "key.csv": ("text/csv", lambda q: ex.to_download_bytes(ex.answer_key_csv(q), bom=True), "answer-key.csv"),
    "questions.json": ("application/json", lambda q: ex.quiz_to_json(q).encode("utf-8"), "questions.json"),
    "questions.csv": ("text/csv", lambda q: ex.to_download_bytes(ex.quiz_to_csv(q), bom=True), "questions.csv"),
    "blank.json": ("application/json", lambda q: ex.quiz_to_json(q, include_answers=False).encode("utf-8"),
                   "questions-without-answers.json"),
    "blank.csv": ("text/csv", lambda q: ex.to_download_bytes(ex.quiz_to_csv(q, include_answers=False), bom=True),
                  "questions-without-answers.csv"),
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "quiz"


def _download(content: bytes, mimetype: str, filename: str) -> Response:
    return Response(content, mimetype=mimetype,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@bp.get("/quizzes/<quiz_id>/export/<kind>")
def export_quiz(quiz_id: str, kind: str):
    quiz = ctx().repo.get_quiz(quiz_id)
    if quiz is None:
        return fail("Quiz not found.", 404)
    if kind not in EXPORTS:
        return fail("Unknown export format.", 404)
    mimetype, build, suffix = EXPORTS[kind]
    return _download(build(quiz), mimetype, f"{slug(quiz.title)}-{suffix}")


@bp.post("/export/<kind>")
def export_unsaved(kind: str):
    """Exports for a draft that is not saved yet (the quiz comes in the request body)."""
    if kind not in EXPORTS:
        return fail("Unknown export format.", 404)
    try:
        quiz = Quiz.model_validate(body().get("quiz") or {})
    except ValidationError as exc:
        return fail(pydantic_message(exc))
    mimetype, build, suffix = EXPORTS[kind]
    return _download(build(quiz), mimetype, f"{slug(quiz.title)}-{suffix}")


# ----------------------------------------------------------------------------------------------- attempts
@bp.post("/quizzes/<quiz_id>/score")
def score(quiz_id: str):
    """Grade answers against the stored key. A retry of only the missed questions is graded but not stored."""
    c, data = ctx(), body()
    quiz = c.repo.get_quiz(quiz_id)
    if quiz is None:
        return fail("Quiz not found.", 404)
    answers = data.get("answers") or {}
    if not isinstance(answers, dict):
        return fail("Answers must be an object of question id -> option id.")
    subset = data.get("question_ids")
    if subset:
        wanted = set(subset)
        quiz = quiz.model_copy(update={"questions": [q for q in quiz.questions if q.id in wanted],
                                       "title": f"{quiz.title} (missed questions)"})
    duration = data.get("duration_s")
    attempt = score_quiz(quiz, {str(k): str(v) for k, v in answers.items()},
                         float(duration) if isinstance(duration, (int, float)) else None)
    saved = not subset and data.get("save", True)
    if saved:
        c.repo.save_attempt(attempt)
        c.repo.log_activity("attempt", f"Scored {attempt.percent:.0f}% on '{quiz.title}'")
        c.repo.clear_active_session()
    return jsonify({
        "attempt": attempt.model_dump(), "saved": bool(saved),
        "recommendations": recommendations(attempt),
        "by_topic": records(performance_by(attempt.details, "topic")),
        "by_difficulty": records(performance_by(attempt.details, "difficulty")),
    })


def _find_attempt(attempt_id: str):
    return next((a for a in ctx().repo.list_attempts() if a.id == attempt_id), None)


@bp.get("/attempts/<attempt_id>/export/<fmt>")
def export_attempt(attempt_id: str, fmt: str):
    attempt = _find_attempt(attempt_id)
    if attempt is None:
        return fail("Attempt not found.", 404)
    quiz = ctx().repo.get_quiz(attempt.quiz_id)
    if quiz is None:
        return fail("The quiz of this attempt was deleted.", 404)
    if fmt == "csv":
        return _download(ex.to_download_bytes(ex.results_to_csv(quiz, attempt), bom=True), "text/csv",
                         f"{slug(quiz.title)}-results.csv")
    if fmt == "json":
        return _download(ex.results_to_json(quiz, attempt).encode("utf-8"), "application/json",
                         f"{slug(quiz.title)}-results.json")
    return fail("Unknown export format.", 404)


@bp.delete("/attempts")
def clear_attempts():
    repo = ctx().repo
    n = repo.clear_attempts()
    repo.log_activity("history", f"Cleared {n} quiz attempt(s)")
    return jsonify({"deleted": n})


@bp.get("/session")
def get_session():
    repo = ctx().repo
    state = repo.load_active_session()
    if not state:
        return jsonify({"session": None})
    quiz = repo.get_quiz(state.get("quiz_id", ""))
    if quiz is None:
        repo.clear_active_session()
        return jsonify({"session": None})
    return jsonify({"session": {**state, "quiz_title": quiz.title, "total": len(quiz.questions)}})


@bp.put("/session")
def put_session():
    state = body().get("session")
    if not isinstance(state, dict) or not state.get("quiz_id"):
        return fail("No quiz session given.")
    ctx().repo.save_active_session(state)
    return jsonify({"ok": True})


@bp.delete("/session")
def delete_session():
    ctx().repo.clear_active_session()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------------- dashboard & analytics
@bp.get("/dashboard")
def dashboard():
    repo = ctx().repo
    draft = repo.load_draft()
    draft_info = None
    if draft and isinstance(draft.get("quiz"), dict):
        draft_info = {"title": draft["quiz"].get("title", "Untitled quiz"),
                      "questions": len(draft["quiz"].get("questions") or [])}
    return jsonify({
        "stats": repo.dashboard_stats(),
        "activity": repo.recent_activity(8),
        "recent_quizzes": repo.list_quizzes()[:4],
        "draft": draft_info,
    })


@bp.get("/analytics")
def analytics():
    repo = ctx().repo
    quiz_id = request.args.get("quiz_id") or None
    attempts = repo.list_attempts(quiz_id)
    details = aggregate_details(attempts)
    answered = [d for d in details if d.selected is not None]
    chrono = list(reversed(attempts))
    return jsonify({
        "summary": {
            "attempts": len(attempts),
            "average": round(sum(a.percent for a in attempts) / len(attempts), 1) if attempts else None,
            "best": max((a.percent for a in attempts), default=None),
            "latest": attempts[0].percent if attempts else None,
            "questions_answered": len(answered),
            "accuracy": round(100 * sum(d.is_correct for d in details) / len(details), 1) if details else None,
        },
        "by_topic": records(performance_by(details, "topic")),
        "by_difficulty": records(performance_by(details, "difficulty")),
        "trend": [{"at": a.submitted_at, "percent": a.percent, "quiz": a.quiz_title} for a in chrono[-30:]],
        "attempts": [{"id": a.id, "quiz_id": a.quiz_id, "quiz": a.quiz_title, "percent": a.percent,
                      "correct": a.correct, "total": a.total, "duration_s": a.duration_s, "at": a.submitted_at}
                     for a in attempts[:50]],
        "recommendations": recommendations(details),
        "quizzes": [{"id": q["id"], "title": q["title"]} for q in repo.list_quizzes() if q["n_attempts"]],
    })


# --------------------------------------------------------------------------------------------------- errors
@bp.errorhandler(404)
def not_found(_exc):
    return fail("Not found.", 404)


@bp.errorhandler(Exception)
def unexpected(exc):
    current_app.logger.exception("API error")
    code = getattr(exc, "code", 500)
    if not isinstance(code, int):
        code = 500
    return fail(f"Something went wrong: {getattr(exc, 'description', None) or exc}", code)
