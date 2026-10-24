"""Exports: JSON, CSV, printable HTML question paper, answer key, results."""
from __future__ import annotations

import csv
import html
import io
import json

from models.schemas import Attempt, Quiz


# ---------------------------------------------------------------------- JSON
def quiz_to_json(quiz: Quiz, include_answers: bool = True) -> str:
    questions = []
    for q in quiz.questions:
        item = {
            "id": q.id,
            "question": q.question,
            "options": [o.model_dump() for o in q.options],
            "difficulty": q.difficulty,
            "topic": q.topic,
        }
        if include_answers:
            item.update(
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                source_excerpt=q.source_excerpt,
                source_section=q.source_section,
            )
        questions.append(item)
    doc = {"title": quiz.title, "subject": quiz.subject,
           "created_at": quiz.created_at, "questions": questions}
    return json.dumps(doc, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------- CSV
def _csv(rows: list[list], header: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def quiz_to_csv(quiz: Quiz, include_answers: bool = True) -> str:
    letters = sorted({o.id for q in quiz.questions for o in q.options})
    header = ["no", "question", *[f"option_{l}" for l in letters], "difficulty", "topic"]
    if include_answers:
        header += ["correct_answer", "explanation", "source_excerpt"]
    rows = []
    for i, q in enumerate(quiz.questions, 1):
        opts = {o.id: o.text for o in q.options}
        row = [i, q.question, *[opts.get(l, "") for l in letters], q.difficulty, q.topic]
        if include_answers:
            row += [q.correct_answer, q.explanation, q.source_excerpt]
        rows.append(_neutralise_row(row))
    return _csv(rows, header)


def answer_key_csv(quiz: Quiz) -> str:
    rows = [[i, q.correct_answer, q.correct_text(), q.explanation] for i, q in enumerate(quiz.questions, 1)]
    return _csv([_neutralise_row(r) for r in rows], ["no", "correct_answer", "correct_text", "explanation"])


def results_to_csv(quiz: Quiz, attempt: Attempt) -> str:
    by_id = {d.question_id: d for d in attempt.details}
    rows = []
    for i, q in enumerate(quiz.questions, 1):
        d = by_id.get(q.id)
        sel = d.selected if d else None
        rows.append([
            i, q.question, sel or "(unanswered)", q.correct_answer,
            "correct" if d and d.is_correct else ("unanswered" if sel is None else "incorrect"),
            q.topic, q.difficulty,
        ])
    return _csv([_neutralise_row(r) for r in rows],
                ["no", "question", "your_answer", "correct_answer", "result", "topic", "difficulty"])


def results_to_json(quiz: Quiz, attempt: Attempt) -> str:
    by_id = {d.question_id: d for d in attempt.details}
    doc = {
        "quiz": quiz.title,
        "submitted_at": attempt.submitted_at,
        "score": {
            "correct": attempt.correct, "incorrect": attempt.incorrect, "unanswered": attempt.unanswered,
            "total": attempt.total, "percent": attempt.percent, "duration_s": attempt.duration_s,
        },
        "answers": [
            {
                "no": i, "question": q.question,
                "selected": by_id[q.id].selected if q.id in by_id else None,
                "correct_answer": q.correct_answer,
                "is_correct": by_id[q.id].is_correct if q.id in by_id else False,
                "explanation": q.explanation,
            }
            for i, q in enumerate(quiz.questions, 1)
        ],
    }
    return json.dumps(doc, indent=2, ensure_ascii=False)


def _neutralise_row(row: list) -> list:
    """Prevent spreadsheet formula injection from model/user supplied text."""
    out = []
    for cell in row:
        if isinstance(cell, str) and cell[:1] in ("=", "+", "-", "@"):
            cell = "'" + cell
        out.append(cell)
    return out


def to_download_bytes(text: str, bom: bool = False) -> bytes:
    """UTF-8 bytes; `bom=True` makes Excel open CSVs with the right encoding."""
    return text.encode("utf-8-sig" if bom else "utf-8")


# ----------------------------------------------------------------------- PDF
_PDF_CSS = """
body { font-family: sans-serif; font-size: 11pt; }
h1 { font-size: 18pt; color: #1f2350; }
.meta { color: #555555; font-size: 10pt; }
.q { font-weight: bold; margin-top: 12pt; }
.o { margin-left: 18pt; }
.a { margin-left: 18pt; color: #1a6b2e; }
.e { margin-left: 18pt; color: #444444; font-size: 10pt; }
"""


def _html_to_pdf(body: str) -> bytes:
    import pymupdf

    story = pymupdf.Story(html=f"<body>{body}</body>", user_css=_PDF_CSS)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    page = pymupdf.paper_rect("a4")
    area = page + (50, 50, -50, -50)
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(area)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def question_paper_pdf(quiz: Quiz) -> bytes:
    parts = [f"<h1>{html.escape(quiz.title)}</h1>",
             f'<p class="meta">{html.escape(quiz.subject)} - {len(quiz.questions)} questions</p>',
             '<p class="meta">Name: ______________________ &nbsp; Date: ____________</p>']
    for i, q in enumerate(quiz.questions, 1):
        parts.append(f'<p class="q">{i}. {html.escape(q.question)}</p>')
        parts += [f'<p class="o">{html.escape(o.id)}. {html.escape(o.text)}</p>' for o in q.options]
    return _html_to_pdf("".join(parts))


def answer_key_pdf(quiz: Quiz) -> bytes:
    parts = [f"<h1>{html.escape(quiz.title)} - Answer key</h1>"]
    for i, q in enumerate(quiz.questions, 1):
        parts.append(f'<p class="q">{i}. {html.escape(q.question)}</p>')
        parts.append(f'<p class="a">Answer: {html.escape(q.correct_answer)}. {html.escape(q.correct_text())}</p>')
        if q.explanation:
            parts.append(f'<p class="e">{html.escape(q.explanation)}</p>')
    return _html_to_pdf("".join(parts))


# ---------------------------------------------------------------------- HTML
_CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:780px;margin:32px auto;padding:0 20px;color:#111;line-height:1.5}
h1{font-size:24px;margin-bottom:4px}.meta{color:#555;font-size:14px;margin-bottom:24px}
.q{margin:0 0 22px;page-break-inside:avoid}.q p{margin:0 0 6px;font-weight:bold}
ol.opts{list-style:none;padding-left:18px;margin:0}ol.opts li{margin:3px 0}
.ans{color:#1a6b2e;font-size:14px;margin-top:6px}.exp{color:#444;font-size:14px}
.name{margin:0 0 20px;font-size:14px}@media print{body{margin:0}}
"""


def _page(title: str, body: str) -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


def question_paper_html(quiz: Quiz) -> str:
    parts = [f"<h1>{html.escape(quiz.title)}</h1>",
             f'<div class="meta">{html.escape(quiz.subject)} &middot; {len(quiz.questions)} questions</div>',
             '<div class="name">Name: ____________________ &nbsp; Date: ____________</div>']
    for i, q in enumerate(quiz.questions, 1):
        opts = "".join(f"<li>{html.escape(o.id)}. {html.escape(o.text)}</li>" for o in q.options)
        parts.append(f'<div class="q"><p>{i}. {html.escape(q.question)}</p><ol class="opts">{opts}</ol></div>')
    return _page(f"{quiz.title} - Question paper", "".join(parts))


def answer_key_html(quiz: Quiz) -> str:
    parts = [f"<h1>{html.escape(quiz.title)} - Answer key</h1>",
             f'<div class="meta">{len(quiz.questions)} questions</div>']
    for i, q in enumerate(quiz.questions, 1):
        exp = f'<div class="exp">{html.escape(q.explanation)}</div>' if q.explanation else ""
        parts.append(
            f'<div class="q"><p>{i}. {html.escape(q.question)}</p>'
            f'<div class="ans">Answer: {html.escape(q.correct_answer)}. {html.escape(q.correct_text())}</div>{exp}</div>'
        )
    return _page(f"{quiz.title} - Answer key", "".join(parts))
