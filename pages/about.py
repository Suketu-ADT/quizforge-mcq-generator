"""About Project: material for the jury presentation."""
from __future__ import annotations

import streamlit as st

from components.metrics import page_header


def render() -> None:
    page_header("About this project", "PS 97 - Automatic MCQ Quiz Generator using a local Qwen 2.5 model")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Problem statement")
        st.write(
            "Teachers and learners spend hours writing practice questions by hand. The task is to read learning "
            "material and automatically produce good multiple-choice questions - with one correct answer, plausible "
            "distractors and explanations - that can be reviewed, practised and exported."
        )
        st.markdown("#### Objectives")
        st.markdown(
            "- Generate MCQs from pasted text or PDF / DOCX / TXT files.\n"
            "- Use only a **local** LLM (Qwen 2.5 via Ollama) - no cloud AI.\n"
            "- Validate every model output before it is shown.\n"
            "- Let a human review, edit, regenerate and delete questions.\n"
            "- Provide an interactive quiz, scoring, analytics and exports."
        )
    with c2:
        st.markdown("#### Technology stack")
        st.markdown(
            "- **Python 3.11+**, **Streamlit** (UI + custom CSS)\n"
            "- **Ollama** + **Qwen 2.5** (3b / 7b / 14b) - local inference\n"
            "- **PyMuPDF**, **python-docx** - text extraction\n"
            "- **Pydantic** - schema validation\n"
            "- **SQLite** - local persistence\n"
            "- **pandas**, **Plotly** - analytics and charts\n"
            "- **pytest** - automated tests (Ollama mocked)"
        )
        st.markdown("#### Privacy design")
        st.markdown(
            "- Study material is sent **only** to the Ollama endpoint you configure (default `localhost:11434`).\n"
            "- No API keys, telemetry or hidden network calls.\n"
            "- Data is stored in a SQLite file on this computer.\n"
            "- Honest note: installing dependencies and *pulling the model* needs internet once; "
            "afterwards generation runs locally."
        )

    st.markdown("#### System architecture")
    st.graphviz_chart(
        """digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#18223d", color="#818cf8", fontcolor="#e6e9f2", fontname="Helvetica"]; edge [color="#818cf8"];
        User -> "Streamlit UI" -> "Document Processor" -> "Question Generator" -> "Ollama / Qwen 2.5" -> "Question Validator" -> SQLite -> "Quiz Engine" -> "Results & Analytics"; }"""
    )

    st.markdown("#### Workflow")
    st.markdown(
        "1. **Input** - paste text or upload a file; the extracted text is shown for review.\n"
        "2. **Chunking** - long text is split at paragraph/heading boundaries (no silent truncation).\n"
        "3. **Generation** - small batches are sent to Qwen with a strict system prompt and a JSON schema.\n"
        "4. **Validation** - parse JSON, check options, answer key, duplicates and that the cited source sentence really exists in the text.\n"
        "5. **Bounded repair** - invalid output triggers at most two repair attempts per batch.\n"
        "6. **Human review** - edit, regenerate or delete; edits are re-validated.\n"
        "7. **Quiz and analytics** - scoring uses the stored answer key, never a new LLM call."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Key features")
        st.markdown(
            "- Source-grounded questions with verified excerpts\n"
            "- Editable review cards, single-question regeneration\n"
            "- Timer, instant-feedback mode, question palette\n"
            "- Topic / difficulty analytics from real attempts\n"
            "- Question paper and answer key exported separately (HTML, CSV, JSON)"
        )
    with c2:
        st.markdown("#### Limitations")
        st.markdown(
            "- The LLM can still write a wrong answer key; automatic checks cannot guarantee correctness - **human review is part of the design**.\n"
            "- Small models (3b) are weaker than 7b/14b and may need more retries.\n"
            "- Scanned PDFs need OCR, which is not included.\n"
            "- Speed depends on your CPU/GPU and RAM."
        )
        st.markdown("#### Future scope")
        st.markdown("- OCR for scans · more question types · spaced repetition · multi-user mode · local embeddings for retrieval.")
