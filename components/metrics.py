"""Small reusable display components."""
from __future__ import annotations

import streamlit as st

from components.styles import esc
from services.ollama_client import OllamaStatus


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="qf-metric"><div class="l">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'
    )


def metric_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)


def status_pill(status: OllamaStatus) -> str:
    if not status.connected:
        return '<span class="qf-pill off"><i></i>Disconnected</span>'
    if not status.model_installed:
        return '<span class="qf-pill warn"><i></i>Connected - model missing</span>'
    return '<span class="qf-pill on"><i></i>Connected</span>'


def fmt_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}%"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<h1 style="margin-bottom:.15rem">{esc(title)}</h1><p class="qf-muted" style="margin-top:0">{esc(subtitle)}</p>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="qf-card" style="text-align:center;padding:2rem"><h4>{esc(title)}</h4><p class="qf-muted">{esc(body)}</p></div>', unsafe_allow_html=True)
