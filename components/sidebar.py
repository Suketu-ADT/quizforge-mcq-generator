"""Sidebar: brand, navigation, live model status."""
from __future__ import annotations

import streamlit as st

from components.context import NAV_PAGES, get_settings, get_status
from components.metrics import status_pill
from components.styles import esc

ICONS = {
    "Dashboard": "▦",
    "Generate Quiz": "✦",
    "My Question Bank": "☰",
    "Take a Quiz": "▶",
    "Analytics": "◔",
    "Settings": "⚙",
    "About Project": "ⓘ",
}


def render_sidebar() -> str:
    if "_nav_request" in st.session_state:
        st.session_state["nav"] = st.session_state.pop("_nav_request")
    st.session_state.setdefault("nav", NAV_PAGES[0])
    with st.sidebar:
        st.markdown(
            '<div class="qf-brand"><div class="qf-logo">Q</div><div><b>QuizForge</b>'
            "<small>Local MCQ generator</small></div></div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            NAV_PAGES,
            key="nav",
            format_func=lambda p: f"{ICONS[p]}  {p}",
            label_visibility="collapsed",
        )
        st.divider()
        status = get_status()
        st.markdown(
            f'{status_pill(status)}<div class="qf-muted" style="margin-top:.5rem">Runs locally on your computer</div>',
            unsafe_allow_html=True,
        )
    return page
