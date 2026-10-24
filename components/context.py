"""Shared Streamlit-side helpers: repository, settings, Ollama client, navigation."""
from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from config import DB_PATH, Settings
from database.db import Database
from database.repository import Repository
from services.ollama_client import OllamaClient, OllamaStatus

NAV_PAGES = [
    "Dashboard",
    "Generate Quiz",
    "My Question Bank",
    "Take a Quiz",
    "Analytics",
    "Settings",
    "About Project",
]


@st.cache_resource(show_spinner=False)
def get_repo() -> Repository:
    return Repository(Database(DB_PATH))


def get_settings() -> Settings:
    """Environment defaults, overridden by whatever the user saved on the Settings page."""
    if "settings" not in st.session_state:
        base = Settings.from_env()
        saved = get_repo().load_setting("settings", {}) or {}
        for key, value in saved.items():
            if hasattr(base, key):
                setattr(base, key, value)
        st.session_state["settings"] = base
    return st.session_state["settings"]


def save_settings(settings: Settings) -> None:
    st.session_state["settings"] = settings
    get_repo().set_setting("settings", asdict(settings))
    get_status.clear()


def get_client(timeout_s: int | None = None) -> OllamaClient:
    s = get_settings()
    return OllamaClient(s.base_url, s.model, s.temperature, timeout_s or s.timeout_s)


@st.cache_data(ttl=10, show_spinner=False)
def _status_cached(base_url: str, model: str) -> OllamaStatus:
    return OllamaClient(base_url, model).status()


def get_status(force: bool = False) -> OllamaStatus:
    s = get_settings()
    if force:
        _status_cached.clear()
    return _status_cached(s.base_url, s.model)


get_status.clear = _status_cached.clear  # type: ignore[attr-defined]


def navigate(page: str) -> None:
    """Request a page change; applied at the top of the next run (before the nav widget exists)."""
    st.session_state["_nav_request"] = page
    st.rerun()
