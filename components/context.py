"""Shared Streamlit-side helpers: repository, settings, Ollama client, navigation."""
from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from config import DB_PATH, Settings
from database.db import Database
from database.repository import Repository
from services.ollama_client import OllamaClient, OllamaStatus, same_model

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


def tier_of(model: str) -> str | None:
    """"Fast" / "Quality" for the two switchable models, None for any other tag."""
    from config import TIER_TAGS

    return next((tier for tier, tag in TIER_TAGS.items() if same_model(tag, model)), None)


def is_installed(model: str, status: OllamaStatus) -> bool:
    return any(same_model(model, name) for name in status.installed_models)


def switch_model(model: str) -> str | None:
    """Make *model* the active model and remember it.

    Returns an error message (and changes nothing) when Ollama is reachable but the model is not installed.
    When Ollama is down the switch is accepted, because installation cannot be checked.
    """
    status = get_status(force=True)
    if status.connected and not is_installed(model, status):
        return (
            "That model is not installed on this computer yet. Open Settings -> Model to see the exact "
            "`ollama pull` command."
        )
    settings = get_settings()
    settings.model = model
    save_settings(settings)
    st.session_state.pop("_warmed", None)  # preload the newly chosen model on the next Generate visit
    return None


def navigate(page: str) -> None:
    """Request a page change; applied at the top of the next run (before the nav widget exists)."""
    st.session_state["_nav_request"] = page
    st.rerun()
