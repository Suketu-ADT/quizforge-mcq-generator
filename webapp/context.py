"""Shared server state: the repository, the user's settings, the Ollama client and a short status cache."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from typing import Callable

from config import ALL_TIERS, AUTO_MODEL_ORDER, DEFAULT_MODEL, MODEL_FILES_DIR, Settings
from database.repository import Repository
from services.ollama_client import OllamaClient, OllamaError, OllamaStatus, same_model
from webapp.jobs import JobManager

ClientFactory = Callable[[Settings], OllamaClient]
STATUS_TTL_S = 5.0


def default_client_factory(s: Settings) -> OllamaClient:
    return OllamaClient(s.base_url, s.model, s.temperature, s.timeout_s)


class AppContext:
    def __init__(self, repo: Repository, client_factory: ClientFactory | None = None):
        self.repo = repo
        self.client_factory = client_factory or default_client_factory
        self.jobs = JobManager()
        self._lock = threading.Lock()
        self._status: tuple[float, str, OllamaStatus] | None = None

    # ------------------------------------------------------------------ settings
    def settings(self) -> Settings:
        """Environment defaults, overridden by what was saved on the Settings page.

        When no model was ever chosen (nothing saved and no OLLAMA_MODEL), the first installed model of
        AUTO_MODEL_ORDER is used, so a team member who imported the fine-tuned models gets them by default."""
        s = Settings.from_env()
        saved = self.repo.load_setting("settings", {}) or {}
        for key, value in saved.items():
            if hasattr(s, key):
                setattr(s, key, value)
        if "model" not in saved and not os.getenv("OLLAMA_MODEL"):
            s.model = self._auto_model(s) or DEFAULT_MODEL
        return s

    def save_settings(self, s: Settings) -> None:
        self.repo.set_setting("settings", asdict(s))
        self.invalidate_status()

    def _auto_model(self, s: Settings) -> str | None:
        st = self._raw_status(s)
        if not st.connected:
            return None
        return next((tag for tag in AUTO_MODEL_ORDER if any(same_model(tag, m) for m in st.installed_models)), None)

    # -------------------------------------------------------------------- client
    def client(self, model: str | None = None) -> OllamaClient:
        s = self.settings()
        if model:
            s.model = model
        return self.client_factory(s)

    # -------------------------------------------------------------------- status
    def invalidate_status(self) -> None:
        with self._lock:
            self._status = None

    def _raw_status(self, s: Settings, force: bool = False) -> OllamaStatus:
        key = s.base_url  # the installed list depends on the server only; status() checks the model itself
        with self._lock:
            cached = self._status
        if not force and cached and cached[1] == key and time.monotonic() - cached[0] < STATUS_TTL_S:
            return cached[2]
        try:
            st = self.client_factory(s).status()
        except OllamaError as exc:  # status() never raises for the real client; fakes might
            st = OllamaStatus(connected=False, error=str(exc))
        with self._lock:
            self._status = (time.monotonic(), key, st)
        return st

    def status(self, force: bool = False) -> dict:
        s = self.settings()
        st = self._raw_status(s, force)
        installed = st.connected and any(same_model(s.model, m) for m in st.installed_models)
        error = st.error
        if st.connected and not installed:
            error = (f"The selected model is not installed. Open Settings to install it or pick another one "
                     f"(the app never downloads models by itself).")
        return {
            "connected": st.connected,
            "ready": installed,
            "model_installed": installed,
            "version": st.version,
            "error": "" if installed else error,
            "model": s.model,
            "model_name": model_name(s.model),
            "installed_models": st.installed_models,
            "base_url": s.base_url,
            "local": is_local_url(s.base_url),
            "models": self.catalogue(s.model, st),
        }

    def catalogue(self, active: str, st: OllamaStatus) -> list[dict]:
        out = []
        for t in ALL_TIERS:
            installed = any(same_model(t["tag"], m) for m in st.installed_models) if st.connected else None
            item = {**t, "active": same_model(t["tag"], active), "installed": installed,
                    "finetuned": "base" in t}
            if item["finetuned"]:
                item["gguf_present"] = (MODEL_FILES_DIR / t["gguf"]).exists()
            out.append(item)
        return out


def model_name(tag: str) -> str:
    """The friendly name the UI shows outside the Settings page."""
    return next((t["tier"] for t in ALL_TIERS if same_model(t["tag"], tag)), "Custom model")


def is_local_url(url: str) -> bool:
    return url.startswith(("http://localhost", "http://127.0.0.1", "http://[::1]", "https://localhost"))
