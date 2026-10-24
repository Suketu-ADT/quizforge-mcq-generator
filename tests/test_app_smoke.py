"""Headless UI smoke tests using Streamlit's AppTest (no browser, no Ollama needed)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from components.context import NAV_PAGES


APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MCQ_DB_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")  # nothing listens here -> "Disconnected"
    import importlib
    import config
    importlib.reload(config)
    import components.context as ctx
    importlib.reload(ctx)
    ctx.get_repo.clear()
    ctx._status_cached.clear()
    yield


def new_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    return at


def test_app_starts_without_ollama_and_shows_disconnected():
    at = new_app()
    assert not at.exception
    assert any("Disconnected" in m.value for m in at.markdown)


@pytest.mark.parametrize("page", NAV_PAGES)
def test_every_page_renders(page):
    at = new_app()
    at.session_state["_nav_request"] = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]
