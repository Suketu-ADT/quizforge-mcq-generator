"""Default model (3b) and the Fast / Quality switch."""
import importlib

import pytest
from streamlit.testing.v1 import AppTest

from config import DEFAULT_MODEL, MODEL_TIERS, TIER_TAGS, Settings
from services.ollama_client import OllamaClient, OllamaStatus
from tests.test_app_flows import APP, env  # noqa: F401  (env is a fixture)


# ------------------------------------------------------------------ defaults
def test_default_model_is_the_3b_one(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert DEFAULT_MODEL == "qwen2.5:3b"
    assert Settings.from_env().model == "qwen2.5:3b"
    assert Settings().model == "qwen2.5:3b"
    assert OllamaClient().model == "qwen2.5:3b"


def test_environment_variable_still_overrides_the_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    assert Settings.from_env().model == "qwen2.5:7b"


def test_the_two_switchable_models():
    assert [t["tier"] for t in MODEL_TIERS] == ["Fast", "Quality"]
    assert TIER_TAGS == {"Fast": "qwen2.5:3b", "Quality": "qwen2.5:7b"}


def test_tier_of_maps_tags_and_ignores_others(env):
    assert env.tier_of("qwen2.5:3b") == "Fast"
    assert env.tier_of("qwen2.5:7b") == "Quality"
    assert env.tier_of("qwen2.5:14b") is None
    assert env.tier_of("llama3:latest") is None


# -------------------------------------------------------------- Settings page
def both_installed(env, monkeypatch, models=("qwen2.5:3b", "qwen2.5:7b")):
    monkeypatch.setattr(
        env, "_status_cached",
        lambda url, model: OllamaStatus(connected=True, model_installed=model in models, installed_models=list(models)),
    )
    env._status_cached.clear = lambda: None


def settings_page(env) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["_nav_request"] = "Settings"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_settings_starts_on_fast_and_can_switch_to_quality_and_back(env, monkeypatch):
    both_installed(env, monkeypatch)
    at = settings_page(env)
    assert at.session_state["settings"].model == "qwen2.5:3b"  # default
    assert next(b for b in at.button if b.key == "use_Fast").disabled  # already in use
    next(b for b in at.button if b.key == "use_Quality").click()
    at.run()
    assert not at.exception
    assert at.session_state["settings"].model == "qwen2.5:7b"
    assert env.get_repo().load_setting("settings")["model"] == "qwen2.5:7b"  # remembered
    assert any("Switched to the Quality model" in s.value for s in at.success)
    next(b for b in at.button if b.key == "use_Fast").click()
    at.run()
    assert at.session_state["settings"].model == "qwen2.5:3b"


def test_a_model_that_is_not_installed_cannot_be_selected(env, monkeypatch):
    both_installed(env, monkeypatch, models=("qwen2.5:3b",))
    at = settings_page(env)
    quality = next(b for b in at.button if b.key == "use_Quality")
    assert quality.disabled
    assert any("ollama pull qwen2.5:7b" in c.value for c in at.code)  # shows how to install it
    assert at.session_state["settings"].model == "qwen2.5:3b"


def test_switch_model_refuses_uninstalled_but_accepts_when_ollama_is_down(env, monkeypatch):
    both_installed(env, monkeypatch, models=("qwen2.5:3b",))
    at = settings_page(env)
    # switch_model needs a Streamlit runtime, so call it through a tiny script
    script = (
        "import streamlit as st\n"
        "from components.context import switch_model\n"
        "st.session_state['refused'] = switch_model('qwen2.5:7b')\n"
        "st.session_state['model_after'] = st.session_state['settings'].model\n"
    )
    probe = AppTest.from_string(script, default_timeout=30)
    probe.session_state["settings"] = at.session_state["settings"]
    probe.run()
    assert "not installed" in probe.session_state["refused"]
    assert probe.session_state["model_after"] == "qwen2.5:3b"

    monkeypatch.setattr(env, "_status_cached", lambda url, model: OllamaStatus(connected=False, error="down"))
    env._status_cached.clear = lambda: None
    probe2 = AppTest.from_string(script, default_timeout=30)
    probe2.session_state["settings"] = at.session_state["settings"]
    probe2.run()
    assert probe2.session_state["refused"] is None and probe2.session_state["model_after"] == "qwen2.5:7b"


def test_a_custom_model_is_reported_and_can_be_left(env, monkeypatch):
    both_installed(env, monkeypatch, models=("qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"))
    env.get_repo().set_setting("settings", {"model": "qwen2.5:14b"})
    at = settings_page(env)
    assert any("custom model is active" in i.value for i in at.info)
    assert not next(b for b in at.button if b.key == "use_Fast").disabled


# ------------------------------------------------------------- Generate page
def test_generate_page_shows_tier_names_not_model_tags(env, monkeypatch):
    both_installed(env, monkeypatch)
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["_nav_request"] = "Generate Quiz"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown) + " ".join(b.label for b in at.button)
    assert "qwen" not in text.lower()
    assert at.session_state["cfg_tier"] == "Fast"
