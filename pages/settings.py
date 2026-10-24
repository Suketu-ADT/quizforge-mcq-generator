"""Settings: Ollama endpoint/model, generation parameters, runtime status, data management."""
from __future__ import annotations

import streamlit as st

from components.context import get_repo, get_settings, get_status, is_installed, save_settings, switch_model
from components.metrics import page_header, status_pill
from components.styles import esc
from config import DB_PATH, MODEL_TIERS, SUPPORTED_MODELS
from services.ollama_client import OllamaClient, same_model

CUSTOM = "Other (type a tag below)"


def _switch(tier: str, tag: str) -> None:
    error = switch_model(tag)
    st.session_state["model_msg"] = (
        ("error", error) if error else ("ok", f"Switched to the {tier} model. It is used from the next generation.")
    )


def _model_cards(settings, status) -> None:
    st.markdown("#### Model")
    st.caption("Switch whenever you like - no restart needed. Both are Qwen 2.5; the larger one is slower but more accurate.")
    if msg := st.session_state.pop("model_msg", None):
        (st.success if msg[0] == "ok" else st.error)(msg[1])
    cols = st.columns(len(MODEL_TIERS))
    for col, t in zip(cols, MODEL_TIERS):
        active = same_model(t["tag"], settings.model)
        installed = is_installed(t["tag"], status) if status.connected else None
        if active:
            pill = '<span class="qf-pill on"><i></i>Active</span>'
        elif installed is False:
            pill = '<span class="qf-pill warn"><i></i>Not installed</span>'
        elif installed:
            pill = '<span class="qf-badge qf-topic">Installed</span>'
        else:
            pill = ""
        with col, st.container(border=True):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<b style="font-size:1.1rem">{esc(t["tier"])}</b>{pill}</div>'
                f'<div class="qf-muted" style="margin:.2rem 0 .5rem"><code>{esc(t["tag"])}</code> · download {esc(t["size"])}</div>'
                f'<div style="font-size:.9rem">{esc(t["blurb"])}</div>',
                unsafe_allow_html=True,
            )
            if installed is False:
                st.code(f"ollama pull {t['tag']}", language="bash")
            st.button(
                "In use" if active else "Use this model", key=f"use_{t['tier']}",
                type="secondary" if active else "primary", disabled=active or installed is False,
                on_click=_switch, args=(t["tier"], t["tag"]), use_container_width=True,
            )
    if not any(same_model(t["tag"], settings.model) for t in MODEL_TIERS):
        st.info(f"A custom model is active ({settings.model}). Pick Fast or Quality above to go back to a standard one.")


def _status_card(status, url: str, model: str) -> None:
    lines = [f"Endpoint: <b>{esc(url)}</b>", f"Configured model: <b>{esc(model)}</b>"]
    if status.version:
        lines.append(f"Ollama version: {esc(status.version)}")
    if status.connected:
        lines.append(f"Installed models: {esc(', '.join(status.installed_models) or 'none')}")
    st.markdown(
        f'<div class="qf-card">{status_pill(status)}<p class="qf-muted" style="margin-top:.7rem">{"<br>".join(lines)}</p></div>',
        unsafe_allow_html=True,
    )
    if status.error:
        st.warning(status.error)


def render() -> None:
    page_header("Settings", "Everything runs against your local Ollama server. No API keys are needed.")
    settings = get_settings()
    status = get_status()

    st.markdown("#### Runtime status")
    _status_card(status, settings.base_url, settings.model)
    if st.button("Refresh status"):
        get_status(force=True)
        st.rerun()

    _model_cards(settings, status)

    st.markdown("#### Connection and generation settings")
    installed_qwen = [m for m in status.installed_models if m.lower().startswith("qwen2.5")]
    options = list(dict.fromkeys([settings.model, *installed_qwen, *SUPPORTED_MODELS]))
    with st.form("settings_form"):
        base_url = st.text_input("Ollama base URL", settings.base_url)
        picked = st.selectbox("Model tag (advanced)", [*options, CUSTOM], index=0,
                              help="Normally use the Fast / Quality cards above. This lists every installed tag and lets you type another.")
        custom = st.text_input("Custom model tag (only used with 'Other')", "", placeholder="e.g. qwen2.5:1.5b")
        c1, c2, c3 = st.columns(3)
        temperature = c1.slider("Temperature", 0.0, 1.2, float(settings.temperature), 0.05,
                                help="Lower = more deterministic questions; higher = more varied.")
        timeout = c2.number_input("Request timeout (seconds)", 30, 3600, int(settings.timeout_s))
        max_chars = c3.number_input("Maximum input size (characters)", 2000, 500_000, int(settings.max_input_chars), step=1000)
        b1, b2, _ = st.columns([1, 1, 3])
        save = b1.form_submit_button("Save settings", type="primary")
        test = b2.form_submit_button("Test connection")
    model = (custom.strip() if picked == CUSTOM else picked).strip()
    url = base_url.strip().rstrip("/")
    if test:
        with st.spinner("Contacting Ollama..."):
            result = OllamaClient(url, model).status()
        if result.ready:
            st.success(f"Connected to {url}. Model '{model}' is installed and ready.")
        elif result.connected:
            st.warning(result.error)
        else:
            st.error(result.error)
    if save:
        if not url.startswith(("http://", "https://")):
            st.error("The base URL must start with http:// or https://")
        elif not model:
            st.error("Please choose or type a model tag.")
        else:
            settings.base_url, settings.model = url, model
            settings.temperature, settings.timeout_s, settings.max_input_chars = float(temperature), int(timeout), int(max_chars)
            save_settings(settings)
            st.success("Settings saved.")
            if not url.startswith(("http://localhost", "http://127.0.0.1", "http://[::1]")):
                st.warning("This endpoint is not on this computer - your study material will be sent there.")
            st.rerun()

    st.markdown("#### Install a Qwen 2.5 model")
    st.markdown(
        "The app never downloads models by itself. Pull the tag you want once (this step needs internet), "
        "then everything runs locally. Pick a size that fits your RAM/VRAM - speed depends on your hardware."
    )
    st.code("ollama pull qwen2.5:3b   # Fast (default)\nollama pull qwen2.5:7b   # Quality\nollama pull qwen2.5:14b  # optional: best quality, needs much more memory\nollama serve             # start the server if it is not already running", language="bash")

    st.markdown("#### Data on this computer")
    st.caption(f"Database file: {DB_PATH}")
    repo = get_repo()
    confirm = st.checkbox("I understand this permanently deletes all quiz attempts (saved quizzes are kept)")
    if st.button("Clear quiz history", disabled=not confirm):
        n = repo.clear_attempts()
        repo.log_activity("history", f"Cleared {n} quiz attempt(s)")
        st.success(f"Deleted {n} attempt(s).")
