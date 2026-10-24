import pytest
import requests

from services.ollama_client import (
    ModelNotFoundError,
    OllamaClient,
    OllamaError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from tests.helpers import FakeResponse, FakeSession, chat_reply, stream_reply

TAGS = FakeResponse(200, {"models": [{"name": "qwen2.5:3b"}, {"name": "llama3:latest"}]})


def client(session, model="qwen2.5:3b"):
    return OllamaClient("http://localhost:11434/", model, session=session, timeout_s=5)


def test_ollama_not_running_is_reported_not_raised_by_status():
    st = client(FakeSession(get=requests.ConnectionError("refused"))).status()
    assert not st.connected and not st.ready and "Cannot reach Ollama" in st.error


def test_list_models_raises_unavailable():
    with pytest.raises(OllamaUnavailableError):
        client(FakeSession(get=requests.ConnectionError("refused"))).list_models()


def test_status_connected_with_model_installed():
    c = client(FakeSession(get=lambda url: TAGS if url.endswith("tags") else FakeResponse(200, {"version": "0.9"})))
    st = c.status()
    assert st.connected and st.model_installed and st.ready and st.version == "0.9"
    assert "qwen2.5:3b" in st.installed_models


def test_status_model_missing_explains_how_to_install():
    st = client(FakeSession(get=TAGS), model="qwen2.5:7b").status()
    assert st.connected and not st.model_installed
    assert "ollama pull" in st.error and "qwen2.5:7b" not in st.error


def test_model_tag_matching_handles_latest():
    assert client(FakeSession(get=FakeResponse(200, {"models": [{"name": "foo:latest"}]})), "foo").has_model()


def test_chat_returns_content_and_sends_schema():
    s = FakeSession(post=chat_reply('{"questions": []}'))
    schema = {"type": "object"}
    assert client(s).chat([{"role": "user", "content": "hi"}], schema=schema) == '{"questions": []}'
    sent = s.posts[0]
    assert sent["stream"] is False and sent["format"] == schema and sent["model"] == "qwen2.5:3b"


def test_chat_timeout_and_connection_errors():
    with pytest.raises(OllamaTimeoutError, match="did not respond"):
        client(FakeSession(post=requests.Timeout("slow"))).chat([])
    with pytest.raises(OllamaUnavailableError):
        client(FakeSession(post=requests.ConnectionError("down"))).chat([])


def test_chat_model_not_found_404():
    s = FakeSession(post=FakeResponse(404, {"error": "model not found"}))
    with pytest.raises(ModelNotFoundError, match="ollama pull"):
        client(s, "qwen2.5:14b").chat([])


def test_chat_falls_back_to_json_mode_when_schema_rejected():
    calls = []

    def handler(url):
        calls.append(1)
        return FakeResponse(400, {"error": "bad format"}) if len(calls) == 1 else chat_reply("{}")

    s = FakeSession(post=handler)
    assert client(s).chat([], schema={"type": "object"}) == "{}"
    assert s.posts[1]["format"] == "json"


def test_chat_http_500_and_bad_payload():
    with pytest.raises(OllamaError, match="HTTP 500"):
        client(FakeSession(post=FakeResponse(500, {"error": "boom"}))).chat([])
    with pytest.raises(OllamaError, match="unexpected format"):
        client(FakeSession(post=FakeResponse(200, {"nope": 1}))).chat([])


def test_streaming_reports_progress_and_joins_text():
    s = FakeSession(post=stream_reply(['{"questions"', ": ", "[]}"]))
    seen = []
    out = client(s).chat([], schema={"type": "object"}, on_progress=seen.append)
    assert out == '{"questions": []}' and seen == [1, 2, 3]
    assert s.posts[0]["stream"] is True


def test_streaming_error_chunk_raises():
    resp = FakeResponse(200, {})
    resp._lines = ['{"error": "model crashed"}']
    with pytest.raises(OllamaError, match="model crashed"):
        client(FakeSession(post=resp)).chat([], on_progress=lambda n: None)
