"""Thin client for the local Ollama HTTP API (http://localhost:11434).

Only the configured local endpoint is ever contacted. No cloud LLM is used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import requests


KEEP_ALIVE = "30m"  # keep the model in memory between requests so later calls start fast


class OllamaError(Exception):
    """Base class for every Ollama-related failure (message is user-friendly)."""


class OllamaUnavailableError(OllamaError):
    pass


class ModelNotFoundError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


@dataclass
class OllamaStatus:
    connected: bool = False
    model_installed: bool = False
    installed_models: list[str] = field(default_factory=list)
    version: str = ""
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.connected and self.model_installed


def _same_model(wanted: str, installed: str) -> bool:
    def norm(tag: str) -> str:
        tag = tag.strip().lower()
        return tag if ":" in tag else f"{tag}:latest"

    return norm(wanted) == norm(installed)


same_model = _same_model  # public alias: "qwen2.5" and "qwen2.5:latest" are the same model


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        temperature: float = 0.4,
        timeout_s: int = 300,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._http = session or requests.Session()

    # ------------------------------------------------------------------ status
    def list_models(self) -> list[str]:
        try:
            resp = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(self._unavailable_msg()) from exc
        except requests.Timeout as exc:
            raise OllamaUnavailableError(
                f"Ollama at {self.base_url} did not answer within 5 seconds."
            ) from exc
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Unexpected reply from Ollama: {exc}") from exc

    def has_model(self, installed: list[str] | None = None) -> bool:
        installed = self.list_models() if installed is None else installed
        return any(_same_model(self.model, name) for name in installed)

    def status(self) -> OllamaStatus:
        """Never raises: used by the UI to show Connected / Disconnected."""
        try:
            installed = self.list_models()
        except OllamaError as exc:
            return OllamaStatus(connected=False, error=str(exc))
        st = OllamaStatus(
            connected=True,
            installed_models=installed,
            model_installed=self.has_model(installed),
        )
        try:
            st.version = self._http.get(f"{self.base_url}/api/version", timeout=3).json().get("version", "")
        except (requests.RequestException, ValueError):
            pass
        if not st.model_installed:
            st.error = self._missing_msg()
        return st

    def warm_up(self) -> bool:
        """Ask Ollama to load the model into memory now (no output is generated). Never raises."""
        try:
            resp = self._http.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": KEEP_ALIVE},
                timeout=max(self.timeout_s, 60),
            )
            return resp.status_code < 400
        except requests.RequestException:
            return False

    # -------------------------------------------------------------------- chat
    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float | None = None,
        num_ctx: int = 8192,
        on_progress: Callable[[int], None] | None = None,
    ) -> str:
        """Call /api/chat and return the assistant message text.

        When *schema* is given it is passed as Ollama's structured-output `format`.
        If the server rejects the schema we fall back to plain JSON mode once.
        When *on_progress* is given the reply is streamed and the callback receives
        the number of tokens produced so far (the timeout then applies between chunks).
        """
        fmt: dict | str = schema if schema else "json"
        try:
            return self._chat_once(messages, fmt, temperature, num_ctx, on_progress)
        except _BadRequest as exc:
            if isinstance(fmt, dict):
                try:
                    return self._chat_once(messages, "json", temperature, num_ctx, on_progress)
                except _BadRequest as exc2:
                    raise OllamaError(f"Ollama rejected the request: {exc2}") from exc2
            raise OllamaError(f"Ollama rejected the request: {exc}") from exc

    def _chat_once(self, messages, fmt, temperature, num_ctx, on_progress=None) -> str:
        stream = on_progress is not None
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "format": fmt,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": num_ctx,
            },
        }
        try:
            if stream:
                resp = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s, stream=True)
            else:
                resp = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(self._unavailable_msg()) from exc
        except requests.Timeout as exc:
            raise OllamaTimeoutError(
                f"The model did not respond within {self.timeout_s} s. Try a smaller model, "
                "fewer questions per run, or raise the request timeout in Settings."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Request to Ollama failed: {exc}") from exc

        if resp.status_code == 404:
            raise ModelNotFoundError(self._missing_msg())
        if resp.status_code == 400:
            raise _BadRequest(_error_text(resp))
        if resp.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {_error_text(resp)}")
        if stream:
            return self._read_stream(resp, on_progress)
        try:
            data = resp.json()
            return data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned a response in an unexpected format.") from exc

    def _read_stream(self, resp, on_progress: Callable[[int], None]) -> str:
        parts: list[str] = []
        count = 0
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError as exc:
                    raise OllamaError("Ollama returned a malformed streaming response.") from exc
                if chunk.get("error"):
                    raise OllamaError(f"Ollama error: {str(chunk['error'])[:300]}")
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    parts.append(piece)
                    count += 1
                    on_progress(count)
                if chunk.get("done"):
                    break
        except requests.Timeout as exc:
            raise OllamaTimeoutError(
                f"The model stopped responding for {self.timeout_s} s. Try a smaller model or raise the timeout in Settings."
            ) from exc
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(self._unavailable_msg()) from exc
        finally:
            close = getattr(resp, "close", None)
            if close:
                close()
        return "".join(parts)

    # ---------------------------------------------------------------- messages
    def _unavailable_msg(self) -> str:
        return (
            f"Cannot reach Ollama at {self.base_url}. Start it (open the Ollama app or run "
            "`ollama serve`) and try again."
        )

    def _missing_msg(self) -> str:
        return (
            "The selected AI model is not installed on this computer. Open Settings to see the exact "
            "`ollama pull` command or to choose a model that is already installed "
            "(the app never downloads models automatically)."
        )


class _BadRequest(Exception):
    pass


def _error_text(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("error", resp.text))[:300]
    except ValueError:
        return resp.text[:300]
