"""LIVE integration test - needs Ollama running and a Qwen 2.5 model installed.

Skipped automatically otherwise. Run explicitly with:
    python -m pytest tests/test_integration_ollama.py -v -m integration
Set OLLAMA_MODEL to choose the tag (default: first installed qwen2.5 tag).
"""
import os

import pytest

from config import Settings
from models.schemas import GenerationRequest
from services.ollama_client import OllamaClient
from services.question_generator import QuestionGenerator

pytestmark = pytest.mark.integration

TEXT = (
    "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
    "It takes place in the chloroplasts, which contain the pigment chlorophyll. "
    "During the light-dependent reactions, water is split and oxygen is released as a by-product. "
    "In the Calvin cycle, carbon dioxide is fixed into glucose using ATP and NADPH.\n\n"
    "Cellular respiration is the opposite process: glucose is broken down in the mitochondria to release ATP. "
    "It consumes oxygen and releases carbon dioxide and water."
)


def _client() -> OllamaClient:
    s = Settings.from_env()
    c = OllamaClient(s.base_url, s.model, s.temperature, timeout_s=600)
    status = c.status()
    if not status.connected:
        pytest.skip("Ollama is not running")
    if not status.model_installed:
        qwen = [m for m in status.installed_models if m.startswith("qwen2.5")]
        if not qwen or os.getenv("OLLAMA_MODEL"):
            pytest.skip(f"model {c.model} not installed")
        c.model = qwen[0]
    return c


def test_live_generation_with_qwen():
    gen = QuestionGenerator(_client())
    res = gen.generate(TEXT, GenerationRequest(count=2, difficulty="Easy", topic="Biology"))
    assert len(res.questions) >= 1, f"no valid questions; rejected={res.rejected}"
    for q in res.questions:
        assert len(q.options) == 4 and q.correct_answer in {o.id for o in q.options}
        assert q.source_excerpt
