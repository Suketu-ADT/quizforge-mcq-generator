"""Central configuration: environment variables, defaults and paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.getenv("MCQ_DB_PATH", DATA_DIR / "quizzes.db"))

APP_NAME = "QuizForge"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
SUPPORTED_MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]

ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt")
MAX_UPLOAD_MB = 10
DIFFICULTIES = ["Easy", "Medium", "Hard"]
QUESTION_STYLES = ["Conceptual", "Factual", "Application-based", "Mixed"]
QUESTION_COUNT_PRESETS = [5, 10, 15, 20]
MAX_QUESTIONS = 50
MAX_GENERATION_RETRIES = 2  # bounded: first try + this many repairs per chunk


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings. Editable from the Settings page (kept in session + SQLite)."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.4
    timeout_s: int = 300
    max_input_chars: int = 60_000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
            temperature=_float("OLLAMA_TEMPERATURE", 0.4),
            timeout_s=_int("OLLAMA_TIMEOUT", 300),
            max_input_chars=_int("MAX_INPUT_CHARS", 60_000),
        )
