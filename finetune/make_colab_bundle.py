r"""Zip the code the Colab notebook needs (option B, when cloning from GitHub is not possible).

    .venv\Scripts\python.exe -m finetune.make_colab_bundle      ->  finetune\quizforge_colab.zip

Only source code goes in: config.py, models/, services/, finetune/ - no database, no .env, no data or model files.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["config.py", "models", "services", "finetune"]
SKIP_DIRS = {"__pycache__", "data", "exports"}
SKIP_SUFFIXES = {".pyc", ".gguf", ".zip", ".jsonl", ".bat"}


def bundle_files() -> list[Path]:
    files = []
    for item in INCLUDE:
        path = ROOT / item
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for p in candidates:
            rel = p.relative_to(ROOT)
            if SKIP_DIRS & set(rel.parts[:-1]) or p.suffix in SKIP_SUFFIXES:
                continue
            files.append(p)
    return files


def main() -> int:
    out = ROOT / "finetune" / "quizforge_colab.zip"
    files = bundle_files()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"Wrote {out} ({len(files)} files, {out.stat().st_size / 1024:.0f} KB) - upload it to Colab and set USE_ZIP = True.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
