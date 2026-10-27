r"""Register a fine-tuned GGUF file with the local Ollama, using the base model's chat template and stop tokens.

    .venv\Scripts\python.exe -m finetune.import_to_ollama --gguf finetune\exports\quizforge-3b.Q4_K_M.gguf --name quizforge-3b --base qwen2.5:3b

The fine-tuned model was trained with Qwen 2.5's chat format, so it must be served with exactly the same template;
it is read from the installed base model (Ollama API /api/show) instead of being retyped. Nothing is downloaded.
Afterwards the model can be picked on the Settings page ("Other" -> custom tag) and evaluated with run_evaluation.bat.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

DEFAULT_OLLAMA = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
# Qwen 2.5 chat-format end tokens; added when the base model declares no stop parameters of its own, so a fine-tuned
# export can never run past the end of its JSON reply
QWEN_STOPS = ['stop "<|im_end|>"', 'stop "<|endoftext|>"']


def base_template(base_url: str, base: str) -> tuple[str, list[str]]:
    """(template, parameter lines) of an installed model."""
    r = requests.post(f"{base_url}/api/show", json={"model": base}, timeout=30)
    if r.status_code == 404:
        raise SystemExit(f"Base model '{base}' is not installed - it is needed for its chat template (ollama pull {base}).")
    r.raise_for_status()
    data = r.json()
    params = [line.strip() for line in (data.get("parameters") or "").splitlines() if line.strip()]
    return data.get("template") or "", params


def build_modelfile(gguf_name: str, template: str, parameters: list[str], temperature: float | None = None) -> str:
    if not template:
        raise ValueError("The base model has no chat template; refusing to create a model without one.")
    if '"""' in template:
        raise ValueError("The chat template contains triple quotes and cannot be embedded in a Modelfile.")
    lines = [f"FROM ./{gguf_name}", f'TEMPLATE """{template}"""']
    if not any(p.split(None, 1)[0] == "stop" for p in parameters):
        parameters = list(parameters) + QWEN_STOPS
    for p in parameters:
        key = p.split(None, 1)[0]
        if key == "temperature" and temperature is not None:
            continue
        lines.append(f"PARAMETER {p}")
    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature}")
    return "\n".join(lines) + "\n"


def ollama_exe() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    if DEFAULT_OLLAMA.exists():
        return str(DEFAULT_OLLAMA)
    raise SystemExit("Could not find the ollama program. Install Ollama, or add it to PATH.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--name", required=True, help="new Ollama model name, e.g. quizforge-3b")
    ap.add_argument("--base", required=True, help="installed base model whose template is reused, e.g. qwen2.5:3b")
    ap.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"))
    ap.add_argument("--dry-run", action="store_true", help="only write the Modelfile")
    args = ap.parse_args(argv)

    gguf = args.gguf.resolve()
    if not gguf.exists():
        raise SystemExit(f"{gguf} does not exist - download it from the Colab notebook first.")
    try:
        template, params = base_template(args.base_url, args.base)
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot reach Ollama at {args.base_url} ({exc}). Start Ollama and try again.")
    modelfile = gguf.parent / f"Modelfile.{args.name}"
    modelfile.write_text(build_modelfile(gguf.name, template, params), encoding="utf-8")
    print(f"Wrote {modelfile}")
    if args.dry_run:
        return 0
    done = subprocess.run([ollama_exe(), "create", args.name, "-f", str(modelfile)], cwd=str(gguf.parent))
    if done.returncode != 0:
        return done.returncode
    print(f"\nCreated Ollama model '{args.name}'. Try it:  ollama run {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
