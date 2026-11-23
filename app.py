"""QuizForge - Automatic MCQ Quiz Generator (PS 97) using a local Qwen 2.5 model via Ollama.

Run with:  python app.py          (or double-click start.bat on Windows)
Then open: http://127.0.0.1:5000

Options:   python app.py --port 5050 --no-browser
"""
from __future__ import annotations

import argparse
import threading
import webbrowser

from webapp import create_app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the QuizForge web app on this computer.")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps the app private to this computer")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--debug", action="store_true", help="Flask debug server with auto-reload (development only)")
    args = ap.parse_args(argv)

    app = create_app()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"QuizForge is running at {url}  (press Ctrl+C to stop)")
    if args.debug:
        app.run(args.host, args.port, debug=True, threaded=True)
        return
    try:
        from waitress import serve
    except ImportError:  # pragma: no cover - waitress is in requirements.txt
        app.run(args.host, args.port, threaded=True)
    else:
        serve(app, host=args.host, port=args.port, threads=8)


if __name__ == "__main__":
    main()
