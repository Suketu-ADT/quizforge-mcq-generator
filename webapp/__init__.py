"""QuizForge web app: a Flask JSON API (webapp/api.py) and a plain HTML/CSS/JS front end (webapp/static)."""
from __future__ import annotations

from pathlib import Path

from flask import Flask, send_from_directory

from config import DB_PATH, MAX_UPLOAD_MB
from database.db import Database
from database.repository import Repository
from webapp.context import AppContext, ClientFactory

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(db_path: str | Path | None = None, client_factory: ClientFactory | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = (MAX_UPLOAD_MB + 1) * 1024 * 1024  # uploads; the exact limit is checked later
    app.config["JSON_SORT_KEYS"] = False
    app.extensions["quizforge"] = AppContext(Repository(Database(db_path or DB_PATH)), client_factory)

    from webapp.api import bp

    app.register_blueprint(bp)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.errorhandler(404)
    def missing(_exc):
        from flask import request

        if request.path.startswith("/api/"):
            return {"error": "Not found."}, 404
        return send_from_directory(STATIC_DIR, "index.html")  # the front end handles its own #routes

    @app.errorhandler(413)
    def too_large(_exc):
        return {"error": f"The file is larger than the {MAX_UPLOAD_MB} MB limit."}, 413

    @app.after_request
    def security_headers(resp):
        # everything is served from this computer: no third-party scripts, styles, fonts or frames
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if resp.mimetype in ("text/html", "application/javascript", "text/javascript", "text/css"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    return app
