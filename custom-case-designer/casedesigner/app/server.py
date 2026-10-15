"""Локальный HTTP-сервер приложения: интерфейс (web/) и API над сессией кейса.

Долгие операции (открыть КТ, совместить, сегментировать, экспортировать)
идут в фоне: API отвечает номером задачи, интерфейс опрашивает /api/jobs/<id>.
Сервер слушает только 127.0.0.1.
"""

import json
import mimetypes
import os
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import errorlog
from .session import Session, mesh_bytes  # noqa: F401 — mesh_bytes: прежний импорт для тестов
from .steps import aesthetics, articulation, casefile, ct, export, landmarks, scans

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
USER_ERRORS = (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError)

# Маршруты шагов (порядок — как раньше в одном списке).
STEP_ROUTES = [ct.route, export.route, scans.route, aesthetics.route, casefile.route, articulation.route,
               landmarks.route]


def routes_source() -> str:
    """Текст всех HTTP-маршрутов (общие — здесь, шагов — steps/<шаг>.route): для контрактных тестов."""
    import inspect
    import sys

    return inspect.getsource(sys.modules[__name__]) + "".join(inspect.getsource(r) for r in STEP_ROUTES)


class Jobs:
    def __init__(self):
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, title: str, fn) -> str:
        job_id = uuid.uuid4().hex[:10]
        job = {"id": job_id, "title": title, "status": "running", "progress": 0.0, "message": "", "info": {},
               "result": None, "error": None}
        with self.lock:
            self.items[job_id] = job

        def progress(fraction, message="", **info):
            job["progress"], job["message"], job["info"] = float(fraction), message, info

        def run():
            try:
                job["result"] = fn(progress)
                job["status"], job["progress"] = "done", 1.0
            except USER_ERRORS as e:
                job["status"], job["error"] = "error", str(e).strip("'")
                errorlog.message(f"{title}: {type(e).__name__}: {e}")
            except Exception as e:  # noqa: BLE001 — показать пользователю, а не уронить сервер
                traceback.print_exc()
                errorlog.error(title, e)
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict:
        return self.items[job_id]


def make_handler(session: Session, jobs: Jobs):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # без журнала каждого запроса
            pass

        # --- ответы ---
        def send(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return True

        def json(self, data, status: int = 200):
            return self.send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def binary(self, data: bytes):
            return self.send(data, "application/octet-stream")

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def job(self, title, fn):
            return self.json({"job": jobs.start(title, fn)})

        # --- маршруты ---
        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def route(self, method: str):
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if not parts or parts[0] != "api":
                    return self.static(url.path)
                self.api(method, parts[1:], query)
            except USER_ERRORS as e:
                errorlog.message(f"{method} {url.path}: {type(e).__name__}: {e}")
                self.json({"error": str(e).strip("'")}, 400)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                errorlog.error(f"{method} {url.path}", e)
                self.json({"error": f"{type(e).__name__}: {e}"}, 500)

        def static(self, path: str):
            rel = os.path.normpath(path.lstrip("/") or "index.html")
            full = os.path.join(WEB, rel)
            if rel.startswith("..") or not os.path.isfile(full):
                return self.send(b"not found", "text/plain", 404)
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if full.endswith(".js"):
                ctype = "text/javascript; charset=utf-8"
            with open(full, "rb") as f:
                self.send(f.read(), ctype)

        def api(self, method, p, q):
            s = session
            if p == ["state"]:
                return self.json(s.state())
            if p == ["jobs", p[-1]] and len(p) == 2:
                return self.json(jobs.get(p[1]))
            if p == ["log"] and method == "POST":  # ошибка в интерфейсе
                b = self.body()
                errorlog.error("интерфейс", details=f"{str(b.get('message', ''))[:2000]}\n{str(b.get('stack', ''))[:6000]}")
                return self.json({"ok": True})
            if p == ["log", "open"] and method == "POST":
                s.open_log_folder()
                return self.json({"ok": True})
            if p == ["settings", "incognito"] and method == "POST":
                return self.json(s.set_incognito(self.body().get("on") is True))
            if p == ["settings", "segment_parts"] and method == "POST":
                return self.json(s.set_segment_parts(self.body().get("parts", [])))
            if p == ["settings", "mount_plane"] and method == "POST":
                return self.json(s.set_default_mount_plane(self.body().get("plane")))
            for route in STEP_ROUTES:  # маршруты шагов — casedesigner/app/steps/<шаг>.py, route()
                if route(self, s, p, method, q):
                    return
            self.json({"error": f"нет такого запроса: {method} /api/{'/'.join(p)}"}, 404)

    return Handler


def serve(session: Session, port: int = 0) -> ThreadingHTTPServer:
    """Запустить сервер в фоне; порт 0 — любой свободный."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, Jobs()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
