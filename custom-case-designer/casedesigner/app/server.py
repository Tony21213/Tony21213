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

from .session import Session

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
USER_ERRORS = (ValueError, FileNotFoundError, KeyError, RuntimeError)


class Jobs:
    def __init__(self):
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, title: str, fn) -> str:
        job_id = uuid.uuid4().hex[:10]
        job = {"id": job_id, "title": title, "status": "running", "progress": 0.0, "message": "", "result": None,
               "error": None}
        with self.lock:
            self.items[job_id] = job

        def progress(fraction, message=""):
            job["progress"], job["message"] = float(fraction), message

        def run():
            try:
                job["result"] = fn(progress)
                job["status"], job["progress"] = "done", 1.0
            except USER_ERRORS as e:
                job["status"], job["error"] = "error", str(e).strip("'")
            except Exception as e:  # noqa: BLE001 — показать пользователю, а не уронить сервер
                traceback.print_exc()
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

        def json(self, data, status: int = 200):
            self.send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def binary(self, data: bytes):
            self.send(data, "application/octet-stream")

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def job(self, title, fn):
            self.json({"job": jobs.start(title, fn)})

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
                self.json({"error": str(e).strip("'")}, 400)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
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
            if p == ["ct"] and method == "POST":
                path = self.body()["path"]
                return self.job("Открываю КТ", lambda progress: s.load_ct(path, progress))
            if p == ["ct", "geometry"]:
                return self.json(s.slice_geometry(q["axis"]))
            if p == ["ct", "slice"]:
                level = float(q["level"]) if "level" in q else None
                width = float(q["width"]) if "width" in q else None
                return self.send(s.slice_png(q["axis"], float(q["pos"]), level, width), "image/png")
            if p == ["ct", "surface"]:
                return self.binary(s.ct_surface())
            if p == ["overlays"] and method == "POST":
                b = self.body()
                return self.json(s.overlays(b["axis"], float(b["pos"]), b.get("visible", [])))
            if p == ["scans"] and method == "POST":
                return self.json(s.add_scan(self.body()["path"]))
            if len(p) == 3 and p[0] == "scans":
                sid, action = p[1], p[2]
                if action == "mesh":
                    return self.binary(s.scan_mesh(sid))
                if action == "colors":
                    return self.binary(s.scan_colors(sid))
                b = self.body() if method == "POST" else {}
                if action == "register":
                    return self.job("Совмещаю скан", lambda progress: s.register(
                        sid, b.get("jaw"), pairs=b.get("pairs"), start=b.get("start"), progress=progress))
                if action == "evaluate":
                    return self.job("Оцениваю положение", lambda progress: s.evaluate(sid, b["transform"]))
                if action == "jaw":
                    s.set_jaw(sid, b.get("jaw"))
                    return self.json(s.scan_info(sid))
                if action == "accept":
                    return self.json(s.accept(sid))
                if action == "remove":
                    s.remove_scan(sid)
                    return self.json({"ok": True})
            if p == ["segment"] and method == "POST":
                b = self.body()
                return self.job("Сегментация", lambda progress: s.segment(b.get("models_dir"), b.get("device", "auto"),
                                                                           progress))
            if len(p) >= 3 and p[0] == "structures" and p[-1] == "mesh":
                return self.binary(s.structure_mesh("/".join(p[1:-1])))
            if p == ["export"] and method == "POST":
                b = self.body()
                return self.job("Экспорт", lambda progress: s.export(b["out_dir"], b.get("bite", "scan"),
                                                                      b.get("frame", "exocad"), b.get("include")))
            self.json({"error": f"нет такого запроса: {method} /api/{'/'.join(p)}"}, 404)

    return Handler


def serve(session: Session, port: int = 0) -> ThreadingHTTPServer:
    """Запустить сервер в фоне; порт 0 — любой свободный."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, Jobs()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
