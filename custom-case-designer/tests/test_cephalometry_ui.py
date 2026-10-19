"""Шаг «Ориентиры» и цефалометрия в программе: анализы по точкам кейса, отчёт PDF/PNG, HTTP; окно — шаг подключён,
точка ставится щелчком по срезу (app.placeLandmark)."""

import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from casedesigner.app import server as app_server
from casedesigner.app.session import Session

WEB = Path(__file__).parents[1] / "casedesigner" / "app" / "web"

# Точки боковой ТРГ в мм (x — вправо, y — назад, z — вверх): SNA ≈ 82°, SNB ≈ 80°
POINTS = {
    "S": (0.0, 20.0, 40.0), "N": (0.0, -50.0, 45.0), "ANS": (0.0, -52.0, 0.0), "PNS": (0.0, -5.0, 0.0),
    "Po_R": (55.0, 20.0, 25.0), "Po_L": (-55.0, 20.0, 25.0), "Or_R": (35.0, -40.0, 25.0), "Or_L": (-35.0, -40.0, 25.0),
}


def session(tmp_path, points=POINTS):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    for key, p in points.items():
        s.set_landmark(key, p)
    return s


def angle_point(vertex, start, deg, length):
    """Точка на луче из vertex под углом deg к лучу vertex→start, в срединной плоскости (x = 0), вниз-вперёд."""
    v, a = np.asarray(vertex, float), np.asarray(start, float)
    d = (a - v) / np.linalg.norm(a - v)
    t = np.radians(deg)
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return tuple(v + length * (rot @ d))


def test_analyses_from_session_points(tmp_path):
    s = session(tmp_path, {key: p for key, p in POINTS.items() if key in ("S", "PNS")})
    info = s.cephalometry_info()
    assert info["analyses"] == {} and "срединная плоскость" in info["error"]
    s = session(tmp_path)
    for key, deg in (("A", 82.0), ("B", 80.0)):
        candidates = [angle_point(POINTS["N"], POINTS["S"], sign * deg, 50.0) for sign in (1, -1)]
        s.set_landmark(key, max(candidates, key=lambda p: -p[2]))  # ниже N
    info = s.cephalometry_info()
    steiner = {r["key"]: r for r in info["analyses"]["steiner"]["results"]}
    assert steiner["SNA"]["value"] == pytest.approx(82.0, abs=0.2) and steiner["SNB"]["value"] == pytest.approx(80.0, abs=0.2)
    assert steiner["ANB"]["text"] == "в норме"
    skipped = {x["key"] for x in info["analyses"]["steiner"]["skipped"]}
    assert "U1_NA_deg" in skipped and "ricketts" in info["analyses"]
    saved = s.cephalometry_save(str(tmp_path / "out"))
    assert saved["files"] == ["cephalometry.png", "cephalometry.pdf"]
    assert (tmp_path / "out" / "cephalometry.pdf").read_bytes().startswith(b"%PDF")


def test_http_routes(tmp_path):
    s = session(tmp_path)
    httpd = app_server.serve(s)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api"

    def post(path, body):
        req = urllib.request.Request(f"{base}/{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    try:
        with urllib.request.urlopen(f"{base}/cephalometry") as r:
            info = json.loads(r.read())
        assert set(info["analyses"]) == {"steiner", "ricketts"} and info["error"] is None
        post("landmarks", {"key": "A", "point": [0.0, -50.0, 0.0]})
        with urllib.request.urlopen(f"{base}/cephalometry") as r:
            assert any(r_["key"] == "SNA" for r_ in json.loads(r.read())["analyses"]["steiner"]["results"])
        with pytest.raises(urllib.error.HTTPError):
            post("cephalometry/open", {})  # отчёта ещё нет
        assert post("cephalometry/save", {"out_dir": str(tmp_path / "rep")})["files"][1] == "cephalometry.pdf"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_step_wired_in_window():
    core = (WEB / "core.js").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    slices = (WEB / "slices.js").read_text(encoding="utf-8")
    step = (WEB / "steps" / "landmarks.js").read_text(encoding="utf-8")
    assert "{ id: 'landmarks', title: 'Ориентиры'" in core
    assert "RENDER.landmarks = renderLandmarks" in app and "app.placeLandmark = placeLandmark" in app
    assert "this.app.placeLandmark?.(p)" in slices
    assert "landmarks/auto" in step and "cephalometry/save" in step and "SNA" not in step  # анализы — с сервера
