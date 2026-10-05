"""Загрузка моделей сегментации из программы: докачка, остановка, проверка контрольной суммы."""

import hashlib
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from casedesigner import model_store as ms

RNG = {name: os.urandom(300_000 + 1000 * i) for i, name in enumerate(ms.MODELS)}


@pytest.fixture
def fake_models(monkeypatch):
    """Маленькие «модели» вместо настоящих: размеры и суммы — свои."""
    table = {n: (f"модель {n}", len(b), hashlib.sha256(b).hexdigest()) for n, b in RNG.items()}
    monkeypatch.setattr(ms, "MODELS", table)
    monkeypatch.setattr(ms, "CHUNK", 16384)
    return table


def serve(files: dict, ranges: bool = True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1]
            if name not in files:
                self.send_error(404)
                return
            data, start = files[name], 0
            if ranges and (r := self.headers.get("Range")):
                start = int(r.split("=")[1].split("-")[0])
                self.send_response(206)
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def files(blobs=RNG):
    return {f"{n}.onnx": b for n, b in blobs.items()}


def test_download_all_and_status(fake_models, tmp_path):
    srv, url = serve(files())
    seen = []
    st = ms.download(str(tmp_path), lambda f, m, **i: seen.append((f, i)), base_url=url)
    srv.shutdown()
    assert st["ready"] and st["left_bytes"] == 0
    for n in ms.MODELS:
        assert (tmp_path / n / "model.onnx").read_bytes() == RNG[n] and (tmp_path / n / "model.json").is_file()
    assert seen[-1][0] == pytest.approx(1.0) and seen[-1][1]["index"] == 3 and seen[-1][1]["done"] == st["total_bytes"]


@pytest.mark.parametrize("ranges", [True, False])
def test_resume_after_interruption(fake_models, tmp_path, ranges):
    """Прерванная загрузка продолжается с того же места (или с начала, если сервер не умеет докачку)."""
    name = next(iter(ms.MODELS))
    (tmp_path / name).mkdir()
    (tmp_path / name / "model.onnx.part").write_bytes(RNG[name][:100_000])
    assert ms.status(str(tmp_path))["left_bytes"] == sum(len(b) for b in RNG.values()) - 100_000
    srv, url = serve(files(), ranges=ranges)
    st = ms.download(str(tmp_path), base_url=url)
    srv.shutdown()
    assert st["ready"] and (tmp_path / name / "model.onnx").read_bytes() == RNG[name]


def test_cancel_keeps_part_and_bad_file_is_rejected(fake_models, tmp_path):
    srv, url = serve(files())
    stop = threading.Event()

    def progress(fraction, message, **info):
        if info["done"] > 50_000:
            stop.set()

    with pytest.raises(ms.Cancelled):
        ms.download(str(tmp_path), progress, stop, base_url=url)
    first = next(iter(ms.MODELS))
    assert (tmp_path / first / "model.onnx.part").is_file() and not ms.status(str(tmp_path))["ready"]
    srv.shutdown()

    bad = dict(RNG)
    bad[first] = bytes(len(RNG[first]))  # подменённый файл того же размера
    for p in tmp_path.glob("*/model.onnx.part"):
        p.unlink()
    srv, url = serve(files(bad))
    with pytest.raises(RuntimeError, match="контрольная сумма"):
        ms.download(str(tmp_path), base_url=url)
    srv.shutdown()
    assert not (tmp_path / first / "model.onnx").exists()


def test_unreachable_server_is_readable(fake_models, tmp_path):
    with pytest.raises(RuntimeError, match="нет связи|ответил 404"):
        ms.download(str(tmp_path), base_url="http://127.0.0.1:9/")
