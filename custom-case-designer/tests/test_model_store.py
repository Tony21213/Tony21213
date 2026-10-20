"""Загрузка моделей сегментации из программы: докачка, остановка, проверка контрольной суммы."""

import hashlib
import json
import os
import sys
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


def test_bundled_specs_match_sources():
    """model.json, который программа кладёт к скачанной модели, описывает те же выходы, что tools/models/*.json.

    Раньше описание cavities отстало (без слуховых проходов), а перевод терял priority.
    """
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    sys.path.insert(0, tools)
    try:
        from export_nnunet_onnx import describe
    finally:
        sys.path.remove(tools)
    for name in ms.MODELS:
        with open(os.path.join(tools, "models", f"{name}.json"), encoding="utf-8") as f:
            spec = json.load(f)
        with open(os.path.join(ms.SPECS, f"{name}.json"), encoding="utf-8") as f:
            bundled = json.load(f)
        expected = describe(spec, {})
        assert {k: bundled.get(k) for k in expected} == expected, name
        assert bundled["priority"] == spec["priority"]
        assert {lab for labs in bundled["outputs"].values() for lab in labs} <= set(bundled["labels"]), name


def test_photo_models_from_zip_and_checksum(monkeypatch, tmp_path):
    """Нейросети для фото: SAM — из архива (кодировщик и декодер), MediaPipe — файлом; подменённый файл не ложится."""
    import io
    import zipfile

    enc, dec, face = os.urandom(40_000), os.urandom(30_000), os.urandom(20_000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("sam21_tiny/sam2.1_hiera_tiny.encoder.onnx", enc)
        z.writestr("sam21_tiny/sam2.1_hiera_tiny.decoder.onnx", dec)
        z.writestr("sam21_tiny/config.yaml", b"input_size: 1024")
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    monkeypatch.setattr(ms, "PHOTO_MODELS", {
        "sam": ("SAM", "https://example.invalid/sam.zip", 90_000,
                {"sam2.1-tiny/sam2.1_hiera_tiny.encoder.onnx": sha(enc), "sam2.1-tiny/sam2.1_hiera_tiny.decoder.onnx": sha(dec)}),
        "face": ("лицо", "https://example.invalid/face_landmarker.task", 20_000, {"face_landmarker.task": sha(face)})})
    monkeypatch.setattr(ms, "CHUNK", 8192)
    st = ms.photo_status(str(tmp_path))
    assert not st["ready"] and st["left_bytes"] == 110_000
    srv, url = serve({"sam.zip": buf.getvalue(), "face_landmarker.task": face})
    seen = []
    st = ms.download_photo(str(tmp_path), lambda f, m, **i: seen.append(f), base_url=url)
    srv.shutdown()
    assert st["ready"] and st["left_bytes"] == 0 and seen[-1] == pytest.approx(1.0)
    assert (tmp_path / "sam2.1-tiny" / "sam2.1_hiera_tiny.encoder.onnx").read_bytes() == enc
    assert (tmp_path / "face_landmarker.task").read_bytes() == face and not list(tmp_path.rglob("*.part"))
    (tmp_path / "face_landmarker.task").unlink()
    srv, url = serve({"face_landmarker.task": bytes(len(face))})
    with pytest.raises(RuntimeError, match="контрольная сумма"):
        ms.download_photo(str(tmp_path), base_url=url)
    srv.shutdown()
    assert not (tmp_path / "face_landmarker.task").exists() and not list(tmp_path.rglob("*.part"))
    assert ms.photo_status(str(tmp_path))["models"][0]["installed"]  # SAM уже был — не тронут


def test_landmark_models_download_and_status(monkeypatch, tmp_path):
    blobs = {"RPo/1.onnx": os.urandom(50_000), "RPo/0-3.onnx": os.urandom(60_000)}
    monkeypatch.setattr(ms, "LANDMARK_FILES", {k: (len(b), hashlib.sha256(b).hexdigest()) for k, b in blobs.items()})
    monkeypatch.setattr(ms, "CHUNK", 8192)
    st = ms.landmarks_status(str(tmp_path))
    assert not st["ready"] and st["left_bytes"] == 110_000
    srv, url = serve({ms._landmark_asset(k): b for k, b in blobs.items()})
    seen = []
    st = ms.download_landmarks(str(tmp_path), lambda f, m, **i: seen.append(f), base_url=url)
    srv.shutdown()
    assert st["ready"] and st["left_bytes"] == 0 and seen[-1] == pytest.approx(1.0)
    for k, b in blobs.items():
        assert (tmp_path / "landmarks" / k).read_bytes() == b
    assert ms._landmark_asset("RPo/0-3.onnx") == "ali-RPo-0-3.onnx"
