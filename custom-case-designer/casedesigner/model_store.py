"""Модели сегментации: где лежат, все ли на месте, загрузка готовых ONNX из программы.

Веса — открытые модели TotalSegmentator, переведённые в ONNX
(tools/prepare_models.py). Перевод требует PyTorch, поэтому программа
скачивает уже переведённые файлы: по model.onnx на модель из релиза
RELEASE репозитория проекта на GitHub. Описание модели (model.json) идёт
вместе с программой (casedesigner/model_specs), размер и SHA-256 каждого
файла записаны здесь — скачанное проверяется, и подменённый или битый файл
не будет использован.

Загрузка докачивает прерванное (файл .part и запрос Range), её можно
остановить и продолжить. Источник можно переопределить переменной
окружения CCD_MODELS_URL (папка, где лежат <модель>.onnx).

После нового перевода моделей таблицу MODELS обновляет
    python -m casedesigner.model_store ПАПКА_МОДЕЛЕЙ
"""

import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

RELEASE = "models-v1"
BASE_URL = f"https://github.com/Tony21213/Tony21213/releases/download/{RELEASE}/"
SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_specs")
CHUNK = 1 << 20
TIMEOUT_S = 30

# Модель → (название для человека, размер model.onnx в байтах, SHA-256).
MODELS = {
    "teeth": ("Зубы, челюсти, каналы, пазухи", 123178062,
              "e4d6e9f99728ef5005e0e9cafa359feb1b5f23fc1860fb218220163eb14aa074"),
    "craniofacial": ("Нижняя челюсть, череп, лобные пазухи", 123168820,
                     "cdff1b9e611cab6cd1fdcfb0df53fa598409919b0076c7bcecc7e0c20628dff5"),
    "cavities": ("Полость носа, нёбо, глотка", 123170404,
                 "d70101ef75387682a8d898f8ed2f304b539a48d230ae8650cbdf70eeac74a4f0"),
}
LICENSE_NOTE = ("TotalSegmentator (Apache-2.0); модель зубов обучена на ToothFairy3 (CC BY-NC-SA 4.0) — "
                "только некоммерческое использование, с указанием авторов")


class Cancelled(RuntimeError):
    pass


def default_dir() -> str:
    """Куда класть модели: рядом с программой, если туда можно писать, иначе в папку пользователя."""
    if getattr(sys, "frozen", False):
        here = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "models")
    else:
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    try:
        os.makedirs(here, exist_ok=True)
        probe = os.path.join(here, ".write_test")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return here
    except OSError:  # Program Files и т.п.
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".casedesigner")
        return os.path.join(base, "CustomCaseDesigner", "models")


def installed(folder: str | None, name: str) -> bool:
    if not folder:
        return False
    onnx = os.path.join(folder, name, "model.onnx")
    return os.path.isfile(onnx) and os.path.isfile(os.path.join(folder, name, "model.json")) and \
        os.path.getsize(onnx) == MODELS[name][1]


def status(folder: str | None) -> dict:
    """Что установлено и сколько осталось скачать (с учётом недокачанного)."""
    items, left = [], 0
    for name, (title, size, _sha) in MODELS.items():
        ok = installed(folder, name)
        part = os.path.join(folder, name, "model.onnx.part") if folder else ""
        have = size if ok else (os.path.getsize(part) if part and os.path.isfile(part) else 0)
        left += size - have
        items.append({"name": name, "title": title, "size": size, "installed": ok, "downloaded": have})
    return {"folder": folder, "ready": all(i["installed"] for i in items), "models": items,
            "total_bytes": sum(s for _t, s, _h in MODELS.values()), "left_bytes": left, "license": LICENSE_NOTE}


def _sha256(path: str) -> "hashlib._Hash":
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h


def download(folder: str, progress=None, cancel: threading.Event | None = None, base_url: str | None = None) -> dict:
    """Скачать недостающие модели в folder; progress(доля, сообщение, **сведения); cancel — остановить.

    Сведения прогресса: done и total (байты всего), model (имя), index и count
    (какая по счёту), speed (байт/с), eta_s (секунд до конца).
    """
    base = (base_url or os.environ.get("CCD_MODELS_URL") or BASE_URL).rstrip("/") + "/"
    todo = [n for n in MODELS if not installed(folder, n)]
    st = status(folder)
    total, done = st["total_bytes"], st["total_bytes"] - st["left_bytes"]
    started, start_done = time.monotonic(), done
    for index, name in enumerate(todo, 1):
        title, size, sha = MODELS[name]
        target = os.path.join(folder, name)
        os.makedirs(target, exist_ok=True)
        part = os.path.join(target, "model.onnx.part")
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        if have > size:  # чужой или испорченный хвост — заново
            os.remove(part)
            have = 0
        h = _sha256(part) if have else hashlib.sha256()
        if have < size:
            req = urllib.request.Request(base + f"{name}.onnx", headers={"User-Agent": "CustomCaseDesigner"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"сервер моделей ответил {e.code} ({base}{name}.onnx)") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"нет связи с сервером моделей: {e.reason}") from e
            with resp:
                if have and resp.status != 206:  # докачка не поддержана — с начала
                    done -= have
                    have, h = 0, hashlib.sha256()
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled("загрузка остановлена — её можно продолжить")
                        try:
                            block = resp.read(CHUNK)
                        except OSError as e:
                            raise RuntimeError(f"связь прервалась: {e} — загрузку можно продолжить") from e
                        if not block:
                            break
                        f.write(block)
                        h.update(block)
                        have += len(block)
                        done += len(block)
                        if progress:
                            elapsed = max(time.monotonic() - started, 1e-3)
                            speed = (done - start_done) / elapsed
                            progress(done / total, title, done=done, total=total, model=name, index=index,
                                     count=len(todo), speed=speed, eta_s=(total - done) / speed if speed > 0 else None)
        if have != size or h.hexdigest() != sha:
            os.remove(part)
            raise RuntimeError(f"{title}: файл повреждён или не тот (не совпала контрольная сумма) — скачайте заново")
        os.replace(part, os.path.join(target, "model.onnx"))
        shutil.copyfile(os.path.join(SPECS, f"{name}.json"), os.path.join(target, "model.json"))
    return status(folder)


def _table(folder: str):
    """Строки для MODELS по папке переведённых моделей."""
    for name in MODELS:
        path = os.path.join(folder, name, "model.onnx")
        with open(os.path.join(folder, name, "model.json"), encoding="utf-8") as f:
            json.load(f)
        print(f'    "{name}": (…, {os.path.getsize(path)}, "{_sha256(path).hexdigest()}"),')


if __name__ == "__main__":
    _table(sys.argv[1])
