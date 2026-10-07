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


def _fetch(url: str, part: str, size: int, sha: str, title: str, on_bytes, cancel) -> None:
    """Скачать url в part (с докачкой) и проверить размер и SHA-256; on_bytes(сколько добавилось, сброс)."""
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have > size:  # чужой или испорченный хвост — заново
        os.remove(part)
        have = 0
    h = _sha256(part) if have else hashlib.sha256()
    if have < size:
        req = urllib.request.Request(url, headers={"User-Agent": "KStomCaseDesigner"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"сервер моделей ответил {e.code} ({url})") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"нет связи с сервером моделей: {e.reason}") from e
        with resp:
            if have and resp.status != 206:  # докачка не поддержана — с начала
                on_bytes(-have, True)
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
                    on_bytes(len(block), False)
    if have != size or h.hexdigest() != sha:
        os.remove(part)
        raise RuntimeError(f"{title}: файл повреждён или не тот (не совпала контрольная сумма) — скачайте заново")


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

        def on_bytes(n, reset, title=title, name=name, index=index):
            nonlocal done
            done += n
            if progress and not reset:
                elapsed = max(time.monotonic() - started, 1e-3)
                speed = (done - start_done) / elapsed
                progress(done / total, title, done=done, total=total, model=name, index=index,
                         count=len(todo), speed=speed, eta_s=(total - done) / speed if speed > 0 else None)

        _fetch(base + f"{name}.onnx", part, size, sha, title, on_bytes, cancel)
        os.replace(part, os.path.join(target, "model.onnx"))
        shutil.copyfile(os.path.join(SPECS, f"{name}.json"), os.path.join(target, "model.json"))
    return status(folder)


# --- модели автоматических ориентиров (ALI-CBCT) ------------------------------------------------
# Файл в папке <модели>/landmarks → (размер, SHA-256). В релизе файлы лежат плоско:
# «ali-<точка>-<масштаб>.onnx». Таблицу печатает tools/prepare_landmarks.py.
LANDMARK_RELEASE = "landmarks-v1"
LANDMARK_URL = f"https://github.com/Tony21213/Tony21213/releases/download/{LANDMARK_RELEASE}/"
LANDMARK_TITLE = "Автоматические ориентиры на КТ"
LANDMARK_FILES: dict[str, tuple[int, str]] = {
    "ANS/1.onnx": (29277027, "029c2346660fdfb88b208f5babf7fe3b9b78d0c94b2bfbaaab57b2415fa623ad"),
    "ANS/0-3.onnx": (29277027, "1cdd2365ac38bfbde233b00472b93bb48c84d432923ec1d90a03c883135d2bdc"),
    "Ba/1.onnx": (29277027, "651582a77a933743d2f8f4a1af1d1ead607ce1ea3faadf5f6ccdea8c39225a0f"),
    "Ba/0-3.onnx": (29277027, "bdcedb4532bd586c8d86f0341dd705604e88f050ef3ee2a4bc72b8f6943450d2"),
    "IF/1.onnx": (29277027, "3926fc39b4766b1af1baf5c476eaf47effa9c4d5fc6d7376bdc171c84d2098dd"),
    "IF/0-3.onnx": (29277027, "321b8c44802da4205e60b509a850c92b252e4fbcf61365f3ff0a9367b491d16b"),
    "LCo/1.onnx": (29277027, "1f4f86d88cf29ec58280d58073ec95430e6027662e123beda9f35cecb8b10a2c"),
    "LCo/0-3.onnx": (29277027, "a9c47a67c31999dcb5e866a2d68679f320b5b92d7664e656c2116cf38f5297b9"),
    "LOr/1.onnx": (29277027, "0944a7cd81fc575c4778a3aecf131e53779000cab2c2165cb8449ca57924b481"),
    "LOr/0-3.onnx": (29277027, "547750dad929317361d4288b2e36ea279188aa836dd0a54a07fc447e5d2b8a77"),
    "LPo/1.onnx": (29277027, "62f0cf80be2f6e0ff51ecf1243e1faf348c37d9671da3648a26f0c8babc2616a"),
    "LPo/0-3.onnx": (29277027, "70fe26ce67638a8ed3749fe74a6b23b5e96eb03574a654b1f132f073aafe6b71"),
    "N/1.onnx": (29277027, "fd6ccf9b677d5852bc59cb153bbe6b7e7efda321a27d7763a2a22b0431ea1a4c"),
    "N/0-3.onnx": (29277027, "86c4931fd4c089bb81d3eee778991be68233bec857229b2401d1de81cafe9c3c"),
    "PNS/1.onnx": (29277027, "e1ed2bfbb70de1fda4375cca252ccc37fd64ce49810ab1618d2a5b7165ef84f0"),
    "PNS/0-3.onnx": (29277027, "857c40e977f8e15fd857bc9f502c11f873623e85fab92bcfa85fadb0ea567fdc"),
    "RCo/1.onnx": (29277027, "184c4b70b0fa988c82837732b4723c9294dfc1ad788e36dabd64042557524128"),
    "RCo/0-3.onnx": (29277027, "9b06e61b524f0b66cab040f082d30bd89fc2f8ca7cb2d295a6d3db3ec7777717"),
    "ROr/1.onnx": (29277027, "ee88fb16e9a3b7b653457bd8131820ad30c83d966005d0f3cb35db818f62b80f"),
    "ROr/0-3.onnx": (29277027, "50e264d7fe5a06ca79186b4fc89785bce256f17e3ecd45957afd4f98d3e538c9"),
    "RPo/1.onnx": (29277027, "be9842ec0505f1c172186e9026e4e43866743dc982e73639e6cfedd4bf8ac826"),
    "RPo/0-3.onnx": (29277027, "652ef52d0c9a190e8d98ae8834246f12e2778c313f29254e0f2a9f239ae3ac5b"),
    "S/1.onnx": (29277027, "4a4d3e062d2fe953a2b209c72d6e0fb4c3b3a5693d20909e2f0c9c68aa9e9f3e"),
    "S/0-3.onnx": (29277027, "3ae6b726a68d05f467d89455aee143166acd76e620194fcd2e5f7aca6472efda"),
}


def landmarks_dir(folder: str | None) -> str | None:
    return os.path.join(folder, "landmarks") if folder else None


def _landmark_asset(rel: str) -> str:
    point, scale = rel[:-len(".onnx")].split("/")
    return f"ali-{point}-{scale}.onnx"


def landmarks_status(folder: str | None) -> dict:
    """Скачаны ли модели ориентиров и сколько осталось (с учётом недокачанного)."""
    root = landmarks_dir(folder)
    left = 0
    for rel, (size, _sha) in LANDMARK_FILES.items():
        path = os.path.join(root, rel) if root else ""
        if path and os.path.isfile(path) and os.path.getsize(path) == size:
            continue
        part = path + ".part" if path else ""
        left += size - (os.path.getsize(part) if part and os.path.isfile(part) else 0)
    total = sum(s for s, _h in LANDMARK_FILES.values())
    return {"folder": root, "ready": bool(LANDMARK_FILES) and left == 0, "total_bytes": total, "left_bytes": left,
            "title": LANDMARK_TITLE, "license": "ALI-CBCT (Gillot M. et al., DCBIA-OrthoLab), лицензия 3D Slicer"}


def download_landmarks(folder: str, progress=None, cancel: threading.Event | None = None,
                       base_url: str | None = None) -> dict:
    """Скачать модели ориентиров в <folder>/landmarks; прогресс — как у download."""
    base = (base_url or os.environ.get("CCD_MODELS_URL") or LANDMARK_URL).rstrip("/") + "/"
    root = landmarks_dir(folder)
    st = landmarks_status(folder)
    total, done = st["total_bytes"], st["total_bytes"] - st["left_bytes"]
    started, start_done = time.monotonic(), done
    todo = [rel for rel, (size, _h) in LANDMARK_FILES.items()
            if not (os.path.isfile(os.path.join(root, rel)) and os.path.getsize(os.path.join(root, rel)) == size)]
    for index, rel in enumerate(todo, 1):
        size, sha = LANDMARK_FILES[rel]
        target = os.path.join(root, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        def on_bytes(n, reset, index=index):
            nonlocal done
            done += n
            if progress and not reset:
                speed = (done - start_done) / max(time.monotonic() - started, 1e-3)
                progress(done / total, LANDMARK_TITLE, done=done, total=total, model="landmarks", index=index,
                         count=len(todo), speed=speed, eta_s=(total - done) / speed if speed > 0 else None)

        _fetch(base + _landmark_asset(rel), target + ".part", size, sha, LANDMARK_TITLE, on_bytes, cancel)
        os.replace(target + ".part", target)
    return landmarks_status(folder)


def _table(folder: str):
    """Строки для MODELS по папке переведённых моделей."""
    for name in MODELS:
        path = os.path.join(folder, name, "model.onnx")
        with open(os.path.join(folder, name, "model.json"), encoding="utf-8") as f:
            json.load(f)
        print(f'    "{name}": (…, {os.path.getsize(path)}, "{_sha256(path).hexdigest()}"),')


if __name__ == "__main__":
    _table(sys.argv[1])
