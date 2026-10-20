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
LANDMARK_RELEASE = "landmarks-v2"  # 14.10.2026: + точки цефалометрии (A, B, Pog, Gn, Me, Go, резцы)
LANDMARK_URL = f"https://github.com/Tony21213/Tony21213/releases/download/{LANDMARK_RELEASE}/"
LANDMARK_TITLE = "Автоматические ориентиры на КТ"
LANDMARK_FILES: dict[str, tuple[int, str]] = {
    "A/1.onnx": (29277027, "d1ed793b9e3a097f971a1106ace556c7c0bff0e0c64ea6dd23b1032f9d6d1d5b"),
    "A/0-3.onnx": (29277027, "6ff4d2f4ec0a4271061c87636895b5c26beddfac0a0b3ccc75777e4d6b727c11"),
    "ANS/1.onnx": (29277027, "795c33063f2e35d3743c3a6313dbd9f8892b1d2df2c079b4c9f028d3bbe16d6b"),
    "ANS/0-3.onnx": (29277027, "1557111bd3f4081ed241c616db10b18e73720f8d01229f0ea02ebd955e636a0f"),
    "B/1.onnx": (29277027, "f448c2c1d6485d10a310528a311f0ce4ba806988996ba448a97bb12d8262f17e"),
    "B/0-3.onnx": (29277027, "077c3aa2d165a58a11c0d170aeab26f9bdea709955d9bfa92dfffc07cc8dafb8"),
    "Ba/1.onnx": (29277027, "eb3e420fda5b87f807eb5ea687395eeaec8574d8c2f72188836bea141b3e8467"),
    "Ba/0-3.onnx": (29277027, "2b6e72a6b0cf7b32e3b77d676907a0b370c5731b610f194e04ade7f1b0c5eb9f"),
    "Gn/1.onnx": (29277027, "5cc7efaac1477ed6deb6ab60c2dafa75626152e183c983f3ea2a2ebf3bb266dc"),
    "Gn/0-3.onnx": (29277027, "02ac8512c58af78f2a88d718a583cf8f580dc25e6bff0914f285170cc9cc7722"),
    "IF/1.onnx": (29277027, "3926fc39b4766b1af1baf5c476eaf47effa9c4d5fc6d7376bdc171c84d2098dd"),
    "IF/0-3.onnx": (29277027, "03711eed6fa445662895ba1820735df5e6fd47317d14d0685084c5ae0eb0f966"),
    "LCo/1.onnx": (29277027, "57eca93fa6225cc354570f0e47eebbdb6b9967095d75585812d83ba402491bba"),
    "LCo/0-3.onnx": (29277027, "b1d2ccb1e6ab2604241ca677b4ad396b21d1cd106285616a0fe1906337723625"),
    "LGo/1.onnx": (29277027, "3df963a2223ed7564760b041d80e4d9185a247b0ed70424290e29f5e180da309"),
    "LGo/0-3.onnx": (29277027, "ae15c114eb38b89c7a29ea291872e362358c405ca476cc644c7ec1058bb2d7c2"),
    "LOr/1.onnx": (29277027, "d5bf1819e870c39d3ba2a5e28d82b53af431add075c10e71a86f8c0d23247884"),
    "LOr/0-3.onnx": (29277027, "22de13b30dc62ea0e5ee59a151f8cd15931cdeed5477a6fe10e720f341d8213b"),
    "LPo/1.onnx": (29277027, "62e4cfdfef5b19da8426674eef06ed74f37bcf367c979563701c48212a669f16"),
    "LPo/0-3.onnx": (29277027, "442aa125ee663b1d5622aa4f12d05f958243e153413ad71338f7f31d6a6bbb08"),
    "LR1O/1.onnx": (29277027, "d4dcebbf4980fd336e28ff4a32bffd96b1af1dde183f42e1c91863ea09e9d6da"),
    "LR1O/0-3.onnx": (29277027, "5b5e6a7ab6f0d61e912029b04e851bb4f747aeaa2c19d3b03550c5c9990cb6d1"),
    "LR1R/1.onnx": (29277027, "799d1410c75119c23ebef47243a5ff5a62040aeeb06858f8fdbf4cf1bbe214de"),
    "LR1R/0-3.onnx": (29277027, "c8ed1e13b98a493cd0ea1eae95f5fc5ebb64303ad1bba941015d513f9cef15cf"),
    "Me/1.onnx": (29277027, "cfcc36f96b7eff9959bd4e91a4fcbcd917c8979b57c80653137665711a620744"),
    "Me/0-3.onnx": (29277027, "28e5ac3cbe962159cb129aa1dadd3214c5e36a34c28181a27954a98adf7d7cf8"),
    "N/1.onnx": (29277027, "03897776563ce60007438c63470deea9e39de43630fa355abdf33cb0f8f0083f"),
    "N/0-3.onnx": (29277027, "c79671d173c9b448a50ed61bec67049037b8177d46f54af53a8c6eca0dafff41"),
    "PNS/1.onnx": (29277027, "d3fe380e0e25b2bcc3c5a66e4ec7f9b004ea4cb7bf00666fa14e4e88967e0b3e"),
    "PNS/0-3.onnx": (29277027, "659a9ca55dcbe79f0b47f3d374cd7866342488c6523649148ec055526331cf17"),
    "Pog/1.onnx": (29277027, "68ed598825cb782d632f5c25303b4598339328511efb708bf4f2e412bf297e72"),
    "Pog/0-3.onnx": (29277027, "b4ba4ffdbb8a78f1b306becc4a1c8f6f6280cf0b488ec7c776c4a093d6b2af87"),
    "RCo/1.onnx": (29277027, "3f525a7719f344556029b86e4af0b969d00999bb4cab84e25286e510a591a19c"),
    "RCo/0-3.onnx": (29277027, "4dd3b455cbaa90006324a9714e6a81f67298a552f4d12ca8d5719cf433c7b86a"),
    "RGo/1.onnx": (29277027, "fa1de7417fd5185ce30c2d7d369024b12900915ed437a6361c3c650846ebdd91"),
    "RGo/0-3.onnx": (29277027, "f40ad61e2940ee9c9edaf5943b1dc05b597294e98c13a9311cd7a5e2ec91dbbb"),
    "ROr/1.onnx": (29277027, "7fbe72729ef6218ac9dd45b2719d699f05a9eefc567c026ed0a6e48ecb50e605"),
    "ROr/0-3.onnx": (29277027, "16fcc84aa367ce1bd789a31ae7cc915a2713cd3706cb3f098c18bcda45769cc7"),
    "RPo/1.onnx": (29277027, "064a1cdd5eb2ba065499232a52f470d6d3eeda062e7d54be7bb310e4d1c926a6"),
    "RPo/0-3.onnx": (29277027, "4b01b921e86372f3031fe31499bd98fd925d71184176a91eef78d97dc25ac7db"),
    "S/1.onnx": (29277027, "372fe9c7b9beee82b27b5b3f03c23e6e4099108093ba17ebadda5f05c2249c4b"),
    "S/0-3.onnx": (29277027, "401b8c728ac0b7d73157ae30013b9c1dc29cd761aa34f7d68ca67bcacf3b9df2"),
    "UR1O/1.onnx": (29277027, "b62d8884fb2f66e0ebebdac601627e25c43ec18c37087b91d95cabc92b47aa0d"),
    "UR1O/0-3.onnx": (29277027, "62fd8ee5600be0772626b476676450d95ab00326df9984714b02ee9bf0f083ba"),
    "UR1R/1.onnx": (29277027, "89b5f0070a27d6d303a9832cd1c78590d7f26db2e29264da21b92a4180fc2242"),
    "UR1R/0-3.onnx": (29277027, "386d3ae46259e34199858c0affb9d9d5bc57eb1afcb0decbbfc84c76d2ae3dfa"),
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


# --- нейросети для фото и сканов (шаг «Эстетика») ---------------------------------------------
# Готовые файлы с хранилищ их авторов: SAM 2.1 — архив ONNX (кодировщик и декодер), MediaPipe — файл .task,
# MoGe-2 — ONNX авторов. Проверяется SHA-256 каждого файла, который ложится в папку моделей; размер — примерный,
# для окна. GeoCalib и DentalModelSeg авторы выкладывают только для PyTorch: адреса нет (None) — наш перевод в ONNX
# (tools/prepare_camera_models.py, tools/prepare_scan_seg.py) ляжет в релиз проекта, тогда — адрес и сумма.
PHOTO_TITLE = "Нейросети для фото"
PHOTO_MODELS: dict[str, tuple[str, str | None, int, dict[str, str | None]]] = {
    # имя → (название, адрес, примерно байт, {файл в папке моделей: SHA-256})
    "sam": ("Контуры зубов — SAM 2.1 Hiera-Tiny (Meta, Apache-2.0)",
            "https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models/resolve/main/"
            "sam2.1_hiera_tiny_20260221.zip", 116_500_000,
            {"sam2.1-tiny/sam2.1_hiera_tiny.encoder.onnx": "667384d1e686de6828b841ac8a24db0fafa2b3452494225f82eeedac56141230",
             "sam2.1-tiny/sam2.1_hiera_tiny.decoder.onnx": "c40f5aa7d37b681cd500481a85d44839fd81c93dce1e86271a2c866470d22105"}),
    "face": ("Точки лица и губы — MediaPipe Face Landmarker (Google, Apache-2.0)",
             "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
             "face_landmarker.task", 3_800_000,
             {"face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"}),
    "moge": ("Расстояние до лица — MoGe-2 ViT-S (Microsoft, MIT)",
             "https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/main/model.onnx", 140_852_051,
             {"moge2-vits/model.onnx": "24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0"}),
    "geocalib": ("Фокус и вертикаль снимка — GeoCalib (ETH Zürich; код Apache-2.0, веса CC BY 4.0)", None,
                 116_238_176, {"geocalib/model.onnx": None}),
    "dentalmodelseg": ("Зубы на скане без КТ — DentalModelSeg (UNC, Univ. of Michigan; Apache-2.0)", None,
                       6_632_040, {"dentalmodelseg/model.onnx": None}),
}
PHOTO_CONVERT = {"geocalib": "tools/prepare_camera_models.py geocalib", "dentalmodelseg": "tools/prepare_scan_seg.py"}


def photo_status(folder: str | None) -> dict:
    """Скачаны ли нейросети для фото и сколько примерно осталось."""
    items = []
    for name, (title, url, approx, files) in PHOTO_MODELS.items():
        ok = bool(folder) and all(os.path.isfile(os.path.join(folder, rel)) for rel in files)
        items.append({"name": name, "title": title, "installed": ok, "approx_bytes": approx,
                      "downloadable": url is not None, "convert": PHOTO_CONVERT.get(name)})
    return {"folder": folder, "ready": all(i["installed"] for i in items), "models": items, "title": PHOTO_TITLE,
            "left_bytes": sum(i["approx_bytes"] for i in items if not i["installed"] and i["downloadable"]),
            "downloadable": any(not i["installed"] and i["downloadable"] for i in items)}


def _get(url: str, part: str, on_bytes, cancel) -> None:
    """Скачать url в part целиком (без докачки: размер и сумма архива заранее не известны — проверяются файлы
    из него); on_bytes(сколько добавилось, сколько всего по ответу сервера)."""
    req = urllib.request.Request(url, headers={"User-Agent": "KStomCaseDesigner"})
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"сервер моделей ответил {e.code} ({url})") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"нет связи с сервером моделей: {e.reason}") from e
    total = int(resp.headers.get("Content-Length") or 0)
    with resp, open(part, "wb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled("загрузка остановлена — её можно начать снова")
            try:
                block = resp.read(CHUNK)
            except OSError as e:
                raise RuntimeError(f"связь прервалась: {e} — скачайте заново") from e
            if not block:
                break
            f.write(block)
            on_bytes(len(block), total)


def _place(source, folder: str, rel: str, sha: str, title: str) -> None:
    """Файл (поток source) — в папку моделей через .part, если совпала SHA-256."""
    target = os.path.join(folder, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    h = hashlib.sha256()
    try:
        with open(target + ".part", "wb") as f:
            for block in iter(lambda: source.read(CHUNK), b""):
                f.write(block)
                h.update(block)
    except BaseException:
        os.remove(target + ".part")
        raise
    if h.hexdigest() != sha:
        os.remove(target + ".part")
        raise RuntimeError(f"{title}: файл {os.path.basename(rel)} повреждён или не тот (не совпала контрольная сумма)")
    os.replace(target + ".part", target)


def download_photo(folder: str, progress=None, cancel: threading.Event | None = None,
                   base_url: str | None = None) -> dict:
    """Скачать нейросети для фото в folder (base_url — другой источник тех же файлов: <base_url>/<имя файла>);
    progress(доля, сообщение, **сведения) — как у download."""
    import tempfile
    import zipfile

    todo = [n for n, i in zip(PHOTO_MODELS, photo_status(folder)["models"]) if not i["installed"] and i["downloadable"]]
    plan = sum(PHOTO_MODELS[n][2] for n in todo) or 1  # примерно байт всего
    done, planned, started = 0, 0, time.monotonic()  # скачано байт; примерно байт в уже скачанных моделях
    for index, name in enumerate(todo, 1):
        title, url, approx, files = PHOTO_MODELS[name]
        if base_url:
            url = base_url.rstrip("/") + "/" + url.rsplit("/", 1)[1]

        def on_bytes(n, total, title=title, name=name, index=index, before=done, planned=planned, approx=approx):
            nonlocal done
            done += n
            if progress:
                share = min(1.0, (done - before) / (total or approx))
                fraction = (planned + share * approx) / plan
                speed = done / max(time.monotonic() - started, 1e-3)
                progress(fraction, title, done=round(fraction * plan), total=plan, model=name, index=index,
                         count=len(todo), speed=speed, eta_s=(1 - fraction) * plan / speed if speed > 0 else None)

        os.makedirs(folder, exist_ok=True)
        fd, part = tempfile.mkstemp(suffix=".part", dir=folder)
        os.close(fd)
        try:
            _get(url, part, on_bytes, cancel)
            if url.lower().endswith(".zip"):
                with zipfile.ZipFile(part) as z:
                    members = {os.path.basename(m): m for m in z.namelist() if not m.endswith("/")}
                    for rel, sha in files.items():
                        member = members.get(os.path.basename(rel))
                        if member is None:
                            raise RuntimeError(f"{title}: в архиве нет {os.path.basename(rel)}")
                        with z.open(member) as src:
                            _place(src, folder, rel, sha, title)
            else:
                (rel, sha), = files.items()
                with open(part, "rb") as src:
                    _place(src, folder, rel, sha, title)
        except zipfile.BadZipFile:
            raise RuntimeError(f"{title}: архив повреждён — скачайте заново") from None
        finally:
            if os.path.isfile(part):
                os.remove(part)
        planned += approx
    return photo_status(folder)


def _table(folder: str):
    """Строки для MODELS по папке переведённых моделей."""
    for name in MODELS:
        path = os.path.join(folder, name, "model.onnx")
        with open(os.path.join(folder, name, "model.json"), encoding="utf-8") as f:
            json.load(f)
        print(f'    "{name}": (…, {os.path.getsize(path)}, "{_sha256(path).hexdigest()}"),')


if __name__ == "__main__":
    _table(sys.argv[1])
