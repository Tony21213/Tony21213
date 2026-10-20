"""Шаг «Эстетика» (docs/handoff/archive/aesthetics-photo.md): фото, пары «фото — скан», камера, точки лица,
монтаж по фото, фото в выгрузке для exocad, кейс, HTTP — на фантоме с синтетической камерой."""

import json
import os
import urllib.error
import urllib.request

import numpy as np
import pytest
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

import phantom
from casedesigner import aesthetic as ae
from casedesigner.app import server as app_server
from casedesigner.app.session import Session
from casedesigner.register import apply

W, H = 1200, 900
ROLL_DEG = 4.0  # крен камеры: истинная горизонталь в кадре — под этим углом к краю


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("aesthetics")
    for jaw in ("upper", "lower"):
        v, f = phantom.make_scan(jaw, resolution=0.4)
        trimesh.Trimesh(v, f).export(d / f"{jaw}.stl")
    for name in ("smile", "retracted"):
        Image.new("RGB", (W, H), (180, 140, 120)).save(d / f"{name}.jpg")
    return d


def _session(files, tmp_path):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    for jaw in ("upper", "lower"):
        sid = s.add_scan(str(files / f"{jaw}.stl"))["id"]
        s.register(sid, jaw)
    return s


def _camera(s):
    """Камера спереди в 300 мм от резцов, с креном ROLL_DEG и лёгким разворотом (как в тестах exocad_views)."""
    base = s._base_frame()
    x_f, y_f, z_f = base[:3, :3]
    R0 = np.array([-x_f, -z_f, -y_f])
    R = Rotation.from_euler("z", ROLL_DEG, degrees=True).as_matrix() @ R0 @ \
        Rotation.from_rotvec(np.radians(6) * z_f).as_matrix()
    centre = s._upper_incisal(base) + 300 * y_f
    K = np.array([[2400.0, 0, W / 2], [0, 2400.0, H / 2], [0, 0, 1]])
    return ae.Camera(K, R, -R @ centre, 0.0, size=(W, H)), base


def _front_points(s, cam, n=8):
    """Вершины верхнего скана, видимые камере, — разнесённые по ширине дуги (как точки на зубах)."""
    up = s._jaw_items()[0]
    v = apply(np.asarray(up["reg"].transform, float), up["scan"].vertices)
    centre = -cam.R.T @ cam.t
    view = cam.R.T @ np.array([0, 0, 1.0])
    near = v[(centre - v) @ -view < np.quantile((centre - v) @ -view, 0.15)]  # ближние к камере
    x = cam.axis((1, 0))
    order = np.argsort(near @ x)
    return near[order[np.linspace(0, len(order) - 1, n).round().astype(int)]]


def _paired(files, tmp_path, kind="smile"):
    s = _session(files, tmp_path)
    s.aesthetic_add_photo(kind, str(files / f"{kind}.jpg"))
    s.aesthetic_set(kind, distance_mm=300)  # камера сцены — в 300 мм от резцов
    cam, base = _camera(s)
    for p in _front_points(s, cam):
        s.aesthetic_pair(kind, cam.project(p[None])[0], p)
    return s, cam, base


def test_pairs_fit_camera_and_overlay(files, tmp_path):
    s = _session(files, tmp_path)
    with pytest.raises(ValueError, match="не загружено"):
        s.aesthetic_pair("smile", (10, 10), (0, 0, 0))
    info = s.aesthetic_add_photo("smile", str(files / "smile.jpg"))
    smile = info["photos"]["smile"]
    assert smile["size"] == [W, H] and not smile["focal"] and smile["need"] == 2 and smile["camera"] is None
    assert smile["distance_mm"] == 1000 and not smile["distance_set"]  # по умолчанию — как exocad: с метра
    info = s.aesthetic_set("smile", distance_mm=300)
    assert info["photos"]["smile"]["distance_mm"] == 300 and info["photos"]["smile"]["distance_set"]
    with pytest.raises(ValueError, match="расстояние"):
        s.aesthetic_set("smile", distance_mm=20)
    assert info["photos"]["retracted"] is None and info["upper"]
    cam, _base = _camera(s)
    points = _front_points(s, cam)
    with pytest.raises(ValueError, match="не на скане"):
        s.aesthetic_pair("smile", (100, 100), points[0] + [0, 0, 30.0])
    with pytest.raises(ValueError, match="вне фото"):
        s.aesthetic_pair("smile", (W + 5, 10), points[0])
    info = s.aesthetic_pair("smile", cam.project(points[:1])[0], points[0])
    assert info["photos"]["smile"]["camera"] is None and "1 из 2" in info["photos"]["smile"]["fit_error"]
    for p in points[1:]:
        info = s.aesthetic_pair("smile", cam.project(p[None])[0], p)
    smile = info["photos"]["smile"]
    assert smile["camera"]["residual_px"] < 1.0 and not smile["camera"]["mirrored"] and not smile["camera"]["check"]
    assert len(smile["pairs"]) == 8 and np.allclose(smile["pairs"][0]["point"], points[0], atol=1e-3)
    fitted = s._camera("smile")[0]
    assert np.allclose(fitted.project(points), cam.project(points), atol=1.0)
    overlay = smile["overlay"]  # номеров зубов (сегментации КТ) нет — видимые точки скана, 3D: окно проецирует само
    assert not overlay["scan"] and len(overlay["points"]) > 100
    uv = fitted.project(np.array(overlay["points"]))
    assert ((uv[:, 0] > -W) & (uv[:, 0] < 2 * W)).mean() > 0.9
    c = smile["camera"]
    assert c["distance_mm"] == pytest.approx(300, abs=1) and len(c["K"]) == 3 and not c["fixed"]
    assert smile["auto"] and not smile["auto_ready"] and not smile["auto_pending"]  # SAM нет — доводки нет
    # ручная поправка, как в exocad: поворот скана вокруг резцов, сдвиг — камера запоминается
    before = fitted.project(points)
    info = s.aesthetic_adjust("smile", {"yaw": 2.0, "dx": 5.0})
    moved = s._camera("smile")[0]
    assert info["photos"]["smile"]["camera"]["fixed"] and not np.allclose(moved.project(points), before, atol=1.0)
    info = s.aesthetic_reset_camera("smile")
    assert not info["photos"]["smile"]["camera"]["fixed"]
    assert np.allclose(s._camera("smile")[0].project(points), before, atol=0.01)
    with pytest.raises(ValueError, match="SAM"):
        s.aesthetic_refine("smile")
    info = s.aesthetic_unpair("smile", 0)
    assert len(info["photos"]["smile"]["pairs"]) == 7
    assert s.aesthetic_unpair("smile")["photos"]["smile"]["camera"] is None


def test_auto_alignment_without_pairs(files, tmp_path, monkeypatch):
    """Без пар точек — автоматически, как «Automatic alignment» exocad Smile Creator: грубо по силуэту зубов, затем
    доводка; считает фоновая задача, окно не ждёт; пары — по ним, убрали пары — снова автоматическое, без пересчёта
    (нейросети подменены: проверяются порядок, кэш и переходы)."""
    from casedesigner import photo_teeth as pt

    s = _session(files, tmp_path)
    s.aesthetic_add_photo("smile", str(files / "smile.jpg"))
    s.aesthetic_set("smile", distance_mm=300)
    cam, _base = _camera(s)
    n = len(s._upper_ct()[2]["v"])
    calls = []
    monkeypatch.setattr(s, "_sam", lambda: object())
    monkeypatch.setattr(s, "_aes_teeth", lambda compute=True: (np.full(n, 11, np.int32), np.zeros((n, 3))))
    monkeypatch.setattr(pt, "coarse_camera", lambda *a, **k: calls.append("грубо") or (cam, 0.7))

    def align(sam, image, camera, *a, **k):
        calls.append("доводка")
        return pt.Refined(camera, {11: 0.5, 21: 0.5}, {11: 0.8, 21: 0.8}, 1.5, [11, 21], {}, (0, 0, 10, 10), 1)
    monkeypatch.setattr(pt, "align", align)
    smile = s.aesthetics_info()["photos"]["smile"]
    assert smile["camera"] is None and smile["auto_pending"] and calls == []
    smile = s.aesthetic_auto("smile")["photos"]["smile"]
    assert calls == ["грубо", "доводка", "доводка"]  # вторая доводка не лучше первой — берётся первая
    assert smile["camera"] and smile["free"] and not smile["auto_pending"] and smile["auto_info"]["free"]
    points = _front_points(s, cam)
    assert np.allclose(s._camera("smile")[0].project(points), cam.project(points), atol=1e-6)
    s.aesthetic_auto("smile")
    assert calls.count("грубо") == 1  # посчитано — не заново
    for p in points:
        s.aesthetic_pair("smile", cam.project(p[None])[0], p)
    smile = s.aesthetics_info()["photos"]["smile"]
    assert not smile["free"] and smile["camera"]["residual_px"] < 1.0
    smile = s.aesthetic_unpair("smile")["photos"]["smile"]
    assert smile["free"] and smile["camera"] and calls.count("грубо") == 1
    s.aesthetic_set("smile", auto=False)  # доводка выключена — без пар камеры нет
    assert s.aesthetics_info()["photos"]["smile"]["camera"] is None


def test_camera_from_photo_sets_distance_and_horizon(files, tmp_path, monkeypatch):
    """Камера по снимку (GeoCalib, MoGe-2) — фоновой задачей, один раз: расстояние до рта — по снимку, пока врач
    не задал своё; горизонт — по вертикали снимка по кнопке; фокус по зубам сверяется с фокусом снимка; камера по
    снимку хранится в кейсе (нейросети подменены)."""
    from casedesigner import photo_camera as pc

    s = _session(files, tmp_path)
    s.aesthetic_add_photo("smile", str(files / "smile.jpg"))
    calls = []
    monkeypatch.setattr(s, "_lens_ready", lambda: True)
    monkeypatch.setattr(pc, "estimate", lambda image, models, point=None: calls.append(point) or pc.Lens(
        size=(W, H), focal_px=2400.0, roll_deg=-3.0, pitch_deg=1.0, distance_mm=620.0))
    p = s.aesthetics_info()["photos"]["smile"]
    assert p["auto_pending"] and p["lens"] is None and p["distance_source"] == "как exocad"
    p = s.aesthetic_auto("smile")["photos"]["smile"]
    assert len(calls) == 1 and p["distance_mm"] == 620 and p["distance_source"] == "по снимку (MoGe-2)"
    assert not p["auto_pending"] and p["lens"]["horizon_deg"] == 3.0 and p["lens"]["focal_35mm"] > 0
    s.aesthetic_auto("smile")
    assert len(calls) == 1  # посчитано — не заново
    p = s.aesthetic_set("smile", horizon_lens=True)["photos"]["smile"]
    assert p["horizon_deg"] == 3.0 and p["horizon_source"] == "вертикаль"
    p = s.aesthetic_set("smile", distance_mm=300)["photos"]["smile"]
    assert p["distance_mm"] == 300 and p["distance_source"] == "вручную"
    assert s.aesthetic_set("smile", distance_mm=0)["photos"]["smile"]["distance_mm"] == 620
    cam, _base = _camera(s)  # сцена — в 300 мм, фокус 2400 px: при 620 мм фокус по зубам вдвое больше снимочного
    for pnt in _front_points(s, cam):
        s.aesthetic_pair("smile", cam.project(pnt[None])[0], pnt)
    c = s.aesthetics_info()["photos"]["smile"]["camera"]
    assert c["lens_check"] and c["lens_share"] > 0.5
    s.aesthetic_set("smile", distance_mm=300)
    assert not s.aesthetics_info()["photos"]["smile"]["camera"]["lens_check"]
    path = s.save_case(str(tmp_path / "case.ccdcase"))["path"]
    other = Session(memory_path=str(tmp_path / "memory2.jsonl"))
    other.open_case(path)
    assert other.aesthetics_info()["photos"]["smile"]["lens"]["distance_mm"] == 620


def test_mouth_only_on_open_smile():
    from casedesigner.app.steps.aesthetics import AestheticsSteps

    lips = {"outer": [[0, 0], [100, 0], [50, 30]], "inner": [[10, 10], [90, 10], [50, 20]]}
    assert AestheticsSteps._mouth("smile", {"lips": lips}) is not None
    assert AestheticsSteps._mouth("retracted", {"lips": lips}) is None  # с ретрактором губы растянуты
    closed = {"outer": lips["outer"], "inner": [[10, 10], [90, 10], [50, 12]]}
    assert AestheticsSteps._mouth("smile", {"lips": closed}) is None and AestheticsSteps._mouth("smile", {}) is None


def test_horizon_midline_and_face_widths(files, tmp_path):
    s, cam, base = _paired(files, tmp_path)
    inc = s._upper_incisal(base)
    # горизонт — край кадра: функциональная горизонталь наклонена к нему на крен камеры
    a = s.aesthetics_info()["photos"]["smile"]["aesthetic"]
    assert a is not None and abs(a["roll_deg"]) == pytest.approx(ROLL_DEG, abs=0.6)
    # горизонт — по линии, параллельной функциональной горизонтали: наклона нет
    line = cam.project(np.array([inc - 20 * base[0, :3], inc + 20 * base[0, :3]]))
    info = s.aesthetic_set("smile", horizon_deg=ae.horizon_from_points(*line))
    assert abs(info["photos"]["smile"]["aesthetic"]["roll_deg"]) < 0.3
    # зрачки — на глубине FACE_DEPTH_MM за резцами, межзрачковое 62 мм
    view = cam.R.T @ np.array([0, 0, 1.0])
    x = np.cross(view, base[2, :3])
    x /= np.linalg.norm(x)
    eyes = [inc + ae.FACE_DEPTH_MM["pupil"] * view + s_ * 31.0 * x + 45 * base[2, :3] for s_ in (1, -1)]
    uv = cam.project(np.array(eyes))
    info = s.aesthetic_set("smile", face={"pupil_right": uv[0].tolist(), "pupil_left": uv[1].tolist()})
    smile = info["photos"]["smile"]
    assert smile["widths"]["межзрачковое"] == pytest.approx(62.0, abs=1.0)
    guide = {g["name"]: g for g in smile["guide"]}
    assert guide["ширина центрального резца (по зрачкам)"]["mm"] == pytest.approx(62.0 / 6.6, abs=0.3)
    assert "Cesario" in guide["ширина центрального резца (по зрачкам)"]["why"]
    info = s.aesthetic_set("smile", midline_px=[cam.project(inc[None] + 40 * base[2, :3])[0].tolist(),
                                                cam.project(inc[None] - 20 * base[2, :3])[0].tolist()])
    assert len(info["photos"]["smile"]["midline_px"]) == 2
    assert info["photos"]["smile"]["aesthetic"]["midline_offset_mm"] is not None
    assert s.aesthetic_set("smile", face={"pupil_left": None})["photos"]["smile"]["widths"] == {}
    with pytest.raises(ValueError, match="нет такой точки лица"):
        s.aesthetic_set("smile", face={"nose": [1, 1]})
    with pytest.raises(ValueError, match="горизонт"):
        s.aesthetic_set("smile", horizon_deg=60)


def test_mount_by_photo_without_face_bones(files, tmp_path, monkeypatch):
    """Шаг «Артикулятор»: совмещённое фото с улыбкой — эстетические плоскости доступны и без костей лица,
    в монтаж уходит photo_aesthetic (наклон по горизонту фото)."""
    s, cam, base = _paired(files, tmp_path)
    hinge = {"right": np.array([50.0, 0, 0]), "left": np.array([-50.0, 0, 0])}
    s._articulation_ready = True
    monkeypatch.setattr(s, "_mount_condyles", lambda: (hinge, hinge, None))
    monkeypatch.setattr(s, "_face_points", lambda: None)
    monkeypatch.setattr(s, "_condyle_signature", lambda: None)
    calls = {}
    monkeypatch.setattr(s.jaw, "mount", lambda method, **kw: calls.update(method=method, **kw) or {})
    s.articulation_mount("ct", "aesthetic_hinge")
    assert calls["plane"] == "aesthetic_hinge" and calls["photo_aesthetic"] is not None
    a = calls["photo_aesthetic"](base)
    assert isinstance(a, ae.Aesthetic) and abs(a.roll_deg) == pytest.approx(ROLL_DEG, abs=0.6)
    s.aesthetic_remove_photo("smile")
    with pytest.raises(ValueError, match="недоступна"):
        s.articulation_mount("ct", "aesthetic_hinge")  # без фото и костей лица — как раньше


def test_export_photos_and_case_file(files, tmp_path):
    s, _cam, _base = _paired(files, tmp_path)
    s.aesthetic_add_photo("retracted", str(files / "retracted.jpg"))
    photos = s._export_photos()
    assert [p["kind"] for p in photos] == ["smile"] and photos[0]["image"].shape == (H, W, 3)
    s.aesthetic_set("smile", horizon_deg=2.5, midline_px=[[600, 100], [602, 700]])
    path = s.save_case(str(tmp_path / "case.ccdcase"))["path"]
    other = Session(memory_path=str(tmp_path / "memory2.jsonl"))
    other.open_case(path)
    info = other.aesthetics_info()
    assert info["photos"]["smile"]["horizon_deg"] == 2.5 and len(info["photos"]["smile"]["pairs"]) == 8
    assert info["photos"]["smile"]["camera"]["residual_px"] < 1.0 and info["photos"]["retracted"]["pairs"] == []
    assert info["missing"] == []
    # фото переместили — кейс открывается, фото помечено
    (files / "retracted.jpg").rename(files / "retracted-moved.jpg")
    try:
        other.open_case(path)
        assert other.aesthetics_info()["missing"] == ["с ретрактором"]
    finally:
        (files / "retracted-moved.jpg").rename(files / "retracted.jpg")
    # входной контроль видит фото
    by = {f.title: f.level for f in s._intake_findings() if f.section == "Фото"}
    assert by == {"Фото с улыбкой": "ok", "Фото с ретрактором": "ok"}


def test_incognito_blurs_face_keeps_mouth(files, tmp_path):
    s, _cam, _base = _paired(files, tmp_path)
    face = np.zeros((H, W, 3), np.uint8)
    face[::2, ::2] = 255  # мелкая сетка: размытие делает её ровной серой
    s.photos["smile"]["photo"].image = face
    import io

    sharp = np.asarray(Image.open(io.BytesIO(s.aesthetic_image("smile"))).convert("L"), float)
    s.settings["incognito"] = True
    blurred = np.asarray(Image.open(io.BytesIO(s.aesthetic_image("smile"))).convert("L"), float)
    px = np.array([p["px"] for p in s.photos["smile"]["pairs"]])
    u, v = px.mean(0).astype(int)
    assert blurred[5:60, 5:60].std() < 0.2 * sharp[5:60, 5:60].std()  # угол кадра (глаза, лоб) — размыт
    assert blurred[v - 5:v + 5, u - 5:u + 5].std() > 0.5 * sharp[v - 5:v + 5, u - 5:u + 5].std()  # рот — резкий


def test_http_routes(files, tmp_path):
    s = _session(files, tmp_path)
    httpd = app_server.serve(s)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api"

    def post(path, body):
        req = urllib.request.Request(f"{base}/{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    try:
        info = post("aesthetics/photo", {"kind": "smile", "path": str(files / "smile.jpg")})
        assert info["photos"]["smile"]["size"] == [W, H]
        cam, _b = _camera(s)
        p = _front_points(s, cam)[0]
        info = post("aesthetics/pair", {"kind": "smile", "px": cam.project(p[None])[0].tolist(), "point": p.tolist()})
        assert len(info["photos"]["smile"]["pairs"]) == 1
        info = post("aesthetics/set", {"kind": "smile", "horizon_deg": 1.5})
        assert info["photos"]["smile"]["horizon_deg"] == 1.5
        assert post("aesthetics/pair/remove", {"kind": "smile", "index": 0})["photos"]["smile"]["pairs"] == []
        with urllib.request.urlopen(f"{base}/aesthetics/smile/image") as r:
            assert r.headers["Content-Type"] == "image/jpeg" and r.read(2) == b"\xff\xd8"
        with urllib.request.urlopen(f"{base}/aesthetics") as r:
            assert json.loads(r.read())["photos"]["smile"]["horizon_deg"] == 1.5
        with pytest.raises(urllib.error.HTTPError) as err:
            post("aesthetics/photo", {"kind": "profile", "path": str(files / "smile.jpg")})
        assert "нет такого фото" in json.loads(err.value.read())["error"]
        assert post("aesthetics/photo/remove", {"kind": "smile"})["photos"]["smile"] is None
    finally:
        httpd.shutdown()
        httpd.server_close()


FACE_MODELS = os.environ.get("CCD_MODELS", os.path.join(os.path.dirname(__file__), "..", "models"))


@pytest.mark.skipif(not os.path.isfile(os.path.join(FACE_MODELS, "face_landmarker.task")), reason="нет модели MediaPipe")
def test_face_found_automatically_like_smile_creator(files, tmp_path):
    """Лицо — само при загрузке фото (MediaPipe), как в exocad Smile Creator: точки лица, средняя линия, губы;
    горизонт — по линии глаз; «0» — по кадру; «Найти лицо» — заново."""
    pytest.importorskip("mediapipe")
    from skimage import data

    Image.fromarray(data.astronaut()).save(tmp_path / "portrait.jpg")  # фото NASA, общественное достояние
    s = _session(files, tmp_path)
    s.models_dir = FACE_MODELS
    info = s.aesthetic_add_photo("smile", str(tmp_path / "portrait.jpg"))
    p = info["photos"]["smile"]
    assert p["face_ready"] and p["face_auto"] and p["horizon_source"] == "глаза"
    assert {"pupil_right", "pupil_left", "endocanthion_right", "ala_left"} <= set(p["face"])
    assert len(p["midline_px"]) >= 5 and len(p["lips"]["inner"]) == 20 and abs(p["horizon_deg"]) < 10
    p = s.aesthetic_set("smile", horizon_deg=0)["photos"]["smile"]
    assert p["horizon_source"] == "кадр"
    s.aesthetic_set("smile", face={"pupil_left": None})
    p = s.aesthetic_face("smile")["photos"]["smile"]  # заново — всё по MediaPipe
    assert p["horizon_source"] == "глаза" and "pupil_left" in p["face"]
