"""Контракт выбора плоскости монтажа (casedesigner/mount_planes.py, docs/handoff/mount-plane.md).

Система кейса в тестах — анатомическая: x вправо пациента, y вперёд, z вверх; окклюзионная плоскость z ≈ −30
горизонтальна, мыщелки — в (±50, 0, 0). Тесты с xfail(strict=True) — задачи ведомого.
"""

import inspect
import json
import re
from pathlib import Path
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from casedesigner import exocad_webview as ew
from casedesigner.guidance import CONDYLE_RADIUS_MM
from casedesigner import mount_planes as mp
from casedesigner.jawcase import JawCase, Mesh
from casedesigner.register import apply, axis_angle

from test_exocad_webview import arch_ribbon
from test_motion import temporal_bone

CO = {"right": np.array([50.0, 0.0, 0.0]), "left": np.array([-50.0, 0.0, 0.0])}
ALPHA = {"right": 40.0, "left": 30.0}  # бугорки: ССП к окклюзионной плоскости


def scans():
    upper = arch_ribbon(-29.9)
    top, base = arch_ribbon(-30.0), arch_ribbon(-38.0)
    lower = (np.vstack([top[0], base[0]]), np.vstack([top[1], base[1] + len(top[0])]))
    return upper, lower


def skull():
    bones = [temporal_bone(ALPHA[s], x=(CO[s][0] - 8, CO[s][0] + 8)) for s in ("right", "left")]
    return SimpleNamespace(vertices=np.vstack([b[0] for b in bones]),
                           faces=np.vstack([bones[0][1], bones[1][1] + len(bones[0][0])]))


def landmarks(pitch_deg=10.0):
    """Ориентиры черепа: франкфуртская передом выше окклюзионной на pitch_deg; Кемпера ей параллельна."""
    t = np.tan(np.radians(pitch_deg))

    def fh(x, y):
        return [x, y, 5.0 + (y + 10.0) * t]

    return {"Po_R": fh(60, -10), "Po_L": fh(-60, -10), "Or_R": fh(33, 70), "Or_L": fh(-33, 70),
            "Tr_R": [65, -5, -10.0], "Tr_L": [-65, -5, -10.0], "Al_R": [15, 95, -10.0], "Al_L": [-15, 95, -10.0],
            "HN_R": [20, 30, -25.0], "HN_L": [-20, 30, -25.0], "IP": [0, 85, -25.0]}


def mount(plane="occlusal", condyles=CO, lm=None, bone=True):
    upper, lower = scans()
    return ew.mount(upper[0], lower[0], None, skull() if bone else None, condyles=condyles,
                    plane=plane, landmarks=landmarks() if lm is None else lm)


def test_occlusal_is_default_and_unchanged():
    upper, lower = scans()
    a0, s0, _e, _n = ew.mount(upper[0], lower[0], None, skull(), condyles=CO)
    a1, s1, _e, notes = mount("occlusal")
    assert np.allclose(a0.frame, a1.frame) and not notes
    assert np.allclose(a1.frame[2, :3], [0, 0, 1], atol=1e-3)  # горизонталь — окклюзионная
    for side in ("right", "left"):
        assert s1.get("sagittal", side) == pytest.approx(ALPHA[side], abs=0.7)


def test_frankfurt_mount_and_angles_from_plane():
    """Франкфуртская передом выше окклюзионной на 10°: путь мыщелка к ней круче на те же 10°."""
    a, s, _e, notes = mount("frankfurt")
    lm = landmarks()
    po, orb = np.array(lm["Po_R"]), np.array(lm["Or_R"])
    assert apply(a.frame, po[None])[0][2] == pytest.approx(apply(a.frame, orb[None])[0][2], abs=1e-6)  # горизонталь
    # шарнир — центры головок: середина между Co — на радиус головки выше начала системы
    assert apply(a.frame, ((CO["right"] + CO["left"]) / 2)[None])[0] == pytest.approx([0, 0, CONDYLE_RADIUS_MM], abs=1e-9)
    for side in ("right", "left"):
        assert s.get("sagittal", side) == pytest.approx(ALPHA[side] + 10.0, abs=0.7)
    assert any("передом вверх на 10.0°" in n for n in notes)
    assert "франкфуртская" in a.source


def test_axis_orbital_hinge_through_both_condyles():
    """Мыщелки на разной высоте: в оси-орбитальной системе оба на оси x (шарнир артикулятора проходит через них)."""
    tilted = {"right": np.array([50.0, 1.0, 3.0]), "left": np.array([-50.0, -1.0, -3.0])}
    occl, *_ = mount("occlusal", condyles=tilted, bone=False)
    axis, *_ = mount("axis_orbital", condyles=tilted, bone=False)
    half = np.linalg.norm(tilted["right"] - tilted["left"]) / 2
    r = CONDYLE_RADIUS_MM  # Co — на радиус головки над шарниром
    assert apply(axis.frame, tilted["right"][None])[0] == pytest.approx([half, 0, r], abs=1e-6)
    assert apply(axis.frame, tilted["left"][None])[0] == pytest.approx([-half, 0, r], abs=1e-6)
    for side in ("right", "left"):  # центры головок — на оси x
        assert apply(axis.frame, axis.points[f"condyle_{side}"][None])[0][1:] == pytest.approx([0, 0], abs=1e-6)
    # у окклюзионной шарнир — горизонтально через середину: центр головки от оси на hypot(1, 3)
    assert np.hypot(*apply(occl.frame, occl.points["condyle_right"][None])[0][1:]) == pytest.approx(np.hypot(1, 3),
                                                                                                     abs=0.05)
    # орбиталь — в горизонтальной плоскости шарнира (через центры головок)
    lm = landmarks()
    orb = (np.array(lm["Or_R"]) + np.array(lm["Or_L"])) / 2
    assert apply(axis.frame, orb[None])[0][2] == pytest.approx(0, abs=1e-6)


def test_hinge_at_head_centres_below_co():
    """Шарнир — центры головок на радиус ниже Co по вертикали монтажа; ССП — по бугорку от Co (не меняется)."""
    a, s, _e, _n = mount("frankfurt")
    up = a.frame[2, :3]
    for side in ("right", "left"):
        assert a.points[f"condyle_{side}"] == pytest.approx(CO[side] - CONDYLE_RADIUS_MM * up, abs=1e-9)
    assert np.linalg.norm(a.points["condyle_right"] - a.points["condyle_left"]) == pytest.approx(100.0)
    assert "центры головок" in a.source


def test_camper_and_hip_build_from_their_points():
    camper, *_ = mount("camper", bone=False)
    assert np.allclose(camper.frame[2, :3], [0, 0, 1], atol=1e-6)  # точки Кемпера лежат в z = −10
    hip, *_ = mount("hip", bone=False)
    assert hip.frame[2, :3] @ np.array([0, 0, 1.0]) > 0.99


def test_missing_points_and_availability():
    info = mp.available({"Or_L": [-33, 70, 0]}, CO)
    assert set(info) == set(mp.PLANES) and info["occlusal"]["available"]
    assert info["axis_orbital"]["available"]  # мыщелки — шарнир, орбиталь — хотя бы одна
    assert not info["frankfurt"]["available"] and "Порион правый" in info["frankfurt"]["missing"]
    assert not mp.available({}, CO)["axis_orbital"]["available"]
    with pytest.raises(ValueError, match="Козелок"):
        mount("camper", lm={}, bone=False)
    with pytest.raises(ValueError, match="нет такой плоскости"):
        mount("bonwill", bone=False)


def test_jawcase_plane_in_state_undo_and_file(tmp_path):
    upper, lower = scans()
    jc = JawCase()
    jc.set_scans(Mesh(*upper, "верх"), Mesh(*lower, "низ"))
    jc.skull = Mesh(skull().vertices, skull().faces, "череп")
    with pytest.raises(ValueError, match="только для монтажа по КТ"):
        jc.mount("average", plane="frankfurt", landmarks=landmarks())
    jc.mount("ct", condyles=CO, ct_condyles=CO)
    state = jc.mount("ct", condyles=CO, ct_condyles=CO, plane="frankfurt", landmarks=landmarks())
    assert state["mounting"]["plane"] == {"key": "frankfurt", "name": mp.PLANES["frankfurt"]["name"]}
    assert jc.settings.get("sagittal", "right") == pytest.approx(ALPHA["right"] + 10.0, abs=0.7)
    path = str(tmp_path / "case.npz")
    jc.save(path)
    assert JawCase.load(path).plane == "frankfurt"
    jc.undo()
    assert jc.plane == "occlusal" and jc.state()["mounting"]["plane"]["key"] == "occlusal"


def face_cloud(roll_deg=3.0, n=1500, seed=0):
    """«Лицевой скелет», зеркально симметричный относительно плоскости, повёрнутой на roll_deg вокруг оси «вперёд»."""
    rng = np.random.default_rng(seed)
    half = np.c_[rng.uniform(5, 60, n), rng.uniform(-20, 100, n), rng.uniform(-40, 40, n)]
    sym = np.vstack([half, half * [-1, 1, 1]]) + [0.0, 0.0, 10.0]
    R = axis_angle(np.array([0, 1.0, 0]), np.radians(roll_deg))
    return sym @ R.T


def test_aesthetic_plane_from_face_symmetry():
    """Лицо симметрично плоскости с креном 3° к окклюзионной: модели — по ней; ССП — от базовой горизонтали."""
    upper, lower = scans()
    jc = JawCase()
    jc.set_scans(Mesh(*upper, "верх"), Mesh(*lower, "низ"))
    jc.skull = Mesh(skull().vertices, skull().faces, "череп")
    st = jc.mount("ct", condyles=CO, ct_condyles=CO, plane="aesthetic", face_points=face_cloud(3.0))
    x = jc.anatomy.frame[0, :3]
    assert abs(np.degrees(np.arcsin(x[2]))) == pytest.approx(3.0, abs=0.3)  # «вправо» с креном лица
    assert st["mounting"]["plane"]["key"] == "aesthetic" and any("эстетическая" in n for n in st["notes"])
    assert jc.settings.frame is not None  # углы — от базовой (окклюзионной) горизонтали
    for side in ("right", "left"):
        assert jc.settings.get("sagittal", side) == pytest.approx(ALPHA[side], abs=0.7)
    # с точками франкфуртской базовая — она
    jc.mount("ct", condyles=CO, ct_condyles=CO, plane="aesthetic", landmarks=landmarks(), face_points=face_cloud(3.0))
    assert "франкфуртская" in jc.anatomy.source
    assert jc.settings.get("sagittal", "right") == pytest.approx(ALPHA["right"] + 10.0, abs=0.7)
    with pytest.raises(ValueError, match="кости лица"):
        JawCase.mount(_bare(), "ct", condyles=CO, plane="aesthetic")


def _bare():
    upper, lower = scans()
    jc = JawCase()
    jc.set_scans(Mesh(*upper, "верх"), Mesh(*lower, "низ"))
    return jc


def test_default_plane_resolves_with_fallback():
    """Плоскость по умолчанию (настройка программы): нет точек — следующая по цепочке, в конце окклюзионная."""
    lm = landmarks()
    assert mp.resolve(None, {}, CO) == ("occlusal", None)
    assert mp.resolve("frankfurt", lm, CO) == ("frankfurt", None)
    kind, note = mp.resolve("axis_orbital", {}, CO)
    assert kind == "occlusal" and "Оси-орбитальная" in note and "Франкфуртская" in note
    assert mp.resolve("aesthetic", lm, CO, face=True) == ("aesthetic", None)
    kind, note = mp.resolve("aesthetic", lm, CO, face=False)
    assert kind == "frankfurt" and "кости лица" in note
    assert mp.base_for_aesthetic(lm, CO) == "frankfurt" and mp.base_for_aesthetic({}, CO) == "occlusal"
    with pytest.raises(ValueError, match="нет такой плоскости"):
        mp.resolve("bonwill", lm, CO)


def test_aesthetic_hinge_rotation_keeps_condyle_axis():
    """Ось «вправо» — линия мыщелков (шарнир через оба), «вверх» — эстетическое поперёк неё."""
    tilted = {"right": np.array([50.0, 1.0, 3.0]), "left": np.array([-50.0, -1.0, -3.0])}
    face_up = axis_angle(np.array([1.0, 0, 0]), np.radians(-8)) @ np.array([0, 0, 1.0])  # лицо передом вниз на 8°
    R = mp.rotation("aesthetic_hinge", [0, 0, 1], [0, 1, 0], condyles=tilted, face_up=face_up)
    line = (tilted["right"] - tilted["left"]) / np.linalg.norm(tilted["right"] - tilted["left"])
    assert R[0] == pytest.approx(line)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12) and np.linalg.det(R) == pytest.approx(1)
    up = face_up - (face_up @ line) * line
    assert R[2] == pytest.approx(up / np.linalg.norm(up))
    # по направлению: центры головок (Co на радиус ниже) дают ту же систему
    centres = {s: p - CONDYLE_RADIUS_MM * R[2] for s, p in tilted.items()}
    assert mp.rotation("aesthetic_hinge", [0, 0, 1], [0, 1, 0], condyles=centres, face_up=face_up) == pytest.approx(R)
    with pytest.raises(ValueError, match="мыщелкам и костям лица"):
        mp.rotation("aesthetic_hinge", [0, 0, 1], [0, 1, 0], condyles=tilted)
    with pytest.raises(ValueError, match="перепутаны"):
        mp.rotation("aesthetic_hinge", [0, 0, 1], [0, 1, 0], condyles={"right": tilted["left"], "left": tilted["right"]},
                    face_up=face_up)
    info = mp.available({}, CO, face=True)["aesthetic_hinge"]
    assert info["available"]
    assert mp.missing("aesthetic_hinge", {}, None, face=False) == ["кости лица (сегментация черепа)",
                                                                   "Мыщелок правый", "Мыщелок левый"]
    kind, note = mp.resolve("aesthetic_hinge", landmarks(), None, face=True)
    assert kind == "aesthetic" and "Мыщелок правый" in note


def test_aesthetic_hinge_mount_hinge_on_condyles_pitch_from_face(tmp_path):
    """Лицо с креном 3°, мыщелки ровно: шарнир на мыщелках, наклон — от франкфуртской (по эстетике), остаток — 3°."""
    upper, lower = scans()
    jc = JawCase()
    jc.set_scans(Mesh(*upper, "верх"), Mesh(*lower, "низ"))
    jc.skull = Mesh(skull().vertices, skull().faces, "череп")
    jc.mount("ct", condyles=CO, ct_condyles=CO)
    assert jc.aesthetic_axes is None
    st = jc.mount("ct", condyles=CO, ct_condyles=CO, plane="aesthetic_hinge", landmarks=landmarks(),
                  face_points=face_cloud(3.0))
    F = jc.anatomy.frame
    for side in ("right", "left"):  # центры головок — на оси x: шарнир артикулятора через оба мыщелка
        assert apply(F, jc.anatomy.points[f"condyle_{side}"][None])[0][1:] == pytest.approx([0, 0], abs=1e-6)
    assert st["mounting"]["plane"]["key"] == "aesthetic_hinge"
    note = next(n for n in st["notes"] if "отличаются на" in n)
    assert "Вид Smile Design" in note
    assert float(re.search(r"отличаются на ([\d.]+)°", note).group(1)) == pytest.approx(3.0, abs=0.3)
    assert "наклон — по эстетике" in jc.anatomy.source
    for side in ("right", "left"):  # горизонталь — эстетическая с наклоном франкфуртской (10° к окклюзионной)
        assert jc.settings.get("sagittal", side) == pytest.approx(ALPHA[side] + 10.0, abs=0.7)
    E = jc.aesthetic_axes
    assert E @ E.T == pytest.approx(np.eye(3), abs=1e-9)
    assert abs(np.degrees(np.arcsin(E[0][2]))) == pytest.approx(3.0, abs=0.3)  # «вправо» лица — с креном
    assert F[2, :3] @ E[2] == pytest.approx(np.cos(np.radians(3.0)), abs=2e-3)  # «вверх» — поперёк оси мыщелков
    path = str(tmp_path / "case.npz")
    jc.save(path)
    loaded = JawCase.load(path)
    assert loaded.plane == "aesthetic_hinge" and loaded.aesthetic_axes == pytest.approx(E)
    jc.undo()
    assert jc.aesthetic_axes is None and jc.plane == "occlusal"
    # мыщелки на разной высоте: шарнир всё равно через оба
    tilted = {"right": np.array([50.0, 0.0, 2.0]), "left": np.array([-50.0, 0.0, -2.0])}
    bare = _bare()
    bare.mount("ct", condyles=tilted, ct_condyles=tilted, plane="aesthetic_hinge", face_points=face_cloud(0.0))
    for side in ("right", "left"):
        p = apply(bare.anatomy.frame, bare.anatomy.points[f"condyle_{side}"][None])[0]
        assert p[1:] == pytest.approx([0, 0], abs=1e-6)
    with pytest.raises(ValueError, match="кости лица"):
        _bare().mount("ct", condyles=CO, plane="aesthetic_hinge")


# --- задачи ведомого ------------------------------------------------------------------------------------

def test_session_and_api_contract(tmp_path):
    from casedesigner.app import server as app_server
    from casedesigner.app.session import Session
    from casedesigner.fusion import Scan

    assert "plane" in inspect.signature(Session.articulation_mount).parameters
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    assert s.settings["mount_plane"] == "aesthetic_hinge"  # рекомендация владельца для нового кейса
    assert s.articulation_planes()["selected"] == "aesthetic_hinge"
    s.set_articulation_plane("camper")
    assert s.articulation_planes()["selected"] == "camper"
    s.set_default_mount_plane("frankfurt")
    assert Session(memory_path=str(tmp_path / "memory.jsonl")).settings["mount_plane"] == "frankfurt"
    with pytest.raises(ValueError):
        s.set_default_mount_plane("bonwill")
    assert set(s.articulation_planes()) >= {"planes", "current", "selected", "default", "tilt"}
    src = inspect.getsource(app_server)
    assert '"planes"' in src and "mount_plane" in src
    ui = (Path(__file__).parents[1] / "casedesigner" / "app" / "web" / "app.js").read_text(encoding="utf-8")
    assert "data-mount-plane" in ui and "defaultMountPlane" in ui
    s.set_articulation_plane("hip")
    s.scans["phantom"] = {"scan": Scan("phantom", np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]),
                                          np.array([[0, 1, 2]])),
                          "path": "phantom.stl", "color": "#fff", "jaw": "upper", "reg": None,
                          "transform": None, "auto": None, "accepted": False, "role": "jaw"}
    saved = s.save_case(str(tmp_path / "plane.ccdcase"))
    reopened = Session(memory_path=str(tmp_path / "reopen-memory.jsonl"))
    reopened.open_case(saved["path"])
    assert reopened.articulation_planes()["selected"] == "hip"

    httpd = app_server.serve(s)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api/"
    try:
        with opener.open(base + "articulation/planes") as response:
            assert json.load(response)["selected"] == "hip"
        data = json.dumps({"plane": "camper"}).encode()
        req = urllib.request.Request(base + "articulation/planes", data=data,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req) as response:
            assert json.load(response)["selected"] == "camper"
        data = json.dumps({"plane": "unknown"}).encode()
        req = urllib.request.Request(base + "settings/mount_plane", data=data,
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(req)
        assert error.value.code == 400
        data = json.dumps({"method": "ct", "plane": "aesthetic_hinge"}).encode()
        req = urllib.request.Request(base + "articulation/mount", data=data,
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(req)
        assert error.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
