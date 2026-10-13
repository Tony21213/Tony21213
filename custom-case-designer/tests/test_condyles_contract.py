"""Контракт ручной проверки и поправки мыщелков в шаге «Артикулятор» (casedesigner/condyles.py, docs/handoff/archive/condyle-adjustment.md).

Тесты с xfail(strict=True) — задачи ведомого: после реализации они проходят, и пометку нужно снять.
"""

import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner import condyles as cond
from casedesigner.app.server import serve
from casedesigner.app.session import Session
from casedesigner.guidance import CONDYLE_RADIUS_MM
from casedesigner.jawcase import JawCase, Mesh
from casedesigner.register import apply, axis_angle, rigid

# Мыщелки в мм КТ (LPS: x — влево пациента, y — назад, z — вверх): правый — с меньшим x.
CO_R, CO_L = np.array([-49.0, 10.0, 0.0]), np.array([49.0, 10.0, 0.0])
STATE_KEYS = {"available", "bite", "points", "icd_mm", "checks", "ready", "confirmed", "applied", "segmentation"}
POINT_KEYS = {"key", "ct", "view", "source", "source_label", "confirmed"}


def bite_motion():
    """Нижняя челюсть с КТ в прикус сканов: поворот 3° вокруг оси сустава и сдвиг на 1 мм."""
    R = axis_angle(np.array([1.0, 0.0, 0.0]), np.radians(3.0))
    hinge = (CO_R + CO_L) / 2
    return rigid(np.eye(3), [0.0, -1.0, 0.0]) @ rigid(R, hinge - R @ hinge)


def mandible_mesh():
    """Синтетическая нижняя челюсть: тело и две ветви с головками мыщелков в CO_R, CO_L (верхушки)."""
    parts = [trimesh.creation.box((110, 20, 10), trimesh.transformations.translation_matrix((0, -40, -45)))]
    for co in (CO_R, CO_L):
        parts.append(trimesh.creation.box((8, 10, 40), trimesh.transformations.translation_matrix(co + (0, 0, -24))))
        parts.append(trimesh.creation.icosphere(subdivisions=3, radius=5.0).apply_translation(co + (0, 0, -5)))
    return trimesh.util.concatenate(parts)


@pytest.fixture
def session(jaw_ct, tmp_path):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.case = jaw_ct  # точки ставятся только при открытом КТ
    return s


# --- ядро: приоритет, координаты, проверки, состояние --------------------------------------------------

def test_priority_user_confirmed_over_automatic():
    landmarks = {"Co_R": CO_R, "Co_L": CO_L}
    points = cond.resolve(landmarks, suggested={"Co_R"}, sources={"Co_R": cond.SOURCE_ALI, "Co_L": cond.SOURCE_USER})
    assert points["right"].source == cond.SOURCE_ALI and not points["right"].confirmed
    assert points["left"].source == cond.SOURCE_USER and points["left"].confirmed
    # точки прежних кейсов без записи об источнике: предложенная — по нижней челюсти, остальные — врач
    old = cond.resolve(landmarks, suggested={"Co_L"}, sources={})
    assert old["right"].source == cond.SOURCE_USER and old["left"].source == cond.SOURCE_MANDIBLE
    empty = cond.resolve({}, set(), {})
    assert empty["right"].point is None and empty["right"].source is None


def test_points_follow_lower_jaw_into_scan_bite():
    M = bite_motion()
    view = cond.to_view(CO_R, M)
    assert np.allclose(view, apply(M, CO_R[None])[0])
    assert np.allclose(cond.from_view(view, M), CO_R)
    assert np.allclose(cond.to_view(CO_R, None), CO_R) and np.allclose(cond.from_view(CO_R, None), CO_R)


def test_icd_and_sides_checks():
    def run(r, l):
        pts = cond.resolve({"Co_R": r, "Co_L": l}, set(), {})
        found = {c.id: c for c in cond.checks(pts)}
        return pts, found

    pts, found = run(CO_R, CO_L)
    assert found["icd"].level == "ok" and found["sides"].level == "ok" and cond.ready(pts, list(found.values()))
    _, found = run(CO_R + (20, 0, 0), CO_L)  # межмыщелковое 78 мм — необычно
    assert found["icd"].level == "warn"
    pts, found = run(CO_R + (60, 0, 0), CO_L)  # 38 мм — точки не на мыщелках
    assert found["icd"].level == "error" and not cond.ready(pts, list(found.values()))
    pts, found = run(CO_L, CO_R)  # стороны перепутаны
    assert found["sides"].level == "error" and not cond.ready(pts, list(found.values()))
    pts, found = run(CO_R, None)
    assert found["icd"].level == "unknown" and not cond.ready(pts, list(found.values()))


def test_state_is_json_contract():
    pts = cond.resolve({"Co_R": CO_R, "Co_L": CO_L}, {"Co_R"}, {"Co_R": cond.SOURCE_MANDIBLE})
    st = cond.state(pts, cond.checks(pts), available=True, bite_motion=bite_motion(), applied=False,
                    mandible=True, skull=False)
    json.dumps(st)
    assert set(st) == STATE_KEYS and set(st["points"]) == {"right", "left"}
    assert all(set(p) == POINT_KEYS for p in st["points"].values())
    assert st["bite"] == "scan" and st["ready"] and not st["confirmed"] and not st["applied"]
    assert np.allclose(st["points"]["right"]["ct"], CO_R)
    assert np.allclose(st["points"]["right"]["view"], cond.to_view(CO_R, bite_motion()), atol=1e-3)
    assert {c["level"] for c in st["checks"]} <= set(cond.LEVELS)


# --- сессия: поставить, подтвердить, сбросить; приоритет над автоматикой; применение к монтажу ---------

def test_session_set_confirm_reset(session):
    st = session.articulation_condyles()
    assert st["available"] and st["points"]["right"]["ct"] is None and not st["segmentation"]["mandible"]
    session.structures["mandible"] = mandible_mesh()
    st = session.reset_condyle("both")  # предложение по нижней челюсти
    for side, co in (("right", CO_R), ("left", CO_L)):
        p = st["points"][side]
        assert p["source"] == cond.SOURCE_MANDIBLE and not p["confirmed"]
        assert np.linalg.norm(np.asarray(p["ct"]) - co) < 1.0
    session.bite_motion = bite_motion()
    moved = cond.to_view(CO_R + (0, 0, -1.0), session.bite_motion)
    st = session.set_condyle("right", moved.tolist(), space="view")  # поставлено в окне «Артикулятор»
    assert np.allclose(st["points"]["right"]["ct"], CO_R + (0, 0, -1.0), atol=1e-3)
    assert st["points"]["right"]["source"] == cond.SOURCE_USER and st["points"]["right"]["confirmed"]
    left = st["points"]["left"]["ct"]
    st = session.confirm_condyle("left")
    assert st["points"]["left"]["confirmed"] and st["points"]["left"]["ct"] == left  # подтверждение не сдвигает
    session.suggest_landmarks()  # автоматика не трогает точки врача
    assert np.allclose(session.landmarks["Co_R"], CO_R + (0, 0, -1.0))
    st = session.reset_condyle("right")
    assert st["points"]["right"]["source"] == cond.SOURCE_MANDIBLE and not st["points"]["right"]["confirmed"]
    with pytest.raises(ValueError):
        session.set_condyle("middle", CO_R.tolist())


def test_session_mount_points_follow_bite(session):
    session.landmarks.update({"Co_R": CO_R, "Co_L": CO_L})
    hinge, ct_points, why = session._mount_condyles()
    assert why == "" and np.allclose(hinge["right"], CO_R) and np.allclose(ct_points["left"], CO_L)
    session.bite_motion = bite_motion()  # прикус сканов: шарнир едет с нижней челюстью, бугорок — по КТ
    hinge, ct_points, _ = session._mount_condyles()
    assert np.allclose(hinge["left"], cond.to_view(CO_L, session.bite_motion)) and np.allclose(ct_points["left"], CO_L)
    session.landmarks.pop("Co_L")  # нет сегментации — второй точке неоткуда взяться: монтаж не по КТ
    hinge, ct_points, why = session._mount_condyles()
    assert hinge is None and ct_points is None and "left" in why


def test_mount_uses_given_condyles_without_mandible():
    up_v, up_f = phantom.make_scan("upper", resolution=0.6)
    lo_v, lo_f = phantom.make_scan("lower", resolution=0.6)
    case = JawCase()
    case.set_scans(Mesh(up_v, up_f, "верх"), Mesh(lo_v, lo_f, "низ"))
    right, left = np.array([-48.0, -70.0, 22.0]), np.array([50.0, -71.0, 24.0])
    case.mount("auto", condyles={"right": right, "left": left})  # кости нет, точки есть — монтаж по КТ
    assert case.mounting == "ct"
    # шарнир — центры головок: на радиус головки ниже точек Co по вертикали монтажа
    up = case.anatomy.frame[2, :3]
    r = CONDYLE_RADIUS_MM
    assert np.allclose(case.anatomy.points["condyle_right"], right - r * up)
    assert np.allclose(case.anatomy.points["condyle_left"], left - r * up)
    origin = apply(case.anatomy.frame, ((right + left) / 2)[None])[0]
    assert np.allclose(origin, [0, 0, r], atol=1e-9)  # начало системы монтажа — середина шарнирной оси


def test_case_file_keeps_sources(tmp_path):
    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(tmp_path / "ct.nii.gz"))
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(tmp_path / "ct.nii.gz"))
    s.set_condyle("right", CO_R.tolist())
    s.landmarks["Co_L"], s.suggested, s.landmark_sources["Co_L"] = CO_L, {"Co_L"}, cond.SOURCE_ALI
    path = s.save_case(str(tmp_path / "кейс"))["path"]
    t = Session(memory_path=str(tmp_path / "memory.jsonl"))
    t.open_case(path)
    st = t.articulation_condyles()
    assert st["points"]["right"]["source"] == cond.SOURCE_USER and st["points"]["right"]["confirmed"]
    assert st["points"]["left"]["source"] == cond.SOURCE_ALI and not st["points"]["left"]["confirmed"]
    assert np.allclose(st["points"]["left"]["ct"], CO_L) and not st["applied"]


def test_server_routes():
    calls = []

    class Fake:
        def articulation_condyles(self):
            calls.append(("state",))
            return {"ok": 1}

        def set_condyle(self, side, point, space):
            calls.append(("set", side, point, space))
            return {"ok": 2}

        def confirm_condyle(self, side):
            calls.append(("confirm", side))
            return {"ok": 3}

        def reset_condyle(self, side):
            calls.append(("reset", side))
            return {"ok": 4}

    srv = serve(Fake())
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/articulation/condyles"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path="", body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with opener.open(req) as r:
            return json.loads(r.read())

    try:
        assert call() == {"ok": 1}
        assert call("", {"side": "right", "point": [1, 2, 3], "space": "view"}) == {"ok": 2}
        assert call("/confirm", {"side": "left"}) == {"ok": 3}
        assert call("/reset", {}) == {"ok": 4}
    finally:
        srv.shutdown()
        time.sleep(0.01)
    assert calls == [("state",), ("set", "right", [1, 2, 3], "view"), ("confirm", "left"), ("reset", "both")]


# --- задачи ведомого ------------------------------------------------------------------------------------

def test_point_off_mandible_is_flagged():
    mesh = np.asarray(mandible_mesh().vertices)
    pts = cond.resolve({"Co_R": CO_R, "Co_L": CO_L + (0, 0, 7.0)}, set(), {})
    found = {c.id: c for c in cond.checks(pts, mesh)}
    assert found["on_mandible_right"].level == "ok"
    assert found["on_mandible_left"].level == "error" and found["on_mandible_left"].value > cond.ON_MANDIBLE_ERROR_MM


def test_asymmetry_is_flagged():
    frame = np.eye(4)
    frame[:3, :3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]  # мм КТ (LPS) → вправо, вперёд, вверх
    pts = cond.resolve({"Co_R": CO_R + (0, 0, 7.0), "Co_L": CO_L}, set(), {})
    found = {c.id: c for c in cond.checks(pts, None, frame)}
    assert found["asymmetry"].level == "warn" and found["asymmetry"].value == pytest.approx(7.0, abs=0.1)


def test_surface_cache_reused_and_invalidated(session):
    mesh = mandible_mesh()
    session.structures["mandible"] = mesh
    session.landmarks.update(Co_R=CO_R.copy(), Co_L=CO_L.copy())
    session.articulation_condyles()
    first = cond.cached_surface(mesh)
    session.set_condyle("right", (CO_R + (0, 0, -0.1)).tolist())
    assert cond.cached_surface(mesh) is first
    mesh.vertices += (0, 0, 200)
    moved = session.articulation_condyles()
    assert cond.cached_surface(mesh) is not first
    assert next(c for c in moved["checks"] if c["id"] == "on_mandible_right")["level"] == "error"
    changed = cond.cached_surface(mesh)
    mesh.faces = mesh.faces[:10]
    assert cond.cached_surface(mesh) is not changed
    session.structures["mandible"] = mandible_mesh()
    restored = session.articulation_condyles()
    assert next(c for c in restored["checks"] if c["id"] == "on_mandible_right")["level"] == "ok"


def test_cached_surface_projects_onto_sparse_faces():
    mesh = trimesh.Trimesh(vertices=[[-20, -20, 0], [20, -20, 0], [0, 20, 0]],
                           faces=[[0, 1, 2]], process=False)
    surface = cond.cached_surface(mesh)
    assert surface.distance([0, 0, 0]) == pytest.approx(0)
    assert surface.distance([0, 0, 2]) == pytest.approx(2)
    assert surface.distance([0, 0, 6]) == pytest.approx(6)


# --- интеграция ведомого -------------------------------------------------------------------------------

def _http(server):
    root = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(root + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with opener.open(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    return call


def test_http_set_mount_applied_in_scan_bite(session):
    up_v, up_f = phantom.make_scan("upper", resolution=0.6)
    lo_v, lo_f = phantom.make_scan("lower", resolution=0.6)
    session.jaw.set_scans(Mesh(up_v, up_f, "верх"), Mesh(lo_v, lo_f, "низ"))
    session._articulation_ready = True
    session.bite_motion = bite_motion()
    server = serve(session)
    call = _http(server)
    try:
        for side, point in (("right", cond.to_view(CO_R, session.bite_motion)),
                            ("left", cond.to_view(CO_L, session.bite_motion))):
            status, state = call("/api/articulation/condyles",
                                 {"side": side, "point": point.tolist(), "space": "view"})
            assert status == 200 and not state["applied"]
        status, started = call("/api/articulation/mount", {"method": "ct"})
        assert status == 200 and "job" in started
        limit = time.monotonic() + 15
        while time.monotonic() < limit:
            status, job = call(f"/api/jobs/{started['job']}")
            if job["status"] != "running":
                break
            time.sleep(0.02)
        assert status == 200 and job["status"] == "done", job.get("error")
        status, state = call("/api/articulation/condyles")
        assert status == 200 and state["applied"] and state["bite"] == "scan"
        up = session.jaw.anatomy.frame[2, :3]  # шарнир — центр головки под точкой Co в прикусе сканов
        assert np.allclose(session.jaw.anatomy.points["condyle_right"],
                           cond.to_view(CO_R, session.bite_motion) - CONDYLE_RADIUS_MM * up)
    finally:
        server.shutdown()


def test_condyle_api_errors_are_400(tmp_path, jaw_ct):
    session = Session(memory_path=str(tmp_path / "memory.jsonl"))
    server = serve(session)
    call = _http(server)
    try:
        status, answer = call("/api/articulation/condyles", {"side": "right", "point": CO_R.tolist()})
        assert status == 400 and "КТ" in answer["error"]
        session.case = jaw_ct
        for body in ({"side": "middle", "point": CO_R.tolist()},
                     {"side": "right", "point": CO_R.tolist(), "space": "screen"}):
            status, answer = call("/api/articulation/condyles", body)
            assert status == 400 and answer["error"]
    finally:
        server.shutdown()


def test_ct_bone_surface_route_and_default(session):
    from casedesigner.app.session import mesh_bytes

    server = serve(session)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root = f"http://127.0.0.1:{server.server_address[1]}/api/ct/surface"
    try:
        surfaces = session.case.surfaces(step=2)
        for query, name in (("", "ct_teeth"), ("?part=bone", "ct_bone")):
            with opener.open(root + query) as response:
                mesh = surfaces[name]
                assert len(mesh.vertices) > 0
                assert response.read() == mesh_bytes(mesh.vertices, mesh.faces)
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(root + "?part=unknown")
        assert error.value.code == 400
    finally:
        server.shutdown()


def test_http_ct_point_stays_on_ct_with_scan_bite(session):
    session.bite_motion = bite_motion()
    server = serve(session)
    try:
        point = CO_R + (0, 0, -2)
        status, state = _http(server)("/api/articulation/condyles",
                                     {"side": "right", "point": point.tolist(), "space": "ct"})
        assert status == 200 and not state["segmentation"]["mandible"]
        assert np.allclose(session.landmarks["Co_R"], point)
        assert np.allclose(state["points"]["right"]["ct"], point)
        assert np.allclose(state["points"]["right"]["view"], cond.to_view(point, session.bite_motion), atol=1e-3)
        guides = session.articulation_guides()
        for key, axis in guides["axes"].items():
            assert np.allclose(session.bite_motion[:3, :3] @ guides["ct_axes"][key], axis, atol=1e-6)
    finally:
        server.shutdown()


def test_guides_paths_with_built_movements(session):
    """Линии основы динамики при построенных движениях: путь каждой точки — по всем кадрам (N×3)."""
    from casedesigner import motion as mo

    session.jaw.anatomy = mo.Anatomy(np.eye(4), {"condyle_right": CO_R, "condyle_left": CO_L,
                                                 "incisal": np.array([0.0, -50.0, -30.0])}, "тест")
    session._articulation_ready = True
    frames = np.stack([np.eye(4), rigid(np.eye(3), [0.0, 1.0, 0.0]), rigid(np.eye(3), [0.0, 2.0, -1.0])])
    session.jaw.movements = {"m": SimpleNamespace(id="m", recorded=False,
                                                  recording=SimpleNamespace(name="Протрузия", transforms=frames))}
    guides = session.articulation_guides()
    assert guides["available"] and len(guides["paths"]) == 1
    path = guides["paths"][0]
    assert np.allclose(path["condyle_right"], [CO_R, CO_R + (0, 1, 0), CO_R + (0, 2, -1)])
    assert np.asarray(path["incisal"]).shape == (3, 3)
    json.dumps(guides)


def test_old_case_without_landmark_sources(tmp_path, jaw_ct):
    session = Session(memory_path=str(tmp_path / "memory.jsonl"))
    session.case = jaw_ct
    session.landmarks = {"Co_R": CO_R, "Co_L": CO_L}
    session.suggested = {"Co_L"}
    path = session.save_case(str(tmp_path / "case"))["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    meta = json.loads(str(arrays["meta"]))
    meta.pop("landmark_sources")
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    with open(path, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    restored = Session(memory_path=str(tmp_path / "memory2.jsonl"))
    restored.open_case(path)
    state = restored.articulation_condyles()
    assert state["points"]["right"]["source"] == cond.SOURCE_USER
    assert state["points"]["left"]["source"] == cond.SOURCE_MANDIBLE


def test_ali_search_keeps_confirmed_condyle(monkeypatch, session, tmp_path):
    from casedesigner import auto_landmarks as auto

    session.vol = object()
    session.landmarks = {"Co_R": CO_R.copy(), "Co_L": CO_L.copy()}
    session.suggested = {"Co_L"}
    session.landmark_sources = {"Co_R": cond.SOURCE_USER, "Co_L": cond.SOURCE_MANDIBLE}
    monkeypatch.setattr(session, "_landmark_folder", lambda: str(tmp_path))

    class Models:
        def __init__(self, _folder):
            pass

        @staticmethod
        def available():
            return ["test"]

    monkeypatch.setattr(auto, "Models", Models)
    monkeypatch.setattr(auto, "find", lambda *_a, **_k: {
        "Co_R": SimpleNamespace(point=CO_R + (0, 0, 20), note=""),
        "Co_L": SimpleNamespace(point=CO_L + (0, 0, 2), note=""),
    })
    session.auto_landmarks(keys=["Co_R", "Co_L"])
    assert np.allclose(session.landmarks["Co_R"], CO_R)
    assert np.allclose(session.landmarks["Co_L"], CO_L + (0, 0, 2))
    assert session.landmark_sources["Co_R"] == cond.SOURCE_USER
    assert session.landmark_sources["Co_L"] == cond.SOURCE_ALI
