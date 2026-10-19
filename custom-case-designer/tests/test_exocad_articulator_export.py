"""Контракт выгрузки «Для exocad с артикулятором» (casedesigner/exocad_articulator_export.py, docs/handoff.md).

Тесты с xfail(strict=True) — задачи ведомого: после реализации они проходят, и пометку нужно снять.
"""

import json
import os
import xml.etree.ElementTree as ET
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

import phantom
from casedesigner import exocad_articulator_export as xa
from casedesigner import exocad_facebow as ef
from casedesigner import kinematics as kin
from casedesigner import motion as mo
from casedesigner.register import apply, axis_angle, rigid

MARKS = np.array([[0, 0, 0], [25, 0, 30.5], [-25, 0, 30.5]], float)
# Координаты кейса фантома: x — вправо пациента, y — вперёд, z — вверх (система монтажа — единичная).
CONDYLES = {"right": np.array([49.0, -70.0, 24.0]), "left": np.array([-50.0, -71.0, 26.0])}


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _s, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return rigid(R, cb - R @ ca)


@pytest.fixture
def register():
    fork = trimesh.creation.box((62, 12, 60), trimesh.transformations.translation_matrix((0, 2.7, 20)))
    return ef.Register(fork, MARKS.copy())


@pytest.fixture
def case():
    up_v, up_f = phantom.make_scan("upper", resolution=0.6)
    lo_v, lo_f = phantom.make_scan("lower", resolution=0.6)
    incisal = lo_v[np.argmax(lo_v[:, 1])]
    anatomy = mo.Anatomy(np.eye(4), {"condyle_right": CONDYLES["right"], "condyle_left": CONDYLES["left"],
                                     "incisal": incisal}, "тест")
    colors = np.tile(np.array([[200, 150, 120, 255]], np.uint8), (len(up_v), 1))
    hinge = (CONDYLES["right"] + CONDYLES["left"]) / 2
    opening = np.array([rigid(axis_angle(np.array([1.0, 0, 0]), -np.radians(a)),
                              hinge - axis_angle(np.array([1.0, 0, 0]), -np.radians(a)) @ hinge)
                        for a in np.linspace(0, 15, 31)])
    t = np.linspace(0, 12, 49)
    path = np.c_[t * np.cos(np.radians(30)), -t * np.sin(np.radians(30))]
    bone = trimesh.creation.box((80, 60, 10), trimesh.transformations.translation_matrix((0, -10, -20)))
    return xa.ArticulatorCase(
        upper=xa.JawScan(up_v, up_f, colors), lower=xa.JawScan(lo_v, lo_f), anatomy=anatomy,
        settings=kin.Settings(sagittal_right_deg=30, sagittal_left_deg=32),
        bites=[xa.JawScan(lo_v[:200], lo_f[:100])],
        bones={"upper": xa.JawScan(bone.vertices + (0, 0, 40), bone.faces), "lower": xa.JawScan(bone.vertices, bone.faces)},
        condylar_paths={"right": path, "left": path}, movements=[("Открывание", opening, 30.0)])


@pytest.fixture
def project(tmp_path):
    d = tmp_path / "case-001"
    d.mkdir()
    text = ("<?xml version=\"1.0\" encoding=\"utf-8\"?>\r\n<Treatment>\r\n  <AntagonistType>MushBite</AntagonistType>\r\n"
            "  <MovementMarkerScan>false</MovementMarkerScan>\r\n  <Other>оставить как есть</Other>\r\n</Treatment>\r\n")
    (d / "case-001.dentalProject").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    (d / "case-001.scanInfo").write_text("<ScanInfo/>", encoding="utf-8")
    return d


def exocad_app(tmp_path, custom: bool):
    app = tmp_path / ("app-custom" if custom else "app")
    (app / "config").mkdir(parents=True)
    body = "<Mappings><Antagonists>" + ("<Antagonist><Type>ArticulatorCustomCaseDesigner</Type>"
                                        "<FolderName>Custom Case Designer</FolderName></Antagonist>"
                                        if custom else "") + "</Antagonists></Mappings>"
    (app / "config" / "articulatormappings.xml").write_text(body, encoding="utf-8")
    return str(app)


def test_export_writes_exocad_project_copy_in_articulator_coordinates(case, project, register, tmp_path):
    before = {p.name: p.read_bytes() for p in project.iterdir()}
    res = xa.export(case, str(project), register=register, exocad_app=exocad_app(tmp_path, custom=True))
    json.dumps({k: v for k, v in res.items() if k != "folder"})
    out = project.parent / ("case-001" + xa.COPY_SUFFIX)
    stem = "case-001" + xa.COPY_SUFFIX
    assert res["folder"] == str(out)
    assert {p.name: p.read_bytes() for p in project.iterdir()} == before  # исходный проект не тронут
    names = {p.name for p in out.iterdir()}
    for suffix in (".dentalProject", "-upperjaw.ply", "-lowerjaw.ply", "-TotalJaw0.ply", "-movementmarker.stl",
                   ".jawmotion", "-upperjaw-partialframework_cad.stl", "-lowerjaw.partialInfo"):
        assert stem + suffix in names, suffix
    assert not any(n.endswith((".scanInfo", ".matrix4")) for n in names)  # сцена = файл
    # флаги проекта: наш тип антагониста, скан-маркер; остальное — как было
    proj = (out / f"{stem}.dentalProject").read_bytes()
    assert proj.startswith(b"\xef\xbb\xbf")
    text = proj.decode("utf-8-sig")
    assert "<AntagonistType>ArticulatorCustomCaseDesigner</AntagonistType>" in text
    assert "<MovementMarkerScan>true</MovementMarkerScan>" in text and "оставить как есть" in text
    # сканы — в координатах артикулятора, цвет сохранён
    to_art = np.array(res["case_to_articulator"])
    up = trimesh.load_mesh(str(out / f"{stem}-upperjaw.ply"), process=False)
    assert np.abs(np.asarray(up.vertices) - apply(to_art, case.upper.vertices)).max() < 1e-3
    assert np.array_equal(np.asarray(up.visual.vertex_colors)[:, :3], case.upper.colors[:, :3])
    mid = (np.array(res["condyles_articulator"]["right"]) + np.array(res["condyles_articulator"]["left"])) / 2
    assert np.abs(mid - [30, -80, 60]).max() < 1e-3
    # мыщелки фантома — на разной высоте: ось через середину, предупреждение — в сводке
    assert max(res["off_axis_mm"].values()) > 1.0 and any("несимметрично" in n for n in res["notes"])
    # плоскость гипсовки — горизонталь на высоте резцовой точки; папка артикулятора с вставками
    # папка — под тем именем, на которое наш тип указывает в конфигурации exocad (установка под прежним названием)
    assert res["articulator_folder"] == "Custom Case Designer"
    art = out / xa.SUBFOLDER / "articulator" / "Custom Case Designer"
    root = ET.parse(art / "articulatorparameters.xml").getroot()
    plane = float(root.findtext("ArticulationPlaneIncisalNeedle/z")) + float(root.findtext("ArticulatorPosition/z"))
    assert plane == pytest.approx(apply(to_art, case.anatomy.points["incisal"][None])[0, 2], abs=0.01)
    assert root.findtext("CurrentCondylarInsertColorRight") == ef.INSERT_ID
    assert float(root.findtext("IntercondylarDistance")) == pytest.approx(res["icd_mm"])
    assert (out / xa.SUBFOLDER / "condyles.stl").is_file() and (out / xa.SUBFOLDER / "articulator.json").is_file()
    # exocad: вилка библиотеки на вилке маркера и метки из файла движений → «файл → регистратор»; с переводом
    # регистратор → артикулятор это тождество: модели остаются, где лежат в файлах
    fb = ef.facebow(case.anatomy.frame, CONDYLES["right"], CONDYLES["left"], case.anatomy.points["incisal"], register)
    marker = trimesh.load_mesh(str(out / f"{stem}-movementmarker.stl"))
    fork = apply(to_art @ fb.fork_pose, register.fork.vertices)
    assert cKDTree(marker.vertices).query(fork)[0].max() < 1e-3
    assert cKDTree(marker.vertices).query(apply(to_art, case.upper.vertices[::50]))[0].max() < 1e-3
    jm = ef.read_jawmotion(str(out / f"{stem}.jawmotion"))
    file_to_register = kabsch(apply(to_art @ fb.fork_pose, register.marks), jm["marks"])
    assert np.abs(ef.register_to_articulator() @ file_to_register - np.eye(4)).max() < 1e-3
    # движение из кейса — в файле: открывание идёт вниз (y регистратора — вверх)
    track = jm["movements"]["opening"][0]["mark_1"]
    assert track[-1][1] < track[0][1] - 5
    # повторно в ту же папку — отказ, без перезаписи
    with pytest.raises(FileExistsError):
        xa.export(case, str(project), register=register, exocad_app=exocad_app(tmp_path, custom=True))


def test_antagonist_falls_back_to_sam_2p(case, project, register, tmp_path):
    case.mounting = "average"
    res = xa.export(case, str(project), str(tmp_path / "copy"), register=register,
                    exocad_app=exocad_app(tmp_path, custom=False))
    assert res["antagonist"] == xa.SAM_TYPE and any("SAM 2P" in n for n in res["notes"])
    assert res["articulator_folder"] == ef.ARTICULATOR_NAME
    assert any("не по мыщелкам КТ" in n for n in res["notes"])  # средний артикулятор — с предупреждением
    text = (tmp_path / "copy" / "copy.dentalProject").read_text(encoding="utf-8-sig")
    assert "<AntagonistType>ArticulatorSendling</AntagonistType>" in text


def test_articulator_values_from_mount_horizontal(case):
    s = kin.Settings(sagittal_right_deg=30, sagittal_left_deg=30, bennett_right_deg=12, bennett_left_deg=12)
    raw = xa.articulator_values(case.anatomy, s)
    assert raw["TiltCondylarGuideRight"] == pytest.approx(30, abs=0.05)
    assert raw["BennettAngleLeft"] == pytest.approx(12, abs=0.05)
    # углы от своей горизонтали, наклонённой на 10° к горизонтали монтажа (передом вниз)
    own = s.copy()
    own.frame = rigid(axis_angle(np.array([1.0, 0, 0]), np.radians(-10)), np.zeros(3))
    tilted = xa.articulator_values(case.anatomy, own)
    assert abs(tilted["TiltCondylarGuideRight"] - 30) == pytest.approx(10, abs=0.05)


def test_case_from_session_moves_lower_with_bite(case):
    """Вход из сессии: нижний скан и кость нижней челюсти — в прикусе монтажа, верхние — как на КТ."""
    bite = rigid(np.eye(3), [0.0, 0.0, -1.5])
    jaw = SimpleNamespace(anatomy=case.anatomy, lower_fix=np.eye(4), settings=kin.Settings(), movements={},
                          mounting="ct")
    up_item = {"role": "jaw", "path": None, "reg": SimpleNamespace(jaw="upper", transform=np.eye(4)),
               "scan": SimpleNamespace(vertices=case.upper.vertices, faces=case.upper.faces)}
    lo_item = {"role": "jaw", "path": None, "reg": SimpleNamespace(jaw="lower", transform=np.eye(4)),
               "scan": SimpleNamespace(vertices=case.lower.vertices, faces=case.lower.faces)}
    mandible = trimesh.creation.box((80, 60, 10))
    session = SimpleNamespace(jaw=jaw, bite_motion=bite, scans={"u": up_item, "l": lo_item},
                              structures={"mandible": mandible}, landmarks={}, suggested=set(), landmark_sources={},
                              _jaw_items=lambda: (up_item, lo_item))
    got = xa.case_from_session(session)
    assert np.allclose(got.upper.vertices, case.upper.vertices)
    assert np.allclose(got.lower.vertices, apply(bite, case.lower.vertices))
    assert np.allclose(got.bones["lower"].vertices, apply(bite, mandible.vertices))
    assert got.condylar_paths is None and got.mounting == "ct"  # нет черепа — вставки стандартные
    assert got.movements == []  # движений не строили — и открывания нет


def test_case_from_session_adds_opening_on_patient_axis(case):
    """К построенным движениям — открывание вокруг оси через оба мыщелка, даже если она наклонена к горизонтали."""
    s = kin.Settings()
    jaw = SimpleNamespace(anatomy=case.anatomy, lower_fix=np.eye(4), settings=s, mounting="ct",
                          movements={"p": SimpleNamespace(recording=kin.protrusion(case.anatomy, s), recorded=False)})
    item = {"role": "jaw", "path": None, "reg": SimpleNamespace(transform=np.eye(4)),
            "scan": SimpleNamespace(vertices=case.upper.vertices, faces=case.upper.faces)}
    session = SimpleNamespace(jaw=jaw, bite_motion=None, scans={}, structures={}, landmarks={}, suggested=set(),
                              landmark_sources={}, _jaw_items=lambda: (item, item))
    names = [name for name, _T, _fps in xa.case_from_session(session).movements]
    assert [ef.movement_type(n) for n in names] == ["opening", "protrusion"]
    opening = xa.case_from_session(session).movements[0][1]
    hinge = opening[len(opening) // 4]  # начало открывания — чистый поворот на шарнирной оси
    assert np.degrees(np.arccos((np.trace(hinge[:3, :3]) - 1) / 2)) > 2
    for p in CONDYLES.values():  # мыщелки фантома на разной высоте — оба стоят на месте
        assert np.abs(apply(hinge, p[None])[0] - p).max() < 1e-6
    jaw.movements["p"].recorded = True  # записанные движения не дополняем
    assert len(xa.case_from_session(session).movements) == 1


# --- задачи ведомого ------------------------------------------------------------------------------------

def test_session_and_api_contract():
    import inspect

    from casedesigner.app import server
    from casedesigner.app.session import Session

    sig = inspect.signature(Session.export_exocad_articulator)
    assert list(sig.parameters)[1:3] == ["project_dir", "out_dir"]
    assert "exocad_articulator" in inspect.getsource(server)


@pytest.fixture
def session(case, register, tmp_path, monkeypatch):
    from casedesigner.app.session import Session

    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.case = object()
    to_ct = np.diag([-1., -1., 1., 1.])
    for side, mesh in (("upper", case.upper), ("lower", case.lower)):
        s.scans[side] = {"role": "jaw", "path": None,
                         "reg": SimpleNamespace(jaw=side, transform=to_ct),
                         "scan": SimpleNamespace(vertices=mesh.vertices, faces=mesh.faces)}
    s.bite_motion = rigid(axis_angle(np.array([1., 0., 0.]), np.radians(2)), [0, 1, -1])
    s.set_condyle("right", apply(to_ct, CONDYLES["right"]).tolist())
    s.set_condyle("left", apply(to_ct, CONDYLES["left"]).tolist())
    app = exocad_app(tmp_path, custom=True)
    monkeypatch.setattr(ef, "find_exocad", lambda: app)
    monkeypatch.setattr(ef, "load_register", lambda _path: register)
    return s


def _http(server, path, body=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/api/{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with opener.open(req) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _done(server, started):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status, job = _http(server, "jobs/" + started["job"])
        assert status == 200
        if job["status"] != "running":
            assert job["status"] == "done", job["error"]
            return job["result"]
        time.sleep(.01)
    pytest.fail("фоновая операция не завершилась")


@pytest.mark.parametrize("with_bone", [False, True])
def test_http_mount_then_export_without_skull(session, project, tmp_path, with_bone):
    from casedesigner.app.server import serve

    before = {p.name: p.read_bytes() for p in project.iterdir()}
    server = serve(session)
    try:
        status, started = _http(server, "articulation/mount", {"method": "ct"})
        assert status == 200
        _done(server, started)
        assert session.articulation_condyles()["applied"]
        # Без черепа движения построены стандартной кинематикой.
        session.jaw._add(kin.protrusion(session.jaw.anatomy, session.jaw.settings), recorded=False)
        if with_bone:
            # Кость ВЧ не меняет проверку мыщелков и не заменяет череп для вставок.
            session.structures["maxilla"] = trimesh.creation.box((30, 20, 10))
        out = tmp_path / "export-001"
        status, started = _http(server, "export/exocad-articulator", {"project": str(project), "out": str(out)})
        assert status == 200 and set(started) == {"job"}
        summary = _done(server, started)
        assert summary["movements"] == 2  # протрузия и добавленное открывание
        assert any("стандартные" in note for note in summary["notes"])
        assert summary["files"].get("кости-каркасы", 0) == int(with_bone)
        for role, name in summary["files"].items():
            if isinstance(name, str):
                assert (out / name).is_file(), role
        params = ET.parse(out / xa.SUBFOLDER / "articulator" / summary["articulator_folder"] /
                          "articulatorparameters.xml").getroot()
        assert params.find("ArticulatorMainParts/CondylarInserts") is None
        assert params.findtext("RotationAxisRightParam/Value") == "0"
        lower = trimesh.load_mesh(out / summary["files"]["нижний скан"], process=False)
        expected = xa.case_from_session(session).lower.vertices
        assert np.max(np.abs(lower.vertices - apply(np.asarray(summary["case_to_articulator"]), expected))) < 1e-3
        assert session.last_export == str(out)
        assert {p.name: p.read_bytes() for p in project.iterdir()} == before
    finally:
        server.shutdown()


def test_export_rejections_are_400(session, project, tmp_path):
    from casedesigner.app.server import serve

    server = serve(session)
    out = tmp_path / "copy"
    body = {"project": str(project), "out": str(out)}
    try:
        status, answer = _http(server, "export/exocad-articulator", body)
        assert status == 400 and "перемонтируйте" in answer["error"]
        session.articulation_mount("ct")
        status, answer = _http(server, "export/exocad-articulator", body)
        assert status == 400 and "движения" in answer["error"]
        session.jaw._add(kin.protrusion(session.jaw.anatomy, session.jaw.settings), recorded=False)
        session.set_condyle("right", [-10, 0, 0])
        status, answer = _http(server, "export/exocad-articulator", body)
        assert status == 400 and "мыщелков" in answer["error"]
        assert not out.exists()
        session.set_condyle("right", [-49, 70, 24])
        session.articulation_mount("ct")
        session.jaw._add(kin.protrusion(session.jaw.anatomy, session.jaw.settings), recorded=False)
        status, answer = _http(server, "export/exocad-articulator", {"project": str(tmp_path / "missing")})
        assert status == 400 and "нет проекта" in answer["error"]
        out.mkdir()
        (out / "keep.txt").write_text("keep")
        status, answer = _http(server, "export/exocad-articulator", body)
        assert status == 400 and "не пуста" in answer["error"]
        assert (out / "keep.txt").read_text() == "keep"
    finally:
        server.shutdown()
