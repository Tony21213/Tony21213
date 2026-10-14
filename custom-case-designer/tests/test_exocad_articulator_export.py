"""Контракт выгрузки «Для exocad с артикулятором» (casedesigner/exocad_articulator_export.py, docs/handoff/archive/exocad-articulator-export.md).

Тесты с xfail(strict=True) — задачи ведомого: после реализации они проходят, и пометку нужно снять.
"""

import json
import os
import xml.etree.ElementTree as ET
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
            "  <MovementMarkerScan>false</MovementMarkerScan>\r\n  <Other>оставить как есть</Other>\r\n"
            "  <Teeth>\r\n    <Tooth>\r\n      <Number>11</Number>\r\n      <Parameters>\r\n        <Parameter>\r\n"
            "          <Type>MinThickness</Type>\r\n          <Value>0.5</Value>\r\n        </Parameter>\r\n"
            "        <Parameter>\r\n          <Type>TiltCondylarGuideLeft</Type>\r\n          <Value>24.9</Value>\r\n"
            "        </Parameter>\r\n      </Parameters>\r\n    </Tooth>\r\n    <Tooth>\r\n      <Number>21</Number>\r\n"
            "      <Parameters />\r\n    </Tooth>\r\n  </Teeth>\r\n</Treatment>\r\n")
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
                   ".jawmotion", "-lowerjaw.partialInfo"):
        assert stem + suffix in names, suffix
    assert {"bone_upper.stl", "bone_lower.stl"} <= names and res["files"]["кости"] == ["bone_upper.stl", "bone_lower.stl"]
    assert not any("partialframework" in n for n in names)
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
    # общий артикулятор — одинаковый для всех кейсов (набор вставок, первая — прямая; межмыщелковое библиотеки),
    # под тем именем, на которое наш тип указывает в конфигурации exocad (установка под прежним названием)
    assert res["articulator_folder"] == "Custom Case Designer"
    art = out / xa.SUBFOLDER / "articulator" / "Custom Case Designer"
    root = ET.parse(art / "articulatorparameters.xml").getroot()
    assert root.findtext("CurrentCondylarInsertColorRight") == "Прямая"
    assert float(root.findtext("IntercondylarDistance")) == ef.LIBRARY_ICD_MM
    assert ef.library_version(str(art)) == ef.LIBRARY_VERSION
    assert any("не найден" in n and "один раз" in n for n in res["notes"])  # в библиотеке exocad его ещё нет
    # значения пациента — в параметрах каждого зуба копии проекта: такой же тип заменён, новые добавлены
    params = res["project_parameters"]
    assert params["TiltCondylarGuideRight"] == res["articulator"]["TiltCondylarGuideRight"]
    teeth = ET.fromstring(text.encode("utf-8")).findall("Teeth/Tooth")
    assert len(teeth) == 2
    for tooth in teeth:
        got = {p.findtext("Type"): float(p.findtext("Value")) for p in tooth.findall("Parameters/Parameter")}
        assert {k: got[k] for k in params} == pytest.approx(params)
        assert [p.findtext("Type") for p in tooth.findall("Parameters/Parameter")].count("TiltCondylarGuideLeft") == 1
        assert "IntercondylarDistance" not in got  # межмыщелковое в проект — только при ICD_IN_PROJECT
    assert got.get("MinThickness") is None and teeth[0].findtext("Parameters/Parameter/Type") == "MinThickness"
    assert any("межмыщелковое пациента" in n for n in res["notes"])
    # вставка ССП: путь фантома — прямой под 30° → прямая вставка, ССП 30°
    assert res["inserts"]["right"]["insert"] == "Прямая"
    assert res["inserts"]["right"]["tilt_deg"] == pytest.approx(30, abs=0.15)
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


def test_export_without_bones_or_skull_uses_standard_inserts(case, project, register, tmp_path):
    case.bones = {}
    case.condylar_paths = None
    out = tmp_path / "without-bones"
    result = xa.export(case, str(project), str(out), register=register,
                       exocad_app=exocad_app(tmp_path, custom=True))

    assert "кости" not in result["files"]
    assert any("вставки ССП в артикуляторе стандартные" in note for note in result["notes"])
    values = ET.parse(out / xa.SUBFOLDER / "articulator" / "Custom Case Designer" /
                      "articulatorparameters.xml").getroot()
    # общий артикулятор тот же; подбора вставки нет
    assert values.findtext("CurrentCondylarInsertColorRight") == "Прямая"
    assert result["inserts"] is None


def test_aesthetic_axes_give_smile_design_view(case, project, register, tmp_path):
    """Есть эстетическая система — в копии проекта набор изображений: «Вид Smile Design» по эстетическому «вперёд»."""
    app = exocad_app(tmp_path, custom=True)
    plain = xa.export(case, str(project), str(tmp_path / "plain"), register=register, exocad_app=app)
    assert "эстетический вид" not in plain["files"]
    assert not any(n.endswith(".treatmentImages") for n in os.listdir(tmp_path / "plain"))
    R = axis_angle(np.array([1.0, 0, 0]), np.radians(-8))
    case.aesthetic = (R @ np.eye(3)).T  # строки: вправо, вперёд, вверх (в координатах кейса)
    res = xa.export(case, str(project), str(tmp_path / "views"), register=register, exocad_app=app)
    assert res["files"]["эстетический вид"] == ["views.treatmentImages", "views-smile.jpg"]
    assert any("Вид Smile Design" in n for n in res["notes"])
    im = ET.parse(tmp_path / "views" / "views.treatmentImages").getroot().find("Images/Image")

    def vec(tag):
        return np.array([float(im.find(tag).findtext(k)) for k in "xyz"])

    to_art = np.array(res["case_to_articulator"])
    forward, up = (to_art[:3, :3] @ case.aesthetic[i] for i in (1, 2))
    assert vec("ViewDirection") == pytest.approx(-forward, abs=1e-5)
    assert vec("ViewAxisY") == pytest.approx(up, abs=1e-5)
    centre = np.mean([vec(k) for k in ("TopLeft", "TopRight", "BottomLeft", "BottomRight")], axis=0)
    incisal = apply(to_art, case.anatomy.points["incisal"][None])[0]
    assert centre == pytest.approx(incisal - 100.0 * forward, abs=1e-3)  # перекрестье — на резцовой точке


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
    assert '"exocad-articulator"' in server.routes_source()


def _http_export(server, body):
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/api/export/exocad-articulator",
        data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = raw.decode("utf-8", errors="replace")
        return error.code, body


def _export_session(ready=True, applied=True):
    from casedesigner.app.session import Session

    session = Session.__new__(Session)
    session.articulation_condyles = lambda: {
        "available": True, "ready": ready, "applied": applied,
        "points": {"right": {"ct": [1, 2, 3]}, "left": {"ct": [4, 5, 6]}}, "checks": []}
    session.jaw = SimpleNamespace(anatomy=object())
    session._jaw_items = lambda: ({"reg": object()}, {"reg": object()})
    return session


def test_http_export_creates_project_copy_with_aesthetic_role(case, project, register, tmp_path, monkeypatch):
    import time

    from casedesigner.app.server import serve

    app = exocad_app(tmp_path, custom=True)
    monkeypatch.setattr(ef, "find_exocad", lambda: app)
    monkeypatch.setattr(ef, "load_register", lambda _path: register)
    case.aesthetic = np.eye(3)
    monkeypatch.setattr(xa, "case_from_session", lambda _session: case)
    server = serve(_export_session())
    try:
        status, answer = _http_export(server, {"project": str(project), "out": None})
        assert status == 200 and set(answer) == {"job"}
        for _ in range(200):
            with __import__("urllib.request").request.urlopen(
                    f"http://127.0.0.1:{server.server_address[1]}/api/jobs/{answer['job']}") as response:
                job = json.loads(response.read())
            if job["status"] == "done":
                break
            assert job["status"] != "error", job["error"]
            time.sleep(0.02)
        else:
            raise AssertionError("фоновая выгрузка не завершилась")

        copied = project.parent / (project.name + xa.COPY_SUFFIX)
        assert job["result"]["files"]["эстетический вид"] == ["case-001-KStom.treatmentImages",
                                                                    "case-001-KStom-smile.jpg"]
        assert (copied / "case-001-KStom.dentalProject").is_file()
        assert (copied / "case-001-KStom.jawmotion").is_file()
        assert (copied / "KStomCaseDesigner" / "articulator.json").is_file()
    finally:
        server.shutdown()


@pytest.mark.parametrize(("body", "ready", "applied", "message"), [
    ({}, True, True, "выберите папку проекта"),
    ({"project": "project"}, False, False, "точки мыщелков не готовы"),
    ({"project": "project"}, True, False, "выполните монтаж моделей"),
])
def test_http_export_validation_errors_are_400(body, ready, applied, message):
    from casedesigner.app.server import serve

    server = serve(_export_session(ready, applied))
    try:
        status, answer = _http_export(server, body)
        assert status == 400
        assert message in answer["error"]
    finally:
        server.shutdown()


# --- модульный артикулятор (п. 0.5): значения пациента — в проекте, артикулятор — общий ---------------------------

def test_set_tooth_parameters_replaces_appends_and_opens_empty():
    nl = "\r\n"
    text = nl.join(["<Teeth>", "  <Tooth>", "    <Parameters>", "      <Parameter>", "        <Type>BennettAngleLeft</Type>",
                    "        <Value>3</Value>", "      </Parameter>", "    </Parameters>", "  </Tooth>",
                    "  <Tooth>", "    <Parameters />", "  </Tooth>", "  <ToothColor>A2</ToothColor>", "</Teeth>"])
    params = [("BennettAngleLeft", 12.5, 0, 30, 0.1), ("Retrusion", 0.0, 0, 2, 0.1)]
    out, teeth = xa.set_tooth_parameters(text, params)
    assert teeth == 2 and "\n" not in out.replace("\r\n", "")  # переводы строк — как в файле
    root = ET.fromstring(out)
    for tooth in root.findall("Tooth"):
        got = {p.findtext("Type"): p.findtext("Value") for p in tooth.findall("Parameters/Parameter")}
        assert got == {"BennettAngleLeft": "12.5", "Retrusion": "0"}
    assert root.findtext("ToothColor") == "A2"


def test_project_parameters_clamped_to_exocad_limits():
    params, notes = xa.project_parameters({"TiltCondylarGuideLeft": 65.0, "BennettAngleRight": 12.0, "Unknown": 1})
    assert [(k, v) for k, v, *_ in params] == [("TiltCondylarGuideLeft", 60.0), ("BennettAngleRight", 12.0)]
    assert len(notes) == 1 and "TiltCondylarGuideLeft" in notes[0]
    params, _ = xa.project_parameters({}, icd=101.26)
    assert params == [("IntercondylarDistance", 101.26, 80.0, 130.0, 0.1)]


def test_icd_into_project_when_enabled(case, project, register, tmp_path, monkeypatch):
    monkeypatch.setattr(xa, "ICD_IN_PROJECT", True)
    res = xa.export(case, str(project), str(tmp_path / "icd"), register=register,
                    exocad_app=exocad_app(tmp_path, custom=True))
    text = (tmp_path / "icd" / "icd.dentalProject").read_text(encoding="utf-8-sig")
    assert text.count("<Type>IntercondylarDistance</Type>") == 2 and f"<Value>{res['icd_mm']:g}</Value>" in text
    assert not any("межмыщелковое пациента" in n for n in res["notes"])


def test_same_library_articulator_for_every_case(case, project, register, tmp_path):
    """Папка артикулятора в выгрузке не зависит от кейса: её ставят в exocad один раз; установленная — без заметки."""
    app = exocad_app(tmp_path, custom=True)
    first = xa.export(case, str(project), str(tmp_path / "a"), register=register, exocad_app=app)
    case.condylar_paths = None
    case.anatomy.points["condyle_left"] = case.anatomy.points["condyle_left"] + np.array([4.0, 0, 0])
    second = xa.export(case, str(project), str(tmp_path / "b"), register=register, exocad_app=app)
    assert first["icd_mm"] != second["icd_mm"]
    dirs = [tmp_path / d / xa.SUBFOLDER / "articulator" / "Custom Case Designer" for d in ("a", "b")]
    assert sorted(p.name for p in dirs[0].iterdir()) == sorted(p.name for p in dirs[1].iterdir())
    for p in dirs[0].iterdir():
        assert p.read_bytes() == (dirs[1] / p.name).read_bytes(), p.name
    import shutil
    shutil.copytree(dirs[0], os.path.join(app, "library", "articulator", "Custom Case Designer"))
    third = xa.export(case, str(project), str(tmp_path / "c"), register=register, exocad_app=app)
    assert third["library_installed"] == ef.LIBRARY_VERSION
    assert not any("общий артикулятор" in n for n in third["notes"])


def test_patient_photos_go_to_smile_design_by_camera(case, project, register, tmp_path):
    """«Эстетика», часть 1: фото с улыбкой и с ретрактором — в набор Smile Design по камере совмещения (вместо сетки)."""
    from casedesigner import aesthetic as ae

    R = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])  # спереди: система монтажа фантома — единичная
    centre = case.anatomy.points["incisal"] + [0, 300.0, 0]
    cam = ae.Camera(np.array([[900.0, 0, 60], [0, 900.0, 40], [0, 0, 1]]), R, -R @ centre, 0.4, size=(120, 80))
    image = np.full((80, 120, 3), 180, np.uint8)
    case.aesthetic = np.eye(3)
    case.photos = [{"kind": "smile", "image": image, "camera": cam}, {"kind": "retracted", "image": image, "camera": cam}]
    res = xa.export(case, str(project), str(tmp_path / "ph"), register=register,
                    exocad_app=exocad_app(tmp_path, custom=True))
    assert res["files"]["эстетический вид"] == ["ph.treatmentImages", "ph-smile.jpg", "ph-retracted.jpg"]
    root = ET.parse(tmp_path / "ph" / "ph.treatmentImages").getroot()
    assert [i.findtext("ImageType") for i in root.findall("Images/Image")] == ["SmileImage", "SmileImageRetracted"]
    view = np.array([float(root.findtext(f"Images/Image/ViewDirection/{k}")) for k in "xyz"])
    to_art = np.array(res["case_to_articulator"])
    assert view == pytest.approx(to_art[:3, :3] @ [0, -1.0, 0], abs=1e-6)  # взгляд — назад, в мм артикулятора
    assert any("фото пациента" in n for n in res["notes"])
