"""API приложения: сценарий врача от открытия КТ до экспорта, через HTTP, как в интерфейсе."""

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.app.server import serve
from casedesigner.app.session import Session, group_of
from casedesigner.register import apply


@pytest.fixture(scope="module")
def case_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("case")
    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(d / "ct.nii.gz"))
    verts, faces = phantom.make_scan("lower")
    trimesh.Trimesh(apply(phantom.scan_pose(2), verts), faces).export(d / "lower.stl")
    return d, verts


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    session = Session(memory_path=str(tmp_path_factory.mktemp("mem") / "memory.jsonl"))
    srv = serve(session)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{base}/api/{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        raw = r.read()
        return json.loads(raw) if r.headers["Content-Type"].startswith("application/json") else raw


def wait(base, answer, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        job = call(base, f"jobs/{answer['job']}")
        if job["status"] == "done":
            return job["result"]
        assert job["status"] != "error", job["error"]
        time.sleep(0.2)
    raise TimeoutError


def test_doctor_workflow(server, case_files, tmp_path):
    d, truth = case_files
    page = urllib.request.urlopen(server + "/").read().decode()
    assert "Custom Case Designer" in page and "app.js" in page

    ct = wait(server, call(server, "ct", {"path": str(d / "ct.nii.gz")}))
    assert ct["shape"] and len(ct["focus"]) == 3
    geo = call(server, "ct/geometry?axis=axial")
    png = call(server, f"ct/slice?axis=axial&pos={ct['focus'][2]}")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(png)).size == (geo["width"], geo["height"])
    # Видимая часть среза в разрешении экрана (масштаб, сдвиг): по сетке целого среза — та же картинка.
    du, dv = geo["du"], geo["dv"]
    whole = {"ua": geo["u0"] - du / 2, "ub": geo["u0"] + (geo["width"] - 0.5) * du, "va": geo["v0"] - dv / 2,
             "vb": geo["v0"] + (geo["height"] - 0.5) * dv, "cols": geo["width"], "rows": geo["height"]}
    query = f"ct/slice?axis=axial&pos={ct['focus'][2]}&"
    same = call(server, query + urllib.parse.urlencode(whole))
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(same))), np.asarray(Image.open(io.BytesIO(png))))
    part = dict(whole, ub=geo["u0"] + 0.25 * geo["width"] * du, vb=geo["v0"] + 0.2 * geo["height"] * dv, cols=300, rows=200)
    assert Image.open(io.BytesIO(call(server, query + urllib.parse.urlencode(part)))).size == (300, 200)

    scan = call(server, "scans", {"path": str(d / "lower.stl")})
    raw = call(server, f"scans/{scan['id']}/mesh")
    nv, nf = np.frombuffer(raw[:8], np.uint32)
    assert nv == scan["vertices"] and len(raw) == 8 + nv * 12 + nf * 12

    reg = wait(server, call(server, f"scans/{scan['id']}/register", {}))
    T = np.array(reg["transform"])
    stl = trimesh.load_mesh(str(d / "lower.stl"))
    assert reg["registered"] and reg["jaw"] == "lower" and reg["stats"]["p90_mm"] < 0.1
    assert len(call(server, f"scans/{scan['id']}/colors")) == 3 * nv

    # Ручная коррекция: сдвиг ухудшает метрики, «уточнить» возвращает.
    moved = T.copy()
    moved[2, 3] += 0.5
    worse = wait(server, call(server, f"scans/{scan['id']}/evaluate", {"transform": moved.tolist()}))
    assert worse["stats"]["mean_mm"] > reg["stats"]["mean_mm"] + 0.1 and worse["corrected_mm"] == pytest.approx(0.5, abs=1e-3)
    back = wait(server, call(server, f"scans/{scan['id']}/register", {"start": moved.tolist()}))
    assert np.abs(np.array(back["transform"]) - T).max() < 0.01

    overlays = call(server, "overlays", {"axis": "axial", "pos": 7.0, "visible": [scan["id"]]})
    assert overlays and overlays[0]["id"] == scan["id"] and len(overlays[0]["segments"]) % 4 == 0

    accepted = call(server, f"scans/{scan['id']}/accept", {})
    assert accepted["scan"]["accepted"] and accepted["memory"]["cases"] == 1
    assert accepted["min_cases"] == 3 and isinstance(accepted["record"]["learned"], bool)

    out = tmp_path / "out"
    res = wait(server, call(server, "export", {"out_dir": str(out), "bite": "scan", "frame": "exocad"}))
    assert {"lower.stl", "ct_teeth.stl", "case.json"} <= {p.name for p in out.iterdir()}
    assert "ct_teeth.stl" in res["files"]

    state = call(server, "state")
    assert state["ct"]["name"] == "ct.nii.gz" and state["scans"][0]["accepted"]


def test_segmentation_reregisters_scans(case_files, tmp_path, monkeypatch):
    """После сегментации сканы, совмещённые по одной плотности, совмещаются заново по зубам;
    принятый остаётся где был."""
    from casedesigner.app import session as app_session
    from casedesigner.segment import SegmentationResult

    class Segmenter:  # вместо моделей ONNX — настоящие зубы фантома
        def __init__(self, folder, device="auto"):
            self.models = []

        def plan(self, want=None):
            return []

        def run(self, vol, progress=None, want=None):
            return SegmentationResult(meshes={f"{j}_teeth": phantom.teeth_mesh(j) for j in ("upper", "lower")})

    monkeypatch.setattr(app_session, "Segmenter", Segmenter)
    d, truth = case_files
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(d / "ct.nii.gz"))
    auto, kept = (s.add_scan(str(d / "lower.stl"))["id"] for _ in range(2))
    for sid in (auto, kept):
        assert app_session.UNGUIDED in s.register(sid)["warnings"]
    s.accept(kept)
    before = s.scan_info(kept)["transform"]

    scans = {i["id"]: i for i in s.segment(models_dir="модели")["scans"]}
    assert scans[auto]["registered"] and app_session.UNGUIDED not in scans[auto]["warnings"]
    pose = np.array(scans[auto]["transform"]) @ phantom.scan_pose(2)  # скан → КТ после совмещения
    assert np.linalg.norm(apply(pose, truth) - truth, axis=1).max() < 0.15
    assert scans[kept]["accepted"] and scans[kept]["transform"] == before


def test_bite_scans_go_onto_jaw_scans(case_files, tmp_path):
    """Сканы прикуса не совмещаются с КТ: стоят на сканах челюстей (узнаются по имени и по геометрии),
    выгружаются вместе с ними и сохраняются в кейсе."""
    from test_bite_scans import bite_patch

    d, _truth = case_files
    pose = phantom.scan_pose(2)
    for jaw in ("upper", "lower"):
        v, f = phantom.make_scan(jaw)
        trimesh.Trimesh(apply(pose, v), f).export(tmp_path / f"{jaw}.stl")
    for name, side in (("p-TotalJaw0", 1), ("scan3", -1)):  # второй — без «прикусного» имени
        v, f = bite_patch(side)
        trimesh.Trimesh(apply(pose, v), f).export(tmp_path / f"{name}.stl")

    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(d / "ct.nii.gz"))
    ids = {n: s.add_scan(str(tmp_path / f"{n}.stl"))["id"] for n in ("p-TotalJaw0", "upper", "lower", "scan3")}
    assert s.scan_info(ids["p-TotalJaw0"])["role"] == "bite" and s.scan_info(ids["scan3"])["role"] == "jaw"
    for n in ("upper", "lower", "p-TotalJaw0", "scan3"):
        s.register(ids[n])
    for n in ("p-TotalJaw0", "scan3"):  # scan3 лежит на обоих сканах — тоже скан прикуса
        info = s.scan_info(ids[n])
        assert info["role"] == "bite" and info["jaw"] == "bite" and info["registered"], n
        assert np.allclose(info["transform"], s.scan_info(ids["upper"])["transform"])
        assert any("вместе с ними" in w for w in info["warnings"])
    with pytest.raises(ValueError):
        s.accept(ids["p-TotalJaw0"])

    res = s.export(str(tmp_path / "out"), "scan", "exocad")
    case = json.loads((tmp_path / "out" / "case.json").read_text(encoding="utf-8"))
    assert case["scans"]["p-TotalJaw0"]["placement"] == "jaws" and "p-TotalJaw0.stl" in res["files"]
    got = trimesh.load_mesh(str(tmp_path / "out" / "p-TotalJaw0.stl"), process=False).vertices
    src = trimesh.load_mesh(str(tmp_path / "p-TotalJaw0.stl"), process=False).vertices
    from scipy.spatial import cKDTree
    assert cKDTree(src).query(got)[0].max() < 1e-3  # в координатах сканера — как в файле

    saved = s.save_case(str(tmp_path / "кейс"))
    again = Session(memory_path=str(tmp_path / "memory.jsonl"))
    again.open_case(saved["path"])
    roles = {i["name"]: (i["role"], i["registered"]) for i in again.state()["scans"]}
    assert roles["p-TotalJaw0"] == ("bite", True) and roles["scan3"] == ("bite", True)


def test_export_into_exocad_project(case_files, tmp_path):
    """Папка проекта exocad: всё — в его подпапку, в координатах сцены (скан × матрица сканера)."""
    import shutil

    from test_exocad_project import SCANNER, matrix_xml

    d, _truth = case_files
    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(d / "lower.stl", project / "p-lowerjaw.stl")
    (project / "p.dentalProject").write_text("<Treatment/>", encoding="utf-8")
    (project / "p.matrix4").write_text(matrix_xml("Matrix4", SCANNER), encoding="utf-8")

    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(d / "ct.nii.gz"))
    sid = s.add_scan(str(project / "p-lowerjaw.stl"))["id"]
    s.register(sid)
    plain = s.export(str(tmp_path / "plain"), "scan", "exocad")  # обычная выгрузка: координаты сканера
    res = s.export(str(project), "scan", "exocad")
    out = project / "CustomCaseDesigner"
    assert res["out_dir"] == str(out) and res["frame"].startswith("exocad project scene")
    assert "Проект exocad" in res["notes"][0] and not any("не найден" in n for n in res["notes"])
    for name in ("p-lowerjaw.stl", "ct_teeth.stl"):  # сцена = координаты сканера × матрица сканера
        a = trimesh.load_mesh(str(tmp_path / "plain" / name), process=False).vertices
        b = trimesh.load_mesh(str(out / name), process=False).vertices
        assert np.abs(apply(SCANNER, a) - b).max() < 1e-3, name
    assert sorted(p.name for p in project.iterdir()) == ["CustomCaseDesigner", "p-lowerjaw.stl", "p.dentalProject",
                                                         "p.matrix4"]  # файлы exocad не тронуты
    assert plain["frame"].startswith("scanner coordinates")

    # Скан не из этого проекта (тот же, но в других координатах файла) — предупреждение.
    moved = trimesh.load_mesh(str(d / "lower.stl"))
    moved.apply_transform(phantom.scan_pose(9))
    moved.export(tmp_path / "elsewhere.stl")
    other = s.add_scan(str(tmp_path / "elsewhere.stl"))["id"]
    s.remove_scan(sid)
    s.register(other)
    assert any("не найден" in n for n in s.export(str(project), "scan", "exocad")["notes"])


def test_segment_parts_setting(case_files, tmp_path, monkeypatch):
    """«Что сегментировать»: выбор сохраняется; невыбранное не показывается и не считается,
    а зубы для совмещения нужны всегда."""
    from casedesigner.app import session as app_session
    from casedesigner.segment import SegmentationResult

    asked = []

    class Segmenter:
        def __init__(self, folder, device="auto"):
            self.models = []

        def plan(self, want=None):
            return []

        def run(self, vol, progress=None, want=None):
            asked.extend(k for k in ("mandibular_canal", "mandible", "upper_teeth", "lower_teeth", "skull") if want(k))
            meshes = {f"{j}_teeth": phantom.teeth_mesh(j) for j in ("upper", "lower")}
            meshes["mandibular_canal"] = meshes["mandible"] = phantom.teeth_mesh("lower")
            return SegmentationResult(meshes=meshes)

    monkeypatch.setattr(app_session, "Segmenter", Segmenter)
    d, _truth = case_files
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    assert set(s.segment_parts_info()["selected"]) == {p for p, _t, _k in app_session.SEGMENT_PARTS}  # по умолчанию всё
    with pytest.raises(ValueError):
        s.set_segment_parts(["нет такой"])
    s.set_segment_parts(["jaws"])
    assert Session(memory_path=str(tmp_path / "memory.jsonl")).segment_parts_info()["selected"] == ["jaws"]  # сохранено

    s.load_ct(str(d / "ct.nii.gz"))
    res = s.segment(models_dir="модели")
    assert sorted(asked) == ["lower_teeth", "mandible", "upper_teeth"]  # каналы и череп не считаются
    assert [i["key"] for i in res["structures"]] == ["mandible"]  # зубы не показаны…
    assert s.case.guided == {"upper", "lower"}  # …но опора совмещения — по ним
    s.set_segment_parts([])
    with pytest.raises(ValueError, match="не выбрано"):
        s.segment(models_dir="модели")


def test_incognito_setting(server, tmp_path):
    """Инкогнито (скрыть имена пациентов и пути на экране) помнится между запусками программы."""
    assert call(server, "state")["incognito"] is False
    assert call(server, "settings/incognito", {"on": True}) == {"incognito": True}
    assert call(server, "state")["incognito"] is True
    assert call(server, "settings/incognito", {"on": "да"}) == {"incognito": False}  # включает только true
    Session(memory_path=str(tmp_path / "memory.jsonl")).set_incognito(True)
    assert Session(memory_path=str(tmp_path / "memory.jsonl")).state()["incognito"] is True


def test_errors_are_readable(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(server, "scans", {"path": "/нет/такого/скана.stl"})
    assert e.value.code == 400
    job = call(server, "ct", {"path": "/нет/такого/КТ"})
    for _ in range(50):
        state = call(server, f"jobs/{job['job']}")
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "error" and state["error"]


def test_groups():
    assert group_of("teeth/tooth_36") == "Зубы" and group_of("pulp/pulp_11") == "Зубы"
    assert group_of("mandibular_canal") == "Каналы" and group_of("teeth/implant") == "Ортопедия и импланты"
    assert group_of("nasal_cavity") == "Пазухи и дыхательные пути" and group_of("что-то") == "Другое"


def test_case_file_round_trip(case_files, tmp_path):
    """Кейс в файл и обратно: КТ, скан на своём месте, принятие и автоматическое положение для обучения."""
    d, _verts = case_files
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(d / "ct.nii.gz"))
    sid = s.add_scan(str(d / "lower.stl"))["id"]
    s.register(sid)
    moved = np.asarray(s.scans[sid]["transform"]).copy()
    moved[:3, 3] += [0.3, 0, 0]
    s.evaluate(sid, moved)
    s.accept(sid)
    s.set_landmark("N", [1.0, -60.0, 40.0])
    s.landmarks["S"], s.suggested = np.array([0.0, 10.0, 45.0]), {"S"}  # найдена программой, не проверена
    path = s.save_case(str(tmp_path / "кейс"))["path"]
    assert path.endswith(".ccdcase")

    t = Session(memory_path=str(tmp_path / "memory.jsonl"))
    assert t.recent_cases()[0]["path"] == path
    state = t.open_case(path)
    item = next(iter(t.scans.values()))
    assert np.allclose(item["transform"], moved) and item["accepted"]
    assert not np.allclose(item["auto"].transform, moved)  # что предлагала программа — тоже сохранено
    assert state["case"]["name"] == "кейс" and state["scans"][0]["registered"]
    assert np.allclose(t.landmarks["N"], [1.0, -60.0, 40.0]) and t.suggested == {"S"}
    with pytest.raises(FileNotFoundError, match="не найден"):
        t.open_case(path, ct_path=str(tmp_path / "нет"))
