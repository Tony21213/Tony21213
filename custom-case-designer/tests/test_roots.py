"""Корни зубов из КТ под коронками сканов (casedesigner/roots.py) и их выгрузка в exocad — на синтетических зубах:
капсула (корень и коронка), «скан» — коронка с небольшим смещением и десна кольцом у шейки."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from casedesigner import exocad_project as ep
from casedesigner import roots as rt
from casedesigner.register import apply, rigid

MARGIN_Z = 4.0  # линия десны на зубе: выше — видимая коронка
HALF = 10.0  # капсула: от −HALF−R до +HALF+R по оси
R = 3.5


def tooth(T=np.eye(4)):
    """Зуб КТ: капсула вдоль z (коронка сверху), и скан: коронка (смещена наружу на 0,1 мм) + десна кольцом."""
    t = trimesh.creation.capsule(height=2 * HALF, radius=R, count=[24, 24]).subdivide_to_size(0.6)  # как сетка КТ
    v = np.asarray(t.vertices)
    crown = v[v[:, 2] > MARGIN_Z]
    radial = crown.copy()
    radial[:, 2] = 0
    crown = crown + 0.1 * radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ring = np.vstack([np.c_[r * np.cos(ang), r * np.sin(ang), np.full_like(ang, MARGIN_Z - 0.2)]
                      for r in np.linspace(R + 0.8, 9.0, 8)])  # десна — не ближе 0,8 мм к корню
    scan = np.vstack([crown, ring])
    return apply(T, v), np.asarray(t.faces), apply(T, scan)


def test_root_below_gum_line_and_axis():
    v, f, scan = tooth()
    root, why = rt.split_tooth(v, f, cKDTree(scan), 11)
    assert why is None
    assert root.root_mm == pytest.approx(MARGIN_Z + HALF + R, abs=1.0)
    assert root.crown_mm == pytest.approx(HALF + R - MARGIN_Z, abs=1.0)
    assert root.axis @ [0, 0, 1.0] > 0.995  # от верхушки к коронке
    assert root.vertices[:, 2].max() <= MARGIN_Z + rt.OVERLAP_MM + 0.8  # корень — ниже десны (с запасом под коронку)
    assert root.vertices[:, 2].min() == pytest.approx(-(HALF + R), abs=0.1)
    assert root.apex[2] == pytest.approx(-(HALF + R) + 0.5, abs=1.0) and root.tip[2] == pytest.approx(HALF + R, abs=0.2)


def test_tilted_tooth_follows_its_axis():
    tilt = rigid(Rotation.from_euler("x", 25, degrees=True).as_matrix(), [5.0, -3.0, 2.0])
    v, f, scan = tooth(tilt)
    root, why = rt.split_tooth(v, f, cKDTree(scan), 21)
    assert why is None
    assert root.axis @ (tilt[:3, :3] @ [0, 0, 1.0]) > 0.995
    assert root.root_mm == pytest.approx(MARGIN_Z + HALF + R, abs=1.0)


def test_stray_fragment_short_tooth_and_tooth_off_scan():
    v, f, scan = tooth()
    blob = trimesh.creation.icosphere(radius=2.0)
    v2 = np.vstack([v, np.asarray(blob.vertices) + [0, 40.0, 0]])  # чужой кусок метки — далеко
    f2 = np.vstack([f, np.asarray(blob.faces) + len(v)])
    root, why = rt.split_tooth(v2, f2, cKDTree(scan), 11)
    assert why is None and np.abs(root.vertices[:, 1]).max() < R + 0.1
    stub = trimesh.creation.capsule(height=2.0, radius=2.0)  # обломок 6 мм — сегментирован не целиком
    sv = np.asarray(stub.vertices) + [0, 0, HALF]
    assert rt.split_tooth(sv, stub.faces, cKDTree(np.vstack([sv, scan])), 11) == (None, "зуб КТ сегментирован не целиком")
    far = cKDTree(scan + [0, 0, 100.0])
    assert rt.split_tooth(v, f, far, 11) == (None, "коронки нет на скане")


def test_split_teeth_by_jaw_and_reversed_axis_rejected():
    up_v, up_f, up_scan = tooth(rigid(Rotation.from_euler("x", 180, degrees=True).as_matrix(), [0, 0, 20.0]))
    lo_v, lo_f, lo_scan = tooth(rigid(np.eye(3), [0, 0, -20.0]))
    # нижний зуб с номером верхнего ряда «наоборот»: коронка смотрит вниз, а должна — к верхнему скану
    bad_v, bad_f, bad_scan = tooth(rigid(Rotation.from_euler("x", 180, degrees=True).as_matrix(), [15.0, 0, -20.0]))
    structures = {"teeth/tooth_11": SimpleNamespace(vertices=up_v, faces=up_f),
                  "teeth/tooth_41": SimpleNamespace(vertices=lo_v, faces=lo_f),
                  "teeth/tooth_46": SimpleNamespace(vertices=bad_v, faces=bad_f),
                  "mandible": SimpleNamespace(vertices=lo_v, faces=lo_f)}
    found, notes = rt.split_teeth(structures, {"upper": up_scan, "lower": np.vstack([lo_scan, bad_scan])})
    assert sorted(found) == [11, 41]
    assert found[11].axis @ [0, 0, -1.0] > 0.99 and found[41].axis @ [0, 0, 1.0] > 0.99
    assert any("против дуги" in n and "46" in n for n in notes)
    v, faces = rt.jaw_mesh(found, "lower")
    assert len(v) == len(found[41].vertices) and faces.max() < len(v)
    av, _af = rt.axes_mesh(found, "upper")
    assert av[:, 2].min() == pytest.approx(20 - HALF - R - 3.0, abs=0.5)  # ось верхнего зуба — за край коронки на 3 мм
    assert [s["fdi"] for s in rt.summary(found)] == [11, 41]


def test_partial_info_lists_bone_roots_axes(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "p.dentalProject").write_text("<Treatment/>", encoding="utf-8")
    proj = ep.ExocadProject(str(folder), [], np.eye(4), "test", stem="p")
    box = trimesh.creation.box()
    files, notes = ep.write_bone_frameworks(proj, {"lower": box}, {"lower": box, "upper": box}, {"lower": box})
    assert notes == []
    assert files == ["bone_lower.stl", "roots_lower.stl", "tooth_axes_lower.stl", "p-lowerjaw.partialInfo",
                     "roots_upper.stl", "p-upperjaw.partialInfo"]
    items = ET.parse(folder / "p-lowerjaw.partialInfo").getroot().findall("PartialFileList/PartialFile")
    assert [i.findtext("FileName") for i in items] == ["bone_lower.stl", "roots_lower.stl", "tooth_axes_lower.stl"]
    assert [i.findtext("MaterialName") for i in items] == [ep.BONE_MATERIAL, ep.ROOTS_MATERIAL, ep.AXES_MATERIAL]
    upper = ET.parse(folder / "p-upperjaw.partialInfo").getroot()
    assert upper.findtext("PartialFileList/PartialFile/FileName") == "roots_upper.stl"  # кости нет — корни первыми


def test_case_from_session_roots_follow_bite():
    """Выгрузка с артикулятором: корни верхних зубов — как на КТ, нижних — в прикусе выгрузки (как кость НЧ)."""
    from casedesigner import exocad_articulator_export as xa
    from casedesigner import kinematics as kin
    from casedesigner import motion as mo

    up_v, up_f, up_scan = tooth(rigid(Rotation.from_euler("x", 180, degrees=True).as_matrix(), [0, 0, 20.0]))
    lo_v, lo_f, lo_scan = tooth(rigid(np.eye(3), [0, 0, -20.0]))
    bite = rigid(np.eye(3), [0.0, 0.0, -1.5])
    anatomy = mo.Anatomy(np.eye(4), {"incisal": np.zeros(3), "condyle_right": np.array([50.0, -80, 30]),
                                     "condyle_left": np.array([-50.0, -80, 30])}, "тест")
    jaw = SimpleNamespace(anatomy=anatomy, lower_fix=np.eye(4), settings=kin.Settings(), movements={}, mounting="ct")
    no_faces = np.zeros((0, 3), int)
    up_item = {"role": "jaw", "path": None, "reg": SimpleNamespace(jaw="upper", transform=np.eye(4)),
               "scan": SimpleNamespace(vertices=up_scan, faces=no_faces)}
    lo_item = {"role": "jaw", "path": None, "reg": SimpleNamespace(jaw="lower", transform=np.eye(4)),
               "scan": SimpleNamespace(vertices=lo_scan, faces=no_faces)}
    session = SimpleNamespace(jaw=jaw, bite_motion=bite, scans={"u": up_item, "l": lo_item},
                              structures={"teeth/tooth_11": SimpleNamespace(vertices=up_v, faces=up_f),
                                          "teeth/tooth_41": SimpleNamespace(vertices=lo_v, faces=lo_f)},
                              landmarks={}, suggested=set(), landmark_sources={},
                              _jaw_items=lambda: (up_item, lo_item))
    case = xa.case_from_session(session)
    found, _notes = rt.split_teeth(session.structures, {"upper": up_scan, "lower": lo_scan})
    assert np.allclose(case.roots["upper"].vertices, rt.jaw_mesh(found, "upper")[0])
    assert np.allclose(case.roots["lower"].vertices, apply(bite, rt.jaw_mesh(found, "lower")[0]))
    assert set(case.tooth_axes) == {"upper", "lower"}
    assert [t["fdi"] for t in case.roots_info["teeth"]] == [11, 41]


def test_session_roots_http_and_cache(tmp_path):
    """Шаг «Экспорт»: по зубам и сетки корней и осей для окна 3D (GET export/roots); расчёт — один раз на положение."""
    import json
    import urllib.request

    from casedesigner.app import server as app_server
    from casedesigner.app.common import Registration, Scan
    from casedesigner.app.session import Session

    up_v, up_f, up_scan = tooth(rigid(Rotation.from_euler("x", 180, degrees=True).as_matrix(), [0, 0, 20.0]))
    lo_v, lo_f, lo_scan = tooth(rigid(np.eye(3), [0, 0, -20.0]))
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.structures = {"teeth/tooth_11": trimesh.Trimesh(up_v, up_f, process=False),
                    "teeth/tooth_41": trimesh.Trimesh(lo_v, lo_f, process=False)}
    for jaw, pts in (("upper", up_scan), ("lower", lo_scan)):
        scan = Scan(jaw, pts, np.zeros((0, 3), int))
        reg = Registration(scan, jaw, np.eye(4), np.full(len(pts), np.nan))
        s.scans[jaw] = {"scan": scan, "path": None, "color": "#fff", "jaw": jaw, "reg": reg, "auto": reg, "role": "jaw",
                        "lower_bite": None, "transform": np.eye(4), "accepted": True, "guided": True}
    first = s.tooth_roots()
    assert s.tooth_roots() is first  # кэш: то же положение сканов
    httpd = app_server.serve(s)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api"
    try:
        with urllib.request.urlopen(f"{base}/export/roots") as r:
            info = json.loads(r.read())
        assert [t["fdi"] for t in info["teeth"]] == [11, 41] and info["notes"] == []
        with urllib.request.urlopen(f"{base}/export/roots?jaw=lower&part=axes") as r:
            head = np.frombuffer(r.read(8), np.uint32)
        assert head[0] > 0 and head[1] > 0  # сетка: число вершин и граней
    finally:
        httpd.shutdown()
        httpd.server_close()
    web = (__import__("pathlib").Path(rt.__file__).parent / "app" / "web" / "steps" / "export.js").read_text(encoding="utf-8")
    assert "export/roots?jaw=" in web and "Корни из КТ" in web
