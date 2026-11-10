"""Дистракция: жёсткое движение, неподвижные резцы, вертикаль монтажа."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from casedesigner import motion as mo, rehab as rh
from casedesigner.register import apply, rigid


def anatomy():
    return mo.Anatomy(np.eye(4), {"condyle_right": np.array([50., -90., 0.]),
                                 "condyle_left": np.array([-50., -90., 0.]),
                                 "incisal": np.array([0., 0., -30.])}, "тест")


@pytest.mark.parametrize("right,left", [(0, 0), (3, 3), (3, 0), (0, 3), (1.2, 2.4)])
def test_distraction_rigid_with_fixed_incisal(right, left):
    a = anatomy()
    T, report = rh.distraction_transform(a, right, left)
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1)
    assert np.allclose(apply(T, a.points["incisal"][None])[0], a.points["incisal"], atol=1e-9)
    for side, mm in (("right", right), ("left", left)):
        p = a.points[f"condyle_{side}"]
        assert (p - apply(T, p[None])[0])[2] == pytest.approx(mm, abs=.02)
        assert report[f"{side}_mm"] == pytest.approx(mm, abs=.02)
    if right != left:
        axis = apply(T, np.array([a.points["condyle_right"], a.points["condyle_left"]]))
        assert abs(axis[0, 2] - axis[1, 2]) > 1
    if not right and not left:
        assert np.array_equal(T, np.eye(4))


def test_distraction_respects_oblique_mount():
    a = anatomy()
    Q = rigid(Rotation.from_euler("xyz", [23, -17, 39], degrees=True).as_matrix(), [30, 70, -12])
    moved = mo.Anatomy(a.frame @ np.linalg.inv(Q), {k: apply(Q, v[None])[0] for k, v in a.points.items()}, "тест")
    T, _ = rh.distraction_transform(a, 2, .5)
    got, _ = rh.distraction_transform(moved, 2, .5)
    assert np.allclose(got, Q @ T @ np.linalg.inv(Q), atol=1e-8)


@pytest.mark.parametrize("value", [-.1, 3.1, float("nan"), float("inf")])
def test_distraction_rejects_invalid(value):
    with pytest.raises(ValueError, match="от 0 до 3"):
        rh.distraction_transform(anatomy(), value, 0)


def session(tmp_path):
    from types import SimpleNamespace
    from casedesigner.app.session import Session

    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    vertices = np.array([[x, y, -30.] for x in np.linspace(-25, 25, 51) for y in np.linspace(-35, 0, 36)])
    s.jaw.set_scans(SimpleNamespace(vertices=vertices + [0, 0, 40], faces=np.zeros((0, 3), int), name="u"),
                    SimpleNamespace(vertices=vertices, faces=np.zeros((0, 3), int), name="l"))
    s.jaw.anatomy = anatomy()
    s._articulation_ready = True
    return s


def test_session_composes_raise_after_distraction(tmp_path):
    s = session(tmp_path)
    a = s.jaw.anatomy
    D, _ = rh.distraction_transform(a, 2, 1)
    moved = rh.moved_anatomy(a, D)
    R, _ = rh.raise_transform(moved, 4)
    info = s.articulation_rehab(raise_mm=4, distraction_right_mm=2, distraction_left_mm=1)
    assert np.allclose(s.rehab_transform(), R @ D)
    assert np.allclose(info["display_bite"], R @ D)
    for side in ("right", "left"):
        p = a.points[f"condyle_{side}"]
        assert np.allclose(apply(R @ D, p[None]), apply(D, p[None]), atol=1e-8)
    assert info["report"]["incisal_mm"] == pytest.approx(4, abs=.01)
    before = dict(s.rehab)
    with pytest.raises(ValueError):
        s.articulation_rehab(distraction_right_mm=4)
    assert s.rehab == before
    s.articulation_rehab(raise_mm=0, distraction_right_mm=0, distraction_left_mm=0)
    assert s.rehab_transform() is None


def test_case_roundtrip_and_legacy_defaults(tmp_path):
    import trimesh
    from casedesigner.app.session import Session

    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    for jaw, dz in (("upper", 10), ("lower", 0)):
        path = tmp_path / f"{jaw}jaw.stl"
        mesh = trimesh.creation.box((30, 30, 3))
        mesh.apply_translation([0, 0, dz])
        mesh.export(path)
        sid = s.add_scan(str(path))["id"]
        s.register(sid, jaw=jaw)
    s.jaw.anatomy = anatomy()
    s.articulation_rehab(distraction_right_mm=2.3, distraction_left_mm=.7, raise_mm=1)
    saved = s.save_case(str(tmp_path / "case.ccdcase"))
    again = Session(memory_path=str(tmp_path / "memory2.jsonl"))
    again.open_case(saved["path"])
    assert again._rehab() == s.rehab
    again.save_case(str(tmp_path / "case2.ccdcase"))
    s.open_case(str(tmp_path / "case2.ccdcase"))
    assert s._rehab()["distraction_right_mm"] == 2.3
    s.rehab.pop("distraction_right_mm")
    s.rehab.pop("distraction_left_mm")
    s.save_case(str(tmp_path / "old.ccdcase"))
    again.open_case(str(tmp_path / "old.ccdcase"))
    assert again._rehab()["distraction_right_mm"] == again._rehab()["distraction_left_mm"] == 0


def test_movements_start_at_distracted_condyles(tmp_path, monkeypatch):
    import trimesh
    from casedesigner import kinematics as kin
    from casedesigner.jawcase import Mesh

    s = session(tmp_path)
    # Небольшие замкнутые поверхности: расчёт ведения доступен и без КТ.
    mesh = trimesh.creation.box((40, 30, 2))
    s.jaw.upper = Mesh(mesh.vertices + [0, -15, -25], mesh.faces, "верх")
    s.jaw.lower = Mesh(mesh.vertices + [0, -15, -30], mesh.faces, "низ")
    a = s.jaw.anatomy
    s.articulation_rehab(distraction_right_mm=2, distraction_left_mm=1, raise_mm=1)
    base = s.rehab_transform()
    heads = rh.moved_anatomy(a, base)
    observed = []
    original = kin.protrusion

    def capture(anatomy, settings, travel, occ):
        observed.append((anatomy, occ))
        return original(anatomy, settings, travel, None)

    monkeypatch.setattr(kin, "protrusion", capture)
    s.articulation_generate(travel=1)
    assert len(s.jaw.movements) == 3
    assert s.jaw.anatomy is a and np.array_equal(s.jaw.lower_fix, np.eye(4))
    for key in ("condyle_right", "condyle_left", "incisal"):
        assert np.allclose(observed[0][0].points[key], heads.points[key])
    guides = s.articulation_guides()
    assert np.allclose(guides["points"]["condyle_right"], heads.points["condyle_right"], atol=.001)
    path = next(p for p in guides["paths"] if "протруз" in p["name"].lower())
    assert np.allclose(path["condyle_right"][0], heads.points["condyle_right"], atol=.001)
    assert np.linalg.norm(np.array(path["condyle_right"][-1]) - path["condyle_right"][0]) > .5
    s.articulation_rehab(distraction_right_mm=0)
    assert not s.jaw.movements  # старые рассчитанные пути больше не соответствуют положению


from test_exocad_articulator_export import case as export_case, project, register, exocad_app


def test_distraction_in_export(export_case, project, register, tmp_path):
    import json
    from casedesigner import exocad_articulator_export as xa
    import trimesh
    from types import SimpleNamespace

    s = session(tmp_path)
    s.jaw.anatomy = export_case.anatomy
    s.articulation_rehab(distraction_right_mm=2, distraction_left_mm=.5, raise_mm=1)
    T = s.rehab_transform()
    up = {"reg": SimpleNamespace(jaw="upper", transform=np.eye(4)), "scan": export_case.upper, "path": None}
    lo = {"reg": SimpleNamespace(jaw="lower", transform=np.eye(4)), "scan": export_case.lower, "path": None}
    bone = trimesh.creation.box((80, 60, 10))
    fake = SimpleNamespace(jaw=s.jaw, bite_motion=None, scans={"u": up, "l": lo},
                           structures={"mandible": bone}, landmarks={}, suggested=set(), landmark_sources={},
                           _jaw_items=lambda: (up, lo), rehab_transform=s.rehab_transform,
                           rehab_anatomy=s.rehab_anatomy, rehab_info=s.rehab_info, rehab_template=s.rehab_template)
    case = xa.case_from_session(fake)
    assert np.allclose(case.lower.vertices, apply(T, export_case.lower.vertices))
    assert np.allclose(case.bones["lower"].vertices, apply(T, bone.vertices))
    assert np.allclose(case.upper.vertices, export_case.upper.vertices)
    assert case.rehab_info["distraction_right_mm"] == 2
    result = xa.export(case, str(project), str(tmp_path / "distraction-export"), register=register,
                       exocad_app=exocad_app(tmp_path, custom=True))
    saved = json.loads((tmp_path / "distraction-export" / xa.SUBFOLDER / "articulator.json").read_text(encoding="utf-8"))
    assert saved["rehab"]["distraction_left_mm"] == .5
    assert any("дистракция мыщелков" in note and "справа 2 мм" in note for note in result["notes"])
    lower = trimesh.load_mesh(tmp_path / "distraction-export" / "distraction-export-lowerjaw.ply", process=False)
    expected = apply(np.asarray(result["case_to_articulator"]), case.lower.vertices)
    assert np.allclose(lower.vertices, expected, atol=.001)


def test_http_distraction_validation_and_pose(tmp_path):
    import json
    from urllib.request import build_opener, ProxyHandler, Request
    from urllib.error import HTTPError
    from casedesigner.app.server import serve

    s = session(tmp_path)
    server = serve(s)
    client = build_opener(ProxyHandler({}))
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/articulation/rehab"
        body = json.dumps({"distraction_right_mm": 2, "distraction_left_mm": 1}).encode()
        with client.open(Request(url, body, {"Content-Type": "application/json"})) as response:
            info = json.load(response)
        assert info["distraction"]["right_mm"] == pytest.approx(2, abs=.02)
        assert info["report"]["incisal_mm"] == pytest.approx(0, abs=.01)
        assert not np.allclose(info["display_bite"], np.eye(4))
        with pytest.raises(HTTPError) as err:
            client.open(Request(url, b'{"distraction_left_mm":4}', {"Content-Type": "application/json"}))
        assert err.value.code == 400
        assert s.rehab["distraction_left_mm"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_export_builds_movements_from_new_axis(tmp_path):
    import trimesh
    from types import SimpleNamespace
    from casedesigner import exocad_articulator_export as xa
    from casedesigner.jawcase import Mesh

    s = session(tmp_path)
    box = trimesh.creation.box((40, 30, 2))
    s.jaw.upper = Mesh(box.vertices + [0, -15, -25], box.faces, "верх")
    s.jaw.lower = Mesh(box.vertices + [0, -15, -30], box.faces, "низ")
    for side, own in (("upper", s.jaw.upper), ("lower", s.jaw.lower)):
        s.scans[side] = {"scan": own, "role": "jaw", "path": None,
                         "reg": SimpleNamespace(jaw=side, transform=np.eye(4))}
    s.articulation_rehab(distraction_right_mm=2, distraction_left_mm=1)
    current = s.rehab_anatomy()
    case = xa.case_from_session(s)
    assert len(case.movements) == 4 and not s.jaw.movements
    opening = next(frames for name, frames, fps in case.movements if "открыв" in name.lower())
    for side in ("right", "left"):
        point = current.points[f"condyle_{side}"]
        assert np.allclose(apply(opening[1], point[None])[0], point, atol=1e-8)
    assert np.allclose(case.lower.vertices, apply(s.rehab_transform(), s.jaw.lower.vertices))
