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
