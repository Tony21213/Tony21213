import numpy as np
import pytest

import phantom
from casedesigner import articulators as arts
from casedesigner import landmarks as lmk
from casedesigner.fusion import Scan, export_case
from casedesigner.register import apply, axis_angle, rigid


def skull_points(tilt_deg=7.0):
    """Ориентиры «черепа», наклонённого вокруг поперечной оси на tilt_deg (LPS: x — влево, y — назад, z — вверх)."""
    pts = {
        "Po_R": [-60, 10, 0], "Po_L": [60, 10, 0], "Or_R": [-33, -70, 0], "Or_L": [33, -70, 0],
        "Co_R": [-50, 5, -12], "Co_L": [50, 5, -12],
        "HN_R": [-20, -20, -35], "HN_L": [20, -20, -35], "IP": [0, -62, -38],
        "N": [0, -85, 15], "ANS": [0, -78, -30], "PNS": [0, -30, -32], "Ba": [0, 15, -20],
    }
    R = axis_angle(np.array([1.0, 0, 0]), np.radians(tilt_deg))
    return {k: R @ np.array(v, float) + [3, -4, 100] for k, v in pts.items()}, R


def test_reference_frame_axes():
    pts, R = skull_points()
    T = lmk.reference_frame(pts, "frankfurt")
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3)) and np.linalg.det(T[:3, :3]) == pytest.approx(1)
    hinge = (pts["Co_R"] + pts["Co_L"]) / 2
    assert apply(T, hinge[None])[0] == pytest.approx([0, 0, 0], abs=1e-9)
    # Порионы и орбитали лежат в плоскости z = const (Франкфурт горизонтален).
    z = apply(T, np.array([pts["Po_R"], pts["Po_L"], (pts["Or_R"] + pts["Or_L"]) / 2]))[:, 2]
    assert np.ptp(z) < 1e-9 and z[0] == pytest.approx(12.0, abs=1e-6)
    # X — вправо пациента, Y — вперёд.
    assert apply(T, pts["Co_R"][None])[0][0] > 0 and apply(T, pts["Or_R"][None])[0][1] > 0
    # Наклон HIP к Франкфурту одинаков при любом наклоне головы на снимке.
    a = lmk.plane_angles(pts)["frankfurt/hip"]
    assert a == pytest.approx(lmk.plane_angles(skull_points(-20)[0])["frankfurt/hip"], abs=1e-6)


def test_missing_points_reported():
    pts, _ = skull_points()
    del pts["Or_L"]
    with pytest.raises(ValueError, match="Орбиталь левая"):
        lmk.reference_frame(pts, "frankfurt")
    assert lmk.plane(pts, "camper") is None


def test_articulator_calibration_roundtrip():
    pts, _ = skull_points()
    art = next(a for a in arts.load() if a.key == "reference_sl")
    truth = arts.Articulator(art.key, art.name, art.maker, "frankfurt", (1.5, 95.0, -20.0), 9.0, True)
    sample = arts.articulator_frame(pts, truth)  # «образец из exocad»
    cal = arts.calibrate(art, pts, sample)
    assert cal.calibrated and cal.tilt_deg == pytest.approx(9.0, abs=1e-6)
    assert cal.hinge_mm == pytest.approx((1.5, 95.0, -20.0), abs=1e-3)
    assert np.allclose(arts.articulator_frame(pts, cal), sample, atol=1e-6)


def test_articulator_list_and_saved_calibration(tmp_path):
    names = {a.maker for a in arts.load()}
    assert {"Amann Girrbach", "Gamma Dental", "KaVo", "Ivoclar", "Whip Mix", "Zirkonzahn"} <= names
    path = str(tmp_path / "articulators.json")
    custom = arts.Articulator("artex_cr", "Artex CR", "Amann Girrbach", "camper", (0, 90, -10), 3.0, True)
    arts.save(path, [custom])
    loaded = {a.key: a for a in arts.load(path)}
    assert loaded["artex_cr"].calibrated and loaded["artex_cr"].plane == "camper"
    assert len(loaded) == len(arts.BUILTIN)


def test_suggest_condyles_and_porion():
    rng = np.random.default_rng(0)
    body = rng.uniform([-45, -40, -40], [45, 10, -15], (3000, 3))
    heads = {"Co_R": np.array([-50.0, 5, 0]), "Co_L": np.array([50.0, 5, 0])}
    caps = [c + rng.normal(0, 0.4, (300, 3)) for c in heads.values()]
    found = lmk.suggest_condyles(np.vstack([body, *caps]))
    for k, c in heads.items():
        assert found[k] == pytest.approx(c + [0, 0, 0.6], abs=1.0)
    canal = np.c_[rng.uniform(-70, -50, 500), rng.uniform(5, 12, 500), rng.uniform(-4, 4, 500)]
    po = lmk.suggest_porion(canal, "Po_R", 0.0)["Po_R"]
    assert po[0] < -60 and po[2] > 3.5


def test_export_in_reference_frame(tmp_path, jaw_ct):
    verts, faces = phantom.make_scan("lower")
    reg = jaw_ct.register(Scan("lower", apply(phantom.scan_pose(2), verts), faces), jaw="lower")
    F = rigid(axis_angle(np.array([0, 0, 1.0]), 0.3), np.array([5.0, -7, 2]))  # «КТ → плоскость»
    for bite in ("scan", "ct"):
        out = tmp_path / bite
        report = export_case(str(out), [reg], bite=bite, frame="reference", ct=jaw_ct, reference=F,
                             reference_name="Франкфурт")
        import trimesh

        got = np.asarray(trimesh.load_mesh(str(out / "lower.stl"), process=False).vertices)
        expected = apply(F, truth := verts)[faces].reshape(-1, 3)
        assert np.abs(got - expected).max() < 0.05, bite
        assert report["frame"].startswith("Франкфурт")
    with pytest.raises(ValueError):
        export_case(str(tmp_path / "x"), [reg], frame="reference")
