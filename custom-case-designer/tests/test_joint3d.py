"""Суставной путь в 3D (casedesigner/joint3d.py) на искусственном суставе известной формы: головка — эллипсоид,
крыша ямки — плоская, скат бугорка — под ALPHA, внутренняя стенка — в W0 от головки и уходит внутрь под BETA."""

import numpy as np
import pytest
import trimesh

from casedesigner import joint3d as j3

ALPHA, BETA, GAP, W0 = 40.0, 12.0, 1.0, 1.0
HEAD = (9.0, 4.0, 5.0)  # полуоси головки: вбок, вперёд, вверх


def head_points():
    s = trimesh.creation.icosphere(subdivisions=4)
    v = np.asarray(s.vertices) * HEAD
    return v + [0, 0, -HEAD[2]]  # верхушка (Co) — в начале координат


def fossa(slope=ALPHA, beta=BETA, wall=True):
    """Поверхность кости (сетка): крыша и скат — высотой над головкой; стенка — внутри (−x, правый сустав)."""
    xs = np.arange(-14.0, 14.01, 0.5)
    ys = np.arange(-10.0, 24.01, 0.5)

    # скат — касательная к головке на зазоре GAP: точка эллипсоида с нормалью «вперёд-вверх» под углом ската
    a, b, c = HEAD
    n = np.array([0.0, np.sin(np.radians(slope)), np.cos(np.radians(slope))])
    p = np.array([0.0, 0.0, -c]) + np.array([a * a * n[0], b * b * n[1], c * c * n[2]]) / np.sqrt(
        (a * n[0]) ** 2 + (b * n[1]) ** 2 + (c * n[2]) ** 2)
    q = p + GAP * n
    tan = np.tan(np.radians(slope))

    def zf(y):
        slope_z = q[2] - (y - q[1]) * tan
        crest = q[2] - 12 * tan  # вершина бугорка — на 12 мм вперёд
        return min(GAP, max(slope_z, crest))

    grid = np.array([[(x, y, zf(y)) for x in xs] for y in ys])
    ny, nx = grid.shape[:2]
    idx = np.arange(nx * ny).reshape(ny, nx)
    q = np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
    faces = np.r_[q[:, [0, 2, 1]], q[:, [0, 3, 2]]]  # нормали — вниз, к суставу
    parts = [trimesh.Trimesh(grid.reshape(-1, 3), faces, process=False)]
    if wall:
        zs = np.arange(-15.0, GAP + 0.01, 0.5)
        wg = np.array([[(-(HEAD[0] + W0) - max(y, 0.0) * np.tan(np.radians(beta)), y, z) for z in zs] for y in ys])
        ny, nz = wg.shape[:2]
        idx = np.arange(ny * nz).reshape(ny, nz)
        q = np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
        wf = np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
        w = trimesh.Trimesh(wg.reshape(-1, 3), wf, process=False)
        if w.face_normals[:, 0].mean() < 0:  # нормаль стенки — к головке (+x)
            w = trimesh.Trimesh(wg.reshape(-1, 3), wf[:, ::-1], process=False)
        parts.append(w)
    m = trimesh.util.concatenate(parts)
    return np.asarray(m.vertices), np.asarray(m.faces)


def joint(**kw):
    v, f = fossa(**kw)
    return j3.joint(v, f, head_points(), np.eye(4), np.zeros(3), "right")


def test_known_slope_and_medial_wall():
    r = j3.analyze(joint())
    assert r["rest_gap_mm"] == pytest.approx(GAP, abs=0.15)
    assert r["sagittal_deg"] == pytest.approx(ALPHA, abs=4.0)
    assert r["side_shift_mm"] == pytest.approx(W0, abs=0.3)
    assert r["bennett_deg"] == pytest.approx(BETA, abs=3.0)
    path = r["path"]
    assert path[0, 0] == 0 and path[-1, 1] < 0  # вперёд и вниз
    flat = j3.analyze(joint(slope=20.0))
    assert flat["sagittal_deg"] < r["sagittal_deg"] - 10


def test_no_medial_wall_and_anatomic_disc():
    r = j3.analyze_both(joint(wall=False))
    assert r["bennett_deg"] is None and any("стенки" in n for n in r["notes"])
    assert r["anatomic"]["sagittal_deg"] is not None
    with pytest.raises(ValueError, match="кости ямки"):
        j3.joint(*fossa(), head_points() + [0, 0, -60.0], np.eye(4), np.array([0, 0, -60.0]) + [0, 0, 80.0], "right")
