"""Срезы и контуры для интерфейса: быстрый путь даёт то же, что трёхмерная интерполяция и trimesh."""

import numpy as np
import pytest
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from casedesigner.app.session import SLICES, Sectioner, Session
from casedesigner.register import axis_angle
from casedesigner.volume import Volume

DIRECTIONS = {"тождественная": np.eye(3), "перевёрнутая": np.diag([1, -1, 1.0]),
              "переставленная": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])}


@pytest.mark.parametrize("name", DIRECTIONS)
def test_fast_slice_matches_trilinear(tmp_path, name):
    data = ndimage.gaussian_filter(np.random.default_rng(0).normal(0, 300, (60, 70, 80)), 2).astype(np.float32)
    vol = Volume(data, np.array([0.3, 0.25, 0.4]), np.array([-20.0, 15, -30]), DIRECTIONS[name])
    s = Session(memory_path=str(tmp_path / "m.jsonl"))
    s.vol = vol
    for axis in SLICES:
        g, sl = s.slice_geometry(axis), SLICES[axis]
        for pos in np.linspace(*g["range"], 5)[1:-1]:
            region = (g["u0"] + 1, g["u0"] + 9, g["v0"] + np.sign(g["dv"]), g["v0"] + 7 * np.sign(g["dv"]))
            fast = s.slice_values(axis, pos, region, (57, 41))
            us = region[0] + (np.arange(57) + 0.5) * (region[1] - region[0]) / 57
            vs = region[2] + (np.arange(41) + 0.5) * (region[3] - region[2]) / 41
            uu, vv = np.meshgrid(us, vs)
            pts = np.zeros((uu.size, 3))
            pts[:, sl["u"]], pts[:, sl["v"]], pts[:, sl["normal"]] = uu.ravel(), vv.ravel(), pos
            assert np.abs(fast - vol.sample(pts).reshape(uu.shape)).max() < 1e-2
    raw = s.slice_raw("axial", 0.0)
    cols, rows = np.frombuffer(raw[:8], np.uint32)
    assert len(raw) == 8 + int(cols) * int(rows)


def test_sections_match_trimesh():
    m = trimesh.creation.icosphere(subdivisions=4, radius=12)
    m.apply_transform(np.r_[np.c_[axis_angle(np.array([1, 2, 3.0]), 0.7), [1, 2, 3]], [[0, 0, 0, 1]]])
    sec = Sectioner(m.vertices, m.faces)
    for n in range(3):
        uv = [k for k in range(3) if k != n]
        for pos in (-5.3, 0.0, 7.7):
            a = np.array(sec.cut(n, pos, uv)).reshape(-1, 2, 2)
            b = trimesh.intersections.mesh_plane(m, np.eye(3)[n], np.eye(3)[n] * pos)[:, :, uv]
            assert len(a) == len(b)
            d, i = cKDTree(b.mean(1)).query(a.mean(1))
            assert d.max() < 1e-3 and np.abs(np.sort(a, 1) - np.sort(b[i], 1)).max() < 1e-3
    assert sec.cut(2, 100.0, [0, 1]) == []
