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
