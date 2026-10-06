"""Исправление прикуса: нижняя «дуга» с буграми садится в верхнюю, как в Bite-Finder.

Синтетические челюсти: зубная дуга (парабола шириной 10 мм) с буграми и
фиссурами; верхняя — та же поверхность, поэтому в правильном прикусе они
прилегают друг к другу целиком. Прикус портится так, как ошибается сканер, —
провал, раскрытие, перекос на сторону, — и должен вернуться. Проверка на
настоящих выгрузках exocad — в test_private_cases.py (CCD_PRIVATE_CASES).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from casedesigner import bite
from casedesigner.register import apply


def arch_surface(n_s=220, n_w=24, cusp_mm=0.8):
    """Дуга: y = −x²/40 (резцы спереди, +y), ширина 10 мм, бугры через 5 мм вдоль и поперёк."""
    s = np.linspace(-1.0, 1.0, n_s)
    x = 26 * s
    y = 18 - x ** 2 / 40.0
    t = np.c_[np.gradient(x), np.gradient(y)]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.c_[t[:, 1], -t[:, 0]]
    w = np.linspace(-5, 5, n_w)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.c_[x, y], axis=0), axis=1))]
    P = np.zeros((n_s, n_w, 3))
    for j, wj in enumerate(w):
        P[:, j, 0] = x + wj * nrm[:, 0]
        P[:, j, 1] = y + wj * nrm[:, 1]
        P[:, j, 2] = cusp_mm * np.cos(2 * np.pi * arc / 5.0) * np.cos(2 * np.pi * wj / 5.0)
    idx = np.arange(n_s * n_w).reshape(n_s, n_w)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    return P.reshape(-1, 3), faces


def hinge_turn(points, angle_deg, axis_point=(0.0, -80.0, 30.0)):
    """Поворот вокруг шарнирной оси (вправо, за дугой и выше её) — так ошибается прикус сканера."""
    R = Rotation.from_rotvec(np.radians(angle_deg) * np.array([1.0, 0, 0])).as_matrix()
    c = np.asarray(axis_point)
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, c - R @ c
    return M


@pytest.fixture(scope="module")
def jaws():
    upper, faces = arch_surface()
    return upper + [0, 0, 0.0], faces, upper.copy()  # в правильном прикусе нижняя совпадает с верхней


@pytest.mark.parametrize("error", [
    ("раскрыт", hinge_turn, -0.35), ("провален", hinge_turn, 0.12),
    ("перекошен", lambda p, a: _roll(a), 0.25)], ids=["open", "sunk", "tilted"])
def test_bite_returns_to_intercuspation(jaws, error):
    upper, faces, lower_true = jaws
    _name, make, amount = error
    P = make(lower_true, amount)
    lower = apply(P, lower_true)
    condyles = (np.array([50.0, -80.0, 30.0]), np.array([-50.0, -80.0, 30.0]))  # «по КТ»
    res = bite.correct_bite(upper, faces, lower, condyles=condyles)
    back = apply(res.transform, lower)
    err = np.linalg.norm(back - lower_true, axis=1)
    assert err.max() < 0.08, (err.max(), res.report)
    assert res.report["after"]["penetration_mm"] <= 0.05
    assert res.report["after"]["points"] >= res.report["before"]["points"] * 0.5


def _roll(deg):
    R = Rotation.from_rotvec(np.radians(deg) * np.array([0, 1.0, 0])).as_matrix()  # вокруг оси вперёд — перекос
    c = np.array([0.0, 18.0, 0.0])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, c - R @ c
    return M


def test_correct_bite_stays_near_scanner_bite(jaws):
    """Хороший прикус почти не трогается; сканы не в прикусе — понятная ошибка."""
    upper, faces, lower_true = jaws
    res = bite.correct_bite(upper, faces, lower_true + [0, 0, -0.02])
    assert res.report["incisal_mm"] < 0.1 and res.report["turn_deg"] < 0.2
    with pytest.raises(ValueError, match="не в прикусе"):
        bite.correct_bite(upper, faces, lower_true + [0, 0, -10.0])
