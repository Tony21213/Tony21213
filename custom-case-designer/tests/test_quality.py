import numpy as np
import pytest

import phantom
from casedesigner import quality
from casedesigner.fusion import Scan
from casedesigner.register import Target, apply, axis_angle


def bent(verts, degrees):
    """Ошибка склейки скана: часть дуги с x > 8 мм повёрнута вокруг вертикальной оси."""
    out = verts.copy()
    part = verts[:, 0] > 8
    pivot = np.array([8.0, 10.0, 0.0])
    out[part] = (verts[part] - pivot) @ axis_angle(np.array([0, 0, 1.0]), np.radians(degrees)).T + pivot
    return out


def register(jaw_ct, verts, faces, **kw):
    return jaw_ct.register(Scan("lower", apply(phantom.scan_pose(2), verts), faces), jaw="lower", **kw)


def test_intact_scan_has_no_warnings(jaw_ct):
    verts, faces = phantom.make_scan("lower")
    reg = register(jaw_ct, verts, faces)
    assert reg.warnings == []
    assert len(reg.segments) == 6 and max(s.shift_mm for s in reg.segments) < 0.02


def test_distorted_arch_is_reported(jaw_ct):
    verts, faces = phantom.make_scan("lower")
    reg = register(jaw_ct, bent(verts, 2.0), faces)
    assert reg.warnings
    worst = max(reg.segments, key=lambda s: s.shift_mm)
    assert worst.shift_mm > 0.3
    assert worst.where.startswith("слева")  # x > 0 — левая сторона пациента (LPS)
    assert worst.where in reg.warnings[0]


def test_partial_scan_found_automatically(jaw_ct):
    verts, faces = phantom.make_scan("lower")
    keep = verts[:, 0] > 8
    faces = faces[keep[faces].all(axis=1)]
    used = np.unique(faces)
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    verts, faces = verts[used], remap[faces]
    reg = register(jaw_ct, verts, faces)
    err = np.linalg.norm(apply(reg.transform, reg.scan.vertices) - verts, axis=1)
    assert err.max() < 0.05


@pytest.mark.parametrize("shift, residual, local_warning, global_warning", [
    (0.20, 0.33, False, False),  # небольшая локальная поправка и фон КЛКТ
    (0.299, 0.399, False, False),
    (0.301, 0.399, True, False),
    (0.299, 0.401, False, True),
    (0.50, 0.50, True, True),
])
def test_warning_thresholds(monkeypatch, shift, residual, local_warning, global_warning):
    # Изолируем решение о предупреждении от численной сходимости ICP;
    # реальную подгонку цельной и искажённой дуги проверяют тесты выше.
    a = np.linspace(0, np.pi, 600)
    vertices = np.column_stack((20 * np.cos(a), 30 * np.sin(a), np.zeros_like(a)))
    target = Target(vertices, np.tile([0., 0., 1.], (len(vertices), 1)))
    local = np.eye(4)
    local[2, 3] = shift
    monkeypatch.setattr(quality, "icp", lambda *args, **kwargs: (local, None))
    segments, warnings = quality.check_scan(vertices, np.eye(4),
                                            np.full(len(vertices), residual), target, "lower")
    assert all(s.shift_mm == pytest.approx(shift) for s in segments)
    assert any("скан искажён" in w for w in warnings) == local_warning
    assert any("Совмещение в целом" in w for w in warnings) == global_warning
    assert len(warnings) == 6 * local_warning + global_warning
