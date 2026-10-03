import numpy as np
import pytest

import phantom
from casedesigner.fusion import Scan
from casedesigner.register import apply, axis_angle


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
    assert worst.shift_mm > 0.1
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
