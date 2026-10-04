import numpy as np
import pytest

import phantom
from casedesigner.fusion import Scan
from casedesigner.register import apply, kabsch, rigid
from casedesigner.teeth import crown_surface, split_jaws


def true_error(reg, verts):
    return np.linalg.norm(apply(reg.transform, reg.scan.vertices) - verts, axis=1)


def test_levels_and_crowns(jaw_ct):
    lv = jaw_ct.levels
    assert phantom.SOFT < lv.hard < phantom.BONE < lv.dense < phantom.TOOTH
    upper, lower = split_jaws(crown_surface(jaw_ct.vol, lv, step=2))
    # Коронки — только то, что выше кости (у нижней челюсти кость до z = 2 мм).
    assert lower.points[:, 2].min() > 1.5
    assert upper.points[:, 2].max() < 2 * phantom.OCCLUSAL_Z - 1.5
    assert abs(len(upper.points) - len(lower.points)) < 0.1 * len(lower.points)


@pytest.mark.parametrize("jaw, seed", [("lower", 2), ("upper", 3), ("lower", 11)])
def test_automatic_registration(jaw_ct, jaw, seed):
    verts, faces = phantom.make_scan(jaw)
    pose = phantom.scan_pose(seed)
    reg = jaw_ct.register(Scan(jaw, apply(pose, verts), faces))
    assert reg.jaw == jaw
    err = true_error(reg, verts)
    assert err.max() < 0.15, err.max()
    # Десна не участвует: на коронках лежит около половины точек скана.
    assert 0.35 < reg.stats["matched_fraction"] < 0.7
    assert reg.stats["within_0_2_mm"] > 0.9


def test_registration_from_point_pairs(jaw_ct):
    verts, faces = phantom.make_scan("lower")
    pose = phantom.scan_pose(5)
    scan = apply(pose, verts)
    picks = [0, len(verts) // 3, 2 * len(verts) // 3, len(verts) - 1]
    clicked = verts[picks] + np.random.default_rng(0).normal(0, 0.5, (4, 3))  # неточные клики в КТ
    reg = jaw_ct.register(Scan("lower", scan, faces), jaw="lower", pairs=(scan[picks], clicked))
    assert true_error(reg, verts).max() < 0.15


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(10, 3)) * 20
    T = phantom.scan_pose(4)
    assert kabsch(src, apply(T, src)) == pytest.approx(T, abs=1e-9)
    with pytest.raises(ValueError):
        kabsch(src[:2], src[:2])
    assert np.allclose(rigid(np.eye(3), np.zeros(3)), np.eye(4))


def test_upper_scan_with_palate(jaw_ct):
    """Нёбо на скане верхней челюсти не мешает: совмещаются только коронки."""
    verts, faces = phantom.make_scan("upper", palate=True)
    reg = jaw_ct.register(Scan("upper", apply(phantom.scan_pose(3), verts), faces))
    assert reg.jaw == "upper"
    assert true_error(reg, verts).max() < 0.02
    assert reg.warnings == []
    # Нёбо — не коронки: кандидатов в коронки заметно меньше половины точек скана.
    assert reg.scan.crowns.mean() < 0.6


@pytest.fixture(scope="module")
def models():
    return {jaw: phantom.make_model(jaw) for jaw in ("lower", "upper")}


@pytest.mark.parametrize("jaw, seed, closed", [("lower", 2, False), ("upper", 7, False), ("lower", 3, True)])
def test_plaster_model_with_socle(jaw_ct, models, jaw, seed, closed):
    """Скан гипсовой модели с настольного сканера: цоколь со стенками, дно не снято (или закрыто) —
    совмещается так же, как внутриротовой."""
    from casedesigner import scan_teeth

    verts, faces = phantom.make_model(jaw, closed=True) if closed else models[jaw]
    scan = Scan(jaw, apply(phantom.scan_pose(seed), verts), faces)
    assert scan_teeth.is_closed(scan.normals) == closed
    # цоколь не коронки: кандидаты — только выше десны
    crowns_z = verts[scan.crowns][:, 2] if jaw == "lower" else 2 * phantom.OCCLUSAL_Z - verts[scan.crowns][:, 2]
    assert crowns_z.min() > -3
    reg = jaw_ct.register(scan)
    assert reg.jaw == jaw
    assert true_error(reg, verts).max() < 0.15


def test_models_in_occlusion(models):
    """Модели, отсканированные сомкнутыми, — в прикусе; разнесённые — нет."""
    from casedesigner.fusion import in_occlusion

    (lv, lf), (uv, uf) = models["lower"], models["upper"]
    T = phantom.scan_pose(4)
    assert in_occlusion(Scan("u", apply(T, uv), uf), Scan("l", apply(T, lv), lf))
    assert not in_occlusion(Scan("u", apply(T, uv + [0, 0, 8.0]), uf), Scan("l", apply(T, lv), lf))
