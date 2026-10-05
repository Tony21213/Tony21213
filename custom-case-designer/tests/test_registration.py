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


def test_segmented_teeth_guide_registration():
    """С зубами из сегментации опора совмещения — только зубы: плотная тонкая «кость» у воздуха
    (на реальном КЛКТ — раковины, стенки пазух) за коронку не принимается, скан встаёт точно."""
    from casedesigner.fusion import CaseCT
    from casedesigner.volume import Volume

    base = phantom.make_volume()
    vol = Volume(base.data.copy(), base.spacing, base.origin, base.direction)
    # Пластинка плотности эмали в воздухе между дугами: по порогам — «коронка» с жевательными нормалями.
    corners = vol.to_index(np.array([[x, y, z] for x in (-8, 8) for y in (2, 12) for z in (8, 12)]))
    lo, hi = np.floor(corners.min(axis=0)).astype(int), np.ceil(corners.max(axis=0)).astype(int)
    idx = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + 1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    world = vol.to_world(idx)
    plate = (np.abs(world[:, 0]) < 6) & (world[:, 1] > 4) & (world[:, 1] < 10) & (np.abs(world[:, 2] - 10) < 0.5)
    vol.data[tuple(idx[plate][:, ::-1].T)] = phantom.TOOTH
    on_plate = lambda p: (np.abs(p[:, 0]) < 7) & (p[:, 1] > 3) & (p[:, 1] < 11) & (np.abs(p[:, 2] - 10) < 1.5)

    ct = CaseCT(vol)
    lower_teeth, _ = phantom.make_scan("lower")  # скан нижней челюсти в координатах КТ
    assert on_plate(ct.fine_crowns(lower_teeth).points).sum() > 50  # без сегментации пластинка — «коронка»

    ct.use_teeth({"upper": phantom.teeth_mesh("upper"), "lower": phantom.teeth_mesh("lower")})
    assert ct.guided == {"upper", "lower"}
    for jaw, target in ct.coarse.items():  # опора — поверхность зубов своей челюсти, нормали наружу
        sign = 1 if jaw == "lower" else -1
        assert np.all(sign * (target.points[:, 2] - phantom.OCCLUSAL_Z) < 1.0)
        top = target.points[:, 2] > 8.0 if jaw == "lower" else target.points[:, 2] < 2 * phantom.OCCLUSAL_Z - 8.0
        assert (sign * target.normals[top, 2] > 0).mean() > 0.6
    assert not on_plate(ct.fine_crowns(lower_teeth).points).any()

    for jaw, seed in (("lower", 2), ("upper", 3)):
        verts, faces = phantom.make_scan(jaw)
        reg = ct.register(Scan(jaw, apply(phantom.scan_pose(seed), verts), faces))
        assert reg.jaw == jaw
        assert true_error(reg, verts).max() < 0.15


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
