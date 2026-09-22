import numpy as np
import pytest

from crownai.anatomy import tooth_type_for_fdi
from crownai.design import design_crown
from crownai.margin import detect_margin
from crownai.mesh import raycast


def test_fdi_mapping():
    assert tooth_type_for_fdi(36).name == "molar"
    assert tooth_type_for_fdi(45).name == "premolar"
    assert tooth_type_for_fdi(13).name == "canine"
    assert tooth_type_for_fdi(11).name == "incisor"
    assert tooth_type_for_fdi(41).name == "lower_incisor"
    with pytest.raises(ValueError):
        tooth_type_for_fdi(19)


def test_margin_detection_finds_finish_line(prep):
    m = detect_margin(prep, n_points=64)
    r = np.hypot(m[:, 0], m[:, 1])
    # synthetic margin: radius 4.8 * (1 +/- 0.06), height 0 +/- 0.4 (scalloped)
    assert r.min() > 4.3 and r.max() < 5.2
    assert np.abs(m[:, 2]).max() < 0.6
    assert m[:, 2].max() - m[:, 2].min() > 0.4  # follows the scallop


@pytest.fixture(scope="module")
def result(prep, antagonist, fast_params):
    return design_crown(prep, tooth=36, antagonist=antagonist, params=fast_params)


def test_crown_is_closed_solid(result):
    crown = result.crown
    assert crown.is_watertight()
    assert crown.volume() > 100
    assert result.report["warnings"] == []


def test_crown_respects_minimum_thickness(result, fast_params):
    assert result.report["min_axial_thickness_mm"] >= fast_params.min_axial - 0.02
    assert result.report["min_occlusal_thickness_mm"] >= fast_params.min_occlusal - 0.02


def test_crown_keeps_cement_gap_off_the_die(result, prep, fast_params):
    inner = result.intaglio[:, 8:].reshape(-1, 3)
    center = result.frame.to_world(np.array([0.0, 0.0, 1.5]))
    to_center = center - inner
    to_center /= np.linalg.norm(to_center, axis=1, keepdims=True)
    t = raycast(inner, to_center, prep)
    # every fitting-surface point lies outside the die (a ray inwards hits it) ...
    assert np.isfinite(t).all()
    # ... at about the cement gap (rays are oblique, so slightly more than the gap)
    assert np.median(t) == pytest.approx(fast_params.cement_gap, abs=0.03)
    assert t.min() > 0.5 * fast_params.cement_gap


def test_crown_clears_antagonist(result, antagonist, fast_params):
    z = np.array([0.0, 0.0, 1.0])
    pts = result.crown.vertices
    t = raycast(pts - z * 30, np.tile(z, (len(pts), 1)), antagonist)
    gap = t[np.isfinite(t)] - 30
    assert gap.min() >= fast_params.occlusal_clearance - 0.05


def test_crown_dimensions_are_anatomical(result):
    rep = result.report
    assert 9 < rep["mesiodistal_mm"] < 13
    assert 9 < rep["buccolingual_mm"] < 13
    assert 5.5 < rep["height_mm"] < 9


def test_explicit_margin_gives_same_crown(prep, fast_params, result):
    res2 = design_crown(prep, tooth=36, margin=result.margin[::2], params=fast_params)
    assert res2.crown.is_watertight()
    assert abs(res2.report["margin_length_mm"] - result.report["margin_length_mm"]) < 1.0


def test_premolar_is_narrower(fast_params):
    from crownai.synthetic import make_prepared_molar

    small = make_prepared_molar(n_theta=64)
    small.vertices[:, :2] *= 0.7
    res = design_crown(small, tooth=45, params=fast_params)
    assert res.crown.is_watertight()
    assert res.report["tooth_type"] == "premolar"
    assert res.report["mesiodistal_mm"] < 9
