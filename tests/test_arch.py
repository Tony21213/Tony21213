import numpy as np
import pytest

from crownai.arch import analyze_neighbors
from crownai.design import CrownParameters, design_crown
from crownai.margin import detect_margin
from crownai.mesh import nearest_distance
from crownai.synthetic import make_lower_arch, make_prep_at


@pytest.fixture(scope="module")
def arch_case():
    jaw, teeth = make_lower_arch(missing=33)
    t = teeth[33]
    prep = make_prep_at(t["center"], t["md"], t["bl"])
    margin = detect_margin(prep, n_points=64)
    return jaw, teeth, prep, margin, analyze_neighbors(jaw, margin, tooth=33)


def test_finds_adjacent_teeth_and_space(arch_case):
    _, teeth, _, _, na = arch_case
    assert len(na.neighbors) == 2
    # the gap left by 33 (plus the tapering of the synthetic neighbours)
    assert na.space == pytest.approx(teeth[33]["md"], abs=1.2)


def test_centre_and_arch_direction(arch_case):
    _, teeth, _, _, na = arch_case
    assert np.linalg.norm(na.target_center[:2] - teeth[33]["center"]) < 0.6
    cos = abs(na.md_direction[:2] @ teeth[33]["tangent"])
    assert cos > 0.97  # within ~14 degrees of the true arch tangent


def test_contralateral_is_mirrored_into_place(arch_case):
    _, teeth, _, _, na = arch_case
    assert na.contralateral is not None
    # the tooth used is 43: its crown sits on the other side of the midline
    src = na.contralateral.mesh.vertices[:, :2].mean(0)
    assert np.linalg.norm(src - teeth[43]["center"]) < 1.5
    assert na.contralateral.md_width == pytest.approx(teeth[43]["md"], abs=1.0)
    placed = na.template.vertices[:, :2].mean(0)
    assert np.linalg.norm(placed - teeth[33]["center"]) < 1.0


def test_crown_touches_neighbours(arch_case):
    _, _, prep, margin, na = arch_case
    params = CrownParameters(n_theta=64, n_v=28)
    res = design_crown(prep, tooth=33, margin=margin, neighbors=na, params=params)
    assert res.crown.is_watertight()
    assert res.report["anatomy_source"] == "anatomical model fitted to the mirrored contralateral tooth"
    for nb in na.neighbors:
        d, _ = nearest_distance(res.outer.reshape(-1, 3), nb.mesh.vertices)
        assert d.min() < 0.35  # in contact (vertex spacing of the scan ~0.3 mm)


def test_without_contralateral_builds_anatomy_by_rules_in_the_gap():
    jaw, teeth = make_lower_arch(missing=35)
    t = teeth[35]
    prep = make_prep_at(t["center"], t["md"], t["bl"])
    margin = detect_margin(prep, n_points=64)
    na = analyze_neighbors(jaw, margin, tooth=None)  # unknown tooth: no contralateral search
    assert na.contralateral is None and len(na.neighbors) == 2
    res = design_crown(prep, tooth=35, margin=margin, neighbors=na,
                       params=CrownParameters(n_theta=64, n_v=28))
    assert res.crown.is_watertight()
    assert res.report["anatomy_source"].startswith("anatomical rules")
    assert res.report["anatomy_fit"]["md_from"] == "space between neighbours"
    for nb in na.neighbors:  # fills the gap: touches both neighbours
        d, _ = nearest_distance(res.outer.reshape(-1, 3), nb.mesh.vertices)
        assert d.min() < 0.35
    res = design_crown(prep, tooth=35, margin=margin, neighbors=na,
                       params=CrownParameters(n_theta=64, n_v=28, posterior_anatomy="mirror"))
    assert res.report["anatomy_source"] == "parametric library"


def test_anterior_anatomy_model_looks_like_a_canine():
    from crownai.anatomy_model import default_anterior, radial_distance

    m = default_anterior(33)
    # labial view: the cusp tip is the highest point, the mesial angle higher than the distal one
    assert m.top(np.array(m.tip_u)) == pytest.approx(m.height, abs=0.01)
    assert m.top(np.array(m.md / 2)) > m.top(np.array(-m.md / 2))
    # widest at the contacts, narrower at the cervix
    M, D = m.mesiodistal(np.array([0.1, m.contact_m * m.height]))
    assert M[1] > M[0]
    # proximal view: labial height of contour near the cervix, lingual concavity above the cingulum
    lab, lin = m.labiolingual(np.array([m.labial_hc, 0.9]))
    assert lab[0] > lab[1]
    c = np.array([0.0, 0.0, 0.4 * m.height])
    r = radial_distance(m, c, np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]))
    assert 2.5 < r[0] < 5 and 2.5 < r[1] < 5
