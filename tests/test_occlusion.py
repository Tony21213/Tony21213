import numpy as np
import pytest

from crownai.design import CrownParameters, design_crown
from crownai.mesh import Mesh
from crownai.occlusion import SlavicekConcept, _gaps, excursion_interference, fix_bite
from crownai.synthetic import make_antagonist, make_prepared_molar

Z = np.array([0.0, 0.0, 1.0])
ANTERIOR, BUCCAL = np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
PARAMS = CrownParameters(n_theta=64, n_v=28)


def test_sequential_guidance_steepens_towards_the_canine():
    c = SlavicekConcept(condylar_inclination=35.0, sequence_step=5.0)
    angles = [c.guidance_angle(10 * 3 + pos, "protrusion") for pos in (7, 6, 5, 4, 3)]
    assert angles == sorted(angles) and angles[0] == 35.0 and angles[-1] == 55.0
    assert c.guidance_angle(36, "mediotrusion") > c.guidance_angle(36, "protrusion")  # Fischer angle


@pytest.fixture(scope="module")
def tight_case():
    prep = make_prepared_molar(n_theta=64)
    # antagonist cusps low enough to hit the crown in every excursion
    antagonist = make_antagonist(z_level=6.6, n=30)
    # nearly flat guidance: the bumpy antagonist must collide unless the crown is carved
    concept = SlavicekConcept(condylar_inclination=5.0, sequence_step=0.0, fischer_angle=0.0,
                              excursion=2.0, steps=6)
    res = design_crown(prep, tooth=36, antagonist=antagonist, occlusion=concept,
                       arch_orientation=(ANTERIOR, BUCCAL), params=PARAMS)
    return prep, antagonist, concept, res


def test_excursions_are_free_of_interference(tight_case):
    _, antagonist, concept, res = tight_case
    occ = res.report["occlusion"]
    assert occ["orientation"] == "given" and occ["crown_on_mandible"]
    assert max(m["interference_removed_mm"] for m in occ["movements"].values()) > 0.05  # there was work to do
    movements = concept.movements(36, ANTERIOR, BUCCAL)
    top = res.outer[:, res.outer.shape[1] // 2:].reshape(-1, 3)
    left = excursion_interference(top, antagonist, Z, movements, crown_on_mandible=True,
                                  clearance=PARAMS.occlusal_clearance, excursion=concept.excursion,
                                  steps=concept.steps)
    if not any("excursive interference" in w for w in res.report["warnings"]):
        assert left < 0.1


def test_centric_contacts_exist(tight_case):
    _, antagonist, _, res = tight_case
    g = _gaps(res.outer.reshape(-1, 3), antagonist, Z, np.zeros(3))
    g = g[np.isfinite(g)]
    assert g.min() < 0.2  # the crown reaches the antagonist in centric
    assert res.crown.is_watertight()


def test_fix_bite_removes_penetration():
    lower = make_prepared_molar(n_theta=48)
    upper = make_antagonist(z_level=4.0, n=24)  # sinks ~0.7 mm into the preparation
    fixed, info = fix_bite(lower, upper, (0, 0, 1), max_tilt_deg=1.0, sample=1500)
    assert info["initial_penetration_mm"] > 0.3
    g = _gaps(lower.vertices, fixed, Z, np.zeros(3))
    g = g[np.isfinite(g)]
    assert g.min() == pytest.approx(0.0, abs=0.03)
    assert info["contact_points"] >= 1
    # moving apart a bite that is too open closes it again
    apart = Mesh(upper.vertices + [0, 0, 2.0], upper.faces)
    closed, info2 = fix_bite(lower, apart, (0, 0, 1), max_tilt_deg=0.0, sample=1500)
    g2 = _gaps(lower.vertices, closed, Z, np.zeros(3))
    assert np.nanmin(np.where(np.isfinite(g2), g2, np.nan)) == pytest.approx(0.0, abs=0.03)
