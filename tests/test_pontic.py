import numpy as np

from crownai.mesh import Mesh
from crownai.pontic import design_missing_tooth, find_implant_holes
from crownai.synthetic import make_lower_arch


def _with_hole(jaw: Mesh, center_xy, radius: float) -> Mesh:
    keep = np.linalg.norm(jaw.vertices[:, :2] - center_xy, axis=1) > radius
    faces = jaw.faces[keep[jaw.faces].all(axis=1)]
    return Mesh(jaw.vertices, faces)


def test_find_implant_hole():
    jaw, teeth = make_lower_arch(missing=45, res=0.3)
    c = teeth[45]["center"]
    holes = find_implant_holes(_with_hole(jaw, c, 2.2))
    assert len(holes) == 1
    assert np.linalg.norm(holes[0]["center"][:2] - c) < 0.5
    assert 1.8 < holes[0]["radius"] < 2.6


def test_design_missing_premolar_on_implant():
    jaw, teeth = make_lower_arch(missing=45, res=0.3)
    c = teeth[45]["center"]
    res, solid, na = design_missing_tooth(_with_hole(jaw, c, 2.2), 45)
    assert solid.is_watertight() and solid.volume() > 0
    assert res.report["site"].startswith("implant")
    assert na.contralateral is not None  # 35 found on the other side of the arch
    loc = res.frame.to_local(solid.vertices)
    # fills the gap between 44 and 46 roughly at the missing tooth's width
    assert 6.0 < np.ptp(loc[:, 0]) < 9.5
    centre = res.frame.to_local(np.array([*c, 0.0]))[:2]
    assert np.linalg.norm(loc[:, :2].mean(axis=0) - centre) < 1.5
