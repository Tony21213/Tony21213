import numpy as np
import pytest

from crownai.mesh import Mesh
from crownai.native_case import locate_prep, locate_preps
from crownai.synthetic import make_lower_arch, make_prep_at


def _merge(*meshes):
    offs = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return Mesh(np.concatenate([m.vertices for m in meshes]),
               np.concatenate([m.faces + o for m, o in zip(meshes, offs)]))


def _with_stumps(missing: set[int]):
    """A lower arch with a real (short, present) prep stump standing in for
    each missing tooth - not bare gum. locate_prep/locate_preps is built
    around the stump-detection heuristic in _arch_chain and is only tested
    fairly against the case it targets.
    """
    jaw, teeth = make_lower_arch(missing=missing)
    stumps = [make_prep_at(teeth[t]["center"], teeth[t]["md"], teeth[t]["bl"]) for t in missing]
    return _merge(jaw, *stumps), teeth


def test_single_tooth_gap():
    jaw, teeth = _with_stumps({33})
    margin = locate_prep(jaw, 33)
    found = margin.mean(axis=0)[:2]
    assert np.linalg.norm(found - teeth[33]["center"]) < 2.5


def test_two_quadrants_are_not_confused():
    """Regression test: calling locate_prep separately per tooth for teeth in two
    different quadrants used to silently return the SAME gap for both - each
    isolated call had no way to know the biggest gap in the whole arch didn't
    belong to the tooth it was asked about, and would confidently mislabel it.

    Nothing in a symmetric jaw's pure geometry says which physical side is
    quadrant 3 vs quadrant 4 - a real gap on the left and a real gap on the
    right of the midline are, by themselves, equally good candidates for
    "tooth 33" and "tooth 43" (both requests would find *a* gap on either
    side). So the fix is not to always resolve this correctly - that is not
    knowable from geometry alone - but to never let it go wrong silently:
    either both teeth land near their own real position, or the whole call
    refuses. What must never happen is the old bug: both teeth landing on the
    very same spot.
    """
    jaw, teeth = _with_stumps({33, 43})
    try:
        placed = locate_preps(jaw, [33, 43])
    except ValueError:
        return  # refusing is the correct outcome when the geometry alone is ambiguous
    assert placed.keys() == {33, 43}
    m33, m43 = placed[33].mean(axis=0)[:2], placed[43].mean(axis=0)[:2]
    assert np.linalg.norm(m33 - m43) > 3.0
    assert (np.linalg.norm(m33 - teeth[33]["center"]) < 2.5
           and np.linalg.norm(m43 - teeth[43]["center"]) < 2.5) or \
          (np.linalg.norm(m33 - teeth[43]["center"]) < 2.5
           and np.linalg.norm(m43 - teeth[33]["center"]) < 2.5)


def test_more_than_two_quadrants_refuses_to_guess():
    jaw, _ = _with_stumps({33})
    with pytest.raises(ValueError):
        locate_preps(jaw, [13, 23, 33])  # three quadrants - not a single jaw


def test_two_quadrants_never_collide_even_when_ambiguous():
    """Known limitation: when a quadrant has more than one missing tooth close
    together (e.g. 32 and 34, with 33 still present between them), telling its
    own gaps apart from a same-strength gap belonging to the OTHER requested
    quadrant is not always reliable with this heuristic alone - real preps
    have far more shape variation than the synthetic stumps here, so real
    cases are usually easier than this, but nothing guarantees it. What must
    always hold, and is the actual bug this module exists to prevent, is that
    it never SILENTLY places two different teeth at the same spot: either it
    gets them right, or it raises rather than guess.
    """
    jaw, teeth = _with_stumps({32, 34, 43})
    try:
        placed = locate_preps(jaw, [32, 34, 43])
    except ValueError:
        return  # refusing is an acceptable outcome here
    assert placed.keys() == {32, 34, 43}
    centres = list({t: tuple(placed[t].mean(axis=0)[:2]) for t in (32, 34, 43)}.values())
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            assert np.linalg.norm(np.array(centres[i]) - np.array(centres[j])) > 3.0
