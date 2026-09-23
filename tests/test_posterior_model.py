import numpy as np
import pytest

from crownai.anatomy_model import PlacedAnatomy
from crownai.posterior_model import default_posterior


def _top(m, u, w):
    return float(m.occlusal(np.array(u), np.array(w)))


@pytest.mark.parametrize("fdi", [14, 15, 16, 17, 34, 35, 36, 37])
def test_cusps_fissures_and_ridges(fdi):
    m = default_posterior(fdi)
    tips = m.cusp_tips()
    # every cusp tip is (close to) the surface there and stands above the central fossa
    for (u, w, z) in tips:
        assert _top(m, u, w) == pytest.approx(z, abs=0.45)  # smooth-max adds a little
    w_mid = 0.5 * (tips[tips[:, 1] > 0, 1].min() + tips[tips[:, 1] < 0, 1].max())
    # lower first premolar: a transverse ridge joins its cusps, the pits lie mesial and distal of it
    fossa = _top(m, m.md / 4 if fdi == 34 else 0.0, w_mid)
    assert fossa < tips[:, 2].max() - 1.0
    # marginal ridges between the fossa and the cusp tips, the mesial one higher
    mesial = _top(m, m.md / 2 - 1.1, 0.0)
    distal = _top(m, -(m.md / 2 - 1.1), 0.0)
    assert fossa < mesial < tips[:, 2].max() and mesial > distal
    # a solid crown: inside at the core, outside above the cusps and beyond the outline
    assert m.inside(np.array([0.0, 0.0, 0.3 * m.height]))
    assert not m.inside(np.array([0.0, 0.0, m.height + 0.5]))
    assert not m.inside(np.array([m.md, 0.0, 0.5 * m.height]))


def test_textbook_cusp_arrangement():
    lower6 = default_posterior(36)
    assert len(lower6.cusps) == 5 and len(default_posterior(37).cusps) == 4
    h = {c.name: c.dh for c in lower6.cusps}
    assert h["ML"] > h["MB"] and h["DL"] > h["DB"]  # lower molars: lingual cusps higher
    upper6 = default_posterior(26)
    assert max(upper6.cusps, key=lambda c: c.dh).name == "MP"  # mesiopalatal the largest
    assert upper6.oblique_ridge and not default_posterior(36).oblique_ridge
    # oblique ridge: the surface between MP and DB stays high (no deep groove across it)
    mp = next(c for c in upper6.cusps if c.name == "MP")
    db = next(c for c in upper6.cusps if c.name == "DB")
    mid = 0.5 * (upper6._cusp_xy(mp) + upper6._cusp_xy(db))
    assert _top(upper6, *mid) > upper6.height - 2.0
    # functional (supporting) cusps: palatal above, buccal below
    assert all(c.functional == (c.w < 0) for c in upper6.cusps)
    assert all(c.functional == (c.w > 0) for c in lower6.cusps)


def test_occlusal_table_stays_level_on_an_inclined_margin():
    m = default_posterior(36)
    th = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    margin = np.column_stack([4.5 * np.cos(th), 4.5 * np.sin(th), 0.4 * 4.5 * np.sin(th)])  # 3.6 mm tilt
    placed = PlacedAnatomy.on_margin(m, 1.0, np.zeros(2), margin, fade=0.55 * m.height)
    top = []
    for u, w in ((2.5, 2.5), (2.5, -2.5)):
        z = np.linspace(0, 12, 1201)
        ins = placed.inside(np.column_stack([np.full_like(z, u), np.full_like(z, w), z]))
        top.append(z[ins].max())
    # buccal and lingual cusps keep their anatomical relation instead of following the margin
    assert abs(top[0] - top[1]) < 1.0
