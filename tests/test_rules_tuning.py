"""Tuning the posterior rules to finished crowns (synthetic "technician" crowns)."""

from dataclasses import replace

import numpy as np
import pytest

from crownai.design import CrownParameters
from crownai.mesh import Mesh, save_stl
from crownai.posterior_model import _textbook_posterior, default_posterior, model_mesh
from crownai.rules_tuning import build_profile, buccal_from_fdi, fit_rules_to_crown, load_profile, tune_rules


def _technician_36():
    """A lab's 36: steeper inclines, a higher distobuccal cusp and a supplemental
    groove across the mesial fossa (secondary anatomy the rules lack)."""
    m = _textbook_posterior(36)
    cusps = tuple(replace(c, dh=-0.2) if c.name == "DB" else c for c in m.cusps)
    g = np.linspace(-1, 1, 33)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    groove = -0.35 * np.exp(-((GX - 0.45) / 0.07) ** 2) * (np.abs(GY) < 0.5)
    return replace(m, slope_in=0.95, cusps=cusps, detail=groove)


def _margin(m):
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    return np.column_stack([m.md_cervix / 2 * np.cos(th), m.bl_cervix / 2 * np.sin(th), np.zeros_like(th)])


def test_buccal_side_from_the_tooth_number():
    z, mesial = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])  # lower arch, molar region: mesial = anterior
    assert buccal_from_fdi(46, z, mesial) @ [1, 0, 0] > 0.99  # patient's right = +x here
    assert buccal_from_fdi(36, z, mesial) @ [-1, 0, 0] > 0.99
    down = -z  # upper teeth point down
    assert buccal_from_fdi(16, down, mesial) @ [1, 0, 0] > 0.99
    assert buccal_from_fdi(26, down, mesial) @ [-1, 0, 0] > 0.99


@pytest.fixture(scope="module")
def fit36():
    tech = _technician_36()
    crown = model_mesh(tech, 240, 160)  # model coordinates = world: margin at z=0, mesial +x, buccal +y
    return tech, fit_rules_to_crown(crown, _margin(tech), (0, 0, 1), (1, 0, 0), 36)


def test_fit_recovers_the_labs_rules(fit36):
    tech, f = fit36
    # what remains after the fit is mostly the groove, which no rule describes
    assert f["rms_after_mm"] < 0.6 * f["rms_before_mm"] and f["rms_after_mm"] < 0.2
    assert f["slope_in"] == pytest.approx(0.95, abs=0.12)
    assert f["md"] == pytest.approx(tech.md, abs=0.4) and f["bl"] == pytest.approx(tech.bl, abs=0.4)
    db = next(c for c, t in zip(f["cusps"], tech.cusps) if t.name == "DB")
    assert db["dh"] > -0.6  # raised from the textbook -0.7 towards the lab's -0.2
    # the supplemental groove is not a rule: it ends up in the secondary anatomy layer
    d = np.array(f["detail"])
    g = np.linspace(-1, 1, 33)
    col = int(np.argmin(np.abs(g - 0.45)))
    assert d[col, 12:21].mean() < -0.12
    assert abs(d[8, 12:21].mean()) < 0.1


def test_profile_is_the_median_and_is_applied(fit36):
    _, f = fit36
    other = dict(f, fossa_depth=f["fossa_depth"] + 1.0, slope_in=0.5)
    profile = build_profile([f, f, other], min_cases=3)
    assert profile["lower_6"]["n_cases"] == 3
    assert profile["lower_6"]["fossa_depth"] == pytest.approx(f["fossa_depth"], abs=1e-3)  # median, not mean
    m = default_posterior(46, profile)  # the same tooth type on the other side
    assert m.slope_in == pytest.approx(f["slope_in"], abs=1e-3) and m.detail is not None
    assert m.occlusal(np.array(0.45 * m.md / 2), np.array(0.0)) < _textbook_posterior(46).occlusal(
        np.array(0.45 * m.md / 2), np.array(0.0)) + 0.2  # the learned groove is there
    assert default_posterior(36, build_profile([f], min_cases=3)).detail is None  # too few cases: textbook


def test_tune_rules_on_an_exocad_archive(tmp_path):
    # a case folder like exocad writes it, with the technician's crown exported back
    case = tmp_path / "Case_0012"
    case.mkdir()
    tech = _technician_36()
    crown = model_mesh(tech, 120, 80)
    shift = np.array([10.0, 20.0, 5.0])
    save_stl(Mesh(crown.vertices + shift, crown.faces), case / "Case-36-crown_cad.stl")
    jaw = Mesh(np.array([[-9, -9, -1], [9, -9, -1], [0, 9, -1.0]]) + shift, np.array([[0, 1, 2]]))
    save_stl(jaw, case / "Case-LowerJaw.stl")
    vec = lambda tag, v: f"<{tag}><x>{v[0]}</x><y>{v[1]}</y><z>{v[2]}</z></{tag}>"
    eye = "".join(f"<_{r}{c}>{float(r == c)}</_{r}{c}>" for r in range(4) for c in range(4))
    margin = "".join(vec("Vec3", p) for p in _margin(tech) + shift)
    (case / "Case.constructionInfo").write_text(f"""<ConstructionInfo><ScanFiles>
      <ScanFile><FileName>Case-LowerJaw.stl</FileName><TransformationMatrix>{eye}</TransformationMatrix>
        <PartType>PreparationScan</PartType><ToothNumbers><int>36</int></ToothNumbers></ScanFile></ScanFiles>
      <Teeth><Tooth><Number>36</Number><ReconstructionType>AnatomicCrown</ReconstructionType>
        <ToothScanFileName>Case-LowerJaw.stl</ToothScanFileName>{vec("Axis", [0, 0, 1])}{vec("AxisMesial", [1, 0, 0])}
        <Margin>{margin}</Margin></Tooth></Teeth></ConstructionInfo>""")
    out = tmp_path / "profile.json"
    profile = tune_rules(tmp_path, out, holdout=0.0, eval_limit=0, min_cases=1, log=lambda *_: None)
    assert profile["lower_6"]["slope_in"] == pytest.approx(0.95, abs=0.15)
    assert load_profile(out).keys() == {"lower_6"}
    fits = (tmp_path / "profile.fits.jsonl").read_text()
    assert "Case_0012" not in fits and str(tmp_path) not in fits  # no names or paths, only hashes
    # a second run resumes instead of refitting
    tune_rules(tmp_path, out, holdout=0.0, eval_limit=0, min_cases=1, log=lambda *_: None)
    assert len((tmp_path / "profile.fits.jsonl").read_text().splitlines()) == 1
    assert CrownParameters(rules_profile=load_profile(out)).rules_profile["lower_6"]["n_cases"] == 1
