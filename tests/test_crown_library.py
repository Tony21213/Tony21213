"""The lab crown library: real crowns as templates (synthetic "technician" crowns)."""

from dataclasses import replace

import numpy as np
import pytest

from crownai.arch import analyze_neighbors
from crownai.crown_library import CrownLibrary, TemplateCrown, build_library, canonical_crown
from crownai.design import CrownParameters, design_crown
from crownai.margin import detect_margin
from crownai.mesh import Mesh, save_stl
from crownai.posterior_model import _textbook_posterior, model_mesh
from crownai.synthetic import make_lower_arch, make_prep_at


def _margin(m):
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    return np.column_stack([m.md_cervix / 2 * np.cos(th), m.bl_cervix / 2 * np.sin(th), np.zeros_like(th)])


def _crown(fdi, **kw):
    m = replace(_textbook_posterior(fdi), **kw)
    return m, model_mesh(m, 200, 140)  # model coordinates = world: mesial +x, buccal +y (quadrant 3)


def test_a_crown_survives_the_trip_into_the_library():
    m, mesh = _crown(36, fissure_depth=0.6)
    c = canonical_crown(mesh, _margin(m), (0, 0, 1), (1, 0, 0), 36, "a/36")
    assert c.key == "lower_6" and c.md == pytest.approx(m.md, abs=0.4) and c.bl == pytest.approx(m.bl, abs=0.4)
    t = TemplateCrown(c)
    p = np.random.default_rng(0).uniform([-6, -6, 0], [6, 6, m.height + 0.5], (20000, 3))
    # the library keeps the crown centred on itself; the model stands on its axis
    differ = m.inside(p) != t.inside(p - [*c.offset, 0.0])
    assert differ.mean() < 0.03  # the same solid, up to sampling right at the surface


def test_library_choice_variants_and_exclusion(tmp_path):
    lib = CrownLibrary(tmp_path / "lib")
    for i, md_scale in enumerate((0.9, 1.0, 1.1)):
        m, mesh = _crown(46)
        mesh = Mesh(mesh.vertices * [md_scale, 1, 1], mesh.faces)
        th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        margin = np.column_stack([md_scale * m.md_cervix / 2 * np.cos(th), m.bl_cervix / 2 * np.sin(th), 0 * th])
        lib.add(canonical_crown(mesh, margin, (0, 0, 1), (-1, 0, 0), 46, f"case{i}/46"))  # quadrant 4: mesial -x
    lib = CrownLibrary(tmp_path / "lib")  # reload from disk
    assert len(lib) == 3 and lib.ids() == {"case0/46", "case1/46", "case2/46"}
    md = lib.entries[2]["md"]
    assert lib.choose("lower_6", md, None).id == "case2/46"  # closest width
    assert lib.choose("lower_6", md, None, variant=1).id == "case1/46"  # next closest
    assert lib.choose("lower_6", md, None, exclude=("case2/46",)).id == "case1/46"
    assert lib.choose("upper_6", md, None) is None  # no crowns of that type


def test_design_with_the_library_uses_the_real_crown(tmp_path):
    jaw, teeth = make_lower_arch(missing=45)
    t = teeth[45]
    prep = make_prep_at(t["center"], t["md"], t["bl"])
    margin = detect_margin(prep, n_points=64)
    na = analyze_neighbors(jaw, margin, tooth=45)
    lib = CrownLibrary(tmp_path / "lib")
    for i, fd in enumerate((0.2, 0.9)):  # two "technicians": shallow and deep fissures
        m, mesh = _crown(35, fissure_depth=fd)
        lib.add(canonical_crown(mesh, _margin(m), (0, 0, 1), (1, 0, 0), 35, f"t{i}/35"))
    fast = CrownParameters(n_theta=64, n_v=28, crown_library=lib)
    res = design_crown(prep, tooth=45, margin=margin, neighbors=na, params=fast)
    assert res.crown.is_watertight()
    assert res.report["anatomy_source"].startswith("lab crown library")
    first = res.report["anatomy_fit"]["library_crown"]
    res2 = design_crown(prep, tooth=45, margin=margin, neighbors=na, params=replace(fast, library_variant=1))
    assert res2.report["anatomy_fit"]["library_crown"] != first  # browse to the other crown
    # the space between the neighbours still rules the width
    assert res.report["anatomy_fit"]["md"] == pytest.approx(na.space, abs=0.05)


def test_build_library_from_an_exocad_archive(tmp_path):
    case = tmp_path / "Case_0099_Surname"
    case.mkdir()
    m, crown = _crown(36)
    shift = np.array([10.0, 20.0, 5.0])
    save_stl(Mesh(crown.vertices + shift, crown.faces), case / "Case-36-crown_cad.stl")
    save_stl(Mesh(np.array([[-9, -9, -1], [9, -9, -1], [0, 9, -1.0]]) + shift, np.array([[0, 1, 2]])),
             case / "Case-LowerJaw.stl")
    vec = lambda tag, v: f"<{tag}><x>{v[0]}</x><y>{v[1]}</y><z>{v[2]}</z></{tag}>"
    eye = "".join(f"<_{r}{c}>{float(r == c)}</_{r}{c}>" for r in range(4) for c in range(4))
    margin = "".join(vec("Vec3", p) for p in _margin(m) + shift)
    (case / "Case.constructionInfo").write_text(f"""<ConstructionInfo><ScanFiles>
      <ScanFile><FileName>Case-LowerJaw.stl</FileName><TransformationMatrix>{eye}</TransformationMatrix>
        <PartType>PreparationScan</PartType><ToothNumbers><int>36</int></ToothNumbers></ScanFile></ScanFiles>
      <Teeth><Tooth><Number>36</Number><ReconstructionType>AnatomicCrown</ReconstructionType>
        <ToothScanFileName>Case-LowerJaw.stl</ToothScanFileName>{vec("Axis", [0, 0, 1])}{vec("AxisMesial", [1, 0, 0])}
        <Margin>{margin}</Margin></Tooth></Teeth></ConstructionInfo>""")
    lib = build_library(tmp_path, tmp_path / "lib", log=lambda *_: None)
    assert len(lib) == 1 and lib.entries[0]["key"] == "lower_6"
    index = (tmp_path / "lib" / "index.json").read_text()
    assert "Surname" not in index and str(tmp_path) not in index  # hashed ids only
    build_library(tmp_path, tmp_path / "lib", log=lambda *_: None)  # resumes: nothing new
    assert len(CrownLibrary(tmp_path / "lib")) == 1
