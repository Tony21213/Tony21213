"""Case context for library choice, and the archive benchmark (synthetic exocad archive)."""

import json

import numpy as np
import pytest

from crownai.benchmark import failure_cause, run_benchmark
from crownai.crown_library import CrownLibrary, LibraryCrown, case_context
from crownai.margin import detect_margin, make_frame
from crownai.mesh import Mesh, save_stl
from crownai.posterior_model import _textbook_posterior, model_mesh
from crownai.synthetic import make_lower_arch, make_prep_at


def _merge(*meshes):
    offs = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return Mesh(np.concatenate([m.vertices for m in meshes]), np.concatenate([m.faces + o for m, o in zip(meshes, offs)]))


@pytest.fixture(scope="module")
def site():
    jaw, teeth = make_lower_arch(missing=35)
    t = teeth[35]
    prep = make_prep_at(t["center"], t["md"], t["bl"])
    margin = detect_margin(prep, n_points=64)
    mesial = np.r_[teeth[34]["center"] - t["center"], 0.0]
    mesial /= np.linalg.norm(mesial)
    upper = Mesh(jaw.vertices * [1, 1, -1] + [0, 0, 14.5], jaw.faces[:, ::-1])
    return jaw, prep, margin, mesial, upper, t


def test_case_context_sees_the_neighbours_and_the_antagonist(site):
    jaw, prep, margin, mesial, upper, t = site
    frame = make_frame(margin.mean(axis=0), (0, 0, 1), mesial)
    ctx = case_context(_merge(jaw, prep), upper, frame, t["md"], 35)
    assert {"mesial_level", "distal_level", "mesial_relief", "distal_relief", "room"} <= set(ctx)
    assert 4.0 < ctx["mesial_level"] < 9.0 and 4.0 < ctx["room"] < 14.0
    flat = Mesh(jaw.vertices * [1, 1, 0.2], jaw.faces)  # a worn, flat dentition
    assert case_context(flat, None, frame, t["md"], 35).get("mesial_relief", 0) < ctx["mesial_relief"]


def test_library_prefers_a_crown_from_a_similar_mouth(tmp_path):
    lib = CrownLibrary(tmp_path / "lib")
    m = _textbook_posterior(36)
    top = np.zeros((141, 141))
    outline = np.full((40, 180), 5.0)
    for cid, relief in (("sharp/36", 2.5), ("worn/36", 0.8)):
        lib.add(LibraryCrown(cid, "lower_6", 36, m.md, m.bl, m.height, top, outline,
                             context={"mesial_relief": relief, "mesial_level": 6.0}))
    lib = CrownLibrary(tmp_path / "lib")
    assert lib.choose("lower_6", m.md, m.bl, context={"mesial_relief": 0.9}).id == "worn/36"
    assert lib.choose("lower_6", m.md, m.bl, context={"mesial_relief": 2.3}).id == "sharp/36"


def test_failure_causes():
    assert failure_cause("12829/15360 intaglio rays missed the preparation") == "die"
    assert failure_cause("tooth 36: no margin line recorded") == "margin"
    assert failure_cause("[Errno 2] No such file or directory: '<path>'") == "files"
    assert failure_cause("something odd") == "other"


def test_benchmark_on_an_exocad_archive(site, tmp_path):
    jaw, prep, margin, mesial, upper, t = site
    case = tmp_path / "Case_0042_Surname"
    case.mkdir()
    save_stl(_merge(jaw, prep), case / "Case-LowerJaw.stl")
    save_stl(upper, case / "Case-UpperJaw.stl")
    # "the technician's crown": the textbook 35 standing on the preparation
    model = _textbook_posterior(35)
    z = np.array([0, 0, 1.0])
    buccal = np.cross(z, mesial)  # quadrant 3
    local = model_mesh(model, 120, 80)
    v = margin.mean(axis=0) + local.vertices @ np.stack([mesial, buccal, z])
    save_stl(Mesh(v, local.faces), case / "Case-35-crown_cad.stl")
    vec = lambda tag, p: f"<{tag}><x>{p[0]}</x><y>{p[1]}</y><z>{p[2]}</z></{tag}>"
    eye = "".join(f"<_{r}{c}>{float(r == c)}</_{r}{c}>" for r in range(4) for c in range(4))
    (case / "Case.constructionInfo").write_text(f"""<ConstructionInfo><ScanFiles>
      <ScanFile><FileName>Case-LowerJaw.stl</FileName><TransformationMatrix>{eye}</TransformationMatrix>
        <PartType>PreparationScan</PartType><ToothNumbers><int>35</int></ToothNumbers></ScanFile>
      <ScanFile><FileName>Case-UpperJaw.stl</FileName><TransformationMatrix>{eye}</TransformationMatrix>
        <PartType>Antagonist</PartType><ToothNumbers /></ScanFile></ScanFiles>
      <Teeth><Tooth><Number>35</Number><ReconstructionType>AnatomicCrown</ReconstructionType>
        <ToothScanFileName>Case-LowerJaw.stl</ToothScanFileName>{vec("Axis", z)}{vec("AxisMesial", mesial)}
        <Margin>{"".join(vec("Vec3", p) for p in margin)}</Margin></Tooth></Teeth></ConstructionInfo>""")
    out = tmp_path / "report"
    summary = run_benchmark(tmp_path, out, n=5, variants=("rules",), pictures=1, log=lambda *_: None)
    assert summary["rules"]["crowns"] == 1 and summary["rules"]["succeeded"] == 1
    row = json.loads((out / "results.jsonl").read_text().splitlines()[0])
    assert row["ok"] and row["tech_to_crown_mm"] < 1.5 and len(row["contact_gaps_mm"]) == 2
    report = (out / "results.jsonl").read_text() + (out / "summary.json").read_text()
    assert "Surname" not in report and str(tmp_path) not in report  # shareable: hashes only
    assert len(list((out / "pictures").glob("*.png"))) == 1
