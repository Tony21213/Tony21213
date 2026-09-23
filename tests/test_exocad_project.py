"""exocad .constructionInfo import, final-crown lookup and gap-based prep location (synthetic data)."""

import numpy as np
import pytest

from crownai.design import CrownParameters
from crownai.exocad_project import (design_construction_case, find_final_crown, learn_construction_case,
                                    parse_construction_info)
from crownai.learning import CrownLearner
from crownai.margin import detect_margin
from crownai.mesh import Mesh, load_mesh, save_stl
from crownai.native_case import locate_prep
from crownai.synthetic import make_lower_arch, make_prep_at

SHIFT = np.array([5.0, -3.0, 2.0])  # scan file -> project frame (row-vector convention)
MESIAL = np.array([0.0, 1.0, 0.0])  # deliberately not the arch direction the neighbour analysis finds
FAST = CrownParameters(n_theta=64, n_v=28)


def _merge(*meshes):
    offs = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return Mesh(np.concatenate([m.vertices for m in meshes]),
                np.concatenate([m.faces + o for m, o in zip(meshes, offs)]))


def _matrix_xml(t):
    m = np.eye(4)
    m[3, :3] = t  # translation in the last row: world = [local, 1] @ M
    return "".join(f"<_{r}{c}>{m[r, c]}</_{r}{c}>" for r in range(4) for c in range(4))


def _vec(tag, v):
    return f"<{tag}><x>{v[0]}</x><y>{v[1]}</y><z>{v[2]}</z></{tag}>"


@pytest.fixture(scope="module")
def case_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("Case_0007_2025-03-26")
    jaw, teeth = make_lower_arch(missing=35)
    t = teeth[35]
    prep = make_prep_at(t["center"], t["md"], t["bl"])
    scan_local = _merge(jaw, prep)
    save_stl(scan_local, folder / "Case-LowerJaw.stl")  # stored in scan coordinates
    margin_world = detect_margin(prep, n_points=48) + SHIFT
    upper = Mesh(jaw.vertices * [1, 1, -1] + [0, 0, 14.5], jaw.faces[:, ::-1])  # flat-ish antagonist arch
    save_stl(upper, folder / "Case-UpperJaw.stl")
    margin_xml = "".join(_vec("Vec3", p) for p in margin_world)
    xml = f"""<ConstructionInfo>
      <ScanFiles>
        <ScanFile><FileName>Case-LowerJaw.stl</FileName><TransformationMatrix>{_matrix_xml(SHIFT)}</TransformationMatrix>
          <PartType>PreparationScan</PartType><ToothNumbers><int>35</int></ToothNumbers></ScanFile>
        <ScanFile><FileName>Case-UpperJaw.stl</FileName><TransformationMatrix>{_matrix_xml(SHIFT)}</TransformationMatrix>
          <PartType>Antagonist</PartType><ToothNumbers /></ScanFile>
      </ScanFiles>
      <Teeth>
        <Tooth><Number>35</Number><ReconstructionType>AnatomicCrown</ReconstructionType>
          <ToothScanFileName>Case-LowerJaw.stl</ToothScanFileName>
          {_vec("Axis", [0, 0, 1])}{_vec("AxisMesial", MESIAL)}<Margin>{margin_xml}</Margin></Tooth>
        <Tooth><Number>36</Number><ReconstructionType>Pontic</ReconstructionType>
          <ToothScanFileName>Case-LowerJaw.stl</ToothScanFileName><Margin /></Tooth>
      </Teeth>
    </ConstructionInfo>"""
    (folder / "Case.constructionInfo").write_text(xml)
    return folder, teeth, prep, margin_world


def test_parse_construction_info(case_folder):
    folder, _, prep, margin_world = case_folder
    case = parse_construction_info(folder / "Case.constructionInfo")
    assert set(case.scans) == {"Case-LowerJaw.stl", "Case-UpperJaw.stl"}
    assert case.crown_teeth() == [35]  # the pontic has no margin and is not a crown
    info = case.teeth[35]
    assert np.allclose(info.margin, margin_world) and np.allclose(info.md_direction, MESIAL)
    assert case.antagonist_scan(35).filename == "Case-UpperJaw.stl"
    # the scan is placed in the project frame by its matrix
    world = case.load_prep_scan(35)
    raw = load_mesh(folder / "Case-LowerJaw.stl")
    assert np.allclose(world.vertices, raw.vertices + SHIFT, atol=1e-4)


def test_find_final_crown_ignores_dates_and_case_numbers(tmp_path):
    for name in ["26_Case-2025-03-26-36-crown_cad.stl",  # case number and date both look like tooth 26
                 "Case-45-46-waxup_cad.stl", "Case-LowerJaw.stl", "notes-26.stl"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "crownai").mkdir()
    (tmp_path / "crownai" / "x-11-crown_cad.stl").write_bytes(b"")  # crownai's own output
    assert find_final_crown(tmp_path, 36).name == "26_Case-2025-03-26-36-crown_cad.stl"
    assert find_final_crown(tmp_path, 26) is None
    assert find_final_crown(tmp_path, 46).name == "Case-45-46-waxup_cad.stl"
    assert find_final_crown(tmp_path, 11) is None


def test_design_uses_the_technicians_margin_and_mesial_direction(case_folder):
    folder, teeth, _, _ = case_folder
    res, _, na = design_construction_case(folder, 35, params=FAST, compare_reference=False)
    assert res.crown.is_watertight()
    assert res.report["margin_source"] == "constructionInfo"
    # exocad's mesial direction wins over the arch estimate, so design and learning share a frame
    assert abs(res.frame.x @ MESIAL) > 0.999
    assert na is not None and len(na.neighbors) == 2


def test_learn_from_the_exported_final_crown(case_folder, tmp_path):
    folder, _, _, _ = case_folder
    res, _, _ = design_construction_case(folder, 35, params=FAST, compare_reference=False)
    save_stl(res.crown, folder / "Case-35-crown_cad.stl")  # "the technician's" finished crown
    lib = CrownLearner(tmp_path / "lib")
    out = learn_construction_case(folder, lib)
    assert len(out) == 1 and "error" not in out[0] and out[0]["examples"] == 1
    # and it is found again as the reference when designing
    res2, _, _ = design_construction_case(folder, 35, params=FAST)
    assert res2.report["reference_file"] == "Case-35-crown_cad.stl"
    assert res2.report["reference"]["crown_to_reference_mean_mm"] < 0.3


def test_locate_prep_finds_the_gap_in_the_arch():
    jaw, teeth = make_lower_arch(missing=35)
    t = teeth[35]
    scan = _merge(jaw, make_prep_at(t["center"], t["md"], t["bl"]))
    margin = locate_prep(scan, 35)
    assert np.linalg.norm(margin[:, :2].mean(0) - t["center"]) < 1.5
