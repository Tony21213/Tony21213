import json

import numpy as np

from crownai.exocad import discover_case, learn_case, process_case, teeth_from_xml, watch
from crownai.learning import CrownLearner
from crownai.mesh import save_stl


def _make_case(folder, prep, antagonist):
    folder.mkdir()
    save_stl(prep, folder / "Case01-Prep_36.stl")
    save_stl(antagonist, folder / "Case01-Antagonist.stl")
    (folder / "Case01.constructionInfo").write_text(
        "<ConstructionInfo><Teeth><Tooth><Number>36</Number><Type>Crown</Type></Tooth></Teeth>"
        "<Scanner><Serial>12345</Serial></Scanner></ConstructionInfo>")
    return folder


def test_teeth_from_xml(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text('<Project><Tooth number="46"/><ToothNumber>47</ToothNumber><Version>21</Version></Project>')
    assert teeth_from_xml(p) == [46, 47]
    (tmp_path / "bad.xml").write_text("<not xml")
    assert teeth_from_xml(tmp_path / "bad.xml") == []


def test_process_and_learn_case(tmp_path, prep, antagonist, fast_params):
    case = _make_case(tmp_path / "Case01", prep, antagonist)
    info = discover_case(case)
    assert info.prep.name == "Case01-Prep_36.stl"
    assert info.antagonist.name == "Case01-Antagonist.stl"
    assert info.teeth == [36]

    reports = process_case(case, params=fast_params, preview=False)
    assert reports[0]["tooth"] == 36 and reports[0]["watertight"]
    out = case / "crownai"
    assert (out / "crown_36.stl").exists() and (out / "done.json").exists()

    lib = CrownLearner(tmp_path / "lib")
    assert learn_case(case, lib) == []  # nothing finished yet

    # technician exports the finished crown back into the case
    (out / "crown_36.stl").rename(out / "crown_36_final.stl")
    learned = learn_case(case, lib)
    assert len(learned) == 1 and learned[0]["examples"] == 1
    assert learn_case(case, lib) == []  # same file is not learned twice


def test_margin_file_and_full_arch_crop(tmp_path, prep, antagonist, fast_params):
    from crownai.margin import detect_margin
    from crownai.mesh import Mesh

    # simulate a full-arch scan: the die plus a distant neighbour tooth
    neighbour = Mesh(prep.vertices + [15.0, 0, 0], prep.faces)
    arch = Mesh(np.vstack([prep.vertices, neighbour.vertices]),
                np.vstack([prep.faces, neighbour.faces + len(prep.vertices)]))
    case = tmp_path / "Arch"
    case.mkdir()
    save_stl(arch, case / "LowerJaw.stl")
    np.savetxt(case / "margin_36.xyz", detect_margin(prep, n_points=64))
    reports = process_case(case, params=fast_params, preview=False)
    assert reports[0]["tooth"] == 36
    assert reports[0]["mesiodistal_mm"] < 14  # neighbour was cropped away


def test_watch_once(tmp_path, prep, antagonist, fast_params):
    root = tmp_path / "inbox"
    root.mkdir()
    _make_case(root / "A", prep, antagonist)
    watch(root, once=True, params=fast_params, preview=False)
    done = json.loads((root / "A" / "crownai" / "done.json").read_text())
    assert done["crowns"][0]["tooth"] == 36
