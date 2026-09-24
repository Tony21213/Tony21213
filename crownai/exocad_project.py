"""Read exocad's own construction data instead of guessing at file naming.

Many exocad exports (anything routed through DentalDB/case management, not
just simple STL dumps) carry a ``.constructionInfo`` XML file next to the
case. It stores, per tooth: the technician's exact preparation margin line,
the insertion axis and mesiodistal direction - and, for every raw scan file,
the 4x4 matrix that places it in the same shared project coordinate frame.
``.dentalProject`` does not carry this (checked: no ``Margin``/``ScanFile``
elements), so ``.constructionInfo`` is required.

This is strictly more accurate than the two other ways crownai locates a
margin: it needs no explicit ``*margin*.pts`` file (unlike ``exocad.py``'s
pattern matching) and no gap-detection heuristic (unlike
:mod:`crownai.native_case`) - the line is the one the technician actually
drew. Use it whenever a ``.constructionInfo`` is present; fall back to the
other two only when it is not.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import Mesh, load_mesh, save_stl

_SINGLE_CROWN_TYPES = {"AnatomicCrown", "FullAnatomicCrown", "AnatomicCrownCutback", "ThimbleCrown"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")  # strip first: a case-number prefix or date can look like a tooth number
_FINAL_RE = re.compile(r"(?:^|[-_])((?:\d{1,2}[-,])*\d{1,2})[-_](crown_cad|waxup\w*cad|final\w*)", re.I)
OUTPUT_DIR = "crownai"


@dataclass
class ScanFile:
    filename: str
    matrix: np.ndarray  # 4x4; world = [local, 1] @ matrix (row-vector convention)
    part_type: str | None
    teeth: list[int]


@dataclass
class ToothInfo:
    number: int
    margin: np.ndarray  # (n, 3) world/design space
    axis: np.ndarray  # insertion direction, world/design space
    md_direction: np.ndarray  # mesial direction, world/design space
    scan_filename: str | None
    reconstruction_type: str | None


@dataclass
class ConstructionCase:
    path: Path
    folder: Path
    scans: dict[str, ScanFile] = field(default_factory=dict)
    teeth: dict[int, ToothInfo] = field(default_factory=dict)

    def crown_teeth(self) -> list[int]:
        """Tooth numbers that are single-crown restorations with a usable margin."""
        return sorted(t for t, info in self.teeth.items()
                      if info.reconstruction_type in _SINGLE_CROWN_TYPES and len(info.margin) >= 8)

    def antagonist_scan(self, tooth: int) -> ScanFile | None:
        own = self.teeth[tooth].scan_filename
        for s in self.scans.values():
            if s.part_type == "Antagonist" and s.filename != own:
                return s
        return None

    def load_world(self, scan: ScanFile) -> Mesh:
        """Load a raw scan STL and place it in the shared project coordinate frame."""
        mesh = load_mesh(self.folder / scan.filename)
        vh = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
        world = (vh @ scan.matrix)[:, :3]
        return Mesh(world, mesh.faces.copy())

    def load_prep_scan(self, tooth: int) -> Mesh:
        fn = self.teeth[tooth].scan_filename
        if fn is None or fn not in self.scans:
            raise FileNotFoundError(f"tooth {tooth}: no scan file recorded in {self.path.name}")
        return self.load_world(self.scans[fn])


def _matrix(el: ET.Element) -> np.ndarray:
    vals = {c.tag: float(c.text) for c in el}
    return np.array([[vals[f"_{r}{c}"] for c in range(4)] for r in range(4)])


def _vec3(el: ET.Element) -> np.ndarray:
    return np.array([float(el.find("x").text), float(el.find("y").text), float(el.find("z").text)])


def find_construction_info(folder: str | Path) -> Path | None:
    matches = sorted(Path(folder).glob("*.constructionInfo"))
    return matches[0] if matches else None


def parse_construction_info(path: str | Path) -> ConstructionCase:
    path = Path(path)
    root = ET.parse(path).getroot()
    case = ConstructionCase(path, path.parent)

    for sf in root.iter("ScanFile"):
        fn_el, tm_el = sf.find("FileName"), sf.find("TransformationMatrix")
        if fn_el is None or tm_el is None or fn_el.text is None:
            continue
        pt_el, tn_el = sf.find("PartType"), sf.find("ToothNumbers")
        teeth = [int(e.text) for e in tn_el] if tn_el is not None else []
        case.scans[fn_el.text] = ScanFile(fn_el.text, _matrix(tm_el), pt_el.text if pt_el is not None else None, teeth)

    for tooth_el in root.iter("Tooth"):
        num_el, margin_el = tooth_el.find("Number"), tooth_el.find("Margin")
        if num_el is None or margin_el is None or not (num_el.text or "").strip().isdigit():
            continue
        pts = np.array([_vec3(v) for v in margin_el.findall("Vec3")], dtype=np.float64)
        axis_el, mesial_el = tooth_el.find("Axis"), tooth_el.find("AxisMesial")
        axis = _vec3(axis_el) if axis_el is not None else np.array([0.0, 0.0, 1.0])
        mesial = _vec3(mesial_el) if mesial_el is not None else np.array([1.0, 0.0, 0.0])
        scan_el, type_el = tooth_el.find("ToothScanFileName"), tooth_el.find("ReconstructionType")
        case.teeth[int(num_el.text)] = ToothInfo(
            int(num_el.text), pts, axis, mesial,
            scan_el.text if scan_el is not None else None,
            type_el.text if type_el is not None else None,
        )
    return case


def find_final_crown(folder: str | Path, tooth: int) -> Path | None:
    """The technician's finished crown for ``tooth``, if one was exported back into the folder.

    Matches this common lab convention (``*-<tooth>-crown_cad.stl``,
    ``*-<tooth>-<tooth2>-waxup_cad.stl``) plus the generic ``*final*``
    convention used elsewhere in crownai. Only the run of tooth numbers
    immediately before the crown/waxup/final keyword counts - a case number
    or a date elsewhere in the filename can coincidentally equal a tooth
    number (e.g. a case folder named ``26_...`` restoring tooth 26 itself),
    so a bare substring search would grab the wrong file. Files already
    written by crownai itself are skipped.
    """
    folder = Path(folder)
    for p in sorted(list(folder.glob("*.stl")) + list(folder.glob("*.ply"))):
        if OUTPUT_DIR in p.relative_to(folder).parts:
            continue
        name = _DATE_RE.sub("_", p.stem)
        m = _FINAL_RE.search(name)
        if not m:
            continue
        teeth_in_name = {int(t) for t in re.split(r"[-,]", m.group(1)) if t.isdigit()}
        if tooth in teeth_in_name:
            return p
    return None


def design_construction_case(folder: str | Path, tooth: int, *, learner=None, params=None,
                             use_neighbors: bool = True, occlusion=None, compare_reference: bool = True,
                             case: ConstructionCase | None = None):
    """Design one crown straight from exocad's ``.constructionInfo`` - exact margin, no guessing.

    Returns ``(CrownResult, ConstructionCase, NeighborAnalysis | None)``. When
    the technician's finished crown was exported back into the folder
    (:func:`find_final_crown`), ``report["reference"]`` compares the two.
    """
    from .anatomy import tooth_type_for_fdi
    from .arch import analyze_neighbors
    from .design import design_crown
    from .exocad import crop_to_margin
    from .metrics import compare_to_reference

    folder = Path(folder)
    if case is None:
        ci = find_construction_info(folder)
        if ci is None:
            raise FileNotFoundError(f"{folder}: no .constructionInfo file")
        case = parse_construction_info(ci)
    if tooth not in case.teeth:
        raise ValueError(f"tooth {tooth} not found in {case.path.name}")
    info = case.teeth[tooth]
    if len(info.margin) < 8:
        raise ValueError(f"tooth {tooth}: no margin line recorded in {case.path.name}")

    jaw_world = case.load_prep_scan(tooth)
    # a crowded arch's crop cylinder can catch part of a still-connected neighbour
    # tooth well above where a real prepared stump would reach (see crop_to_margin)
    height_cap = tooth_type_for_fdi(tooth).height + 2.0
    die = crop_to_margin(jaw_world, info.margin, info.axis, height_cap=height_cap)
    ant_scan = case.antagonist_scan(tooth)
    antagonist = case.load_world(ant_scan) if ant_scan is not None else None
    neighbors = None
    if use_neighbors:
        try:
            neighbors = analyze_neighbors(jaw_world, info.margin, info.axis, tooth=tooth)
        except Exception:
            neighbors = None

    res = design_crown(die, tooth=tooth, margin=info.margin, antagonist=antagonist, axis=info.axis,
                       md_direction=info.md_direction, neighbors=neighbors, learner=learner,
                       occlusion=occlusion, jaw=jaw_world, params=params,
                       # the technician's mesial direction, the same one learn_construction_case
                       # trains with - not the arch estimate from the neighbour analysis
                       trust_md_direction=True)
    res.report["margin_source"] = "constructionInfo"

    ref_path = find_final_crown(folder, tooth) if compare_reference else None
    if ref_path is not None:
        # the reference file can be a multi-tooth waxup/splint (this lab names them
        # "<t1>-<t2>-waxup_cad.stl"): crop to a cylinder around this tooth's margin so the
        # comparison isn't polluted by neighbouring teeth in the same file (matches the crop
        # learn_construction_case already applies before training on the same kind of file).
        reference = crop_to_margin(load_mesh(ref_path), info.margin, info.axis, radial_pad=2.5, depth=1.0)
        top = res.frame.to_local(info.margin)[:, 2].max()
        res.report["reference"] = compare_to_reference(res.outer[:, 4:].reshape(-1, 3), res.crown,
                                                        reference, res.frame, top)
        res.report["reference_file"] = ref_path.name
    return res, case, neighbors


def process_construction_case(folder: str | Path, *, teeth: list[int] | None = None, out_dir=None,
                              params=None, learner=None, use_neighbors: bool = True, occlusion=None,
                              preview: bool = True) -> list[dict]:
    """Design every restored tooth of a case folder from its ``.constructionInfo``.

    Writes ``crown_<tooth>.stl`` + report (+ preview) to ``out_dir``
    (default: ``<folder>/crownai``, matching :func:`crownai.exocad.process_case`).
    """
    folder = Path(folder)
    ci = find_construction_info(folder)
    if ci is None:
        raise FileNotFoundError(f"{folder}: no .constructionInfo file")
    case = parse_construction_info(ci)
    tooth_list = teeth or case.crown_teeth()
    out_dir = Path(out_dir) if out_dir is not None else folder / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for tooth in tooth_list:
        stem = f"crown_{tooth}"
        try:
            res, _, _ = design_construction_case(folder, tooth, learner=learner, params=params,
                                                  use_neighbors=use_neighbors, occlusion=occlusion,
                                                  case=case)
        except Exception as exc:
            results.append({"tooth": tooth, "error": str(exc)})
            continue
        save_stl(res.crown, out_dir / f"{stem}.stl")
        report = {"tooth": tooth, **res.report}
        (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if preview:
            try:
                from .viz import render_preview

                ant_scan = case.antagonist_scan(tooth)
                antagonist = case.load_world(ant_scan) if ant_scan is not None else None
                render_preview(res, case.load_prep_scan(tooth), out_dir / f"{stem}_preview.png",
                               antagonist, title=stem)
            except ImportError:
                pass
        results.append(report)
    (out_dir / "done.json").write_text(json.dumps({"crowns": results}, indent=2, ensure_ascii=False))
    return results


def learn_construction_case(folder: str | Path, learner, *, teeth: list[int] | None = None) -> list[dict]:
    """Add every finished crown exported back into a case folder to the training library.

    Uses the exact margin from ``.constructionInfo`` and the technician's own
    finished crown (:func:`find_final_crown`) as the training example - no
    intermediate crownai design step needed, unlike
    :func:`crownai.exocad.learn_case`.
    """
    from .anatomy import tooth_type_for_fdi
    from .exocad import crop_to_margin
    from .learning import learn_from_crown

    folder = Path(folder)
    ci = find_construction_info(folder)
    if ci is None:
        raise FileNotFoundError(f"{folder}: no .constructionInfo file")
    case = parse_construction_info(ci)
    tooth_list = teeth or case.crown_teeth()

    results = []
    for tooth in tooth_list:
        info = case.teeth[tooth]
        if len(info.margin) < 8:
            continue
        ref_path = find_final_crown(folder, tooth)
        if ref_path is None:
            continue
        try:
            crown_full = load_mesh(ref_path)
            # the reference file can be a multi-tooth waxup/splint (this lab names them
            # "<t1>-<t2>-waxup_cad.stl"): crop to a cylinder around this tooth's margin so
            # neighbouring teeth in the same file do not pollute the learned shape.
            crown = crop_to_margin(crown_full, info.margin, info.axis, radial_pad=2.5, depth=1.0)
            jaw_world = case.load_prep_scan(tooth)
            height_cap = tooth_type_for_fdi(tooth).height + 2.0
            prep = crop_to_margin(jaw_world, info.margin, info.axis, height_cap=height_cap)
            info_out = learn_from_crown(learner, crown, prep=prep, margin=info.margin, tooth=tooth,
                                        axis=info.axis, md_direction=info.md_direction,
                                        case_id=f"{folder.name}/{tooth}")
        except Exception as exc:  # keep learning from the rest of this case / the rest of the batch
            results.append({"tooth": tooth, "reference_file": ref_path.name, "error": str(exc)})
            continue
        results.append({"tooth": tooth, "reference_file": ref_path.name, **info_out})
    return results
