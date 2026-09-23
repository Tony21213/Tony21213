"""File-based bridge to exocad DentalCAD.

exocad does not offer a public plugin SDK, so crownai integrates through the
case folder instead:

1. In exocad, export the case (scans as STL and, ideally, the margin line).
2. ``crownai exocad-case <folder>`` (or the ``exocad-watch`` loop) finds the
   preparation scan, antagonist, margin and tooth numbers, designs the crowns
   and writes them to ``<folder>/crownai/``.
3. Import the resulting STL back into exocad as an external mesh.
4. After the technician finishes the crown in exocad, export the final crown
   into the case folder (``crownai/crown_<tooth>_final.stl`` or any
   ``*final*.stl`` / ``*<tooth>*final*.stl``); ``crownai learn-case`` (or the
   watch loop with ``--library``) adds it to the training library.

File discovery is pattern based and deliberately tolerant; every pattern can be
overridden from the command line because naming differs between exocad
versions, scanners and lab conventions.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .design import CrownParameters, design_crown
from .margin import load_margin, make_frame
from .mesh import Mesh, load_stl, nearest_distance, save_stl

PREP_PATTERNS = ("*prep*.stl", "*die*.stl", "*stump*.stl", "*stumpf*.stl", "*upperjaw*.stl", "*lowerjaw*.stl")
ANTAGONIST_PATTERNS = ("*antag*.stl", "*opposing*.stl", "*gegenkiefer*.stl")
MARGIN_PATTERNS = ("*margin*.pts", "*margin*.xyz", "*margin*.txt", "*margin*.csv")
INFO_PATTERNS = ("*.constructionInfo", "*.dentalProject", "*.xml")
OUTPUT_DIR = "crownai"


@dataclass
class ExocadCase:
    folder: Path
    prep: Path | None = None
    antagonist: Path | None = None
    margins: list[Path] = field(default_factory=list)
    teeth: list[int] = field(default_factory=list)


def _match_ci(folder: Path, patterns) -> list[Path]:
    """Case-insensitive glob (exocad exports mix upper/lower case)."""
    regexes = [re.compile(re.escape(p).replace(r"\*", ".*") + "$", re.I) for p in patterns]
    out = []
    for p in sorted(folder.rglob("*"), key=lambda q: str(q).lower()):
        if p.is_file() and OUTPUT_DIR not in p.relative_to(folder).parts:
            if any(r.match(p.name) for r in regexes):
                out.append(p)
    return out


def teeth_from_xml(path: Path) -> list[int]:
    """Collect FDI tooth numbers from an exocad project/construction XML.

    The schema is not public, so this looks at elements whose name mentions
    "tooth" and takes their text, number attributes or ``<Number>`` children
    when they hold a valid FDI number (11-48).
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    teeth: list[int] = []

    def add(value: str | None):
        if value and value.strip().isdigit():
            n = int(value.strip())
            if 1 <= n // 10 <= 4 and 1 <= n % 10 <= 8 and n not in teeth:
                teeth.append(n)

    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()
        if "tooth" not in tag:
            continue
        if "number" in tag or tag == "tooth":
            add(el.text)
        for k, val in el.attrib.items():
            if "number" in k.lower() or "tooth" in k.lower():
                add(val)
        for child in el:
            if child.tag.split("}")[-1].lower() in ("number", "toothnumber"):
                add(child.text)
    return teeth


def discover_case(folder: str | Path, prep_patterns=PREP_PATTERNS,
                  antagonist_patterns=ANTAGONIST_PATTERNS) -> ExocadCase:
    folder = Path(folder)
    case = ExocadCase(folder)
    antagonists = _match_ci(folder, antagonist_patterns)
    preps = [p for p in _match_ci(folder, prep_patterns) if p not in antagonists]
    case.prep = preps[0] if preps else None
    case.antagonist = antagonists[0] if antagonists else None
    case.margins = _match_ci(folder, MARGIN_PATTERNS)
    for info in _match_ci(folder, INFO_PATTERNS):
        for t in teeth_from_xml(info):
            if t not in case.teeth:
                case.teeth.append(t)
    return case


def crop_to_margin(scan: Mesh, margin: np.ndarray, axis=(0.0, 0.0, 1.0),
                   radial_pad: float = 1.0, depth: float = 3.0, height_cap: float | None = None) -> Mesh:
    """Cut the region of one preparation out of a full-arch scan.

    Keeps triangles inside a cylinder around the margin (plus ``radial_pad``)
    and above ``depth`` mm below the lowest margin point. Intraoral scans are
    one continuous surface (gum tissue connects every tooth), so on a crowded
    arch this cylinder can also catch part of a neighbouring, unprepared
    tooth still rising well above the margin - it stays topologically
    connected via the gum, so a connected-component filter cannot separate
    it. ``height_cap``, when given (e.g. the tooth's expected natural crown
    height + a couple mm), additionally excludes anything higher than that
    above the margin's own top, which a real prepared stump should not reach.
    """
    frame = make_frame(margin.mean(axis=0), axis)
    m = frame.to_local(margin)
    r_max = np.hypot(m[:, 0], m[:, 1]).max() + radial_pad
    q = frame.to_local(scan.triangles.reshape(-1, 3)).reshape(-1, 3, 3)
    c = q.mean(axis=1)
    keep = (np.hypot(c[:, 0], c[:, 1]) <= r_max) & (c[:, 2] >= m[:, 2].min() - depth)
    if height_cap is not None:
        keep &= c[:, 2] <= m[:, 2].max() + height_cap
    faces = scan.faces[keep]
    used, inv = np.unique(faces, return_inverse=True)
    cropped = Mesh(scan.vertices[used], inv.reshape(-1, 3))
    return _largest_component_near_margin(cropped, margin)


def _largest_component_near_margin(mesh: Mesh, margin: np.ndarray) -> Mesh:
    """Keep only the connected surface patch that the margin line actually sits on."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = mesh.faces
    if len(f) == 0:
        return mesh
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = len(mesh.vertices)
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)
    if n_comp <= 1:
        return mesh
    _, idx = nearest_distance(margin, mesh.vertices)
    keep_label = np.bincount(labels[idx]).argmax()
    keep_v = labels == keep_label
    faces = mesh.faces[keep_v[mesh.faces].all(axis=1)]
    used, inv = np.unique(faces, return_inverse=True)
    return Mesh(mesh.vertices[used], inv.reshape(-1, 3))


def process_case(folder: str | Path, *, teeth: list[int] | None = None, prep: str | Path | None = None,
                 antagonist: str | Path | None = None, margin: str | Path | None = None,
                 axis=(0.0, 0.0, 1.0), md_direction=(1.0, 0.0, 0.0),
                 params: CrownParameters | None = None, shape_model=None, learner=None,
                 preview: bool = True) -> list[dict]:
    """Design one crown per margin file (or per tooth on a segmented die) in an exocad case."""
    folder = Path(folder)
    case = discover_case(folder)
    prep_path = Path(prep) if prep else case.prep
    if prep_path is None:
        raise FileNotFoundError(f"{folder}: no preparation scan found (use --prep)")
    ant_path = Path(antagonist) if antagonist else case.antagonist
    margin_paths = [Path(margin)] if margin else case.margins
    tooth_list = teeth or case.teeth

    scan = load_stl(prep_path)
    ant_mesh = load_stl(ant_path) if ant_path else None
    out_dir = folder / OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    jobs = []
    if margin_paths:
        for k, mp in enumerate(margin_paths):
            tooth = _tooth_from_name(mp.name) or (tooth_list[k] if k < len(tooth_list) else None)
            jobs.append((tooth, load_margin(mp)))
    else:  # segmented die: one crown, margin detected automatically
        jobs.append((tooth_list[0] if tooth_list else None, None))

    results = []
    for tooth, margin_pts in jobs:
        die = crop_to_margin(scan, margin_pts, axis) if margin_pts is not None else scan
        res = design_crown(die, tooth=tooth, margin=margin_pts, antagonist=ant_mesh, axis=axis,
                           md_direction=md_direction, shape_model=shape_model, learner=learner,
                           params=params)
        stem = f"crown_{tooth}" if tooth else "crown"
        save_stl(res.crown, out_dir / f"{stem}.stl")
        np.savetxt(out_dir / f"{stem}_margin.xyz", res.margin, fmt="%.4f")
        report = {"tooth": tooth, "prep_scan": prep_path.name,
                  "antagonist_scan": ant_path.name if ant_path else None,
                  "crown_stl": f"{OUTPUT_DIR}/{stem}.stl", **res.report}
        (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if preview:
            try:
                from .viz import render_preview

                render_preview(res, die, out_dir / f"{stem}_preview.png", ant_mesh, title=stem)
            except ImportError:
                pass
        results.append(report)
    (out_dir / "done.json").write_text(json.dumps({"crowns": results}, indent=2, ensure_ascii=False))
    return results


def learn_case(folder: str | Path, learner, *, axis=(0.0, 0.0, 1.0), md_direction=(1.0, 0.0, 0.0)) -> list[dict]:
    """Add every finished crown of a processed case to the training library.

    Needs the ``crownai/crown_<tooth>_report.json`` + ``_margin.xyz`` written by
    :func:`process_case` and a final crown STL exported from exocad.
    """
    from .learning import learn_from_crown

    folder = Path(folder)
    out_dir = folder / OUTPUT_DIR
    learned_path = out_dir / "learned.json"
    learned = json.loads(learned_path.read_text()) if learned_path.exists() else {}
    results = []
    for report_path in sorted(out_dir.glob("crown*_report.json")):
        report = json.loads(report_path.read_text())
        stem = report_path.name[: -len("_report.json")]
        tooth = report.get("tooth")
        final = _final_crown(folder, stem, tooth)
        if final is None:
            continue
        stamp = final.stat().st_mtime
        if learned.get(stem) == stamp:
            continue  # already learned this exact file
        margin = load_margin(out_dir / f"{stem}_margin.xyz")
        prep = None
        prep_file = folder / report["prep_scan"] if report.get("prep_scan") else None
        if prep_file and prep_file.exists():
            prep = crop_to_margin(load_stl(prep_file), margin, axis)
        info = learn_from_crown(learner, load_stl(final), prep=prep, margin=margin, tooth=tooth,
                                axis=axis, md_direction=md_direction, case_id=f"{folder.name}/{stem}")
        learned[stem] = stamp
        results.append({"crown": stem, "final": final.name, **info})
    if results:
        learned_path.write_text(json.dumps(learned, indent=2))
    return results


def _final_crown(folder: Path, stem: str, tooth: int | None) -> Path | None:
    exact = folder / OUTPUT_DIR / f"{stem}_final.stl"
    if exact.exists():
        return exact
    for p in sorted(folder.rglob("*.stl"), key=lambda q: str(q).lower()):
        name = p.name.lower()
        if OUTPUT_DIR in p.relative_to(folder).parts or "final" not in name:
            continue
        if tooth is None or _tooth_from_name(p.name) in (tooth, None):
            return p
    return None


def _tooth_from_name(name: str) -> int | None:
    for m in re.finditer(r"(?<!\d)([1-4][1-8])(?!\d)", name):
        return int(m.group(1))
    return None


def watch(root: str | Path, interval: float = 5.0, once: bool = False, learner=None, **kw) -> None:
    """Poll ``root`` for case folders and process them.

    * a case without ``crownai/done.json`` gets its crowns designed, once its
      files have not changed for one polling interval (so half-written exports
      are not picked up);
    * with a ``learner``, final crowns exported back into processed cases are
      learned automatically - the tool improves with every finished case.
    """
    root = Path(root)
    seen: dict[Path, float] = {}
    while True:
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if (case_dir / OUTPUT_DIR / "done.json").exists():
                if learner is not None:
                    try:
                        for info in learn_case(case_dir, learner, axis=kw.get("axis", (0.0, 0.0, 1.0)),
                                               md_direction=kw.get("md_direction", (1.0, 0.0, 0.0))):
                            print(f"[crownai] learned {case_dir.name}/{info['crown']}: "
                                  f"{info['examples']} {info['class']} cases, {info['regressor']}")
                    except Exception as exc:
                        print(f"[crownai] {case_dir.name}: learning failed - {exc}")
                continue
            files = [f for f in case_dir.rglob("*") if f.is_file()]
            if not files:
                continue
            stamp = max(f.stat().st_mtime for f in files)
            if once or seen.get(case_dir) == stamp:
                try:
                    reports = process_case(case_dir, learner=learner, **kw)
                    print(f"[crownai] {case_dir.name}: {len(reports)} crown(s) designed")
                except Exception as exc:  # keep watching other cases
                    (case_dir / OUTPUT_DIR).mkdir(exist_ok=True)
                    (case_dir / OUTPUT_DIR / "done.json").write_text(json.dumps({"error": str(exc)}))
                    print(f"[crownai] {case_dir.name}: FAILED - {exc}")
            seen[case_dir] = stamp
        if once:
            return
        time.sleep(interval)
