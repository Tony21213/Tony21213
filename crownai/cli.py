"""Command line interface.

``crownai design | demo | train-ssm | learn | learn-case | library | exocad-case | exocad-watch |
webview | design-webview | learn-webview``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .anatomy import ShapeModel, synthetic_library, tooth_type_for_fdi
from .design import CrownParameters, design_crown
from .margin import load_margin
from .mesh import load_stl, save_stl


def _vec(text: str) -> tuple[float, float, float]:
    parts = [float(t) for t in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three numbers, e.g. 0,0,1")
    return tuple(parts)


def _add_design_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("crown parameters (mm)")
    d = CrownParameters()
    g.add_argument("--cement-gap", type=float, default=d.cement_gap)
    g.add_argument("--margin-gap", type=float, default=d.margin_gap)
    g.add_argument("--min-axial", type=float, default=d.min_axial)
    g.add_argument("--min-occlusal", type=float, default=d.min_occlusal)
    g.add_argument("--clearance", type=float, default=d.occlusal_clearance,
                   help="distance to keep from the antagonist (negative = contact)")
    p.add_argument("--axis", type=_vec, default=(0.0, 0.0, 1.0), help="insertion axis (default 0,0,1)")
    p.add_argument("--md-dir", type=_vec, default=(1.0, 0.0, 0.0), help="mesiodistal direction (default 1,0,0)")
    p.add_argument("--ssm", type=Path, help="trained shape model (.npz) instead of the parametric library")
    p.add_argument("--library", type=Path, help="learning library folder: design with the anatomy learned from past cases")
    o = p.add_argument_group("functional occlusion (Slavicek sequential guidance)")
    o.add_argument("--slavicek", action="store_true",
                   help="centric contacts + no interference in protrusion / latero- / mediotrusion")
    o.add_argument("--condylar-inclination", type=float, default=35.0,
                   help="deg vs occlusal plane (axiography value, default mean 35)")
    o.add_argument("--bennett", type=float, default=10.0, help="Bennett angle, deg")


def _params(a) -> CrownParameters:
    return CrownParameters(cement_gap=a.cement_gap, margin_gap=a.margin_gap, min_axial=a.min_axial,
                           min_occlusal=a.min_occlusal, occlusal_clearance=a.clearance)


def _occlusion(a):
    if not getattr(a, "slavicek", False):
        return None
    from .occlusion import SlavicekConcept

    return SlavicekConcept(condylar_inclination=a.condylar_inclination, bennett_angle=a.bennett)


def _learner(a):
    if getattr(a, "library", None) is None:
        return None
    from .learning import CrownLearner

    return CrownLearner(a.library)


def _preview(result, prep, path, antagonist=None, title=""):
    try:
        from .viz import render_preview
    except ImportError:
        print("matplotlib not installed: skipping preview", file=sys.stderr)
        return
    render_preview(result, prep, path, antagonist, title)


def cmd_design(a) -> int:
    prep = load_stl(a.prep)
    ant = load_stl(a.antagonist) if a.antagonist else None
    margin = load_margin(a.margin) if a.margin else None
    ssm = ShapeModel.load(a.ssm) if a.ssm else None
    res = design_crown(prep, tooth=a.tooth, margin=margin, antagonist=ant, axis=a.axis,
                       md_direction=a.md_dir, shape_model=ssm, learner=_learner(a), occlusion=_occlusion(a),
                       params=_params(a))
    save_stl(res.crown, a.out)
    if a.report:
        Path(a.report).write_text(json.dumps(res.report, indent=2, ensure_ascii=False))
    if a.preview:
        _preview(res, prep, a.preview, ant, f"Tooth {a.tooth or ''}")
    print(json.dumps(res.report, indent=2, ensure_ascii=False))
    return 0


def cmd_demo(a) -> int:
    from .synthetic import make_antagonist, make_prepared_molar

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prep, ant = make_prepared_molar(), make_antagonist()
    save_stl(prep, out / "prep_36.stl")
    save_stl(ant, out / "antagonist.stl")
    res = design_crown(prep, tooth=36, antagonist=ant)
    save_stl(res.crown, out / "crown_36.stl")
    (out / "crown_36_report.json").write_text(json.dumps(res.report, indent=2))
    _preview(res, prep, out / "crown_36_preview.png", ant, "Demo: crown on a synthetic prepared 36")
    print(json.dumps(res.report, indent=2))
    print(f"files written to {out}/")
    return 0


def cmd_train(a) -> int:
    if a.synthetic:
        ttype = tooth_type_for_fdi(a.tooth)
        sigs = synthetic_library(a.synthetic, np.random.default_rng(a.seed), ttype)
        model = ShapeModel.fit(sigs, 64, 32, 0.45)
    else:
        files = sorted(Path(a.library).glob("*.stl"))
        if len(files) < 3:
            print(f"need at least 3 crown STLs in {a.library}", file=sys.stderr)
            return 2
        model = ShapeModel.from_meshes([load_stl(f) for f in files])
    model.save(a.out)
    print(f"shape model: {len(model.stddev)} components -> {a.out}")
    return 0


def cmd_exocad_case(a) -> int:
    from .exocad import process_case

    ssm = ShapeModel.load(a.ssm) if a.ssm else None
    reports = process_case(a.folder, teeth=a.tooth or None, prep=a.prep, antagonist=a.antagonist,
                           margin=a.margin, axis=a.axis, md_direction=a.md_dir, params=_params(a),
                           shape_model=ssm, learner=_learner(a), preview=not a.no_preview)
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


def cmd_exocad_watch(a) -> int:
    from .exocad import watch

    ssm = ShapeModel.load(a.ssm) if a.ssm else None
    print(f"[crownai] watching {a.root} (Ctrl+C to stop)")
    try:
        watch(a.root, interval=a.interval, once=a.once, axis=a.axis, md_direction=a.md_dir,
              params=_params(a), shape_model=ssm, learner=_learner(a))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_learn(a) -> int:
    from .learning import CrownLearner, learn_from_crown

    info = learn_from_crown(CrownLearner(a.library), load_stl(a.crown),
                            prep=load_stl(a.prep) if a.prep else None,
                            margin=load_margin(a.margin) if a.margin else None, tooth=a.tooth,
                            axis=a.axis, md_direction=a.md_dir, case_id=a.case_id)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_learn_case(a) -> int:
    from .exocad import learn_case
    from .learning import CrownLearner

    learner = CrownLearner(a.library)
    total = 0
    for folder in a.folders:
        for info in learn_case(folder, learner, axis=a.axis, md_direction=a.md_dir):
            print(json.dumps(info, ensure_ascii=False))
            total += 1
    print(f"learned {total} crown(s)")
    return 0


def cmd_library(a) -> int:
    from .learning import CrownLearner

    lib = CrownLearner(a.library)
    status = lib.status()
    if a.evaluate:
        for cls in status:
            status[cls].update(lib.evaluate(cls))
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


def cmd_webview(a) -> int:
    from .webview import classify, load_webview

    objects = load_webview(a.file)
    if a.export:
        a.export.mkdir(parents=True, exist_ok=True)
    for i, o in enumerate(objects):
        c = classify(o)
        teeth = "-".join(map(str, c["teeth"])) or "-"
        label = f"{i:02d}_{c['kind']}_{c['jaw'] or 'na'}_{teeth}"
        name = f"  {o.name}" if a.names else ""  # names may contain the patient's name
        print(f"{label:34} {len(o.mesh.vertices):8d} vertices{name}")
        if a.export:
            save_stl(o.mesh, a.export / f"{label}.stl")
    return 0


def cmd_design_webview(a) -> int:
    from .webview import design_from_webview

    res, case, na = design_from_webview(a.file, a.tooth, learner=_learner(a), params=_params(a),
                                        use_neighbors=not a.no_neighbors, occlusion=_occlusion(a),
                                        fix_bite_first=a.fix_bite)
    save_stl(res.crown, a.out)
    if a.report:
        Path(a.report).write_text(json.dumps(res.report, indent=2, ensure_ascii=False))
    if a.preview:
        _preview(res, case.prep, a.preview, case.antagonist, f"Tooth {a.tooth}")
    print(json.dumps(res.report, indent=2, ensure_ascii=False))
    return 0


def cmd_learn_webview(a) -> int:
    from .learning import CrownLearner
    from .webview import learn_from_webview

    learner = CrownLearner(a.library)
    total = 0
    for f in a.files:
        for info in learn_from_webview(f, learner, teeth=a.tooth):
            print(json.dumps(info, ensure_ascii=False))
            total += 1
    print(f"learned {total} tooth/teeth from {len(a.files)} file(s)")
    return 0


def cmd_fix_bite(a) -> int:
    from .occlusion import fix_bite

    fixed, info = fix_bite(load_stl(a.jaw), load_stl(a.antagonist), a.axis, contact=a.contact)
    save_stl(fixed, a.out)
    print(json.dumps(info, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crownai", description="Automatic dental crown design")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("design", help="design a crown on a preparation scan")
    p.add_argument("--prep", required=True, type=Path, help="segmented die / preparation STL")
    p.add_argument("--out", required=True, type=Path, help="output crown STL")
    p.add_argument("--tooth", type=int, help="FDI tooth number, e.g. 36")
    p.add_argument("--margin", type=Path, help="margin line points (x y z per line); auto-detected if omitted")
    p.add_argument("--antagonist", type=Path, help="opposing jaw STL for occlusal clearance")
    p.add_argument("--report", type=Path, help="write a JSON QA report")
    p.add_argument("--preview", type=Path, help="write a PNG preview")
    _add_design_options(p)
    p.set_defaults(func=cmd_design)

    p = sub.add_parser("demo", help="run the full pipeline on synthetic scans")
    p.add_argument("--out-dir", default="out/demo")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("train-ssm", help="train a statistical crown shape model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--library", type=Path, help="folder of aligned crown STLs (occlusal +z, mesiodistal +x)")
    src.add_argument("--synthetic", type=int, metavar="N", help="train on N synthetic variants (demo)")
    p.add_argument("--tooth", type=int, default=36)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("learn", help="add an approved crown to the learning library")
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--crown", required=True, type=Path, help="final crown STL")
    p.add_argument("--prep", type=Path, help="preparation / die STL")
    p.add_argument("--margin", type=Path, help="margin line points")
    p.add_argument("--tooth", type=int, required=True)
    p.add_argument("--case-id")
    p.add_argument("--axis", type=_vec, default=(0.0, 0.0, 1.0))
    p.add_argument("--md-dir", type=_vec, default=(1.0, 0.0, 0.0))
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("learn-case", help="learn final crowns exported into processed exocad case folders")
    p.add_argument("folders", type=Path, nargs="+")
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--axis", type=_vec, default=(0.0, 0.0, 1.0))
    p.add_argument("--md-dir", type=_vec, default=(1.0, 0.0, 0.0))
    p.set_defaults(func=cmd_learn_case)

    p = sub.add_parser("library", help="show what the learning library contains")
    p.add_argument("library", type=Path)
    p.add_argument("--evaluate", action="store_true", help="cross-validate the learned models")
    p.set_defaults(func=cmd_library)

    p = sub.add_parser("exocad-case", help="design crowns for an exported exocad case folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--tooth", type=int, action="append", help="FDI number(s); read from the case XML if omitted")
    p.add_argument("--prep", type=Path, help="override preparation scan")
    p.add_argument("--antagonist", type=Path, help="override antagonist scan")
    p.add_argument("--margin", type=Path, help="override margin file")
    p.add_argument("--no-preview", action="store_true")
    _add_design_options(p)
    p.set_defaults(func=cmd_exocad_case)

    p = sub.add_parser("exocad-watch", help="watch a folder of exocad case exports")
    p.add_argument("root", type=Path)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--once", action="store_true", help="process pending cases and exit")
    _add_design_options(p)
    p.set_defaults(func=cmd_exocad_watch)

    p = sub.add_parser("webview", help="list (and export) the objects of an exocad webview HTML")
    p.add_argument("file", type=Path)
    p.add_argument("--export", type=Path, help="write every object as STL into this folder")
    p.add_argument("--names", action="store_true", help="show exocad object names (may contain patient data)")
    p.set_defaults(func=cmd_webview)

    p = sub.add_parser("design-webview", help="design a crown from an exocad webview HTML")
    p.add_argument("file", type=Path)
    p.add_argument("--tooth", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p.add_argument("--preview", type=Path)
    p.add_argument("--no-neighbors", action="store_true", help="ignore adjacent/contralateral teeth")
    p.add_argument("--fix-bite", action="store_true", help="correct the jaw relation before designing")
    _add_design_options(p)
    p.set_defaults(func=cmd_design_webview)

    p = sub.add_parser("learn-webview", help="learn the technician's wax-ups from exocad webview HTML files")
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--tooth", type=int, action="append")
    p.set_defaults(func=cmd_learn_webview)

    p = sub.add_parser("fix-bite", help="move the antagonist scan so the jaws meet without penetrating")
    p.add_argument("--jaw", required=True, type=Path)
    p.add_argument("--antagonist", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="corrected antagonist STL")
    p.add_argument("--axis", type=_vec, default=(0.0, 0.0, 1.0), help="direction from jaw to antagonist")
    p.add_argument("--contact", type=float, default=0.0, help="gap at the closest contact, mm")
    p.set_defaults(func=cmd_fix_bite)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
