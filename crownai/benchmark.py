"""Archive benchmark: does the designer work on every case, and how close is it to the technician?

``crownai benchmark ROOT --out report`` designs a sample of the lab's finished
crowns (exocad ``.constructionInfo`` cases) with each requested anatomy source
and records, per crown:

* whether a crown came out, and if not, *why* - grouped into a few causes
  (the die, the margin, the neighbour analysis, the surface checks, files);
* the distance to the technician's crown (both ways, mean and 90th percentile);
* contacts to the neighbours, room to the antagonist, wall thickness, time.

The report folder holds ``summary.json`` (per variant: success rate, causes,
distances, per tooth type), ``results.jsonl`` (one row per crown and variant)
and small pictures of each crown next to the technician's (occlusal relief and
two sections; crowns only - no scans).  Case ids are hashes; error texts have
paths removed - the folder can be shared to discuss the results.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

CAUSES = (
    ("die", ("intaglio rays", "does not rise above the margin", "does not surround the preparation", "die")),
    ("margin", ("margin",)),
    ("neighbours", ("neighbour", "contralateral", "arch")),
    ("surface", ("spike", "collapse", "self-intersect", "wall", "watertight")),
    ("files", ("no such file", "errno", "not found", "cannot read", "scan file")),
)


def failure_cause(message: str) -> str:
    m = message.lower()
    for cause, keys in CAUSES:
        if any(k in m for k in keys):
            return cause
    return "other"


def _variants(library, rules_profile, names):
    from .design import CrownParameters

    out = {}
    for name in names:
        if name == "rules":
            out[name] = lambda cid: CrownParameters()
        elif name == "profile" and rules_profile:
            out[name] = lambda cid: CrownParameters(rules_profile=rules_profile)
        elif name == "library" and library is not None:
            out[name] = lambda cid: CrownParameters(crown_library=library, library_exclude=(cid,))
    return out


def _render(path: Path, results: dict, ref, frame, die=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LightSource

    from .mesh import raycast
    from .viz import slice_mesh

    meshes = [(k, r.crown) for k, r in results.items()] + [("technician", ref)]
    g = np.arange(-6.5, 6.5, 0.1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    xy = np.column_stack([X.ravel(), Y.ravel()])
    ls = LightSource(315, 40)
    n = len(meshes)
    fig, axs = plt.subplots(2, max(n, 2), figsize=(3.4 * max(n, 2), 6.6))
    for i, (name, m) in enumerate(meshes):
        o = frame.to_world(np.column_stack([xy, np.full(len(xy), 25.0)]))
        t = raycast(o, np.broadcast_to(-frame.z, o.shape), m)
        Z = (25.0 - t).reshape(X.shape)
        if np.isfinite(Z).any():
            Zf = np.where(np.isfinite(Z), Z, np.nanmin(Z[np.isfinite(Z)]))
            img = np.where(np.isfinite(Z), ls.hillshade(Zf, vert_exag=2, dx=0.1, dy=0.1), 1.0)
            axs[0, i].imshow(img.T, origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap="gray", vmin=0, vmax=1)
        axs[0, i].set_title(name, fontsize=10)
        axs[0, i].set_xticks([]), axs[0, i].set_yticks([])
    colours = ["black", "tab:green", "tab:purple", "tab:orange"]
    for k, (normal, ix, title) in enumerate(((frame.x, 1, "buccolingual"), (frame.y, 0, "mesiodistal"))):
        ax = axs[1, k]
        items = ([("die", die, "tab:red", 1.0)] if die is not None else []) + \
            [(name, m, colours[i % 4] if name != "technician" else "tab:orange", 2.0 if name == "technician" else 1.4)
             for i, (name, m) in enumerate(meshes)]
        for name, m, col, lw in items:
            s = slice_mesh(m, frame.origin, normal)
            if len(s):
                s = frame.to_local(s.reshape(-1, 3)).reshape(-1, 2, 3)[:, :, [ix, 2]]
                ax.add_collection(LineCollection(s, colors=col, linewidths=lw, label=name))
        ax.set_xlim(-7, 7), ax.set_ylim(-3, 11), ax.set_aspect("equal"), ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10), ax.legend(fontsize=7)
    for j in range(2, axs.shape[1]):
        axs[1, j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=55)
    plt.close(fig)


def run_benchmark(root, out, *, n: int = 40, variants=("rules", "library"), library=None,
                  rules_profile: dict | None = None, pictures: int = 20, seed: int = 2, slavicek: bool = True,
                  log=print) -> dict:
    from .crown_library import CrownLibrary
    from .occlusion import SlavicekConcept
    from .exocad import crop_to_margin
    from .exocad_project import design_construction_case
    from .mesh import load_mesh, nearest_distance
    from .metrics import compare_to_reference
    from .rules_tuning import _case_id, _safe_error, archive_crowns, isolate_crown

    out = Path(out)
    (out / "pictures").mkdir(parents=True, exist_ok=True)
    if library is not None and not isinstance(library, CrownLibrary):
        library = CrownLibrary(library)
    makers = _variants(library, rules_profile, variants)
    occlusion = SlavicekConcept() if slavicek else None
    items = list(archive_crowns(root))
    random.Random(seed).shuffle(items)
    items = items[:n]
    log(f"benchmark: {len(items)} crowns x {list(makers)}")
    rows = []
    with (out / "results.jsonl").open("w") as fh:
        for i, (folder, case, tooth, ref_path) in enumerate(items):
            cid = _case_id(folder, tooth)
            info = case.teeth[tooth]
            try:
                ref = isolate_crown(crop_to_margin(load_mesh(ref_path), info.margin, info.axis, radial_pad=2.5,
                                                   depth=1.0), info.margin, info.axis, info.md_direction, tooth)
            except Exception as exc:
                ref = None
                log(f"  {cid}: technician's crown unreadable ({_safe_error(exc)})")
            done, frame = {}, None
            for name, make in makers.items():
                row = {"case_id": cid, "tooth": int(tooth), "variant": name}
                t0 = time.time()
                try:
                    res, _, na = design_construction_case(folder, tooth, params=make(cid), case=case,
                                                          compare_reference=False, occlusion=occlusion)
                    row.update(ok=True, seconds=round(time.time() - t0, 1),
                               source=res.report.get("anatomy_source"),
                               warnings=[_safe_error(Exception(w)) for w in res.report.get("warnings", [])],
                               min_axial_mm=res.report.get("min_axial_thickness_mm"),
                               min_occlusal_mm=res.report.get("min_occlusal_thickness_mm"))
                    occ = res.report.get("occlusion") or {}
                    row["interference_mm"] = occ.get("remaining_interference_mm")
                    if na is not None and na.neighbors:
                        row["contact_gaps_mm"] = [round(float(nearest_distance(res.outer.reshape(-1, 3),
                                                                               nb.mesh.vertices)[0].min()), 3)
                                                  for nb in na.neighbors]
                    if ref is not None:
                        top = res.frame.to_local(res.margin)[:, 2].max()
                        cmp = compare_to_reference(res.outer[:, 4:].reshape(-1, 3), res.crown, ref, res.frame, top)
                        row.update(tech_to_crown_mm=cmp["reference_to_crown_mean_mm"],
                                   tech_to_crown_p90_mm=cmp["reference_to_crown_p90_mm"],
                                   crown_to_tech_mm=cmp["crown_to_reference_mean_mm"])
                    lib_crown = (res.report.get("anatomy_fit") or {}).get("library_crown")
                    if lib_crown:
                        row["library_crown"] = lib_crown
                    done[name] = res
                    frame = res.frame
                except Exception as exc:
                    msg = _safe_error(exc)
                    row.update(ok=False, seconds=round(time.time() - t0, 1), cause=failure_cause(msg), error=msg)
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            if done and ref is not None and i < pictures:
                try:
                    _render(out / "pictures" / f"{cid.replace('/', '_')}.png", done, ref, frame)
                except Exception as exc:
                    log(f"  picture failed: {_safe_error(exc)}")
            log(f"  [{i + 1}/{len(items)}] {tooth}: " + ", ".join(
                f"{r['variant']} " + (f"{r.get('tech_to_crown_mm', '-')} mm" if r["ok"] else f"FAILED ({r['cause']})")
                for r in rows if r["case_id"] == cid))
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary.items():
        log(f"{name}: {s['succeeded']}/{s['crowns']} designed, failures {s['failures']}, "
            f"technician-to-crown {s.get('tech_to_crown_mean_mm')} mm (median {s.get('tech_to_crown_median_mm')})")
    return summary


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == name]
        ok = [r for r in rs if r["ok"]]
        d = [r["tech_to_crown_mm"] for r in ok if r.get("tech_to_crown_mm") is not None]
        causes = {}
        for r in rs:
            if not r["ok"]:
                causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        by_type = {}
        for r in ok:
            if r.get("tech_to_crown_mm") is None:
                continue
            arch = "upper" if r["tooth"] // 10 in (1, 2) else "lower"
            by_type.setdefault(f"{arch}_{min(r['tooth'] % 10, 7)}", []).append(r["tech_to_crown_mm"])
        gaps = [g for r in ok for g in r.get("contact_gaps_mm", [])]
        out[name] = {
            "crowns": len(rs), "succeeded": len(ok), "failures": causes,
            "tech_to_crown_mean_mm": round(float(np.mean(d)), 3) if d else None,
            "tech_to_crown_median_mm": round(float(np.median(d)), 3) if d else None,
            "by_tooth_type_mm": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_type.items())},
            "contact_gap_median_mm": round(float(np.median(gaps)), 3) if gaps else None,
            "seconds_median": round(float(np.median([r["seconds"] for r in rs])), 1) if rs else None,
        }
    return out
