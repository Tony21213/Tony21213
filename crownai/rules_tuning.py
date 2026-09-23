"""Tune the posterior anatomy rules to a lab's own finished crowns.

The rule model (:mod:`crownai.posterior_model`) gives every premolar and molar
a correct but textbook-generic form.  What makes the lab's crowns look alive
is how *its* technicians shape them: cusp heights and positions, how steep
the inclines are, how deep the fissures run, where the heights of contour
sit - and the secondary anatomy (supplemental grooves, ridge shapes) the
rules do not describe.

For every finished crown in the archive (exocad ``.constructionInfo`` +
the crown exported back into the case folder) this module

1. places the rule model on the case's margin exactly as the designer
   would (same frame, same cervical reference);
2. fits the rule parameters to the technician's occlusal surface
   (cusp heights and positions, inclines, triangular ridges, fossa depth,
   marginal ridges) and measures the outline directly (heights of contour,
   contacts, proportions, crown inclination, squareness);
3. keeps the residual - technician surface minus fitted rule surface - on a
   normalised occlusal grid: the secondary anatomy.

:func:`build_profile` reduces the fits to one robust (median) profile per
tooth type; :func:`crownai.posterior_model.default_posterior` applies it:
tuned rules plus the typical secondary anatomy layered on the occlusal
surface.  Nothing patient-identifying is stored: case ids are hashes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from .anatomy_model import PlacedAnatomy
from .margin import make_frame
from .mesh import Mesh, raycast
from .metrics import surface_samples
from .posterior_model import PROFILE_FIELDS, _textbook_posterior, profile_key, scaled

DETAIL_N = 33  # detail grid resolution over the occlusal table


def buccal_from_fdi(fdi: int, axis, mesial) -> np.ndarray:
    """Buccal direction from the tooth number, the occlusal axis and the mesial direction.

    With the axis pointing occlusally (as exocad's insertion axis does) the
    buccal side is +axis x mesial in quadrants 1 and 3 and the opposite in
    quadrants 2 and 4 - any rigid placement of the scans keeps this.
    """
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    m = np.asarray(mesial, float)
    m = m - (m @ z) * z
    b = np.cross(z, m / np.linalg.norm(m))
    return b if (fdi // 10) % 2 == 1 else -b


def _isolate_tooth(q: np.ndarray, md_expected: float, step: float = 0.25, return_mask: bool = False):
    """Drop neighbouring teeth that came along in the crown file (splinted wax-ups, bridges).

    Between two teeth the crown narrows to the contact: along the mesiodistal
    axis the buccolingual width has a waist there.  Look for it on each side
    between 0.3 and 1.0 of the expected width from the tooth axis and cut where the
    width grows again beyond it (the next tooth; a bridge connector is a thick waist).
    """
    z = q[:, 2]
    upper = q[z > 0.35 * np.percentile(z, 99)]
    keep = np.ones(len(q), bool)
    for sign in (1.0, -1.0):
        a = sign * upper[:, 0]
        edges = np.arange(0.3 * md_expected, 1.0 * md_expected + step, step)
        widths = []
        for e in edges:
            sl = upper[np.abs(a - e) < step / 2]
            widths.append(np.percentile(sl[:, 1], 97) - np.percentile(sl[:, 1], 3) if len(sl) >= 8 else 0.0)
        widths = np.array(widths)
        centre = upper[np.abs(upper[:, 0]) < 0.15 * md_expected]
        if len(centre) < 8 or not (widths > 0).any():
            continue
        w0 = np.percentile(centre[:, 1], 97) - np.percentile(centre[:, 1], 3)
        i = int(np.argmin(np.where(widths > 0, widths, np.inf)))
        # a waist that widens again beyond it = the next tooth
        beyond = widths[i + 1:]
        if widths[i] < 0.8 * w0 and len(beyond) and beyond.max() > widths[i] + 1.0:
            keep &= sign * q[:, 0] < edges[i]
    return keep if return_mask else q[keep]


def isolate_crown(crown: Mesh, margin: np.ndarray, axis, md_direction, tooth: int) -> Mesh:
    """The part of a crown file that belongs to ``tooth`` (neighbours of a wax-up cut off)."""
    base = _textbook_posterior(tooth)
    frame = make_frame(np.asarray(margin).mean(axis=0), axis, md_direction)
    q = frame.to_local(crown.vertices)
    kept = _isolate_tooth(np.column_stack([q[:, 0], q[:, 1], q[:, 2] - frame.to_local(margin)[:, 2].mean()]),
                          base.md, return_mask=True)
    f = crown.faces[kept[crown.faces].all(axis=1)]
    used, inv = np.unique(f, return_inverse=True)
    return Mesh(crown.vertices[used], inv.reshape(-1, 3))


def _outline_measures(q: np.ndarray, H: float) -> dict:
    """Heights of contour, contacts, proportions, inclination and squareness of a crown (model coords)."""
    u, w, z = q[:, 0], q[:, 1], q[:, 2]
    band = (z > 0.2 * H) & (z < 0.9 * H)
    ub, wb = u[band], w[band]
    md = float(np.percentile(ub, 99.5) - np.percentile(ub, 0.5))
    bl = float(np.percentile(wb, 99.5) - np.percentile(wb, 0.5))
    su = 0.5 * float(np.percentile(ub, 99.5) + np.percentile(ub, 0.5))
    sw = 0.5 * float(np.percentile(wb, 99.5) + np.percentile(wb, 0.5))

    def z_at(sel):
        return float(np.median(z[sel])) / H if sel.sum() >= 5 else np.nan

    out = {
        "md": md, "bl": bl, "su": su, "sw": sw,
        "contact_m": z_at(band & (u > np.percentile(ub, 99) - 0.3)),
        "contact_d": z_at(band & (u < np.percentile(ub, 1) + 0.3)),
        "hc_b": z_at(band & (w > np.percentile(wb, 99) - 0.3)),
        "hc_l": z_at(band & (w < np.percentile(wb, 1) + 0.3)),
        "buccal_share": float(np.percentile(wb, 99.5) / max(np.percentile(wb, 99.5) - np.percentile(wb, 0.5), 1e-6)),
    }
    lo, hi = (z > 0.15 * H) & (z < 0.35 * H), (z > 0.6 * H) & (z < 0.8 * H)
    if lo.sum() > 20 and hi.sum() > 20:
        c_lo = 0.5 * (np.percentile(w[lo], 99) + np.percentile(w[lo], 1))
        c_hi = 0.5 * (np.percentile(w[hi], 99) + np.percentile(w[hi], 1))
        out["tilt"] = float((c_hi - c_lo) / max(np.median(z[hi]) - np.median(z[lo]), 1e-3))
    # squareness: superellipse exponent of the section at the height of contour
    zc = np.nanmean([out["hc_b"], out["hc_l"]]) * H if np.isfinite([out["hc_b"], out["hc_l"]]).any() else 0.4 * H
    sec = np.abs(z - zc) < 0.5
    if sec.sum() > 60:
        un, wn = (u[sec] - su) / (md / 2), (w[sec] - sw) / (bl / 2)
        ang = np.arctan2(wn, un)
        rad = np.hypot(un, wn)
        bins = np.digitize(ang, np.linspace(-np.pi, np.pi, 49))
        outer = np.array([np.argmax(np.where(bins == b, rad, -1)) for b in np.unique(bins)])
        un, wn = un[outer], wn[outer]
        es = np.linspace(1.8, 5.0, 33)
        err = [np.mean(((np.abs(un) ** e + np.abs(wn) ** e) ** (1 / e) - 1) ** 2) for e in es]
        out["squareness"] = float(es[int(np.argmin(err))])
    return out


def fit_rules_to_crown(crown: Mesh, margin: np.ndarray, axis, md_direction, tooth: int,
                       buccal=None, detail_n: int = DETAIL_N) -> dict:
    """Rule parameters (and secondary anatomy) of one finished crown.

    ``crown``: the technician's crown, cropped to this tooth; ``margin``,
    ``axis`` (occlusal), ``md_direction`` (mesial) as recorded by exocad.
    """
    from scipy.optimize import least_squares

    base = _textbook_posterior(tooth)
    frame = make_frame(np.asarray(margin).mean(axis=0), axis, md_direction)
    m_loc = frame.to_local(margin)
    if buccal is None:
        buccal = buccal_from_fdi(tooth, axis, md_direction)
    labial = 1.0 if frame.to_local(frame.origin + np.asarray(buccal))[1] >= 0 else -1.0
    placed = PlacedAnatomy.on_margin(base, labial, np.zeros(2), m_loc, fade=0.55 * base.height)

    crown = isolate_crown(crown, margin, axis, md_direction, tooth)
    q = placed.to_model(frame.to_local(surface_samples(crown, 4)))
    q = _isolate_tooth(q[q[:, 2] > 0.3], base.md)
    if len(q) < 300:
        raise ValueError("too little crown surface above the margin")
    H = float(np.percentile(q[:, 2], 99.8))
    meas = _outline_measures(q, H)
    md, bl = meas["md"], meas["bl"]
    tilt = meas.get("tilt", base.tilt)

    # occlusal surface on a grid normalised to the occlusal table, in the model's own
    # coordinates (origin on the tooth axis, like the cusp positions)
    g = np.linspace(-1, 1, detail_n)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    U, W_ray = GX * md / 2, GY * bl / 2 + tilt * 0.8 * H
    top = frame.to_local(crown.vertices)[:, 2].max() + 5.0
    xy_loc = np.column_stack([U.ravel(), labial * W_ray.ravel()])
    t = raycast(frame.to_world(np.column_stack([xy_loc, np.full(len(xy_loc), top)])),
                np.broadcast_to(-frame.z, (len(xy_loc), 3)), crown)
    hit = np.isfinite(t)
    Zs = np.full(len(xy_loc), np.nan)
    if hit.any():
        Zs[hit] = placed.to_model(np.column_stack([xy_loc[hit], top - t[hit]]))[:, 2]
    Z = Zs.reshape(GX.shape)
    W = W_ray - tilt * np.nan_to_num(Z)  # upright, as PosteriorCrown.occlusal expects
    table = np.maximum(np.abs(GX), np.abs(GY)) <= 0.85
    fit_pts = np.isfinite(Z) & table
    if fit_pts.sum() < 60:
        raise ValueError("occlusal surface not visible from above")
    uu, ww, zz = U[fit_pts], W[fit_pts], Z[fit_pts]

    n = len(base.cusps)
    names = ("slope_in", "cusp_ridge_slope", "ridge_ratio", "fossa_depth", "marginal_drop", "mesial_ridge_extra")
    x0 = [H] + [getattr(base, k) for k in names] + [c.dh for c in base.cusps] + [0.0] * (2 * n)
    lo = [0.5 * H] + [0.35, 0.2, 1.0, 1.2, 0.3, -0.5] + [-3.0] * n + [-0.2] * (2 * n)
    hi = [1.5 * H + 1] + [1.4, 1.2, 3.0, 5.0, 2.5, 1.0] + [0.5] * n + [0.2] * (2 * n)
    sigma = [np.inf] + [0.2, 0.2, 0.5, 0.8, 0.5, 0.3] + [1.0] * n + [0.1] * (2 * n)
    shape0 = scaled(base, md=md, bl=bl)

    def build(x):
        kw = dict(zip(names, x[1:1 + len(names)]))
        dh = x[1 + len(names):1 + len(names) + n]
        du = x[1 + len(names) + n:1 + len(names) + 2 * n]
        dw = x[1 + len(names) + 2 * n:]
        cusps = tuple(replace(c, u=c.u + a, w=c.w + b, dh=d) for c, a, b, d in zip(base.cusps, du, dw, dh))
        return replace(shape0, height=x[0], cusps=cusps, **kw)

    x0 = np.clip(np.array(x0, float), np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    lam = 0.05 * np.sqrt(len(zz))

    def residual(x):
        m = build(x)
        occ = m.occlusal(uu, ww)
        # where the model's side wall cuts the occlusal surface the crown top is the
        # outline, not the occlusal function: those points say nothing about the rules
        on_table = m.inside(np.column_stack([uu, ww + m.tilt * (occ - 0.01), occ - 0.01]))
        prior = [lam * (x[i] - x0[i]) / s for i, s in enumerate(sigma) if np.isfinite(s)]
        return np.concatenate([np.where(on_table, occ - zz, 0.0), prior])

    shape0 = replace(shape0, tilt=tilt)
    r0 = residual(x0)[:len(zz)]
    r0 = np.abs(r0[r0 != 0])
    sol = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.3, diff_step=0.01, max_nfev=150)
    model = build(sol.x)
    r1 = residual(sol.x)[:len(zz)]
    r1 = np.abs(r1[r1 != 0])
    # normalise: the highest cusp at dh = 0
    top_dh = max(c.dh for c in model.cusps)
    model = replace(model, height=model.height + top_dh,
                    cusps=tuple(replace(c, dh=c.dh - top_dh) for c in model.cusps))
    occ = model.occlusal(U, W)
    on_table = model.inside(np.stack([U, W + tilt * (occ - 0.01), occ - 0.01], axis=-1))
    detail = np.where(fit_pts & on_table, np.clip(np.nan_to_num(Z) - occ, -1.0, 1.0), 0.0)

    fit = {"key": profile_key(tooth), "tooth": int(tooth),
           "md": md, "bl": bl, "bl_md_ratio": bl / md, "height": float(model.height),
           **{k: float(getattr(model, k)) for k in names},
           **{k: meas[k] for k in ("contact_m", "contact_d", "hc_b", "hc_l", "buccal_share", "squareness")
              if k in meas and np.isfinite(meas[k])},
           "tilt": float(tilt),
           "cusps": [{"u": float(c.u), "w": float(c.w), "dh": float(c.dh)} for c in model.cusps],
           "detail": np.round(detail, 3).tolist(),
           "rms_before_mm": float(np.sqrt(np.mean(r0 ** 2))), "rms_after_mm": float(np.sqrt(np.mean(r1 ** 2))),
           "points": int(len(zz))}
    return fit


def build_profile(fits: list[dict], min_cases: int = 5) -> dict:
    """One robust (median) profile per tooth type from many fitted crowns."""
    groups: dict[str, list[dict]] = {}
    for f in fits:
        if "error" not in f:
            groups.setdefault(f["key"], []).append(f)
    profile = {}
    for key, fs in sorted(groups.items()):
        if len(fs) < min_cases:
            continue
        entry = {"n_cases": len(fs)}
        for k in ("md", "bl_md_ratio", "height") + PROFILE_FIELDS:
            vals = [f[k] for f in fs if k in f and f[k] is not None and np.isfinite(f[k])]
            if vals:
                entry[k] = round(float(np.median(vals)), 4)
        counts = {len(f["cusps"]) for f in fs}
        if len(counts) == 1:
            entry["cusps"] = [{k: round(float(np.median([f["cusps"][i][k] for f in fs])), 4)
                               for k in ("u", "w", "dh")} for i in range(counts.pop())]
        details = np.array([f["detail"] for f in fs if f.get("detail") is not None])
        if len(details):
            entry["detail"] = np.round(np.median(details, axis=0), 3).tolist()
        entry["fit_rms_mm"] = round(float(np.median([f["rms_after_mm"] for f in fs])), 3)
        profile[key] = entry
    return profile


def load_profile(path) -> dict:
    data = json.loads(Path(path).read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _case_id(folder: Path, tooth: int) -> str:
    return hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:12] + f"/{tooth}"


def archive_crowns(root):
    """(folder, case, tooth, crown file) for every finished premolar/molar crown under ``root``."""
    from .exocad_project import find_final_crown, parse_construction_info

    for ci in sorted(Path(root).rglob("*.constructionInfo")):
        try:
            case = parse_construction_info(ci)
        except Exception:
            continue
        for tooth in case.crown_teeth():
            if tooth % 10 < 4:
                continue
            ref = find_final_crown(ci.parent, tooth)
            if ref is not None:
                yield ci.parent, case, tooth, ref


def tune_rules(root, out, *, fits_path=None, limit: int | None = None, holdout: float = 0.15,
               eval_limit: int = 8, min_cases: int = 5, seed: int = 0, log=print) -> dict:
    """Fit the rules to every finished posterior crown under ``root`` and write a profile.

    ``fits_path`` (JSON lines) keeps each fitted crown so an interrupted run
    resumes where it stopped.  A share ``holdout`` of the cases is left out
    of the profile and used to check it: those crowns are designed with and
    without the profile and compared with the technician's crowns.
    """
    from .exocad import crop_to_margin
    from .mesh import load_mesh

    fits_path = Path(fits_path) if fits_path else Path(out).with_suffix(".fits.jsonl")
    done = {}
    if fits_path.exists():
        for line in fits_path.read_text().splitlines():
            if line.strip():
                f = json.loads(line)
                done[f["case_id"]] = f
    rng = random.Random(seed)
    items = list(archive_crowns(root))
    log(f"{len(items)} finished premolar/molar crowns found ({len(done)} already fitted)")
    if limit:
        rng.shuffle(items)
        items = items[:limit]
    new = 0
    with fits_path.open("a") as fh:
        for i, (folder, case, tooth, ref) in enumerate(items):
            cid = _case_id(folder, tooth)
            if cid in done:
                continue
            info = case.teeth[tooth]
            try:
                crown = crop_to_margin(load_mesh(ref), info.margin, info.axis, radial_pad=2.5, depth=1.0)
                f = fit_rules_to_crown(crown, info.margin, info.axis, info.md_direction, tooth)
            except Exception as exc:
                f = {"key": profile_key(tooth), "tooth": int(tooth), "error": str(exc)[:200]}
            h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 1000
            f.update(case_id=cid, holdout=h < holdout * 1000)
            done[cid] = f
            fh.write(json.dumps(f) + "\n")
            fh.flush()
            new += 1
            if new % 10 == 0:
                log(f"  fitted {new} ({i + 1}/{len(items)})")
    fits = [f for f in done.values() if "error" not in f]
    errors = sum(1 for f in done.values() if "error" in f)
    train = [f for f in fits if not f.get("holdout")]
    profile = build_profile(train, min_cases=min_cases)
    summary = {k: {"n_cases": v["n_cases"], "fit_rms_mm": v["fit_rms_mm"]} for k, v in profile.items()}
    log(f"profile from {len(train)} crowns ({errors} could not be fitted): {summary}")

    evaluation = []
    if eval_limit:
        evaluation = evaluate_profile(root, profile, [f["case_id"] for f in fits if f.get("holdout")],
                                      limit=eval_limit, log=log)
    Path(out).write_text(json.dumps({**profile, "_summary": summary, "_evaluation": evaluation}, indent=1))
    return profile


def evaluate_profile(root, profile: dict, case_ids: list[str], limit: int = 8, log=print) -> list[dict]:
    """Design held-out crowns with and without the profile; distance to the technician's crowns."""
    from .design import CrownParameters
    from .exocad import crop_to_margin
    from .exocad_project import design_construction_case
    from .mesh import load_mesh
    from .metrics import compare_to_reference

    wanted = set(case_ids)
    rows = []
    for folder, case, tooth, ref_path in archive_crowns(root):
        if len(rows) >= limit:
            break
        if _case_id(folder, tooth) not in wanted:
            continue
        row = {"case_id": _case_id(folder, tooth), "tooth": tooth}
        try:
            info = case.teeth[tooth]
            ref = isolate_crown(crop_to_margin(load_mesh(ref_path), info.margin, info.axis, radial_pad=2.5, depth=1.0),
                                info.margin, info.axis, info.md_direction, tooth)
            for name, params in (("textbook", CrownParameters()), ("profile", CrownParameters(rules_profile=profile))):
                res, _, _ = design_construction_case(folder, tooth, params=params, case=case, compare_reference=False)
                top = res.frame.to_local(res.margin)[:, 2].max()
                cmp = compare_to_reference(res.outer[:, 4:].reshape(-1, 3), res.crown, ref, res.frame, top)
                row[name] = cmp["reference_to_crown_mean_mm"]
        except Exception as exc:
            row["error"] = str(exc)[:200]
        rows.append(row)
        log(f"  check {row}")
    ok = [r for r in rows if r.get("textbook") is not None and r.get("profile") is not None]
    if ok:
        log(f"held-out crowns: technician-to-crown mean {np.mean([r['textbook'] for r in ok]):.3f} mm (textbook rules) "
            f"-> {np.mean([r['profile'] for r in ok]):.3f} mm (lab profile), {len(ok)} crowns")
    return rows
