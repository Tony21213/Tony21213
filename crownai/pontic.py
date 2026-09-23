"""Full-contour tooth for a site with no preparation: implant or pontic.

The crown pipeline in :mod:`crownai.design` needs a die to build on.  For a
missing tooth there is none, so this module puts a *virtual abutment* on the
ridge: an elliptical cervical outline of the tooth's textbook cervical size,
following the gum surface, with a tapered stump above it.  The normal crown
design then runs unchanged (neighbours, contralateral template, Slavicek
occlusion), and its outer surface is closed at the bottom with a slightly
convex (ovate) base resting on the gum instead of an intaglio.

The site is found from the scan when possible: an implant scan usually has a
round hole in the gum where the scan body / healing abutment was cut out.
"""

from __future__ import annotations

import numpy as np

from .anatomy_model import WHEELER
from .margin import make_frame
from .mesh import Mesh, raycast


def _boundary_loops(mesh: Mesh) -> list[np.ndarray]:
    """Vertex index sets of the open-boundary loops of ``mesh``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = mesh.faces
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    b = uniq[count == 1]
    if not len(b):
        return []
    n = len(mesh.vertices)
    _, lab = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n)),
                                  directed=False)
    loops = {}
    for v in np.unique(b):
        loops.setdefault(lab[v], []).append(v)
    return [np.array(v) for v in loops.values()]


def find_implant_holes(jaw: Mesh, axis=(0.0, 0.0, 1.0), min_d: float = 2.5, max_d: float = 8.0) -> list[dict]:
    """Round holes in the scan (cut-out scan bodies / healing abutments).

    Each hole: ``{"center": world point, "radius": mm, "rim": (n, 3) world rim points}``.
    """
    frame = make_frame(jaw.vertices.mean(axis=0), axis)
    out = []
    for loop in _boundary_loops(jaw):
        if len(loop) < 8:
            continue
        q = frame.to_local(jaw.vertices[loop])
        c = q[:, :2].mean(axis=0)
        r = np.linalg.norm(q[:, :2] - c, axis=1)
        d = 2 * r.mean()
        if min_d <= d <= max_d and r.std() < 0.25 * r.mean() and np.ptp(q[:, 2]) < 0.6 * d:
            out.append({"center": frame.to_world(np.array([*c, q[:, 2].mean()])), "radius": float(r.mean()),
                        "rim": jaw.vertices[loop]})
    return out


def _gum_height(jaw: Mesh, frame, xy: np.ndarray) -> np.ndarray:
    """Height (local z) of the jaw surface under local points ``xy``; NaN where the scan has no surface."""
    top = frame.to_local(jaw.vertices)[:, 2].max() + 5.0
    o = frame.to_world(np.column_stack([xy, np.full(len(xy), top)]))
    t = raycast(o, np.repeat(-frame.z[None], len(xy), axis=0), jaw)
    return np.where(np.isfinite(t), top - t, np.nan)


def virtual_site(jaw: Mesh, tooth: int, center, axis=(0.0, 0.0, 1.0), md_direction=(1.0, 0.0, 0.0), *,
                 hole: dict | None = None, n_points: int = 128, depth: float | None = None, stump: float = 2.5):
    """Margin ring and a virtual abutment inside it.

    Implant (``hole`` given): the ring follows the rim of the soft-tissue hole,
    ``depth`` (default 1 mm) below it, where the crown meets the ti-base.
    Pontic: an ellipse of the tooth's cervical size on the ridge, ``depth``
    (default 0.5 mm) into the gum (ovate).  Returns ``(margin (n,3), abutment Mesh)``.
    """
    frame = make_frame(np.asarray(center, float), axis, md_direction)
    th = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    if hole is not None:
        depth = 1.0 if depth is None else depth
        rim = frame.to_local(hole["rim"])
        c = rim[:, :2].mean(axis=0)
        ang = np.arctan2(rim[:, 1] - c[1], rim[:, 0] - c[0])
        o = np.argsort(ang)
        ang, rim = np.concatenate([ang[o] - 2 * np.pi, ang[o], ang[o] + 2 * np.pi]), np.tile(rim[o], (3, 1))
        r = np.interp(th, ang, np.linalg.norm(rim[:, :2] - c, axis=1)) * 0.9
        z = np.interp(th, ang, rim[:, 2])
        xy = c + np.column_stack([r * np.cos(th), r * np.sin(th)])
        md_c = bl_c = 2 * float(r.mean())
    else:
        depth = 0.5 if depth is None else depth
        arch = "upper" if tooth // 10 in (1, 2, 5, 6) else "lower"
        _, _, md_c, _, bl_c = WHEELER[arch][tooth % 10]
        xy = np.column_stack([md_c / 2 * np.cos(th), bl_c / 2 * np.sin(th)])
        z = _gum_height(jaw, frame, xy)
        if np.isnan(z).all():
            raise ValueError("no gum surface under the site")
        z = np.where(np.isnan(z), np.nanmedian(z), z)
    # smooth the ring height a little (scan noise, papillae)
    k = np.ones(9) / 9
    z = np.convolve(np.concatenate([z[-4:], z, z[:4]]), k, mode="valid") - depth
    margin = frame.to_world(np.column_stack([xy, z]))
    cxy = xy.mean(axis=0)
    xy = xy - cxy

    # abutment: rings shrinking inward (0.6 mm shoulder, 6 deg taper), dome on top
    levels = [(-3.0, 0.92), (-0.01, 1.0), (0.05, 0.97)]
    top_z = z.max() + stump
    for h in np.linspace(0.4, stump - 1.0, 6):
        levels.append((h, 1.0 - (min(0.6, 0.15 * md_c) + h * np.tan(np.radians(6))) / (min(md_c, bl_c) / 2)))
    s_top = levels[-1][1]
    for a in np.linspace(0.3, 1.0, 4)[:-1] * np.pi / 2:
        levels.append((stump - 1.0 + 1.0 * np.sin(a), s_top * np.cos(a)))
    rings = []
    for h, s in levels:
        zz = z + h if h <= 0.05 else np.maximum(z + h, z.max() + h - 1.5)  # flatten the stump top
        rings.append(np.column_stack([cxy + xy * s, zz]))
    rings = np.stack(rings)  # (levels, n, 3)
    bottom = np.array([*cxy, z.min() - 3.0])
    top = np.array([*cxy, top_z])
    n_l, n = rings.shape[:2]
    verts = np.concatenate([rings.reshape(-1, 3), bottom[None], top[None]])
    i = np.arange(n)
    faces = []
    for l in range(n_l - 1):
        a, b = l * n + i, l * n + (i + 1) % n
        c, d = (l + 1) * n + (i + 1) % n, (l + 1) * n + i
        faces += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    faces.append(np.stack([(i + 1) % n, i, np.full(n, n_l * n)], 1))
    last = (n_l - 1) * n
    faces.append(np.stack([last + i, last + (i + 1) % n, np.full(n, n_l * n + 1)], 1))
    die = Mesh(frame.to_world(verts), np.concatenate(faces))
    if die.volume() < 0:
        die = die.flipped()
    return margin, die


def _plane_normal(pts: np.ndarray) -> np.ndarray:
    """Unit normal (z > 0) of the least-squares plane z = a x + b y + c through local points."""
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    a, b, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]
    n = np.array([-a, -b, 1.0])
    return n / np.linalg.norm(n)


def _seat_on_antagonist(template: Mesh, antagonist: Mesh, axis, contact: float = 0.05,
                        max_tilt_deg: float = 25.0) -> Mesh:
    """Level the mirrored template to the antagonist and move it into contact.

    With no preparation limiting the height, the contralateral tooth (often
    tipped or at another level: curve of Spee, a tilted scan) is set into the
    patient's bite whole: its occlusal table is turned parallel to the
    antagonist's occlusal surface above the site, then moved along the axis
    until it just touches - instead of having its cusps cut off.
    """
    frame = make_frame(template.vertices.mean(axis=0), axis)
    q = frame.to_local(template.vertices)
    c = q[:, :2].mean(axis=0)
    g = np.linspace(-2.5, 2.5, 15)  # the occlusal table, clear of the cusp slopes' outer flanks
    X, Y = np.meshgrid(g + c[0], g + c[1], indexing="ij")
    xy = np.column_stack([X.ravel(), Y.ravel()])
    hi, lo = q[:, 2].max() + 20.0, q[:, 2].min() - 20.0
    z = frame.z
    t_top = raycast(frame.to_world(np.column_stack([xy, np.full(len(xy), hi)])),
                    np.broadcast_to(-z, (len(xy), 3)), template)
    t_ant = raycast(frame.to_world(np.column_stack([xy, np.full(len(xy), lo)])),
                    np.broadcast_to(z, (len(xy), 3)), antagonist)
    ok = np.isfinite(t_top) & np.isfinite(t_ant)
    if ok.sum() >= 12:
        n_t = _plane_normal(np.column_stack([xy[ok], hi - t_top[ok]]))
        n_a = _plane_normal(np.column_stack([xy[ok], lo + t_ant[ok]]))
        axis_r = np.cross(n_t, n_a)
        s_ang = np.linalg.norm(axis_r)
        ang = np.arctan2(s_ang, n_t @ n_a)
        if s_ang > 1e-9 and np.degrees(ang) <= max_tilt_deg:
            k = axis_r / s_ang
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            pivot = np.array([c[0], c[1], q[:, 2].max()])
            q = (q - pivot) @ R.T + pivot
    v = frame.to_world(q)
    below = raycast(v - 15.0 * z, np.broadcast_to(z, v.shape), antagonist)
    gap = below - 15.0  # >0: free space above the vertex, <0: inside the antagonist
    ok = np.isfinite(gap)
    if ok.sum() < 20:
        return Mesh(v, template.faces)
    shift = float(np.percentile(gap[ok], 0.5)) - contact
    return Mesh(v + shift * z, template.faces)


def close_with_base(result, dip: float = 0.5) -> Mesh:
    """Solid tooth: the crown's outer surface closed by a convex base under the margin."""
    from .design import _stitch

    frame, margin, outer = result.frame, result.margin, result.outer
    outer_pole = result.crown.vertices[-1]
    m = frame.to_local(margin)
    n_v = outer.shape[1]
    v = np.arange(n_v) / n_v
    s = (1 - v)[None, :, None]
    base = m[:, None, :] * s
    w = s[..., 0] ** 2  # rim height at the margin, one common height at the centre
    base[..., 2] = w * m[:, None, 2] + (1 - w) * (m[:, 2].mean() - dip)
    pole = np.array([0.0, 0.0, m[:, 2].mean() - dip])
    return _stitch(margin, frame.to_world(base.reshape(-1, 3)).reshape(base.shape),
                   frame.to_world(pole), outer, outer_pole)


def design_missing_tooth(jaw: Mesh, tooth: int, *, antagonist: Mesh | None = None, center=None,
                         axis=(0.0, 0.0, 1.0), occlusion=None, learner=None, params=None,
                         use_neighbors: bool = True):
    """Design a full-contour tooth (implant crown / pontic) where ``tooth`` is missing.

    ``center``: a point at the site; if omitted, the only round implant hole in
    the scan is used.  Returns ``(CrownResult, solid Mesh, NeighborAnalysis | None)``;
    ``solid`` is the closed tooth (outer anatomy + ovate base).
    """
    from .arch import analyze_neighbors
    from .design import design_crown

    from dataclasses import replace

    from .design import CrownParameters

    hole = None
    holes = find_implant_holes(jaw, axis)
    if center is None:
        if len(holes) != 1:
            raise ValueError(f"found {len(holes)} implant holes in the scan - pass the site centre explicitly")
        hole = holes[0]
        center = hole["center"]
    else:
        near = [h for h in holes if np.linalg.norm(h["center"] - np.asarray(center)) < 4.0]
        hole = near[0] if near else None
    if hole is not None:
        # an implant crown flares from the narrow ti-base to full width within the gum
        params = replace(params or CrownParameters(), emergence_slope=2.5, emergence_height=3.0)
    # no root to centre on, and the abutment is virtual: the tooth lines up with its
    # neighbours and needs no material minimum over the abutment
    params = replace(params or CrownParameters(), min_axial=0.2, min_occlusal=0.2, follow_arch_line=True)
    margin, die = virtual_site(jaw, tooth, center, axis, hole=hole)
    na = analyze_neighbors(jaw, margin, axis, tooth=tooth) if use_neighbors else None
    if na is not None:  # re-orient the cervical outline along the arch
        margin, die = virtual_site(jaw, tooth, center, axis, na.md_direction, hole=hole)
        na = analyze_neighbors(jaw, margin, axis, tooth=tooth)
    mirror = (params or CrownParameters()).posterior_anatomy == "mirror" or tooth % 10 <= 3
    if mirror and na is not None and na.template is not None and antagonist is not None:
        seated = _seat_on_antagonist(na.template, antagonist, axis)
        na.template = seated
    res = design_crown(die, tooth=tooth, margin=margin, antagonist=antagonist, axis=axis, neighbors=na,
                       occlusion=occlusion, learner=learner, jaw=jaw, params=params)
    # the abutment is virtual: advice about reducing a preparation does not apply
    res.report["warnings"] = [w.replace("minimum thickness keeps", "left:")
                               .replace("the preparation needs more occlusal reduction", "check the bite")
                              for w in res.report["warnings"] if "intaglio rays" not in w]
    res.report["site"] = ("implant: crown emerges from the soft-tissue hole" if hole is not None
                          else "pontic: ovate base on the ridge")
    solid = close_with_base(res)
    res.report["solid_volume_mm3"] = round(solid.volume(), 1)
    res.report["solid_watertight"] = solid.is_watertight()
    return res, solid, na


def missing_tooth_from_webview(path, tooth: int, *, center=None, occlusion=None, learner=None, params=None):
    """Design ``tooth`` (implant crown / pontic) from an exocad webview with no preparation.

    Returns ``(CrownResult, solid Mesh, NeighborAnalysis)``.  If the scene holds
    the technician's tooth for this site (a wax-up labelled with the tooth, or an
    unlabelled closed tooth-sized object over the site), the report gains a
    ``reference`` comparison of the outer surfaces.
    """
    from .metrics import compare_to_reference, outer_skin
    from .webview import _merge, classify, load_webview

    objects = load_webview(path)
    info = [classify(o) for o in objects]
    side = "upper" if tooth // 10 in (1, 2) else "lower"
    other = "lower" if side == "upper" else "upper"
    axis = (0.0, 0.0, -1.0) if side == "upper" else (0.0, 0.0, 1.0)
    jaws = [o.mesh for o, c in zip(objects, info)
            if c["kind"] == "jaw" and not c["tooth_specific"] and c["jaw"] in (side, None)]
    ants = [o.mesh for o, c in zip(objects, info)
            if c["kind"] == "antagonist" or (c["kind"] in ("jaw", "waxup", "prep") and c["jaw"] == other)]
    if not jaws:
        raise ValueError("no jaw scan in this webview")
    jaw, antagonist = _merge(jaws), _merge(ants) if ants else None
    res, solid, na = design_missing_tooth(jaw, tooth, antagonist=antagonist, center=center, axis=axis,
                                          occlusion=occlusion, learner=learner, params=params)
    refs = [o.mesh for o, c in zip(objects, info) if c["kind"] == "waxup" and c["teeth"] == [tooth]]
    if not refs:  # exocad often names the restoration after the patient: find it by shape and place
        site = res.frame.origin
        for o, c in zip(objects, info):
            m = o.mesh
            if c["kind"] in ("other", "waxup") and m.is_watertight() and 100 < abs(m.volume()) < 1500 \
                    and np.linalg.norm(m.vertices.mean(axis=0) - site) < 6.0:
                refs.append(m)
    res.report["inputs"] = {"jaw_scans": len(jaws), "antagonist_objects": len(ants), "reference": bool(refs)}
    if refs:
        top = res.frame.to_local(res.margin)[:, 2].max()
        res.report["reference"] = compare_to_reference(res.outer[:, 4:].reshape(-1, 3), solid,
                                                       outer_skin(refs[0]), res.frame, top)
    return res, solid, na
