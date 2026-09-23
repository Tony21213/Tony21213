"""Automatic crown design.

Pipeline
--------
1. Margin line: detected on the die or supplied by the user.
2. Tooth frame: insertion axis + mesiodistal direction, origin at the margin centroid.
3. Ray fan: from a centre inside the preparation, rays sweep from every margin
   point up to the occlusal pole.  Both crown surfaces are sampled on this
   shared (theta, v) grid, which makes stitching them into a watertight solid
   trivial.
4. Intaglio (fitting surface): preparation hit points offset by the cement gap,
   tapering to a tighter gap near the margin.
5. Outer (anatomy): shape prior (parametric library or trained PCA model)
   scaled to the preparation, blended into the margin for the emergence
   profile, trimmed against the antagonist and pushed out wherever the
   material would be thinner than the minimum.
6. Stitch intaglio + outer along the margin and export.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .anatomy import ModelTooth, ParametricTooth, ShapeModel, ToothType, tooth_type_for_fdi
from .margin import ToothFrame, detect_margin, make_frame, order_margin
from .mesh import Mesh, nearest_distance, raycast


@dataclass
class CrownParameters:
    cement_gap: float = 0.05  # mm, virtual die spacer on the axial/occlusal walls
    margin_gap: float = 0.01  # mm, gap at the finish line
    margin_band: float = 1.0  # mm from the margin over which the gap ramps up
    thickness_band: float = 1.5  # mm from the margin over which min thickness ramps up
    min_axial: float = 0.8  # mm, minimum wall thickness
    min_occlusal: float = 1.2  # mm, minimum occlusal thickness
    occlusal_clearance: float = 0.05  # mm to keep from the antagonist (negative = contact)
    n_theta: int = 128  # samples around the margin
    n_v: int = 48  # samples from margin to occlusal pole
    smoothing: int = 4  # Laplacian passes on the outer surface
    emergence_height: float = 3.5  # mm above the margin over which anatomy blends into it
    emergence_slope: float = 1.0  # max widening per mm of height above the margin (1 = 45 deg)
    posterior_anatomy: str = "rules"  # premolars/molars: "rules" (anatomical cusp model) or "mirror"
    implant: bool = False  # implant crown: narrower occlusal table (less lateral load), opt-in
    rules_profile: dict | None = None  # the lab's tuned rules (crownai tune-rules), per tooth type
    follow_arch_line: bool = False  # no root to centre on (implant, pontic): line up with the neighbours
    cusp_fossa_max: float = 0.0  # opt-in: max buccolingual shift to put supporting cusps into antagonist fossae (mm)
    contact_gap: float = 0.0  # mm to the adjacent teeth at the contacts (negative = tight)


@dataclass
class CrownResult:
    crown: Mesh
    margin: np.ndarray
    frame: ToothFrame
    intaglio: np.ndarray  # (n_theta, n_v, 3) grid incl. margin row; pole excluded
    outer: np.ndarray
    report: dict = field(default_factory=dict)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical interpolation between unit vectors a (n,3) and b (3,) at t (m,) -> (n,m,3)."""
    cos = np.clip(a @ b, -1.0, 1.0)
    omega = np.arccos(cos)[:, None, None]
    sin = np.maximum(np.sin(omega), 1e-9)
    t = t[None, :, None]
    out = (np.sin((1 - t) * omega) / sin) * a[:, None, :] + (np.sin(t * omega) / sin) * b[None, None, :]
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _grid_normals(P: np.ndarray, pole: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Outward (away from ``center``) normals of a (n_theta, n_v, 3) wrap-around grid."""
    t_theta = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    ext = np.concatenate([P, np.broadcast_to(pole, (P.shape[0], 1, 3))], axis=1)
    t_v = np.empty_like(P)
    t_v[:, 1:] = ext[:, 2:] - ext[:, :-2]
    t_v[:, 0] = ext[:, 1] - ext[:, 0]
    n = np.cross(t_theta, t_v)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    flip = np.einsum("ijk,ijk->ij", n, P - center) < 0
    n[flip] *= -1
    return n


def _arc_length(P: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    return np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(seg, axis=1)], axis=1)


def _laplacian(P: np.ndarray, passes: int, weight: float = 0.5) -> np.ndarray:
    """Smooth rows 1.. of a wrap-around grid; row 0 (margin) stays fixed."""
    P = P.copy()
    for _ in range(passes):
        avg = 0.25 * (np.roll(P, 1, 0) + np.roll(P, -1, 0))
        up = np.concatenate([P[:, 1:], P[:, -1:]], axis=1)
        down = np.concatenate([P[:, :1], P[:, :-1]], axis=1)
        avg = avg + 0.25 * (up + down)
        P[:, 1:] = (1 - weight) * P[:, 1:] + weight * avg[:, 1:]
    return P


def _smooth_top(P: np.ndarray, pole: np.ndarray, v: np.ndarray, start: float = 0.8,
                iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Relax the rows around the pole (the cusp tip / incisal ridge) to remove folds.

    Umbrella smoothing that includes the pole as the upper neighbour of the
    last row; the weight fades in from ``start`` so the rest of the crown
    keeps its shape.
    """
    P = P.copy()
    w = _smoothstep((v - start) / (1.0 - start))[None, :, None] * 0.5
    for _ in range(iters):
        up = np.concatenate([P[:, 1:], np.broadcast_to(pole, (P.shape[0], 1, 3))], axis=1)
        down = np.concatenate([P[:, :1], P[:, :-1]], axis=1)
        avg = 0.25 * (np.roll(P, 1, 0) + np.roll(P, -1, 0) + up + down)
        P = (1 - w) * P + w * avg
        ring = P[:, -1]
        pole = 0.5 * pole + 0.5 * (ring.mean(axis=0) + (pole - ring.mean(axis=0)) * 0.9)
    return P, pole


def _stitch(margin: np.ndarray, inner: np.ndarray, inner_pole: np.ndarray,
            outer: np.ndarray, outer_pole: np.ndarray) -> Mesh:
    """Build a closed crown from two grids sharing their row-0 margin."""
    n_t, n_v = inner.shape[:2]
    verts = [margin, inner[:, 1:].transpose(1, 0, 2).reshape(-1, 3), inner_pole[None],
             outer[:, 1:].transpose(1, 0, 2).reshape(-1, 3), outer_pole[None]]
    base_in = n_t
    pole_in = base_in + (n_v - 1) * n_t
    base_out = pole_in + 1
    pole_out = base_out + (n_v - 1) * n_t

    def idx(base, j, i):
        i = np.asarray(i) % n_t
        return np.where(j == 0, i, base + (j - 1) * n_t + i)

    faces = []
    i = np.arange(n_t)
    for base, pole, sign in ((base_out, pole_out, 1), (base_in, pole_in, -1)):
        surf = []
        for j in range(n_v - 1):
            a, b = idx(base, j, i), idx(base, j, i + 1)
            c, d = idx(base, j + 1, i + 1), idx(base, j + 1, i)
            surf += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
        a, b = idx(base, n_v - 1, i), idx(base, n_v - 1, i + 1)
        surf.append(np.stack([a, b, np.full(n_t, pole)], 1))
        f = np.concatenate(surf)
        faces.append(f if sign > 0 else f[:, ::-1])
    mesh = Mesh(np.concatenate(verts), np.concatenate(faces))
    return mesh if mesh.volume() > 0 else mesh.flipped()


def design_crown(prep: Mesh, *, tooth: int | ToothType | None = None,
                 margin: np.ndarray | None = None, antagonist: Mesh | None = None,
                 axis=(0.0, 0.0, 1.0), md_direction=(1.0, 0.0, 0.0),
                 shape_model: ShapeModel | None = None, shape_coeffs=None,
                 learner=None, neighbors=None, occlusion=None, arch_orientation=None, jaw: Mesh | None = None,
                 trust_md_direction: bool = False,
                 params: CrownParameters | None = None) -> CrownResult:
    """Design a full-contour anatomical crown on ``prep`` (a segmented die scan, mm).

    Anatomy source: incisors and canines use the anatomical model of
    :mod:`crownai.anatomy_model`, fitted to the mirrored contralateral tooth when
    there is one.  Premolars and molars: an explicit ``shape_model``, a trained
    ``learner`` (:class:`crownai.learning.CrownLearner`) with cases for this
    tooth class, otherwise the rule-based cusp model of
    :mod:`crownai.posterior_model` (``params.posterior_anatomy="rules"``, the
    default) or, with ``"mirror"``, the patient's contralateral tooth mirrored
    into place (``neighbors``, see :func:`crownai.arch.analyze_neighbors`) and
    the parametric library tooth when there is none.  With ``neighbors`` the crown also
    follows the arch direction, fills the space between the adjacent teeth
    and touches them at the contacts.

    ``occlusion`` (e.g. :class:`crownai.occlusion.SlavicekConcept`) shapes the
    occlusal surface functionally against the ``antagonist``: centric
    contacts plus no interference along the simulated excursive movements.
    ``arch_orientation`` = (anterior, buccal) world vectors overrides the
    orientation derived from ``neighbors``.  ``trust_md_direction`` keeps the
    given ``md_direction`` (e.g. the technician's own from exocad) instead of
    the one estimated from the arch - learning and design then share a frame.  With ``jaw`` (the scan of the
    crown's own jaw) the excursive paths are simulated from the patient's
    teeth themselves (relief-guided, see :func:`crownai.occlusion.guided_path`).
    """
    p = params or CrownParameters()
    ttype = tooth if isinstance(tooth, ToothType) else tooth_type_for_fdi(tooth)
    warnings: list[str] = []

    # 1-2. Margin and frame --------------------------------------------------
    if margin is None:
        margin = detect_margin(prep, axis=axis, n_points=p.n_theta)
    else:
        margin = order_margin(np.asarray(margin, dtype=np.float64), axis=axis)
        margin = _resample_loop(margin, p.n_theta)
    if neighbors is not None and not trust_md_direction:
        md_direction = neighbors.md_direction
    frame = make_frame(margin.mean(axis=0), axis, md_direction)
    m_loc = frame.to_local(margin)
    prep_top = frame.to_local(prep.vertices)[:, 2].max()
    if prep_top <= m_loc[:, 2].max():
        raise ValueError("preparation does not rise above the margin along the insertion axis")
    if prep_top - m_loc[:, 2].max() > 0.9 * ttype.height:
        # a real prep is reduced well below the tooth's natural crown height; this
        # tall a rise usually means the scan is still the unprepared tooth (or the
        # margin/axis do not match this scan) - the ray fan below assumes a mostly
        # convex stump and can produce a badly distorted crown on real anatomy
        warnings.append(f"preparation rises {prep_top - m_loc[:, 2].max():.1f} mm above the margin - "
                        f"close to or above a natural {ttype.name} crown height "
                        f"({ttype.height:.1f} mm); this may not be an actual reduced preparation")

    # Anatomy (before the ray fan, which is aimed at the cusp tip) --------------
    half_x = np.abs(m_loc[:, 0]).max()
    half_y = np.abs(m_loc[:, 1]).max()
    A = max(ttype.mesiodistal / 2, half_x * 1.12)
    B = max(ttype.buccolingual / 2, half_y * 1.12)
    H = max(ttype.height, prep_top + p.cement_gap + p.min_occlusal + 0.8)
    z0 = m_loc[:, 2].min() - 1.0
    offset = np.zeros(3)
    if neighbors is not None:
        c = frame.to_local(neighbors.target_center)
        offset = np.array([c[0], c[1], 0.0])
        if neighbors.space is not None:
            A = neighbors.space / 2
    prediction = None
    if shape_model is None and learner is not None:
        from .learning import case_features

        prediction = learner.predict(ttype.name, case_features(m_loc, prep_top))
    fdi = int(tooth) if isinstance(tooth, (int, np.integer)) else None
    anatomy_fit = None
    placed = False
    if fdi is not None and fdi % 10 <= 3 and shape_model is None:
        # Incisors and canines: anatomical model, fitted to the patient's
        # mirrored contralateral tooth when there is one.
        shape, source, anatomy_fit = _anterior_anatomy(fdi, frame, m_loc, offset, neighbors, prep_top, p, prep)
        placed = True
    elif fdi is not None and fdi % 10 >= 4 and shape_model is None and prediction is None \
            and p.posterior_anatomy == "rules":
        # Premolars and molars: cusp model from the rules of anatomy, sized and set
        # by the patient's neighbours and antagonist (the mirror image only lends its size)
        shape, source, anatomy_fit = _posterior_anatomy(fdi, frame, m_loc, offset, neighbors, antagonist,
                                                        arch_orientation, prep_top, p, warnings)
        placed = True
    elif neighbors is not None and neighbors.template is not None:
        from .learning import CENTER_Z, N_PHI, N_THETA, crown_signature

        sig, (tA, tB, tH) = crown_signature(neighbors.template, frame, m_loc)
        own = ShapeModel(N_THETA, N_PHI, CENTER_Z, sig, np.zeros((0, sig.size)), np.zeros(0))
        H = max(tH, prep_top + p.cement_gap + p.min_occlusal + 0.3)
        shape = ModelTooth(own, sig, tA, tB, H, z0)
        source = "mirrored contralateral tooth"
    elif shape_model is not None:
        shape = ModelTooth(shape_model, shape_model.reconstruct(shape_coeffs), A, B, H, z0)
        source = "shape model"
    elif prediction is not None:
        # A real mesiodistal space measured from the patient's own neighbouring
        # teeth (set above) is more reliable than the learned model's prediction,
        # which can extrapolate badly outside its training range - keep it.
        if neighbors is None or neighbors.space is None:
            A = max(prediction.dims[0], half_x * 1.05)
        B = max(prediction.dims[1], half_y * 1.05)
        H = max(prediction.dims[2], prep_top + p.cement_gap + p.min_occlusal + 0.3)
        shape = ModelTooth(prediction.shape_model, prediction.signature, A, B, H, z0)
        source = f"learned ({prediction.kind}, {prediction.n_examples} cases)"
    else:
        shape = ParametricTooth(ttype, A, B, H, cervical_x=half_x / A, cervical_y=half_y / B, z0=z0)
        source = "parametric library"

    if neighbors is not None and neighbors.template is None and np.any(offset) and not placed:
        shape = _Shifted(shape, offset)
    if antagonist is not None:
        # the antagonist bounds the shape itself, so the cut is clean (pressing
        # surface points down afterwards folds thin incisal ridges)
        shape = _BelowAntagonist(shape, antagonist, frame, m_loc, p.occlusal_clearance)
    shape = EmergenceShape(shape, m_loc, p.emergence_height, max_slope=p.emergence_slope)
    # 3. Ray fan --------------------------------------------------------------
    detailed = placed and fdi is not None and fdi % 10 >= 4
    if detailed:
        # fine occlusal anatomy needs rows every ~0.2 mm over the occlusal table too
        from dataclasses import replace as _replace

        p = _replace(p, n_v=max(p.n_v, 96), n_theta=max(p.n_theta, 160))
        margin = _resample_loop(margin, p.n_theta)
        m_loc = frame.to_local(margin)
        v = (np.arange(p.n_v) / p.n_v) ** 1.15
    else:
        v = (np.arange(p.n_v) / p.n_v) ** 1.5  # denser sampling near the margin
    center_loc = np.array([0.0, 0.0, m_loc[:, 2].mean() + 0.35 * (prep_top - m_loc[:, 2].mean())])
    center = frame.to_world(center_loc)
    d0 = margin - center
    t_margin = np.linalg.norm(d0, axis=1)
    d0 /= t_margin[:, None]
    pole_dir = frame.z
    tip, node = None, shape
    while node is not None and tip is None:  # EmergenceShape(_BelowAntagonist(PlacedAnatomy)) ...
        tip = getattr(node, "tip_local", None)
        node = getattr(node, "base", None)
    if tip is not None:
        # aim the fan's pole at the cusp tip so an off-axis tip is sampled cleanly
        d_tip = frame.to_world(tip()) - center
        d_tip /= np.linalg.norm(d_tip)
        if d_tip @ frame.z > np.cos(np.radians(20)):
            pole_dir = d_tip
    dirs = _slerp(d0, pole_dir, v)  # (n_theta, n_v, 3)

    # 4. Intaglio -------------------------------------------------------------
    t_in = raycast(center[None], dirs.reshape(-1, 3), prep).reshape(p.n_theta, p.n_v)
    t_in[:, 0] = t_margin
    missing = ~np.isfinite(t_in)
    if missing.any():
        miss_ratio = missing.sum() / missing.size
        row_ok_counts = np.isfinite(t_in).sum(axis=1)
        if miss_ratio > 0.1 or (row_ok_counts < 2).any():
            # this many missed rays means the ray fan from the design pole does not
            # sweep the die cleanly (concave/undercut anatomy, e.g. an unreduced
            # natural crown rather than a real prepared stump) - interpolating across
            # gaps this large does not reconstruct a real surface, it fabricates one
            # (self-intersecting folds), so refuse rather than ship a bad crown
            raise ValueError(
                f"{int(missing.sum())}/{missing.size} intaglio rays missed the preparation "
                "(die geometry is not a clean, mostly-convex stump around the insertion axis - "
                "check that the scan is actually the reduced/prepared tooth, not the natural one)")
        warnings.append(f"{int(missing.sum())} intaglio rays missed the die; interpolated")
        for i in range(p.n_theta):
            row = t_in[i]
            ok = np.isfinite(row)
            t_in[i] = np.interp(np.arange(p.n_v), np.flatnonzero(ok), row[ok])
    t_pole = raycast(center[None], pole_dir[None], prep)[0]
    if not np.isfinite(t_pole):
        t_pole = prep_top - center_loc[2]
    die = center + dirs * t_in[..., None]
    die_pole = center + pole_dir * t_pole

    s_in = _arc_length(die)
    gap = p.margin_gap + (p.cement_gap - p.margin_gap) * _smoothstep(s_in / p.margin_band)
    gap[:, 0] = 0.0
    n_in = _grid_normals(die, die_pole, center)
    inner = die + n_in * gap[..., None]
    inner_pole = die_pole + pole_dir * p.cement_gap

    # 5. Outer anatomy: shaped above, sampled on the ray fan
    t_out = _shape_radii(shape, frame, center, dirs)
    smoothing = p.smoothing
    # a posterior occlusal table carries fine anatomy (fissures, triangular ridges):
    # keep it - the pole smoothing below is for the single tip of a canine
    if detailed:
        smoothing = min(smoothing, 1)
    # Near the pole the rays graze a thin incisal ridge / cusp: neighbouring
    # rays hit its crest or its flanks and the radius jumps.  Smooth around
    # the circumference there (keeps the heights, removes the zigzag).
    top_rows = v > (0.97 if detailed else 0.75)
    if top_rows.any():
        k = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)
        k /= k.sum()
        ext = np.concatenate([t_out[-4:], t_out, t_out[:4]], axis=0)
        smooth = np.stack([np.convolve(ext[:, j], k, mode="valid") for j in range(t_out.shape[1])], axis=1)
        w_top = _smoothstep((v - (0.97 if detailed else 0.75)) / 0.15)[None, :]
        t_out = t_out * (1 - w_top) + smooth * w_top
    t_out[:, 0] = t_margin
    outer = center + dirs * t_out[..., None]
    outer_pole = center + pole_dir * _shape_radii(shape, frame, center, pole_dir[None, None])[0, 0]
    outer = _laplacian(outer, smoothing)
    outer, outer_pole = _smooth_top(outer, outer_pole, v, **({"start": 0.95, "iters": 4} if detailed else {}))

    # Thickness ramps up from the finish line over ``thickness_band`` (measured on the die).
    occlusal_w = _smoothstep((v - 0.45) / 0.3)[None, :]
    required = ((p.min_axial + (p.min_occlusal - p.min_axial) * occlusal_w)
                * _smoothstep(s_in / p.thickness_band))

    if antagonist is not None:
        outer, outer_pole, n_trim = _trim_to_antagonist(outer, outer_pole, antagonist, frame, p.occlusal_clearance)
        if n_trim:
            outer = _laplacian(outer, 2)

    occlusion_report = None
    if occlusion is not None and antagonist is not None:
        outer, outer_pole, occlusion_report = _functional_occlusion(
            outer, outer_pole, antagonist, frame, v, tooth, neighbors, occlusion, arch_orientation,
            p.occlusal_clearance, warnings, jaw)

    contacts = []
    if neighbors is not None:
        for nb in neighbors.neighbors:
            outer, gap_before = _fit_contact(outer, nb.mesh, frame, center, p.contact_gap)
            contacts.append(round(gap_before, 3))
        outer = _laplacian(outer, 1, weight=0.2)

    inner_pts = np.concatenate([inner.reshape(-1, 3), inner_pole[None]])
    for it in range(12):
        thick, _ = nearest_distance(outer.reshape(-1, 3), inner_pts)
        deficit = (required.reshape(-1) - thick).reshape(outer.shape[:2])
        deficit[:, 0] = 0.0
        if deficit.max() <= 1e-3:
            break
        # Push along the surface normal blended with the ray direction: near the
        # margin the normal is almost tangential to the die, the ray is not.
        push = _grid_normals(outer, outer_pole, center) + dirs
        # never push below the finish line: that would be clamped back into a ledge
        push -= np.minimum(push @ frame.z, 0.0)[..., None] * frame.z
        push /= np.maximum(np.linalg.norm(push, axis=-1, keepdims=True), 1e-9)
        outer = outer + push * (1.1 * np.clip(deficit, 0, None))[..., None]
        if it < 8:
            outer = _laplacian(outer, 1, weight=0.15)
        outer = _clamp_above_margin(outer, frame, m_loc[:, 2])
    # the pole follows its ring of neighbours (moving it on its own makes a spike)
    ring = outer[:, -1]
    mid = ring.mean(axis=0)
    outer_pole = mid + pole_dir * max(float(((ring - mid) @ pole_dir).max()), 0.0)
    pole_thick = float(np.linalg.norm(outer_pole - inner_pole))

    thick, _ = nearest_distance(outer.reshape(-1, 3), inner_pts)
    thick = thick.reshape(outer.shape[:2])
    shortfall = (required - thick)[:, 1:]
    if shortfall.max() > 0.05:
        warnings.append(f"minimum thickness not reached by up to {shortfall.max():.2f} mm")

    if occlusion_report is not None:
        from .occlusion import excursion_interference

        left_over = excursion_interference(
            outer[:, v >= 0.35].reshape(-1, 3), antagonist, frame.z, occlusion_report.pop("_movements"),
            crown_on_mandible=occlusion_report["crown_on_mandible"], clearance=p.occlusal_clearance,
            excursion=occlusion.excursion, steps=occlusion.steps)
        occlusion_report["remaining_interference_mm"] = round(left_over, 3)
        if left_over > 0.05:
            warnings.append(f"minimum thickness keeps {left_over:.2f} mm of excursive interference: "
                            "the preparation needs more occlusal reduction")
    if antagonist is not None:
        lift = _antagonist_penetration(outer, antagonist, frame, p.occlusal_clearance)
        if lift > 0.05:
            warnings.append(f"crown reaches {lift:.2f} mm into the antagonist clearance: "
                            "the preparation needs more occlusal reduction")

    # 6. Stitch -------------------------------------------------------------------
    crown = _stitch(margin, inner, inner_pole, outer, outer_pole)
    loc = frame.to_local(crown.vertices)
    report = {
        "tooth_type": ttype.name,
        "anatomy_source": source,
        **({"anatomy_fit": anatomy_fit} if anatomy_fit else {}),
        **({"neighbors": neighbors.summary(), "contact_gap_before_fit_mm": contacts}
           if neighbors is not None else {}),
        **({"occlusion": occlusion_report} if occlusion_report is not None else {}),
        "volume_mm3": round(crown.volume(), 2),
        "watertight": crown.is_watertight(),
        "vertices": int(len(crown.vertices)),
        "triangles": int(len(crown.faces)),
        "mesiodistal_mm": round(float(np.ptp(loc[:, 0])), 2),
        "buccolingual_mm": round(float(np.ptp(loc[:, 1])), 2),
        "height_mm": round(float(loc[:, 2].max() - m_loc[:, 2].min()), 2),
        # Measured outside the margin ramp, where the full minimum applies.
        "min_axial_thickness_mm": round(float(thick[s_in >= p.thickness_band].min()), 3),
        "min_occlusal_thickness_mm": round(float(min(thick[:, v >= 0.75].min(), pole_thick)), 3),
        "margin_length_mm": round(float(np.linalg.norm(np.diff(
            np.vstack([margin, margin[:1]]), axis=0), axis=1).sum()), 2),
        "warnings": warnings,
    }
    return CrownResult(crown, margin, frame, inner, outer, report)


class EmergenceShape:
    """Adapts an anatomy prior so its surface starts exactly on the margin line.

    The template's cylindrical radius is shifted by ``r_margin - r_template``
    at the margin height, fading out over ``height`` mm, the crown may widen
    by at most ``max_slope`` mm per mm of height above the finish line (a
    straight emergence ramp) and nothing is kept below the margin: the crown
    emerges from the finish line without ledges or overhangs.
    """

    def __init__(self, base, margin_local: np.ndarray, height: float, samples: int = 256,
                 max_slope: float = 1.0):
        self.base = base
        self.height = height
        self.max_slope = max_slope  # emergence: at most 1 mm outwards per 1 mm up (45 deg)
        th = np.arctan2(margin_local[:, 1], margin_local[:, 0])
        order = np.argsort(th)
        th, m = th[order], margin_local[order]
        grid = np.linspace(-np.pi, np.pi, samples, endpoint=False)
        ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
        self.theta = grid
        self.z_m = np.interp(grid, ext, np.tile(m[:, 2], 3))
        r_m = np.interp(grid, ext, np.tile(np.hypot(m[:, 0], m[:, 1]), 3))
        self.r_m = r_m
        # Template radius at the margin height, by bisection along each direction.
        lo, hi = np.zeros(samples), np.full(samples, 30.0)
        u = np.stack([np.cos(grid), np.sin(grid), np.zeros(samples)], 1)
        base_pt = np.stack([np.zeros(samples), np.zeros(samples), self.z_m], 1)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            inside = base.inside(base_pt + u * mid[:, None])
            lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
        self.delta = r_m - 0.5 * (lo + hi)

    def _lookup(self, values, theta):
        n = len(self.theta)
        f = (theta + np.pi) / (2 * np.pi) * n
        i0 = np.floor(f).astype(int) % n
        w = f - np.floor(f)
        return (1 - w) * values[i0] + w * values[(i0 + 1) % n]

    def inside(self, p: np.ndarray) -> np.ndarray:
        theta = np.arctan2(p[..., 1], p[..., 0])
        r = np.hypot(p[..., 0], p[..., 1])
        z_m = self._lookup(self.z_m, theta)
        fade = 1.0 - _smoothstep((p[..., 2] - z_m) / self.height)
        r_base = np.maximum(r - self._lookup(self.delta, theta) * fade, 0.0)
        scale = np.where(r > 1e-9, r_base / np.maximum(r, 1e-9), 1.0)
        q = np.stack([p[..., 0] * scale, p[..., 1] * scale, p[..., 2]], axis=-1)
        # emergence profile: rise from the finish line as a ramp, never a ledge
        ramp = r <= self._lookup(self.r_m, theta) + self.max_slope * np.maximum(p[..., 2] - z_m, 0.0) + 0.05
        return self.base.inside(q) & (p[..., 2] >= z_m - 1e-6) & ramp


def _shape_radii(shape, frame: ToothFrame, center: np.ndarray, dirs: np.ndarray,
                 r_max: float = 25.0, step: float = 0.1) -> np.ndarray:
    """Last exit distance of each ray from the implicit anatomy.

    The last exit is the outer envelope seen from the centre: where a ray
    leaves through a concavity (lingual fossa, occlusal groove) and enters
    the crown again, the first exit would jump between neighbouring rays and
    fold the surface.
    """
    c_loc = frame.to_local(center)
    d_loc = np.stack([dirs @ frame.x, dirs @ frame.y, dirs @ frame.z], axis=-1)
    shp = d_loc.shape[:-1]
    d_loc = d_loc.reshape(-1, 3)
    t = np.arange(0.0, r_max, step)
    out = np.empty(len(d_loc))
    for s in range(0, len(d_loc), 512):
        dc = d_loc[s:s + 512]
        ins = shape.inside(c_loc + dc[:, None, :] * t[None, :, None])
        # index of the last sample inside the shape along each ray
        last_in = len(t) - 1 - np.argmax(ins[:, ::-1], axis=1)
        last_in = np.where(ins.any(1), last_in, 0)
        exit_ = np.minimum(last_in + 1, len(t) - 1)
        lo, hi = t[last_in], t[exit_]
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            m_in = shape.inside(c_loc + dc * mid[:, None])
            lo, hi = np.where(m_in, mid, lo), np.where(m_in, hi, mid)
        out[s:s + 512] = 0.5 * (lo + hi)
    return out.reshape(shp)


def _anterior_anatomy(fdi, frame, m_loc, offset, neighbors, prep_top, p, prep):
    """Anatomical incisor/canine placed on the preparation (and fitted to the contralateral)."""
    from dataclasses import replace

    from .anatomy_model import PlacedAnatomy, default_anterior, fit_to_surface

    base = default_anterior(fdi)
    if neighbors is not None and neighbors.space is not None:
        k = neighbors.space / base.md
        base = replace(base, md=base.md * k, md_cervix=base.md_cervix * k)
    labial = 1.0
    if neighbors is not None and neighbors.buccal is not None:
        labial = 1.0 if frame.to_local(frame.origin + neighbors.buccal)[1] >= 0 else -1.0
    z_cervix = float(m_loc[:, 2].mean())
    shift = np.array([offset[0], labial * offset[1]])
    # the crown's cervical cross-section is the root's: take it from the finish line
    mm = PlacedAnatomy.on_margin(base, labial, shift, m_loc).to_model(m_loc)
    base = replace(base, md_cervix=float(np.ptp(mm[:, 0])) * 0.98, ll_cervix=float(np.ptp(mm[:, 1])) * 0.98)
    shift = shift + np.array([0.5 * (mm[:, 0].max() + mm[:, 0].min()), 0.5 * (mm[:, 1].max() + mm[:, 1].min())])
    placed = PlacedAnatomy.on_margin(base, labial, shift, m_loc)
    # the preparation grown by the minimum thickness: the anatomy must enclose it
    q = frame.to_local(prep.vertices)
    zc = placed.cervix(q[:, 0], q[:, 1])
    above = q[:, 2] > zc + 0.3
    q = q[above]
    rel = q[:, :2] - placed.shift * np.array([1.0, labial])
    radial = np.column_stack([rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-6), np.zeros(len(q))])
    h = (q[:, 2] - zc[above]) / max(prep_top - float(zc.mean()), 1e-3)
    t_min = p.cement_gap + p.min_axial + (p.min_occlusal - p.min_axial) * np.clip((h - 0.5) / 0.4, 0, 1)
    grown = q + radial * t_min[:, None]
    grown[:, 2] += np.where(h > 0.85, t_min, 0.0)  # over the top of the stump
    contain = placed.to_model(grown)
    tpl = placed.to_model(frame.to_local(neighbors.template.vertices)) \
        if neighbors is not None and neighbors.template is not None else None
    fitted, d, stats = fit_to_surface(base, tpl, contain=contain, fix_cervix=True)
    placed = PlacedAnatomy.on_margin(fitted, labial, shift + d[:2], m_loc)
    source = ("anatomical model fitted to the mirrored contralateral tooth" if tpl is not None
              else "anatomical model (textbook proportions, sized to the preparation)")
    # the crown must cover the preparation with the occlusal minimum
    need = prep_top + p.cement_gap + p.min_occlusal + 0.3 - float(placed.margin_z.min())
    if placed.model.height < need:
        placed.model = replace(placed.model, height=need)
    if stats is not None:
        stats = {**stats, **{k: round(float(getattr(placed.model, k)), 2)
                             for k in ("height", "md", "ll", "tip_u", "tip_w", "drop_m", "drop_d")}}
    return placed, source, stats


def _posterior_anatomy(fdi, frame, m_loc, offset, neighbors, antagonist, arch_orientation, prep_top, p, warnings):
    """Premolar/molar from the anatomical cusp model, set into the patient's arch by rules.

    * size: mesiodistal = the space between the neighbours (or the measured
      contralateral width, or Wheeler); buccolingual in the textbook proportion
      to the mesial neighbour's; implant crowns get a narrower occlusal table;
    * height: cusp tips on the neighbours' occlusal level;
    * buccolingual position: the supporting cusps over the antagonist's fossae
      (cusp-fossa relation), close to the line of the neighbouring teeth;
    * each cusp is then raised or lowered to the antagonist - supporting cusps
      into contact, none into it - instead of the crown being cut flat.
    """
    from dataclasses import replace

    from .anatomy_model import WHEELER, PlacedAnatomy
    from .posterior_model import default_posterior, scaled

    pos = min(fdi % 10, 7)
    arch = "upper" if fdi // 10 in (1, 2, 5, 6) else "lower"
    base = default_posterior(fdi, p.rules_profile)
    info = {}
    if p.rules_profile and base.detail is not None:
        info["profile"] = "lab profile"
    md = base.md
    if neighbors is not None and neighbors.space is not None:
        md, info["md_from"] = neighbors.space, "space between neighbours"
    elif neighbors is not None and neighbors.contralateral is not None:
        # the patient's own width, drawn towards the textbook (a scan crop can overstate it)
        md = 0.5 * (float(np.clip(neighbors.contralateral.md_width, 0.9 * base.md, 1.12 * base.md)) + base.md)
        info["md_from"] = "contralateral width"
    else:
        info["md_from"] = "textbook"
    bl = base.bl * (md / base.md) ** 0.5
    # neighbours in the local frame (mesial = +x)
    nbs = []
    if neighbors is not None:
        for nb in neighbors.neighbors:
            q = frame.to_local(nb.mesh.vertices)
            nbs.append(q[q[:, 2] > q[:, 2].max() - 3.5])
    mesial = [q for q in nbs if q[:, 0].mean() > 0]
    if mesial and pos - 1 >= 4:
        q = mesial[0]
        bl_nb = float(np.percentile(q[:, 1], 98) - np.percentile(q[:, 1], 2))
        rule = bl_nb * WHEELER[arch][pos][3] / WHEELER[arch][pos - 1][3]
        if 0.8 * base.bl < rule < 1.25 * base.bl:
            bl = 0.5 * (bl + rule)
            info["bl_from"] = "mesial neighbour proportion"
    if p.implant:
        bl *= 0.9
        info["implant_narrowed"] = True
    model = scaled(base, md=md, bl=bl)

    labial = 1.0
    buccal_vec = None
    if neighbors is not None and neighbors.buccal is not None:
        buccal_vec = neighbors.buccal
    elif arch_orientation is not None:
        buccal_vec = np.asarray(arch_orientation[1], float)
    if buccal_vec is not None:
        labial = 1.0 if frame.to_local(frame.origin + buccal_vec)[1] >= 0 else -1.0
    else:
        # the quadrant fixes the side: buccal = +z x mesial in quadrants 1 and 3 (valid
        # when the mesial direction is real, e.g. exocad's; a guessed one gets a note)
        labial = 1.0 if (fdi // 10) % 2 == 1 else -1.0
        info["buccal_from"] = "tooth number"
        warnings.append("buccal side unknown: taken from the tooth number and the mesial direction")
    shift = np.array([offset[0], labial * offset[1]])
    if mesial and (neighbors is None or neighbors.space is None):
        # free end: touch the mesial neighbour
        q = mesial[0]
        near = q[np.abs(q[:, 1] - offset[1]) < 3.0]
        if len(near):
            shift[0] = float(near[:, 0].min()) - md / 2 - p.contact_gap
            info["md_position"] = "against the mesial neighbour"

    fade = 0.55 * base.height  # the cervical line's course fades out by mid-crown
    if p.follow_arch_line and nbs:
        # the tooth stands in the row: its occlusal centre on the neighbours' line
        shift[1] = labial * float(np.mean([np.median(q[:, 1]) for q in nbs]))
        info["bl_position"] = "in line with the neighbours"
    placed = PlacedAnatomy.on_margin(model, labial, shift, m_loc, fade=fade)
    mm = placed.to_model(m_loc)
    model = replace(model, md_cervix=min(float(np.ptp(mm[:, 0])) * 0.98, 0.95 * md),
                    bl_cervix=min(float(np.ptp(mm[:, 1])) * 0.98, 0.95 * bl))
    placed = PlacedAnatomy.on_margin(model, labial, shift, m_loc, fade=fade)
    cz = float(placed.margin_z.mean())

    # height: on the neighbours' occlusal level, else textbook
    H = model.height
    if nbs:
        tops = [float(q[:, 2].max()) for q in nbs]
        H = float(np.mean(tops)) - cz
        info["height_from"] = "neighbours' cusp level"
    need = prep_top + p.cement_gap + p.min_occlusal + 0.3 - cz  # cusp tips stand above the mean cervix
    H = max(H, need, 0.6 * base.height)
    placed.model = model = replace(model, height=H)

    def local_tips(m, sh):
        t = m.cusp_tips()
        x, y = t[:, 0] + sh[0], labial * (t[:, 1] + sh[1] + m.tilt * t[:, 2])
        return np.column_stack([x, y, placed.z_cervix + t[:, 2]])  # tips: above the mean cervix

    if antagonist is not None:
        ceil = _BelowAntagonist(None, antagonist, frame, m_loc, p.occlusal_clearance, reach=12.0)
        func = np.array([c.functional for c in model.cusps])

        def ceiling(xy):
            i = np.clip(np.round((xy[:, 0] - ceil.x0) / ceil.res).astype(int), 0, ceil.ceiling.shape[0] - 1)
            j = np.clip(np.round((xy[:, 1] - ceil.y0) / ceil.res).astype(int), 0, ceil.ceiling.shape[1] - 1)
            return ceil.ceiling[i, j]

        # cusp-fossa relation: each supporting cusp goes into the antagonist's fossa
        # next to it - a valley between two antagonist cusps in the buccolingual
        # section (not simply where the antagonist is farthest: embrasures, air)
        tips = local_tips(model, shift)
        ys = np.linspace(-3.0, 3.0, 61)
        offsets = []
        for (x_t, y_t, _), f in zip(tips, func):
            if not f:
                continue
            c = ceiling(np.column_stack([np.full(len(ys), x_t), y_t + ys]))
            if not np.isfinite(c).all():
                continue
            c = np.convolve(np.pad(c, 3, mode="edge"), np.ones(7) / 7, mode="valid")
            cand = []
            for i in range(5, len(ys) - 5):
                if c[i] < c[max(i - 6, 0):i + 7].max():
                    continue
                # a fossa: the antagonist comes down again on both sides (its cusps)
                if c[i] - c[:i].min() > 0.4 and c[i] - c[i + 1:].min() > 0.4:
                    cand.append(ys[i])
            if cand:
                offsets.append(min(cand, key=abs))
        best_s = float(np.clip(np.median(offsets), -p.cusp_fossa_max, p.cusp_fossa_max)) if offsets else 0.0
        info["cusp_fossa_shift_mm"] = round(float(best_s), 2)
        shift = shift + np.array([0.0, labial * best_s])  # local y -> model w
        placed = PlacedAnatomy.on_margin(model, labial, shift, m_loc, fade=fade)
        tips = local_tips(model, shift)
        c = ceiling(tips[:, :2])
        if np.isfinite(c[func]).any():
            gap = c - tips[:, 2]
            dz = float(np.nanmin(np.where(func & np.isfinite(gap), gap, np.nan)))
            H = float(np.clip(H + dz, max(need, 0.6 * base.height), 1.8 * base.height))
            model = replace(model, height=H)
            tips = local_tips(model, shift)
            gap = c - tips[:, 2]
            cusps, moved = [], {}
            for cusp, g, f in zip(model.cusps, gap, func):
                dh = cusp.dh
                if np.isfinite(g):
                    if g < 0:  # would reach into the antagonist: lower this cusp (the rest is cut)
                        dh += max(g, -1.0)
                    elif f and g > 0.1:  # supporting cusp short of contact: bring it up
                        dh += min(g - 0.05, 0.6)
                moved[cusp.name] = round(float(dh - cusp.dh), 2)
                cusps.append(replace(cusp, dh=dh))
            top = max(c_.dh for c_ in cusps)
            model = replace(model, height=H + top, cusps=tuple(replace(c_, dh=c_.dh - top) for c_ in cusps))
            info["cusp_height_changes_mm"] = moved
            info["height_from"] = "antagonist contact"
    placed = PlacedAnatomy.on_margin(model, labial, shift, m_loc, fade=fade)
    info.update(md=round(model.md, 2), bl=round(model.bl, 2), height=round(model.height, 2))
    return placed, "anatomical rules (cusp model, set by neighbours and antagonist)", info


class _BelowAntagonist:
    """Anatomy prior limited to the space under the antagonist (occlusal height field)."""

    def __init__(self, base, antagonist: Mesh, frame: ToothFrame, m_loc: np.ndarray, clearance: float,
                 reach: float = 9.0, res: float = 0.2):
        self.base = base
        c = m_loc[:, :2].mean(0)
        self.x0, self.y0 = c[0] - reach, c[1] - reach
        self.res = res
        n = int(2 * reach / res) + 1
        xs = self.x0 + np.arange(n) * res
        ys = self.y0 + np.arange(n) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        low = m_loc[:, 2].min() - 30.0
        origins = frame.to_world(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, low)]))
        t = raycast(origins, np.broadcast_to(frame.z, origins.shape), antagonist)
        self.ceiling = (low + t).reshape(n, n) - clearance  # inf where there is no antagonist

    def inside(self, p: np.ndarray) -> np.ndarray:
        i = np.clip(np.round((p[..., 0] - self.x0) / self.res).astype(int), 0, self.ceiling.shape[0] - 1)
        j = np.clip(np.round((p[..., 1] - self.y0) / self.res).astype(int), 0, self.ceiling.shape[1] - 1)
        return self.base.inside(p) & (p[..., 2] <= self.ceiling[i, j])


class _Shifted:
    """An anatomy prior moved within the occlusal plane (tooth-local offset)."""

    def __init__(self, base, offset: np.ndarray):
        self.base, self.offset = base, offset

    def inside(self, p: np.ndarray) -> np.ndarray:
        return self.base.inside(p - self.offset)


def _fit_contact(outer: np.ndarray, neighbor: Mesh, frame: ToothFrame, center: np.ndarray,
                 gap: float, band: float = 3.0) -> tuple[np.ndarray, float]:
    """Move the proximal surface facing ``neighbor`` so the closest approach is ``gap``.

    The distance is measured along the direction to the neighbour; the whole
    proximal third moves, fading out towards the middle of the crown and
    towards the margin, which stays fixed.
    """
    nb_loc = frame.to_local(neighbor.vertices)
    c_loc = frame.to_local(center)
    d = nb_loc[:, :2].mean(0) - c_loc[:2]
    d = frame.vector_to_world(np.array([d[0], d[1], 0.0]))
    d /= np.linalg.norm(d)
    pts = outer.reshape(-1, 3)
    back = 4.0
    t = raycast(pts - d * back, np.broadcast_to(d, pts.shape), neighbor) - back
    t = t.reshape(outer.shape[:2])
    t[:, :3] = np.inf  # never move the margin region
    if not np.isfinite(t).any():
        return outer, float("nan")
    closest = float(np.min(t))
    proj = (outer - center) @ d
    w = _smoothstep((proj - (proj.max() - band)) / band)
    w *= _smoothstep((np.arange(outer.shape[1]) - 2) / 6.0)[None, :]
    return outer + d * ((closest - gap) * w)[..., None], closest


def _functional_occlusion(outer, outer_pole, antagonist, frame, v, tooth, neighbors, concept,
                          arch_orientation, clearance, warnings, jaw=None):
    """Centric contacts and excursion-free occlusal surface after ``concept``."""
    from .occlusion import arch_directions, carve_excursions, raise_to_centric_contacts

    fdi = tooth if isinstance(tooth, (int, np.integer)) else None
    if arch_orientation is not None:
        anterior, buccal = (np.asarray(a, dtype=float) for a in arch_orientation)
        source = "given"
    elif neighbors is not None and neighbors.buccal is not None:
        anterior, buccal = arch_directions(frame.x, frame.z, neighbors.buccal, fdi, neighbors.midline_normal)
        source = "adjacent teeth" + (" + midline" if neighbors.midline_normal is not None else "")
    else:
        anterior, buccal = arch_directions(frame.x, frame.z, None, fdi)
        source = "assumed"
        warnings.append("arch orientation unknown (no adjacent teeth): excursion directions assumed")
    on_mandible = (fdi // 10 in (3, 4)) if fdi else bool(frame.z[2] > 0)
    movements = concept.movements(fdi, anterior, buccal)
    guidance = {}
    if jaw is not None and getattr(concept, "patient_guidance", False):
        from .occlusion import guided_path

        mandible, maxilla = (jaw, antagonist) if on_mandible else (antagonist, jaw)
        up = frame.z if on_mandible else -frame.z  # mandible -> maxilla
        patient = []
        for name, d, angle in movements:
            try:
                path = guided_path(mandible, maxilla, up, d, name, concept.excursion, concept.steps * 2)
            except ValueError as exc:
                warnings.append(f"relief-guided simulation not possible: {exc}")
                patient = []
                break
            measured = path.summary()
            concept_lift = path.s * np.tan(np.radians(angle))
            # Canines and incisors lead the guidance: they may be as steep as the
            # concept asks even where the other teeth guide flatter.  Premolars
            # and molars must disclude: they stay under the flatter of the two.
            leads = fdi is not None and fdi % 10 <= 3
            lift = np.maximum(path.lift, concept_lift) if leads else np.minimum(path.lift, concept_lift)
            if path.lift[-1] <= 0.1:  # the teeth do not guide this movement at all
                lift = concept_lift
            path.lift = lift
            guidance[name] = {**measured, "concept_angle_deg": round(angle, 1),
                              "used_angle_deg": round(path.angle(), 1),
                              "role": "guiding tooth" if leads else "discluding tooth"}
            patient.append((name, d, angle, path))
        if patient:
            movements = patient
    region = np.broadcast_to((v >= 0.35)[None, :], outer.shape[:2])
    n_centric = 0
    if concept.centric_contacts:
        outer, n_centric = raise_to_centric_contacts(outer, antagonist, frame.z, region, clearance)
    grid = np.concatenate([outer, np.broadcast_to(outer_pole, (outer.shape[0], 1, 3))], axis=1)
    reg = np.concatenate([region, np.ones((outer.shape[0], 1), bool)], axis=1)
    grid, moves = carve_excursions(grid, antagonist, frame.z, movements, crown_on_mandible=on_mandible,
                                   clearance=clearance, excursion=concept.excursion, steps=concept.steps,
                                   region=reg)
    outer = _laplacian(grid[:, :-1], 1, weight=0.2)
    outer_pole = grid[:, -1].mean(axis=0)
    return outer, outer_pole, {
        "concept": "Slavicek sequential guidance",
        "crown_on_mandible": on_mandible,
        "orientation": source,
        "centric_contact_points": n_centric,
        "movements": moves,
        **({"patient_guidance": guidance} if guidance else {}),
        "_movements": movements,
    }


def _clamp_above_margin(outer: np.ndarray, frame: ToothFrame, margin_z: np.ndarray) -> np.ndarray:
    """No material below the finish line: lift outer points to the margin height."""
    below = np.clip(margin_z[:, None] - frame.to_local(outer)[..., 2], 0.0, None)
    return outer + below[..., None] * frame.z


def _trim_to_antagonist(outer, outer_pole, antagonist: Mesh, frame: ToothFrame, clearance: float):
    pts = np.concatenate([outer.reshape(-1, 3), outer_pole[None]])
    below = pts - frame.z * 30.0
    t = raycast(below, np.broadcast_to(frame.z, pts.shape), antagonist)
    excess = np.where(np.isfinite(t), (30.0 - t) + clearance, -np.inf)
    push = np.clip(excess, 0, None)
    pts = pts - frame.z * push[:, None]
    n = int((push > 0).sum())
    return pts[:-1].reshape(outer.shape), pts[-1], n


def _antagonist_penetration(outer, antagonist: Mesh, frame: ToothFrame, clearance: float) -> float:
    pts = outer.reshape(-1, 3)
    t = raycast(pts - frame.z * 30.0, np.broadcast_to(frame.z, pts.shape), antagonist)
    excess = np.where(np.isfinite(t), (30.0 - t) + clearance, 0.0)
    return float(max(excess.max(), 0.0))


def _resample_loop(points: np.ndarray, n: int) -> np.ndarray:
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    target = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(target, s, closed[:, k]) for k in range(3)], axis=1)
