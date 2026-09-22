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
    emergence_height: float = 2.0  # mm above the margin over which anatomy blends into it
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
                 learner=None, neighbors=None, params: CrownParameters | None = None) -> CrownResult:
    """Design a full-contour anatomical crown on ``prep`` (a segmented die scan, mm).

    Anatomy source, in priority order: the patient's own contralateral tooth
    mirrored into place (``neighbors``, see :func:`crownai.arch.analyze_neighbors`),
    an explicit ``shape_model``, a trained ``learner``
    (:class:`crownai.learning.CrownLearner`) with cases for this tooth class,
    otherwise the parametric library tooth.  With ``neighbors`` the crown also
    follows the arch direction, fills the space between the adjacent teeth
    and touches them at the contacts.
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
    if neighbors is not None:
        md_direction = neighbors.md_direction
    frame = make_frame(margin.mean(axis=0), axis, md_direction)
    m_loc = frame.to_local(margin)
    prep_top = frame.to_local(prep.vertices)[:, 2].max()
    if prep_top <= m_loc[:, 2].max():
        raise ValueError("preparation does not rise above the margin along the insertion axis")

    # 3. Ray fan --------------------------------------------------------------
    center_loc = np.array([0.0, 0.0, m_loc[:, 2].mean() + 0.35 * (prep_top - m_loc[:, 2].mean())])
    center = frame.to_world(center_loc)
    d0 = margin - center
    t_margin = np.linalg.norm(d0, axis=1)
    d0 /= t_margin[:, None]
    v = (np.arange(p.n_v) / p.n_v) ** 1.5  # denser sampling near the margin
    dirs = _slerp(d0, frame.z, v)  # (n_theta, n_v, 3)

    # 4. Intaglio -------------------------------------------------------------
    t_in = raycast(center[None], dirs.reshape(-1, 3), prep).reshape(p.n_theta, p.n_v)
    t_in[:, 0] = t_margin
    missing = ~np.isfinite(t_in)
    if missing.any():
        warnings.append(f"{int(missing.sum())} intaglio rays missed the die; interpolated")
        for i in range(p.n_theta):
            row = t_in[i]
            ok = np.isfinite(row)
            t_in[i] = np.interp(np.arange(p.n_v), np.flatnonzero(ok), row[ok])
    t_pole = raycast(center[None], frame.z[None], prep)[0]
    if not np.isfinite(t_pole):
        t_pole = prep_top - center_loc[2]
    die = center + dirs * t_in[..., None]
    die_pole = center + frame.z * t_pole

    s_in = _arc_length(die)
    gap = p.margin_gap + (p.cement_gap - p.margin_gap) * _smoothstep(s_in / p.margin_band)
    gap[:, 0] = 0.0
    n_in = _grid_normals(die, die_pole, center)
    inner = die + n_in * gap[..., None]
    inner_pole = die_pole + frame.z * p.cement_gap

    # 5. Outer anatomy ----------------------------------------------------------
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
    if neighbors is not None and neighbors.template is not None:
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
        A = max(prediction.dims[0], half_x * 1.05)
        B = max(prediction.dims[1], half_y * 1.05)
        H = max(prediction.dims[2], prep_top + p.cement_gap + p.min_occlusal + 0.3)
        shape = ModelTooth(prediction.shape_model, prediction.signature, A, B, H, z0)
        source = f"learned ({prediction.kind}, {prediction.n_examples} cases)"
    else:
        shape = ParametricTooth(ttype, A, B, H, cervical_x=half_x / A, cervical_y=half_y / B, z0=z0)
        source = "parametric library"

    if neighbors is not None and neighbors.template is None and np.any(offset):
        shape = _Shifted(shape, offset)
    shape = EmergenceShape(shape, m_loc, p.emergence_height)
    t_out = _shape_radii(shape, frame, center, dirs)
    t_out[:, 0] = t_margin
    outer = center + dirs * t_out[..., None]
    outer_pole = center + frame.z * _shape_radii(shape, frame, center, frame.z[None, None])[0, 0]
    outer = _laplacian(outer, p.smoothing)

    # Thickness ramps up from the finish line over ``thickness_band`` (measured on the die).
    occlusal_w = _smoothstep((v - 0.45) / 0.3)[None, :]
    required = ((p.min_axial + (p.min_occlusal - p.min_axial) * occlusal_w)
                * _smoothstep(s_in / p.thickness_band))

    if antagonist is not None:
        outer, outer_pole, n_trim = _trim_to_antagonist(outer, outer_pole, antagonist, frame, p.occlusal_clearance)
        if n_trim:
            outer = _laplacian(outer, 2)

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
        push /= np.linalg.norm(push, axis=-1, keepdims=True)
        outer = outer + push * (1.1 * np.clip(deficit, 0, None))[..., None]
        if it < 8:
            outer = _laplacian(outer, 1, weight=0.15)
        outer = _clamp_above_margin(outer, frame, m_loc[:, 2])
    pole_thick = float(np.linalg.norm(outer_pole - inner_pole))
    if pole_thick < p.min_occlusal:
        outer_pole = inner_pole + frame.z * p.min_occlusal
        pole_thick = p.min_occlusal

    thick, _ = nearest_distance(outer.reshape(-1, 3), inner_pts)
    thick = thick.reshape(outer.shape[:2])
    shortfall = (required - thick)[:, 1:]
    if shortfall.max() > 0.05:
        warnings.append(f"minimum thickness not reached by up to {shortfall.max():.2f} mm")

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
        **({"neighbors": neighbors.summary(), "contact_gap_before_fit_mm": contacts}
           if neighbors is not None else {}),
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
    at the margin height, fading out over ``height`` mm, and nothing is kept
    below the margin: the crown emerges from the finish line without ledges
    or overhangs.
    """

    def __init__(self, base, margin_local: np.ndarray, height: float, samples: int = 256):
        self.base = base
        self.height = height
        th = np.arctan2(margin_local[:, 1], margin_local[:, 0])
        order = np.argsort(th)
        th, m = th[order], margin_local[order]
        grid = np.linspace(-np.pi, np.pi, samples, endpoint=False)
        ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
        self.theta = grid
        self.z_m = np.interp(grid, ext, np.tile(m[:, 2], 3))
        r_m = np.interp(grid, ext, np.tile(np.hypot(m[:, 0], m[:, 1]), 3))
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
        return self.base.inside(q) & (p[..., 2] >= z_m - 1e-6)


def _shape_radii(shape, frame: ToothFrame, center: np.ndarray, dirs: np.ndarray,
                 r_max: float = 25.0, step: float = 0.1) -> np.ndarray:
    """First exit distance of each ray from the implicit anatomy."""
    c_loc = frame.to_local(center)
    d_loc = np.stack([dirs @ frame.x, dirs @ frame.y, dirs @ frame.z], axis=-1)
    shp = d_loc.shape[:-1]
    d_loc = d_loc.reshape(-1, 3)
    t = np.arange(0.0, r_max, step)
    out = np.empty(len(d_loc))
    for s in range(0, len(d_loc), 512):
        dc = d_loc[s:s + 512]
        ins = shape.inside(c_loc + dc[:, None, :] * t[None, :, None])
        outside = ~ins
        first = np.where(outside.any(1), outside.argmax(1), len(t) - 1)
        lo, hi = t[np.maximum(first - 1, 0)], t[first]
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            m_in = shape.inside(c_loc + dc * mid[:, None])
            lo, hi = np.where(m_in, mid, lo), np.where(m_in, hi, mid)
        out[s:s + 512] = 0.5 * (lo + hi)
    return out.reshape(shp)


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
