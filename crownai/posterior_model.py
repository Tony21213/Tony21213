"""Premolar and molar crowns built from the rules of dental anatomy.

The crown is modelled the way a technician waxes it up (cusp-by-cusp
technique): every cusp is a rounded cone at its anatomical place, the
occlusal surface is where the cones, the marginal ridges and (upper first
molar) the oblique ridge meet.  The fissures are not drawn - they appear
where neighbouring cusp slopes meet, exactly as in a wax-up, so their
pattern follows the cusp arrangement of each tooth type (the "Y" of the
lower second premolar, the five-cusp pattern of the lower first molar, the
cross of the lower second molar, the oblique ridge of the upper first molar).

Rules encoded (Wheeler / Ash & Nelson, standard wax-up practice):

* dimensions from Wheeler's table; mesial contact higher than distal, both in
  the occlusal third (molars) / at the junction of occlusal and middle thirds
  (premolars);
* buccal height of contour in the cervical third, lingual in the middle third;
  lower posteriors lean lingually, their lingual cusps are higher than the
  buccal ones, upper molars have the mesiopalatal cusp as the largest;
* cusp tips on the buccal and lingual cusp lines, inside the outline, so the
  occlusal table is ~55-65 % of the buccolingual diameter;
* each cusp has a triangular ridge running to the central fossa (its inner
  slope is gentler along the ridge than across it) - this gives the
  triangular ridges and the grooves between them;
* marginal ridges ~1.2 mm below the cusp tips, the mesial one higher;
* the central fossa floor ~2.5 mm below the highest cusp.

Coordinates: ``u`` mesial, ``w`` buccal, ``z`` occlusal from the cervical
line - the same as :mod:`crownai.anatomy_model`, so a posterior crown is
placed with :class:`crownai.anatomy_model.PlacedAnatomy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .anatomy_model import WHEELER


def _smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass(frozen=True)
class Cusp:
    name: str
    u: float  # tip position, fraction of md/2 (+ mesial)
    w: float  # fraction of bl/2 (+ buccal)
    dh: float  # tip height relative to the crown height (0 = highest cusp), mm
    functional: bool  # supporting (centric) cusp


@dataclass
class PosteriorCrown:
    """Premolar / molar crown.  All lengths in mm."""

    height: float  # cervical line to the highest cusp tip
    md: float
    md_cervix: float
    bl: float
    bl_cervix: float
    cusps: tuple[Cusp, ...] = ()
    slope_in: float = 0.75  # inner cusp incline (tan): ~37 deg, unworn
    slope_out: float = 1.2
    cusp_ridge_slope: float = 0.45  # mesial/distal cusp ridges along the buccal and lingual edges
    ridge_ratio: float = 1.7  # >1: triangular ridges (steeper across than along the ridge)
    ridge_lean: float = 0.35  # triangular ridges lean mesially/distally towards the tooth's middle
    tip_radius: float = 0.6
    marginal_drop: float = 1.2  # marginal ridges below the highest cusp
    mesial_ridge_extra: float = 0.3  # mesial marginal ridge higher than the distal one
    fossa_depth: float = 3.2  # deepest point of the central fossa below the highest cusp
    oblique_ridge: bool = False  # upper first/second molar: mesiopalatal to distobuccal
    contact_m: float = 0.80  # contact heights, fraction of the height
    contact_d: float = 0.74
    hc_b: float = 0.22  # buccal height of contour (cervical third)
    hc_l: float = 0.45  # lingual height of contour (middle third)
    buccal_share: float = 0.5  # buccal part of the BL diameter
    lingual_convergence: float = 0.0  # MD narrower towards lingual (lower molars)
    skew: float = 0.0  # rhomboid outline (upper molars): u shift per unit w
    squareness: float = 3.4  # molars: boxy outline (rounded rectangle)
    tilt: float = 0.0  # buccal (+) / lingual (-) crown inclination, w per mm of z
    fissure_k: float = 5.0  # sharpness of the fissures (smooth-max of the cusp slopes)
    tip_u: float = 0.0  # for PlacedAnatomy.tip_local (centre of the occlusal table)
    tip_w: float = 0.0
    # learned secondary anatomy: mean difference between real crowns and the rule
    # surface, on a grid over (u / (md/2), w / (bl/2)) in [-1, 1]^2 (see crownai.rules_tuning)
    detail: np.ndarray | None = field(default=None, repr=False)

    # ---- occlusal surface --------------------------------------------------
    def _cusp_xy(self, c: Cusp) -> np.ndarray:
        return np.array([c.u * self.md / 2, c.w * self.bl / 2])

    def cusp_tips(self) -> np.ndarray:
        """(n, 3) cusp tips in model coordinates."""
        return np.array([[*self._cusp_xy(c), self.height + c.dh] for c in self.cusps])

    def occlusal(self, u, w):
        """Occlusal surface height z_top(u, w) (upright, before the crown tilt)."""
        r = self.tip_radius
        parts = []
        for c in self.cusps:
            cu, cw = self._cusp_xy(c)
            du, dw = u - cu, w - cw
            # the triangular ridge runs across the tooth to the central groove (the
            # mesiodistal fissure between the buccal and lingual cusp rows), leaning
            # a little towards the middle of the tooth
            n = np.array([-self.ridge_lean * np.sign(cu) * min(abs(cu) / max(self.md / 2, 1e-6), 1.0),
                          -np.sign(cw) if cw != 0 else 0.0])
            if not n.any():
                n = -np.array([cu, cw])
            n = n / max(np.linalg.norm(n), 1e-6)
            par = du * n[0] + dw * n[1]
            perp = -du * n[1] + dw * n[0]
            # inside: the triangular ridge (gentle along it, steeper across);
            # outside: the cusp ridges run gently along the buccal/lingual edge
            # (mesial and distal cusp ridges), the outer face drops steeply.
            # Both blend continuously across the cusp tip.
            t_a = _smooth01((par + 0.2) / 0.4)
            t_b = _smooth01((par + 0.3) / 1.8)
            a = self.slope_out + (self.slope_in - self.slope_out) * t_a
            b = self.cusp_ridge_slope + (self.ridge_ratio * self.slope_in - self.cusp_ridge_slope) * t_b
            grade = np.sqrt((a * par) ** 2 + (b * perp) ** 2)
            parts.append(self.height + c.dh - (np.sqrt(grade ** 2 + (self.slope_in * r) ** 2) - self.slope_in * r))
        # marginal ridges: crests across the mesial and distal ends of the occlusal table
        b_tip = max((c.w for c in self.cusps if c.w > 0), default=0.5) * self.bl / 2
        l_tip = min((c.w for c in self.cusps if c.w < 0), default=-0.5) * self.bl / 2
        for sign, extra in ((1.0, self.mesial_ridge_extra), (-1.0, 0.0)):
            u_mr = sign * (self.md / 2 - 1.1)
            h = self.height - self.marginal_drop + extra
            ww = np.clip((w - 0.5 * (b_tip + l_tip)) / max(0.5 * (b_tip - l_tip), 1e-3), -1.5, 1.5)
            along = 0.8 * np.clip(np.abs(ww) - 1.0, 0, None) ** 2  # rounds off into the cusp ridges
            d = np.abs(u - u_mr)
            parts.append(h - 0.15 * ww ** 2 - along - 0.9 * (np.sqrt(d * d + r * r) - r))
        if self.oblique_ridge:
            ends = [c for c in self.cusps if c.name in ("MP", "DB")]
            if len(ends) == 2:
                a, b = self._cusp_xy(ends[0]), self._cusp_xy(ends[1])
                ha, hb = self.height + ends[0].dh, self.height + ends[1].dh
                ab = b - a
                t = np.clip(((u - a[0]) * ab[0] + (w - a[1]) * ab[1]) / (ab @ ab), 0, 1)
                d = np.hypot(u - (a[0] + t * ab[0]), w - (a[1] + t * ab[1]))
                crest = ha + (hb - ha) * t - 1.0 * np.sin(np.pi * t)
                parts.append(crest - 0.9 * (np.sqrt(d * d + r * r) - r))
        parts.append(np.full(np.shape(u), self.height - self.fossa_depth))  # central fossa floor
        P = np.stack(parts)
        k = self.fissure_k
        m = P.max(axis=0)
        z = m + np.log(np.exp(k * (P - m)).sum(axis=0)) / k
        if self.detail is not None:
            z = z + sample_grid(self.detail, u / (self.md / 2), w / (self.bl / 2))
        return z

    # ---- outlines ------------------------------------------------------------
    def _profile(self, z, cervix, hc, contour, tipline, z_occ):
        """Extent from the cervix, out to the height of contour, in to the cusp line."""
        H = self.height
        zc = hc * H
        low = cervix + (contour - cervix) * np.sin(0.5 * np.pi * np.clip(z / zc, 0, 1))
        s = np.clip((z - zc) / max(z_occ - zc, 1e-3), 0, 1)
        high = contour + (tipline - contour) * (1 - np.cos(0.5 * np.pi * s)) ** 1.0
        top = tipline - 0.6 * np.clip(z - z_occ, 0, None)  # above: the cusps' outer slopes take over
        return np.where(z < zc, low, np.where(z < z_occ, high, top))

    def extents(self, z):
        """(mesial, distal, buccal, lingual) extents at height z."""
        H = self.height
        tips = self.cusp_tips() if self.cusps else np.zeros((0, 3))
        b_tips = tips[tips[:, 1] > 0] if len(tips) else tips
        l_tips = tips[tips[:, 1] < 0] if len(tips) else tips
        b_line = float(b_tips[:, 1].max()) + self.tip_radius + 0.6 if len(b_tips) else 0.3 * self.bl
        l_line = float(-l_tips[:, 1].min()) + self.tip_radius + 0.6 if len(l_tips) else 0.3 * self.bl
        z_b = float(b_tips[:, 2].min()) - 1.2 if len(b_tips) else 0.8 * H
        z_l = float(l_tips[:, 2].min()) - 1.2 if len(l_tips) else 0.8 * H
        bs = self.buccal_share
        B = self._profile(z, self.bl_cervix * bs, self.hc_b, self.bl * bs, b_line, z_b)
        L = self._profile(z, self.bl_cervix * (1 - bs), self.hc_l, self.bl * (1 - bs), l_line, z_l)
        out = []
        for contact in (self.contact_m, self.contact_d):
            zc = contact * H
            below = self.md_cervix / 2 + (self.md / 2 - self.md_cervix / 2) * np.sin(0.5 * np.pi * np.clip(z / zc, 0, 1))
            above = self.md / 2 * (1 - 0.18 * np.clip((z - zc) / (H - zc), 0, 1.5) ** 2)
            out.append(np.where(z < zc, below, above))
        return out[0], out[1], B, L

    def inside(self, p: np.ndarray) -> np.ndarray:
        u, z = p[..., 0], p[..., 2]
        w = p[..., 1] - self.tilt * np.clip(z, 0, None)
        M, D, B, L = self.extents(z)
        W = np.where(w >= 0, B, L)
        wn = w / np.maximum(W, 1e-3)
        us = u + self.skew * wn * self.md / 2
        U = np.where(us >= 0, M, D) * (1 - self.lingual_convergence * np.clip(-wn, 0, 1))
        e = self.squareness
        section = np.abs(us / np.maximum(U, 1e-3)) ** e + np.abs(wn) ** e <= 1.0
        return section & (z >= -3.0) & (z <= self.occlusal(u, w))


def sample_grid(grid: np.ndarray, x, y) -> np.ndarray:
    """Bilinear lookup in a square grid spanning [-1, 1]^2 (0 outside, faded at the rim)."""
    n = grid.shape[0]
    fx = (np.clip(x, -1, 1) + 1) / 2 * (n - 1)
    fy = (np.clip(y, -1, 1) + 1) / 2 * (n - 1)
    i0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, n - 2)
    tx, ty = fx - i0, fy - j0
    v = (grid[i0, j0] * (1 - tx) * (1 - ty) + grid[i0 + 1, j0] * tx * (1 - ty)
         + grid[i0, j0 + 1] * (1 - tx) * ty + grid[i0 + 1, j0 + 1] * tx * ty)
    r = np.maximum(np.abs(x), np.abs(y))
    return v * np.clip((1.15 - r) / 0.25, 0, 1)  # no detail beyond the occlusal table


# --------------------------------------------------------------------------
# Tooth types
# --------------------------------------------------------------------------

_CUSPS = {
    # upper premolars: buccal cusp higher; first premolar's buccal tip slightly distal, lingual tip mesial
    ("upper", 4): (Cusp("B", -0.08, 0.52, 0.0, False), Cusp("L", 0.10, -0.52, -1.0, True)),
    ("upper", 5): (Cusp("B", 0.0, 0.50, 0.0, False), Cusp("L", 0.0, -0.50, -0.4, True)),
    # upper molars: mesiopalatal the largest, distopalatal the smallest
    ("upper", 6): (Cusp("MB", 0.48, 0.60, -0.3, False), Cusp("DB", -0.42, 0.62, -0.6, False),
                   Cusp("MP", 0.30, -0.52, 0.0, True), Cusp("DP", -0.55, -0.50, -0.9, True)),
    ("upper", 7): (Cusp("MB", 0.48, 0.60, -0.3, False), Cusp("DB", -0.38, 0.58, -0.8, False),
                   Cusp("MP", 0.22, -0.48, 0.0, True), Cusp("DP", -0.62, -0.42, -1.5, True)),
    # lower premolars: first - dominant buccal cusp over the centre, tiny lingual cusp;
    # second - three cusps (Y groove pattern)
    ("lower", 4): (Cusp("B", 0.05, 0.22, 0.0, True), Cusp("L", 0.18, -0.60, -2.0, False)),
    ("lower", 5): (Cusp("B", 0.0, 0.40, 0.0, True), Cusp("ML", 0.40, -0.50, -0.7, False),
                   Cusp("DL", -0.45, -0.48, -1.1, False)),
    # lower molars: lingual cusps higher than buccal; first molar five cusps, second four
    ("lower", 6): (Cusp("MB", 0.56, 0.60, -0.6, True), Cusp("DB", -0.04, 0.64, -0.7, True),
                   Cusp("D", -0.66, 0.36, -1.1, True), Cusp("ML", 0.50, -0.56, 0.0, False),
                   Cusp("DL", -0.32, -0.58, -0.2, False)),
    ("lower", 7): (Cusp("MB", 0.50, 0.58, -0.6, True), Cusp("DB", -0.46, 0.58, -0.8, True),
                   Cusp("ML", 0.48, -0.56, 0.0, False), Cusp("DL", -0.48, -0.56, -0.25, False)),
}


def profile_key(fdi: int) -> str:
    arch = "upper" if fdi // 10 in (1, 2, 5, 6) else "lower"
    return f"{arch}_{min(fdi % 10, 7)}"


# scalar rule parameters a profile may override (all tooth-shape, none size or placement)
PROFILE_FIELDS = ("slope_in", "slope_out", "cusp_ridge_slope", "ridge_ratio", "fossa_depth", "marginal_drop",
                  "mesial_ridge_extra", "hc_b", "hc_l", "contact_m", "contact_d", "buccal_share",
                  "squareness", "tilt")


def default_posterior(fdi: int, profile: dict | None = None) -> PosteriorCrown:
    """Premolar / molar crown for FDI position 4-8 (8 uses the 7).

    Textbook values, or - with ``profile`` (from :func:`crownai.rules_tuning.tune_rules`,
    learned from the lab's own finished crowns) - the lab's typical proportions,
    cusp arrangement and secondary anatomy for this tooth type.
    """
    base = _textbook_posterior(fdi)
    entry = (profile or {}).get(profile_key(fdi))
    if not entry:
        return base
    kw = {f: float(entry[f]) for f in PROFILE_FIELDS if f in entry}
    if "cusps" in entry and len(entry["cusps"]) == len(base.cusps):
        kw["cusps"] = tuple(replace(c, u=float(e["u"]), w=float(e["w"]), dh=float(e["dh"]))
                            for c, e in zip(base.cusps, entry["cusps"]))
    if "md" in entry:
        md = float(entry["md"])
        base = scaled(base, md=md, bl=md * float(entry.get("bl_md_ratio", base.bl / base.md)))
    if "height" in entry:
        kw["height"] = float(entry["height"])
    if entry.get("detail") is not None:
        kw["detail"] = np.asarray(entry["detail"], dtype=float)
    return replace(base, **kw)


def _textbook_posterior(fdi: int) -> PosteriorCrown:
    arch = "upper" if fdi // 10 in (1, 2, 5, 6) else "lower"
    pos = min(fdi % 10, 7)
    if pos < 4:
        raise ValueError("posterior model covers premolars and molars (positions 4-8)")
    h, md, mdc, bl, blc = WHEELER[arch][pos]
    base = PosteriorCrown(h, md, mdc, bl, blc, cusps=_CUSPS[(arch, pos)])
    if pos <= 5:  # premolars
        base = replace(base, slope_in=0.8, squareness=2.4, contact_m=0.72, contact_d=0.66, fossa_depth=2.9,
                       marginal_drop=1.1, hc_l=0.5 if arch == "lower" else 0.45)
    if arch == "lower":
        base = replace(base, tilt=-0.06, lingual_convergence=0.08 if pos >= 6 else 0.0)
        if pos == 4:  # lingual cusp non-functional and small: occlusal table tilted lingually
            base = replace(base, marginal_drop=1.6, fossa_depth=3.3)
    else:
        base = replace(base, oblique_ridge=pos == 6, skew=0.10 if pos >= 6 else 0.0)
    return base


def scaled(model: PosteriorCrown, md: float | None = None, bl: float | None = None,
           height: float | None = None) -> PosteriorCrown:
    """The same tooth at other dimensions (cusp positions scale with it)."""
    kw = {}
    if md is not None:
        kw.update(md=md, md_cervix=model.md_cervix * md / model.md)
    if bl is not None:
        kw.update(bl=bl, bl_cervix=model.bl_cervix * bl / model.bl)
    if height is not None:
        kw.update(height=height)
    return replace(model, **kw)


def model_mesh(model: PosteriorCrown, n_theta: int = 120, n_phi: int = 70):
    """Closed mesh of a crown model (visualisation, tests, synthetic references)."""
    from .anatomy_model import radial_distance
    from .mesh import Mesh

    c = np.array([0.0, 0.0, 0.4 * model.height])
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ph = np.linspace(0.03, np.pi - 0.03, n_phi)
    T, P = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)], -1).reshape(-1, 3)
    r = radial_distance(model, c, d, r_max=14.0, steps=140)
    V = (c + d * r[:, None]).reshape(n_theta, n_phi, 3)
    faces = []
    for i in range(n_theta):
        i1 = (i + 1) % n_theta
        for j in range(n_phi - 1):
            a, b, cc, dd = i * n_phi + j, i1 * n_phi + j, i1 * n_phi + j + 1, i * n_phi + j + 1
            faces += [[a, cc, b], [a, dd, cc]]
    top = len(V.reshape(-1, 3))
    verts = np.vstack([V.reshape(-1, 3), c + [0, 0, r.reshape(n_theta, n_phi)[:, 0].mean()],
                       c - [0, 0, r.reshape(n_theta, n_phi)[:, -1].mean()]])
    for i in range(n_theta):
        i1 = (i + 1) % n_theta
        faces.append([top, i * n_phi, i1 * n_phi])
        faces.append([top + 1, i1 * n_phi + n_phi - 1, i * n_phi + n_phi - 1])
    mesh = Mesh(verts, np.array(faces))
    return mesh if mesh.volume() > 0 else mesh.flipped()
