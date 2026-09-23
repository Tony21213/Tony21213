"""Anatomical crown model built from dental-anatomy rules, fitted to the patient.

A crown is described the way dental anatomy describes it - by its outlines
seen from each aspect - and every quantity has an anatomical meaning:

* **dimensions** from Wheeler's measurement table (crown length, mesiodistal
  and labio/buccolingual diameters at the contacts and at the cervix);
* **labial view**: cusp tip position, mesial and distal cusp ridges (the
  mesial one shorter on canines), the incisal angles, contact heights
  (mesial higher than distal), mesial outline straighter than distal;
* **proximal view**: labial height of contour near the cervical line,
  lingual cingulum in the cervical third, incisal ridge thickness and its
  labiolingual position;
* **lingual anatomy**: marginal ridges, lingual ridge (canines) and the
  fossae between them;
* **labial anatomy**: labial ridge (pronounced on upper canines, faint on
  lower ones).

The model is smooth and complete (all sides), so it always looks like a
tooth.  To look like *this patient's* tooth, :func:`fit_to_surface` adjusts
its parameters to a partial scan of the contralateral tooth (mirrored into
place) - the scan only steers the anatomy, it is never pasted in.

Coordinates: ``u`` mesial, ``w`` labial/buccal, ``z`` occlusal, origin on
the tooth axis at the cervical level.  Anterior teeth only for now
(incisors and canines); premolars and molars use the cusp model in
:mod:`crownai.anatomy`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

# Wheeler: crown length, MD crown, MD cervix, LL crown, LL cervix (mm) by arch and position
WHEELER = {
    "upper": {1: (10.5, 8.5, 7.0, 7.0, 6.0), 2: (9.0, 6.5, 5.0, 6.0, 5.0), 3: (10.0, 7.5, 5.5, 8.0, 7.0),
              4: (8.5, 7.0, 5.0, 9.0, 8.0), 5: (8.5, 7.0, 5.0, 9.0, 8.0), 6: (7.5, 10.0, 8.0, 11.0, 10.0),
              7: (7.0, 9.0, 7.0, 11.0, 10.0)},
    "lower": {1: (9.0, 5.0, 3.5, 6.0, 5.3), 2: (9.5, 5.5, 4.0, 6.5, 5.8), 3: (11.0, 7.0, 5.5, 7.5, 7.0),
              4: (8.5, 7.0, 5.0, 7.5, 6.5), 5: (8.0, 7.0, 5.0, 8.0, 7.0), 6: (7.5, 11.0, 9.0, 10.5, 9.0),
              7: (7.0, 10.5, 8.0, 10.0, 9.0)},
}


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class AnteriorCrown:
    """Incisor / canine crown.  All lengths in mm."""

    height: float  # cervical line to cusp tip / incisal edge
    md: float  # mesiodistal diameter at the contacts
    md_cervix: float
    ll: float  # labiolingual diameter at the heights of contour
    ll_cervix: float
    tip_u: float = 0.0  # cusp tip mesiodistal position (canine: over the root centre)
    tip_w: float = 0.0  # incisal ridge labiolingual position (+ labial)
    drop_m: float = 1.2  # tip height minus mesio-incisal angle height
    drop_d: float = 2.0  # tip height minus disto-incisal angle height
    contact_m: float = 0.78  # mesial contact height, fraction of the crown height
    contact_d: float = 0.62  # distal contact height
    distal_convexity: float = 0.30  # how much the distal outline curves in above the contact
    mesial_convexity: float = 0.08
    labial_share: float = 0.52  # labial part of the LL diameter (the rest is lingual)
    labial_hc: float = 0.15  # labial height of contour, fraction of the height (near the cervix)
    labial_fullness: float = 1.5  # >1: labial face stays full towards the incisal ridge
    cingulum_h: float = 0.25  # cingulum height, fraction of the height
    ridge_thickness: float = 1.0  # incisal ridge thickness
    fossa_depth: float = 0.16  # lingual fossae depth, fraction of the lingual extent
    lingual_ridge: float = 0.6  # 0 = one fossa (incisor), 1 = two fossae split by a strong ridge (canine)
    labial_ridge: float = 0.02  # vertical labial ridge prominence (fraction)
    tip_radius: float = 0.35  # rounding of the cusp tip / incisal angles
    squareness: float = 2.2  # superellipse exponent of the cross-sections
    tilt: float = 0.0  # labial inclination of the crown (w shift per mm of height)

    # ---- outlines -------------------------------------------------------
    def top(self, u):
        """Incisal ridge height along the mesiodistal axis (labial view)."""
        du = u - self.tip_u
        half_m = max(self.md / 2 - self.tip_u, 1e-3)
        half_d = max(self.md / 2 + self.tip_u, 1e-3)
        slope = np.where(du > 0, self.drop_m / half_m, self.drop_d / half_d)
        r = self.tip_radius
        return self.height - (np.sqrt((slope * du) ** 2 + r * r) - r)

    def mesiodistal(self, z):
        """Mesial and distal extents at height z (labial view outlines)."""
        H = self.height
        out = []
        for contact, convex in ((self.contact_m, self.mesial_convexity), (self.contact_d, self.distal_convexity)):
            zc = contact * H
            below = self.md_cervix / 2 + (self.md / 2 - self.md_cervix / 2) * np.sin(
                0.5 * np.pi * np.clip(z / zc, 0, 1))
            above = self.md / 2 * (1 - convex * np.clip((z - zc) / (H - zc), 0, 1) ** 2)
            out.append(np.where(z < zc, below, above))
        return out[0], out[1]

    def labiolingual(self, zeta):
        """Labial and lingual extents at relative height zeta (proximal view outlines).

        Each is a smooth line from the cervix to the incisal ridge plus a
        bulge: the labial height of contour near the cervix, the lingual
        cingulum in the cervical third.  The labial line is full (convex),
        the lingual line hollow (concave) - the lingual fossa.
        """
        lab_max = self.ll * self.labial_share
        lin_max = self.ll - lab_max
        lab_c = self.ll_cervix * self.labial_share
        lin_c = self.ll_cervix - lab_c
        end_l = self.tip_w + self.ridge_thickness / 2
        end_g = -self.tip_w + self.ridge_thickness / 2
        z = np.clip(zeta, 0, 1)
        lab_line = lab_c + (end_l - lab_c) * z ** self.labial_fullness
        lin_line = lin_c + (end_g - lin_c) * z ** 0.75
        hc, ci = self.labial_hc, self.cingulum_h
        lab_line_hc = lab_c + (end_l - lab_c) * hc ** self.labial_fullness
        lin_line_ci = lin_c + (end_g - lin_c) * ci ** 0.75
        lab = lab_line + (lab_max - lab_line_hc) * np.exp(-((z - hc) / 0.2) ** 2)
        lin = lin_line + (lin_max - lin_line_ci) * np.exp(-((z - ci) / 0.26) ** 2)
        return lab, lin

    def inside(self, p: np.ndarray) -> np.ndarray:
        u, w, z = p[..., 0], p[..., 1] - self.tilt * np.clip(p[..., 2], 0, None), p[..., 2]
        top = self.top(u)
        zeta = np.clip(z / np.maximum(top, 1e-3), 0, 1)
        M, D = self.mesiodistal(z)
        U = np.where(u >= 0, M, D)
        lab, lin = self.labiolingual(zeta)
        un = u / np.maximum(U, 1e-3)
        # lingual anatomy: fossae between the marginal ridges, split by the lingual ridge
        fossa_z = np.exp(-((zeta - 0.6) / 0.25) ** 2)
        split = self.lingual_ridge * np.exp(-((u - self.tip_u) / (0.16 * self.md)) ** 2)
        fossa_u = np.clip(1 - np.abs(un) ** 2.5, 0, 1) * (1 - split)
        lin = lin * (1 - self.fossa_depth * fossa_z * fossa_u)
        # faint vertical labial ridge through the cusp tip
        lab = lab * (1 + self.labial_ridge * np.exp(-((u - self.tip_u) / (0.22 * self.md)) ** 2)
                     * np.sin(np.pi * zeta))
        W = np.where(w >= 0, lab, lin)
        e = self.squareness
        section = np.abs(un) ** e + (np.abs(w) / np.maximum(W, 1e-3)) ** e <= 1.0
        return section & (z >= -3.0) & (z <= top)


def default_anterior(fdi: int) -> AnteriorCrown:
    """Textbook crown for an incisor or canine (FDI position 1-3)."""
    arch = "upper" if fdi // 10 in (1, 2) else "lower"
    pos = fdi % 10
    if pos > 3:
        raise ValueError("anterior model covers incisors and canines (positions 1-3)")
    h, md, mdc, ll, llc = WHEELER[arch][pos]
    base = AnteriorCrown(h, md, mdc, ll, llc)
    if pos == 3:
        if arch == "lower":  # tip over the root, lingual incisal ridge, faint labial ridge
            return replace(base, tip_w=-0.3, drop_m=1.8, drop_d=2.8, labial_ridge=0.02, lingual_ridge=0.45)
        return replace(base, tip_w=0.2, drop_m=2.2, drop_d=3.0, labial_ridge=0.06, lingual_ridge=0.8,
                       cingulum_h=0.28, fossa_depth=0.2)
    # incisors: straight incisal edge, single lingual fossa, thin ridge
    incisal = dict(drop_m=0.15, drop_d=0.45, lingual_ridge=0.0, ridge_thickness=0.9, tip_radius=0.2,
                   contact_m=0.85, contact_d=0.75 if pos == 1 else 0.65, distal_convexity=0.2)
    if arch == "lower":
        return replace(base, tip_w=-0.2, fossa_depth=0.12, labial_ridge=0.0, **incisal)
    return replace(base, tip_w=0.3, fossa_depth=0.2, labial_ridge=0.01, **incisal)


# --------------------------------------------------------------------------
# Fitting to the patient's contralateral tooth
# --------------------------------------------------------------------------

_FIT = ("height", "md", "ll", "tip_u", "tip_w", "drop_m", "drop_d", "labial_share", "labial_fullness",
        "tilt")
_BOUNDS = {"height": (0.5, 1.35), "md": (0.8, 1.25), "ll": (0.8, 1.25), "tip_u": (-0.8, 0.8),
           "tip_w": (-1.0, 1.0), "drop_m": (0.3, 3.0), "drop_d": (0.5, 3.5), "labial_share": (0.4, 0.65),
           "labial_fullness": (0.8, 2.5), "tilt": (-0.25, 0.25)}
_RELATIVE = {"height", "md", "ll"}  # bounds are factors of the textbook value


def radial_distance(model, center: np.ndarray, dirs: np.ndarray, r_max: float = 15.0, steps: int = 90) -> np.ndarray:
    """Distance from ``center`` to the model surface along each direction."""
    t = np.linspace(0, r_max, steps)
    ins = model.inside(center + dirs[:, None, :] * t[None, :, None])
    out = ~ins
    first = np.where(out.any(1), out.argmax(1), steps - 1)
    lo, hi = t[np.maximum(first - 1, 0)], t[first]
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        m_in = model.inside(center + dirs * mid[:, None])
        lo, hi = np.where(m_in, mid, lo), np.where(m_in, hi, mid)
    return 0.5 * (lo + hi)


def model_center(m: AnteriorCrown) -> np.ndarray:
    """A point well inside the crown (middle of its cross-section at 40 % height)."""
    z = 0.4 * m.height
    zeta = z / float(m.top(np.array(m.tip_u)))
    lab, lin = m.labiolingual(np.array(zeta))
    M, D = m.mesiodistal(np.array(z))
    return np.array([0.5 * (float(M) - float(D)), 0.5 * (float(lab) - float(lin)) + m.tilt * z, z])


def fit_to_surface(model: AnteriorCrown, points: np.ndarray | None, *, contain: np.ndarray | None = None,
                   max_points: int = 1500, seed: int = 0, prior_weight: float = 0.03,
                   contain_weight: float = 3.0, fix_cervix: bool = False) -> tuple[AnteriorCrown, np.ndarray, dict]:
    """Adjust the anatomy (and a small shift in u, w) to a partial crown scan.

    ``points`` are in model coordinates (u mesial, w labial, z from the
    cervical level).  Only the scanned part constrains the fit; everything
    else keeps its anatomical default.  ``contain`` are points the crown must
    enclose - the preparation grown by the minimum material thickness - so
    the anatomy is sized to cover the stump instead of being inflated
    locally afterwards.  ``fix_cervix`` keeps the cervical diameters (set
    from the finish line) while the crown above is fitted.  Returns the
    fitted model, the fitted shift and fit statistics.
    """
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    if points is None or len(points) == 0:
        pts = np.zeros((0, 3))
    else:
        pts = points[points[:, 2] > 0.8]  # crown only
        if len(pts) > max_points:
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
    if contain is not None and len(contain) > 600:
        contain = contain[rng.choice(len(contain), 600, replace=False)]
    base = {f.name: getattr(model, f.name) for f in fields(model)}
    x0, lo, hi = [], [], []
    for k in _FIT:
        v = base[k]
        b = _BOUNDS[k]
        if k in _RELATIVE:
            x0.append(1.0); lo.append(b[0]); hi.append(b[1])
        else:
            x0.append(float(np.clip(v, *b))); lo.append(b[0]); hi.append(b[1])
    # start from the scan's own measurements, not the textbook
    ref = pts if len(pts) >= 50 else (contain if contain is not None else pts)
    top = np.percentile(ref[:, 2], 99.5) + (0.0 if len(pts) >= 50 else 1.0)
    pts_for_init = ref
    upper = pts_for_init[pts_for_init[:, 2] > 0.35 * top]
    su = 0.5 * (np.percentile(upper[:, 0], 2) + np.percentile(upper[:, 0], 98))
    sw = 0.5 * (np.percentile(upper[:, 1], 2) + np.percentile(upper[:, 1], 98))
    tip_pts = pts_for_init[pts_for_init[:, 2] > top - 0.6 - (0.0 if len(pts) >= 50 else 1.0)]
    start = {"height": max(top, 0.5) / base["height"],
             "md": (np.percentile(upper[:, 0], 98) - np.percentile(upper[:, 0], 2)) / base["md"],
             "ll": (np.percentile(upper[:, 1], 98) - np.percentile(upper[:, 1], 2)) / base["ll"],
             "tip_u": tip_pts[:, 0].mean() - su, "tip_w": tip_pts[:, 1].mean() - sw}
    for i, k in enumerate(_FIT):
        if k in start:
            x0[i] = float(np.clip(start[k], lo[i] + 1e-6, hi[i] - 1e-6))
    x0 += [float(np.clip(su, -1.99, 1.99)), float(np.clip(sw, -1.99, 1.99))]
    lo += [-2.0, -2.0]
    hi += [2.0, 2.0]

    def build(x):
        kw = dict(base)
        for k, val in zip(_FIT, x[:len(_FIT)]):
            kw[k] = base[k] * val if k in _RELATIVE else val
        if not fix_cervix:
            kw["md_cervix"] = base["md_cervix"] * kw["md"] / base["md"]
            kw["ll_cervix"] = base["ll_cervix"] * kw["ll"] / base["ll"]
        return AnteriorCrown(**kw), np.array([x[-2], x[-1], 0.0])

    # Anatomical prior: the shape (not the size) may leave the textbook only
    # as far as the scan clearly demands - worn or partial scans must not turn
    # a canine into a stump.
    prior_sigma = {"tip_u": 0.5, "tip_w": 0.6, "drop_m": 0.5, "drop_d": 0.6, "labial_share": 0.04,
                   "labial_fullness": 0.35, "tilt": 0.08}
    prior_idx = [(i, base[k], prior_sigma[k]) for i, k in enumerate(_FIT) if k in prior_sigma]

    def residual(x):
        m, shift = build(x)
        q = pts - shift
        center = model_center(m)
        v = q - center
        r = np.linalg.norm(v, axis=1)
        d = v / r[:, None]
        data = r - radial_distance(m, center, d)
        lam = prior_weight * np.sqrt(max(len(pts), 100))
        prior = [lam * (x[i] - mu) / sd for i, mu, sd in prior_idx]
        parts = [data, prior]
        if contain is not None:
            vc = contain - shift - center
            rc = np.linalg.norm(vc, axis=1)
            outside = rc - radial_distance(m, center, vc / rc[:, None])
            parts.append(contain_weight * np.clip(outside, 0.0, None))  # only where not covered
        return np.concatenate(parts)

    x = np.array(x0)
    before = np.abs(residual(x)[:len(pts)])
    all_pts = pts
    for _ in range(3):
        # Trimmed fit: points far from the fitted crown are not crown surface
        # (gingiva, model base, neighbouring teeth) - drop them and refit.
        sol = least_squares(residual, x, bounds=(lo, hi), loss="soft_l1", f_scale=0.3,
                            diff_step=0.02, max_nfev=60)
        x = sol.x
        pts = all_pts
        r = np.abs(residual(x)[:len(pts)])
        keep = r < max(0.5, 2.5 * np.median(r))
        if keep.all() or keep.sum() < 100:
            break
        pts = all_pts[keep]
        all_pts = pts
    fitted, shift = build(x)
    after = np.abs(residual(x)[:len(pts)])
    return fitted, shift, {"points": int(len(pts)), "mean_before_mm": round(float(before.mean()), 3),
                           "mean_after_mm": round(float(after.mean()), 3)}


@dataclass
class PlacedAnatomy:
    """An anatomical model placed in the tooth-local design frame.

    local x = mesial, local y -> labial with ``labial_sign``, local z = occlusal.
    The model's cervical line follows the preparation margin: at each angle
    around the tooth the model height is measured from the margin there, so a
    scalloped finish line gives the curved cervical line of a natural crown
    (higher on the proximal surfaces) instead of a ledge.
    """

    model: AnteriorCrown
    labial_sign: float
    z_cervix: float  # mean cervical level (used where no margin is given)
    shift: np.ndarray  # (u, w) of the model axis in the local frame
    margin_theta: np.ndarray | None = None  # sorted angles of the margin around the local origin
    margin_z: np.ndarray | None = None
    fade: float = 0.0  # >0: the margin's ups and downs fade out over this height (mm) - the
    #                   occlusal table stays level however scalloped or inclined the finish line

    @classmethod
    def on_margin(cls, model, labial_sign, shift, margin_local: np.ndarray, samples: int = 180,
                  fade: float = 0.0):
        th = np.arctan2(margin_local[:, 1], margin_local[:, 0])
        order = np.argsort(th)
        grid = np.linspace(-np.pi, np.pi, samples, endpoint=False)
        ext_t = np.concatenate([th[order] - 2 * np.pi, th[order], th[order] + 2 * np.pi])
        z = np.interp(grid, ext_t, np.tile(margin_local[order, 2], 3))
        # smooth the cervical line (a finish line has small wiggles a CEJ has not)
        k = np.exp(-0.5 * (np.arange(-8, 9) / 4.0) ** 2)
        z = np.convolve(np.concatenate([z[-8:], z, z[:8]]), k / k.sum(), mode="valid")
        return cls(model, labial_sign, float(margin_local[:, 2].mean()), np.asarray(shift, float), grid, z, fade)

    def cervix(self, x, y):
        if self.margin_theta is None:
            return np.full(np.shape(x), self.z_cervix)
        th = np.arctan2(y, x)
        n = len(self.margin_theta)
        f = (th + np.pi) / (2 * np.pi) * n
        i0 = np.floor(f).astype(int) % n
        w = f - np.floor(f)
        return (1 - w) * self.margin_z[i0] + w * self.margin_z[(i0 + 1) % n]

    def level(self, x, y, z):
        """Cervical reference height for a point: the local margin, fading to the mean."""
        zc = self.cervix(x, y)
        if self.fade > 0:
            t = np.clip((z - self.z_cervix) / self.fade, 0, 1)
            t = t * t * (3 - 2 * t)
            zc = zc * (1 - t) + self.z_cervix * t
        return zc

    def to_model(self, p: np.ndarray) -> np.ndarray:
        zc = self.level(p[..., 0], p[..., 1], p[..., 2])
        return np.stack([p[..., 0] - self.shift[0], self.labial_sign * p[..., 1] - self.shift[1],
                         p[..., 2] - zc], axis=-1)

    def inside(self, p: np.ndarray) -> np.ndarray:
        return self.model.inside(self.to_model(p))

    def tip_local(self) -> np.ndarray:
        """Cusp tip / incisal edge centre in the local frame."""
        m = self.model
        x = m.tip_u + self.shift[0]
        y = self.labial_sign * (m.tip_w + self.shift[1] + m.tilt * m.height)
        zc = self.z_cervix if self.fade > 0 else float(self.cervix(np.array(x), np.array(y)))
        return np.array([x, y, zc + m.height])
