"""Anatomical crown shape priors: a parametric tooth library and a statistical
shape model (PCA) that can be trained on existing crown designs.

Both expose the same interface used by the designer: ``inside(points)`` on
points in the tooth-local frame (z = occlusal, x = mesiodistal, origin at the
margin centroid), where the shape spans ``[-A, A] x [-B, B] x [z0, H]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import Mesh, raycast


@dataclass(frozen=True)
class ToothType:
    name: str
    mesiodistal: float  # mm, crown width at the height of contour
    buccolingual: float
    height: float  # mm, cervical line to cusp tip
    exponent: float  # superellipse exponent of the cross-section
    top_taper_md: float  # width at the occlusal edge relative to height of contour
    top_taper_bl: float
    cusps: tuple[tuple[float, float, float], ...] = ()  # (x/A, y/B, height mm)
    fossa_depth: float = 0.0


_MOLAR_CUSPS = ((-0.45, 0.45, 1.8), (0.45, 0.45, 1.6), (-0.45, -0.45, 1.9),
                (0.45, -0.45, 1.7), (0.0, -0.1, 0.0))
_PREMOLAR_CUSPS = ((0.0, 0.45, 2.0), (0.0, -0.45, 1.4))

TOOTH_TYPES = {
    "molar": ToothType("molar", 10.5, 10.5, 7.5, 2.6, 0.84, 0.80, _MOLAR_CUSPS, 1.0),
    "premolar": ToothType("premolar", 7.0, 8.5, 8.0, 2.3, 0.80, 0.72, _PREMOLAR_CUSPS, 0.8),
    "canine": ToothType("canine", 7.5, 7.8, 10.0, 2.0, 0.55, 0.45, ((0.0, 0.0, 2.2),), 0.0),
    "incisor": ToothType("incisor", 8.0, 6.8, 10.0, 2.0, 0.95, 0.25, (), 0.0),
    "lower_incisor": ToothType("lower_incisor", 5.4, 6.0, 9.0, 2.0, 0.95, 0.25, (), 0.0),
}


def tooth_type_for_fdi(fdi: int | None) -> ToothType:
    """Map an FDI tooth number (e.g. 36 = lower-left first molar) to a shape class."""
    if fdi is None:
        return TOOTH_TYPES["molar"]
    quadrant, pos = divmod(int(fdi), 10)
    if quadrant not in (1, 2, 3, 4) or not 1 <= pos <= 8:
        raise ValueError(f"invalid FDI tooth number: {fdi}")
    lower = quadrant in (3, 4)
    if pos >= 6:
        return TOOTH_TYPES["molar"]
    if pos >= 4:
        return TOOTH_TYPES["premolar"]
    if pos == 3:
        return TOOTH_TYPES["canine"]
    return TOOTH_TYPES["lower_incisor" if lower else "incisor"]


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class ParametricTooth:
    """Implicit anatomical crown built from a :class:`ToothType` and fitted dimensions."""

    tooth: ToothType
    A: float  # semi-axis along x (mesiodistal) at the height of contour
    B: float  # semi-axis along y (buccolingual)
    H: float  # z of the highest cusp tip above the margin plane
    cervical_x: float = 0.85  # cervical width relative to the height of contour
    cervical_y: float = 0.85
    z0: float = -2.0
    cusp_scale: float = 1.0

    def _edge_height(self) -> float:
        top = max((h for *_, h in self.tooth.cusps), default=0.0) * self.cusp_scale
        return self.H - top

    def _profile(self, z, cervical, taper):
        he = self._edge_height()
        zn = z / he
        zc = 0.35  # height of contour
        rise = cervical + (1 - cervical) * np.sin(0.5 * np.pi * np.clip(zn / zc, 0, 1))
        fall = 1 - (1 - taper) * np.clip((zn - zc) / (1 - zc), 0, 1) ** 2
        return np.where(zn < zc, rise, fall)

    def top_surface(self, x, y):
        xn, yn = x / self.A, y / self.B
        z = np.full(np.shape(x), self._edge_height(), dtype=np.float64)
        for cx, cy, h in self.tooth.cusps:
            z = z + h * self.cusp_scale * np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / (2 * 0.28 ** 2))
        if self.tooth.fossa_depth:
            z = z - self.tooth.fossa_depth * np.exp(-(xn ** 2 + (yn / 0.6) ** 2) / (2 * 0.3 ** 2))
        # Round the occlusal edge into the axial walls.
        r = np.abs(xn) ** 2 + np.abs(yn) ** 2
        return z - 0.8 * _smoothstep((r - 0.55) / 0.45)

    def inside(self, p: np.ndarray) -> np.ndarray:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        a = self.A * self._profile(z, self.cervical_x, self.tooth.top_taper_md)
        b = self.B * self._profile(z, self.cervical_y, self.tooth.top_taper_bl)
        e = self.tooth.exponent
        section = (np.abs(x) / a) ** e + (np.abs(y) / b) ** e <= 1.0
        return section & (z >= self.z0) & (z <= self.top_surface(x, y))


# --------------------------------------------------------------------------
# Statistical shape model
# --------------------------------------------------------------------------

def sphere_directions(n_theta: int, n_phi: int) -> np.ndarray:
    """Unit directions on a (phi from +z, theta around z) grid, shape (n_phi*n_theta, 3)."""
    phi = (np.arange(n_phi) + 0.5) / n_phi * np.pi
    theta = np.arange(n_theta) / n_theta * 2 * np.pi
    P, T = np.meshgrid(phi, theta, indexing="ij")
    return np.stack([np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)], -1).reshape(-1, 3)


def radial_signature(inside_fn, center, directions, r_max=2.0, steps=400) -> np.ndarray:
    """Distance from ``center`` to the first boundary crossing of an implicit shape."""
    t = np.linspace(0, r_max, steps)
    pts = center + directions[:, None, :] * t[None, :, None]
    ins = inside_fn(pts)
    outside = ~ins
    first = np.where(outside.any(axis=1), outside.argmax(axis=1), steps - 1)
    lo = t[np.maximum(first - 1, 0)]
    hi = t[first]
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        m_in = inside_fn(center + directions * mid[:, None])
        lo = np.where(m_in, mid, lo)
        hi = np.where(m_in, hi, mid)
    return 0.5 * (lo + hi)


@dataclass
class ShapeModel:
    """PCA over radial signatures of crowns normalised to a unit bounding box.

    Each training crown (occlusal = +z, mesiodistal = +x) is scaled to
    ``[-1, 1]^2 x [0, 1]`` and described by the distance from a fixed centre to
    its surface along a spherical grid of directions.  New crowns are drawn
    from, or fitted to partial observations under, the learned distribution.
    """

    n_theta: int
    n_phi: int
    center_z: float
    mean: np.ndarray
    components: np.ndarray  # (k, D)
    stddev: np.ndarray  # (k,)
    directions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.directions = sphere_directions(self.n_theta, self.n_phi)

    # ---- training -------------------------------------------------------
    @classmethod
    def fit(cls, signatures: np.ndarray, n_theta: int, n_phi: int, center_z: float,
            variance: float = 0.98, max_components: int = 20) -> "ShapeModel":
        X = np.asarray(signatures, dtype=np.float64)
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        var = s ** 2 / max(len(X) - 1, 1)
        if var.sum() <= 0:
            k = 0
        else:
            k = int(np.searchsorted(np.cumsum(var) / var.sum(), variance) + 1)
        k = max(0, min(k, max_components, len(var)))
        return cls(n_theta, n_phi, center_z, mean, vt[:k], np.sqrt(var[:k]))

    @classmethod
    def from_meshes(cls, meshes: list[Mesh], n_theta: int = 64, n_phi: int = 32,
                    center_z: float = 0.45, **kw) -> "ShapeModel":
        dirs = sphere_directions(n_theta, n_phi)
        sigs = []
        for m in meshes:
            lo, hi = m.vertices.min(0), m.vertices.max(0)
            half = np.maximum((hi[:2] - lo[:2]) / 2, 1e-6)
            mid = (hi[:2] + lo[:2]) / 2
            v = np.column_stack([(m.vertices[:, :2] - mid) / half,
                                 (m.vertices[:, 2] - lo[2]) / max(hi[2] - lo[2], 1e-6)])
            sig = raycast(np.array([[0.0, 0.0, center_z]]), dirs, Mesh(v, m.faces))
            if not np.isfinite(sig).all():
                raise ValueError("training crown is not star-shaped around its centre")
            sigs.append(sig)
        return cls.fit(np.array(sigs), n_theta, n_phi, center_z, **kw)

    # ---- inference ------------------------------------------------------
    def reconstruct(self, coeffs=None) -> np.ndarray:
        """Signature for standardised coefficients (``None`` = mean shape)."""
        if coeffs is None or len(self.stddev) == 0:
            return self.mean.copy()
        c = np.asarray(coeffs, dtype=np.float64)[: len(self.stddev)]
        return self.mean + (c * self.stddev) @ self.components[: len(c)]

    def sample(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        return rng.standard_normal(len(self.stddev)) * scale

    def fit_partial(self, mask: np.ndarray, observed: np.ndarray, regularization: float = 1.0) -> np.ndarray:
        """MAP estimate of coefficients from a subset of observed radii.

        ``mask`` selects signature entries that are known (e.g. from a pre-op
        scan of the intact tooth or a mirrored contralateral tooth); the model
        predicts the rest.  Returns standardised coefficients.
        """
        mask = np.asarray(mask, dtype=bool)
        if len(self.stddev) == 0:
            return np.zeros(0)
        W = (self.components * self.stddev[:, None])[:, mask].T  # (m, k)
        r = np.asarray(observed, dtype=np.float64) - self.mean[mask]
        lhs = W.T @ W + regularization * np.eye(W.shape[1]) * 1e-3 * len(r)
        return np.linalg.solve(lhs, W.T @ r)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, n_theta=self.n_theta, n_phi=self.n_phi, center_z=self.center_z,
                            mean=self.mean, components=self.components, stddev=self.stddev)

    @classmethod
    def load(cls, path: str | Path) -> "ShapeModel":
        d = np.load(path)
        return cls(int(d["n_theta"]), int(d["n_phi"]), float(d["center_z"]),
                   d["mean"], d["components"], d["stddev"])


@dataclass
class ModelTooth:
    """Adapts a :class:`ShapeModel` signature to the designer's ``inside`` interface."""

    model: ShapeModel
    signature: np.ndarray
    A: float
    B: float
    H: float
    z0: float = -2.0

    def inside(self, p: np.ndarray) -> np.ndarray:
        span = self.H - self.z0
        q = np.stack([p[..., 0] / self.A, p[..., 1] / self.B,
                      (p[..., 2] - self.z0) / span - self.model.center_z], axis=-1)
        r = np.linalg.norm(q, axis=-1)
        phi = np.arccos(np.clip(q[..., 2] / np.maximum(r, 1e-12), -1, 1))
        theta = np.mod(np.arctan2(q[..., 1], q[..., 0]), 2 * np.pi)
        n_t, n_p = self.model.n_theta, self.model.n_phi
        grid = self.signature.reshape(n_p, n_t)
        fp = np.clip(phi / np.pi * n_p - 0.5, 0, n_p - 1)
        ft = theta / (2 * np.pi) * n_t
        p0 = np.floor(fp).astype(int)
        p1 = np.minimum(p0 + 1, n_p - 1)
        t0 = np.floor(ft).astype(int) % n_t
        t1 = (t0 + 1) % n_t
        wp, wt = fp - np.floor(fp), ft - np.floor(ft)
        R = ((1 - wp) * ((1 - wt) * grid[p0, t0] + wt * grid[p0, t1])
             + wp * ((1 - wt) * grid[p1, t0] + wt * grid[p1, t1]))
        return r <= R


def synthetic_library(n: int, rng: np.random.Generator, tooth: ToothType = TOOTH_TYPES["molar"],
                      n_theta: int = 64, n_phi: int = 32, center_z: float = 0.45) -> np.ndarray:
    """Radial signatures of randomly perturbed parametric teeth (for demos/tests)."""
    dirs = sphere_directions(n_theta, n_phi)
    sigs = []
    for _ in range(n):
        cusps = tuple((cx + rng.normal(0, 0.04), cy + rng.normal(0, 0.04), h * rng.uniform(0.7, 1.3))
                      for cx, cy, h in tooth.cusps)
        variant = ToothType(tooth.name, tooth.mesiodistal, tooth.buccolingual, tooth.height,
                            tooth.exponent * rng.uniform(0.9, 1.1),
                            tooth.top_taper_md * rng.uniform(0.95, 1.05),
                            tooth.top_taper_bl * rng.uniform(0.95, 1.05), cusps, tooth.fossa_depth)
        A, B, H, z0 = tooth.mesiodistal / 2, tooth.buccolingual / 2, tooth.height, -1.0
        t = ParametricTooth(variant, A=A, B=B, H=H, z0=z0,
                            cervical_x=rng.uniform(0.8, 0.9), cervical_y=rng.uniform(0.8, 0.9))
        # Sample in the unit-box normalisation used by ModelTooth.
        scale = np.array([A, B, H - z0])
        offset = np.array([0.0, 0.0, z0])
        sigs.append(radial_signature(lambda q: t.inside(q * scale + offset),
                                     np.array([0.0, 0.0, center_z]), dirs))
    return np.array(sigs)
