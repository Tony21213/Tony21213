"""Preparation analysis: local tooth frame and automatic margin-line detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import Mesh


@dataclass
class ToothFrame:
    """Right-handed frame: z = insertion axis (occlusal), x = mesiodistal direction."""

    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def to_local(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p) - self.origin
        return np.stack([q @ self.x, q @ self.y, q @ self.z], axis=-1)

    def to_world(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q)
        return self.origin + q[..., :1] * self.x + q[..., 1:2] * self.y + q[..., 2:3] * self.z

    def vector_to_world(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q)
        return q[..., :1] * self.x + q[..., 1:2] * self.y + q[..., 2:3] * self.z


def make_frame(origin, axis=(0.0, 0.0, 1.0), md_direction=(1.0, 0.0, 0.0)) -> ToothFrame:
    z = np.asarray(axis, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.asarray(md_direction, dtype=np.float64)
    x = x - (x @ z) * z
    if np.linalg.norm(x) < 1e-6:  # md direction parallel to axis: pick any perpendicular
        x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    x = x / np.linalg.norm(x)
    return ToothFrame(np.asarray(origin, dtype=np.float64), x, np.cross(z, x), z)


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    return np.convolve(ext, kernel, mode="valid")[: len(values)]


def _fill_circular(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN bins around a closed loop."""
    good = ~np.isnan(values)
    if good.all():
        return values
    if not good.any():
        raise ValueError("margin detection failed: no vertices around the axis")
    n = len(values)
    idx = np.arange(n)
    gi = idx[good]
    return np.interp(idx, np.concatenate([gi - n, gi, gi + n]),
                     np.tile(values[good], 3))


def detect_margin(prep: Mesh, axis=(0.0, 0.0, 1.0), n_points: int = 128,
                  smooth: int = 5) -> np.ndarray:
    """Detect the preparation margin line of a segmented die.

    Heuristic: on a trimmed die the finish line (shoulder / chamfer edge) is the
    widest point of the tooth in every direction around the insertion axis --
    above it the preparation walls taper inwards, below it the root narrows.
    Returns ``n_points`` world-space points ordered counter-clockwise about ``axis``.
    """
    v = prep.vertices
    frame = make_frame(v.mean(axis=0), axis)
    q = frame.to_local(v)
    # Recentre on the axial centroid so the angular sweep is fair.
    frame.origin = frame.to_world(np.array([q[:, 0].mean(), q[:, 1].mean(), 0.0]))
    q = frame.to_local(v)

    # Exact profile of the die in each half-plane through the axis: intersect
    # every mesh edge with the plane and keep the widest crossing.  Unlike
    # binning vertices, this does not depend on how the scan is tessellated.
    f = prep.faces
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    a, b = q[edges[:, 0]], q[edges[:, 1]]
    ang = (np.arange(n_points) + 0.5) / n_points * 2 * np.pi - np.pi
    r_best = np.full(n_points, np.nan)
    z_best = np.full(n_points, np.nan)
    for k, th in enumerate(ang):
        u = np.array([np.cos(th), np.sin(th)])
        nrm = np.array([-u[1], u[0]])
        da, db = a[:, :2] @ nrm, b[:, :2] @ nrm
        hit = (da > 0) != (db > 0)
        if not hit.any():
            continue
        t = da[hit] / (da[hit] - db[hit])
        pts = a[hit] + t[:, None] * (b[hit] - a[hit])
        rad = pts[:, :2] @ u
        if rad.max() <= 0:
            continue
        j = rad.argmax()
        r_best[k], z_best[k] = rad[j], pts[j, 2]

    r = _circular_smooth(_fill_circular(r_best), smooth)
    z = _circular_smooth(_fill_circular(z_best), smooth)
    local = np.stack([r * np.cos(ang), r * np.sin(ang), z], axis=1)
    return frame.to_world(local)


def order_margin(points: np.ndarray, axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Sort user-supplied margin points counter-clockwise about ``axis``."""
    frame = make_frame(points.mean(axis=0), axis)
    q = frame.to_local(points)
    return points[np.argsort(np.arctan2(q[:, 1], q[:, 0]))]


def load_margin(path: str | Path) -> np.ndarray:
    """Read margin points from a whitespace/comma separated x y z text file."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].replace(",", " ").strip()
        if line:
            rows.append([float(t) for t in line.split()[:3]])
    pts = np.array(rows, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 8:
        raise ValueError(f"{path}: need at least 8 margin points")
    return pts
