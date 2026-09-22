"""Synthetic scans for demos and tests: a prepared molar die and an antagonist."""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _revolve(profile_r: np.ndarray, profile_z: np.ndarray, n_theta: int, radius_fn, z_fn) -> Mesh:
    """Closed surface of revolution; first/last profile points (r=0) become poles."""
    theta = np.arange(n_theta) / n_theta * 2 * np.pi
    rings = []
    for r, z in zip(profile_r[1:-1], profile_z[1:-1]):
        rings.append(np.stack([radius_fn(r, theta) * np.cos(theta),
                               radius_fn(r, theta) * np.sin(theta) * 1.0,
                               z_fn(z, theta)], axis=1))
    ring_v = np.concatenate(rings)
    bottom = np.array([[0.0, 0.0, profile_z[0]]])
    top = np.array([[0.0, 0.0, profile_z[-1]]])
    verts = np.concatenate([bottom, ring_v, top])
    n_rings = len(rings)
    faces = []
    i = np.arange(n_theta)
    ring = lambda k, i: 1 + k * n_theta + (i % n_theta)
    faces.append(np.stack([np.zeros(n_theta, int), ring(0, i + 1), ring(0, i)], 1))
    for k in range(n_rings - 1):
        a, b, c, d = ring(k, i), ring(k, i + 1), ring(k + 1, i + 1), ring(k + 1, i)
        faces += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    top_i = len(verts) - 1
    faces.append(np.stack([ring(n_rings - 1, i), ring(n_rings - 1, i + 1), np.full(n_theta, top_i)], 1))
    return Mesh(verts, np.concatenate(faces))


def make_prepared_molar(n_theta: int = 96, shoulder: float = 0.8, height: float = 4.6,
                        taper_deg: float = 6.0, scallop: float = 0.4) -> Mesh:
    """A chamfer-prepared molar die: root below z=0, margin at z~0, stump above."""
    r_m = 4.8
    wall = r_m - shoulder
    top_wall = wall - (height - 0.8) * np.tan(np.radians(taper_deg))
    pts = [(0.0, -4.0), (3.2, -4.0), (3.6, -3.0), (4.1, -1.8), (4.55, -0.7), (r_m, 0.0),
           (r_m - 0.35, 0.12), (wall + 0.05, 0.35), (wall, 0.6)]
    for z in np.linspace(0.9, height - 0.8, 8):
        pts.append((wall + (top_wall - wall) * (z - 0.6) / (height - 1.4), z))
    for a in np.linspace(0.25, 1.0, 6) * np.pi / 2:
        pts.append((top_wall * np.cos(a) * 0.95, height - 0.8 + 0.8 * np.sin(a)))
    pts[-1] = (0.0, height)
    r, z = np.array(pts).T

    def radius_fn(rr, th):  # slightly oval cross-section, wider buccolingually
        return rr * (1.0 + 0.06 * np.cos(2 * th + np.pi))

    def z_fn(zz, th):  # scalloped margin following the gingiva
        w = np.clip(1.0 - np.abs(zz) / 2.0, 0.0, 1.0)
        return zz + scallop * np.cos(2 * th) * w

    return _revolve(r, z, n_theta, radius_fn, z_fn)


def make_antagonist(z_level: float = 7.0, size: float = 9.0, n: int = 40) -> Mesh:
    """Closed slab whose lower (occlusal) surface is bumpy, above the preparation."""
    x = np.linspace(-size, size, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Z = z_level + 0.5 * np.cos(X * 0.7) * np.cos(Y * 0.7)
    bottom = np.stack([X, Y, Z], -1).reshape(-1, 3)
    top = np.stack([X, Y, np.full_like(X, z_level + 4.0)], -1).reshape(-1, 3)
    verts = np.concatenate([bottom, top])
    off = n * n
    idx = lambda i, j: i * n + j
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [(a, c, b), (a, d, c)]  # bottom faces point down
            faces += [(a + off, b + off, c + off), (a + off, c + off, d + off)]
    border = ([idx(i, 0) for i in range(n)] + [idx(n - 1, j) for j in range(1, n)]
              + [idx(i, n - 1) for i in range(n - 2, -1, -1)] + [idx(0, j) for j in range(n - 2, 0, -1)])
    for k in range(len(border)):
        a, b = border[k], border[(k + 1) % len(border)]
        faces += [(a, b, b + off), (a, b + off, a + off)]
    mesh = Mesh(verts, np.array(faces))
    return mesh if mesh.volume() > 0 else mesh.flipped()
