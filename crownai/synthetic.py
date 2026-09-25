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


# FDI position -> (mesiodistal, buccolingual) crown size of lower teeth, mm
_LOWER_SIZES = {1: (5.4, 5.9), 2: (5.9, 6.2), 3: (6.9, 7.5), 4: (7.0, 7.6), 5: (7.1, 8.0),
                6: (11.0, 10.3), 7: (10.5, 10.0)}


def make_lower_arch(missing: int | set[int] = 33, res: float = 0.3) -> tuple[Mesh, dict]:
    """Height-field scan of a lower dental arch with one or more teeth missing.

    ``missing`` is a single FDI number or a set of them (e.g. across both
    quadrants, to test multi-tooth gap disambiguation). Returns the jaw mesh
    (z up = occlusal) and, per FDI tooth, its crown centre, arch tangent and
    size. Teeth are rounded bumps 6 mm above a gingival ridge, placed along a
    parabolic arch with their real widths.
    """
    missing = {missing} if isinstance(missing, int) else set(missing)
    # arch: y = c - k x^2, teeth laid out along its arc length from the midline
    k, c = 0.028, 22.0
    xs_fine = np.linspace(-30, 30, 6001)
    ys_fine = c - k * xs_fine ** 2
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs_fine), np.diff(ys_fine)))])
    arc -= np.interp(0.0, xs_fine, arc)
    teeth = {}
    for side, quadrant in ((-1, 3), (1, 4)):
        s = 0.0
        for pos in range(1, 8):
            md, bl = _LOWER_SIZES[pos]
            centre_s = side * (s + md / 2)
            x = np.interp(centre_s, arc, xs_fine)
            tangent = np.array([1.0, -2 * k * x])
            tangent /= np.linalg.norm(tangent)
            teeth[quadrant * 10 + pos] = {"center": np.array([x, c - k * x ** 2]), "tangent": tangent,
                                          "md": md, "bl": bl}
            s += md + 0.05
    X, Y = np.meshgrid(np.arange(-30, 30, res), np.arange(-15, 26, res), indexing="ij")
    # gingival ridge along the arch
    d_arch = np.abs(Y - (c - k * X ** 2)) / np.sqrt(1 + (2 * k * X) ** 2)
    Z = 2.0 * np.exp(-(d_arch / 6.0) ** 2)
    for fdi, t in teeth.items():
        if fdi in missing:
            continue
        rel = np.stack([X - t["center"][0], Y - t["center"][1]], -1)
        a = rel @ t["tangent"] / (t["md"] / 2)
        b = rel @ np.array([-t["tangent"][1], t["tangent"][0]]) / (t["bl"] / 2)
        r = np.clip(1 - a ** 4, 0, None) * np.clip(1 - b ** 2, 0, None)
        cusp = 6.0 + (0.8 if fdi % 10 == 3 else 0.0) * np.exp(-(a ** 2 + b ** 2) * 3)
        Z = np.maximum(Z, 2.0 + cusp * np.sqrt(r))
    n0, n1 = X.shape
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(n0 * n1).reshape(n0, n1)
    a, b, c_, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c_], 1), np.stack([a, c_, d], 1)])
    return Mesh(verts, faces), teeth


def make_prep_at(center_xy: np.ndarray, md: float, bl: float, margin_z: float = 2.0) -> Mesh:
    """A small tapered preparation (die) standing at ``center_xy`` with its margin at ``margin_z``."""
    die = make_prepared_molar(n_theta=64, height=4.2, shoulder=0.6, scallop=0.0)
    v = die.vertices.copy()
    v[:, 0] *= (md * 0.8 / 2) / 4.8
    v[:, 1] *= (bl * 0.8 / 2) / 4.8
    v[:, 0] += center_xy[0]
    v[:, 1] += center_xy[1]
    v[:, 2] += margin_z
    return Mesh(v, die.faces)
