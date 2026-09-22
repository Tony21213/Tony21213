"""Neighbour analysis on a jaw scan: adjacent teeth and the contralateral tooth.

Designers copy the patient's own anatomy: the contralateral tooth (mirrored)
is the best template for shape and size, the adjacent teeth fix the
mesiodistal space, the contacts and the buccal line.  This module finds them
on an unsegmented jaw scan, working outwards from the preparation the way a
technician scans the arch by eye:

1. occlusal height map of the jaw along the insertion axis;
2. every tooth has a highest cusp / incisal point: prominent local maxima
   of the smoothed height map;
3. starting at the preparation, the peaks are linked into a chain along the
   arch on each side (next peak = nearest one that keeps the direction);
   the contact between two linked teeth is the lowest point between their
   peaks, which gives each tooth its mesiodistal extent;
4. the contralateral tooth of position p (FDI) is the (2p-1)-th tooth along
   the mesial chain; it is cut from the scan and mirrored into place.

Only local geometry is used, so partial scans, implant cases and model
bases do not disturb it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .margin import ToothFrame, make_frame
from .mesh import Mesh, nearest_distance, raycast


@dataclass(eq=False)
class ToothSegment:
    """One tooth of the arch, measured from its own crown surface."""

    peak: np.ndarray  # local xy of its highest point
    along: np.ndarray  # unit arch direction, pointing away from the preparation
    a_near: float  # extent along ``along`` relative to the peak (mm):
    a_far: float  # contact side towards / away from the preparation
    top: float  # local z of the highest point (mm above the margin centroid)
    mesh: Mesh  # crown surface patch (world coordinates)

    @property
    def md_width(self) -> float:
        return self.a_far - self.a_near

    @property
    def near(self) -> np.ndarray:
        return self.peak + self.a_near * self.along

    @property
    def center(self) -> np.ndarray:
        return self.peak + 0.5 * (self.a_near + self.a_far) * self.along


@dataclass
class NeighborAnalysis:
    frame: ToothFrame  # z = insertion axis, x = mesial direction along the ridge
    md_direction: np.ndarray  # world
    target_center: np.ndarray  # world point where the crown should be centred
    space: float | None  # mesiodistal space between the adjacent teeth (mm)
    neighbors: list[ToothSegment] = field(default_factory=list)
    contralateral: ToothSegment | None = None
    template: Mesh | None = None  # contralateral tooth mirrored into place (world)
    chains: tuple[int, int] = (0, 0)  # teeth found on the mesial / distal side

    def summary(self) -> dict:
        return {
            "teeth_found": list(self.chains),
            "adjacent_teeth": len(self.neighbors),
            "space_mm": None if self.space is None else round(self.space, 2),
            "contralateral_found": self.contralateral is not None,
            "contralateral_md_mm": round(self.contralateral.md_width, 2) if self.contralateral else None,
        }


def _masked_blur(a: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    w = valid.astype(float)
    num = ndimage.gaussian_filter(np.where(valid, a, 0.0), sigma)
    den = ndimage.gaussian_filter(w, sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)


class _HeightMap:
    def __init__(self, jaw: Mesh, frame: ToothFrame, radius: float, res: float):
        self.res = res
        q = frame.to_local(jaw.vertices)
        lo = np.maximum(q.min(0)[:2], -radius)
        hi = np.minimum(q.max(0)[:2], radius)
        self.xs = np.arange(lo[0], hi[0] + res, res)
        self.ys = np.arange(lo[1], hi[1] + res, res)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        top = q[:, 2].max() + 5.0
        origins = frame.to_world(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, top)]))
        t = raycast(origins, np.broadcast_to(-frame.z, origins.shape), jaw)
        H = (top - t).reshape(X.shape)
        self.valid = np.isfinite(H)
        self.H = _masked_blur(H, self.valid, 0.5 / res)
        self.H[~self.valid] = np.nan
        self.Hs = _masked_blur(np.where(self.valid, self.H, 0.0), self.valid, 1.0 / res)  # for peaks
        self.Hs[~self.valid] = np.nan

    def sample(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        i = np.round((xy[:, 0] - self.xs[0]) / self.res).astype(int)
        j = np.round((xy[:, 1] - self.ys[0]) / self.res).astype(int)
        ok = (i >= 0) & (i < len(self.xs)) & (j >= 0) & (j < len(self.ys))
        out = np.full(len(xy), np.nan)
        out[ok] = self.H[i[ok], j[ok]]
        return out


def _peaks(hm: _HeightMap, gum: float, spacing: float = 3.5, prominence: float = 1.0) -> np.ndarray:
    """Local xy of the highest point of every tooth crown."""
    res = hm.res
    Hs = hm.Hs
    hi = np.where(hm.valid, Hs, -1e3)
    lo = np.where(hm.valid, Hs, 1e3)
    peak = (hi == ndimage.maximum_filter(hi, size=int(spacing / res) | 1)) & hm.valid & (Hs > gum)
    peak &= Hs - ndimage.minimum_filter(lo, size=int(10.0 / res) | 1) > prominence
    lab, n = ndimage.label(peak)
    if n == 0:
        return np.zeros((0, 2))
    idx = np.array(ndimage.center_of_mass(peak, lab, range(1, n + 1)))
    return np.column_stack([np.interp(idx[:, 0], np.arange(len(hm.xs)), hm.xs),
                            np.interp(idx[:, 1], np.arange(len(hm.ys)), hm.ys)])


def _chain(peaks: np.ndarray, start: np.ndarray, first_dir: np.ndarray | None,
           max_step: float = 12.0, max_turn: float = 75.0) -> list[int]:
    """Peaks linked from ``start`` along the arch (indices, in order)."""
    order: list[int] = []
    pos, d = start, first_dir
    free = set(range(len(peaks)))
    while free:
        cand = []
        for i in free:
            v = peaks[i] - pos
            dist = np.linalg.norm(v)
            if dist < 1e-6 or dist > max_step:
                continue
            if d is not None:
                cosang = np.dot(v / dist, d)
                if cosang < np.cos(np.radians(max_turn)):
                    continue
            cand.append((dist, i))
        if not cand:
            break
        _, i = min(cand)
        v = peaks[i] - pos
        d = v / np.linalg.norm(v)
        pos = peaks[i]
        order.append(i)
        free.discard(i)
    return order


@dataclass
class _Path:
    """Smoothed path along the arch from the preparation through a chain of peaks."""

    pts: np.ndarray  # (n, 2) local xy, 0.25 mm apart
    t: np.ndarray  # distance along the path
    left: np.ndarray  # (n, 2) unit normals

    @classmethod
    def through(cls, points: np.ndarray, extend: float = 10.0, step: float = 0.25) -> "_Path":
        d_end = points[-1] - points[-2]
        points = np.vstack([points, points[-1] + extend * d_end / np.linalg.norm(d_end)])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        t = np.arange(0, cum[-1], step)
        pts = np.column_stack([np.interp(t, cum, points[:, 0]), np.interp(t, cum, points[:, 1])])
        pts = ndimage.gaussian_filter1d(pts, 1.0 / step, axis=0, mode="nearest")
        tan = np.gradient(pts, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        return cls(pts, t, np.column_stack([-tan[:, 1], tan[:, 0]]))

    def at(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.t, self.pts[:, 0]), np.interp(t, self.t, self.pts[:, 1])])


def _teeth_along(hm: _HeightMap, path: _Path, start: float, gum: float, drop: float = 2.5,
                 lateral: float = 5.0, min_width: float = 3.0) -> tuple[list[tuple[float, float, float, float]], float]:
    """Teeth along a path as (t_near, t_far, t_peak, top), and the path length.

    The crown cross-section (how wide the tooth is ``drop`` mm below its top)
    collapses at every contact: the embrasures are V-shaped notches.  That
    separates even tightly packed incisors whose edges touch.
    """
    step = path.t[1] - path.t[0]
    offs = np.arange(-lateral, lateral + 1e-9, 0.25)
    L = np.stack([hm.sample(path.pts + o * path.left) for o in offs], 1)
    top = np.where(np.isfinite(L).any(1), np.nanmax(np.where(np.isfinite(L), L, -1e3), 1), -1e3)
    run_top = ndimage.maximum_filter1d(top, size=int(6.0 / step) | 1)
    width = (np.where(np.isfinite(L), L, -1e3) > (run_top - drop)[:, None]).sum(1) * 0.25
    width[(top < gum) | (path.t < start)] = 0.0
    width = ndimage.gaussian_filter1d(width, 0.3 / step)
    teeth = []
    inside = width > 1.5
    # split runs at deep notches of the width profile
    k = int(1.5 / step)
    for i in range(k, len(width) - k):
        w0 = width[i]
        if inside[i] and w0 == width[i - k:i + k + 1].min():
            left_max = width[max(0, i - 4 * k):i].max(initial=0)
            right_max = width[i:i + 4 * k].max(initial=0)
            if w0 < 0.7 * min(left_max, right_max):
                inside[i] = False
    lab, n = ndimage.label(inside)
    for j in range(1, n + 1):
        idx = np.flatnonzero(lab == j)
        if len(idx) * step < min_width:
            continue
        pk = idx[np.argmax(top[idx])]
        teeth.append((float(path.t[idx[0]]), float(path.t[idx[-1]]), float(path.t[pk]), float(top[pk])))
    # a run cut off by the end of the path has no far contact
    return teeth, float(path.t[-1])


def _local_max(hm: _HeightMap, p: np.ndarray, radius: float) -> np.ndarray:
    X, Y = np.meshgrid(hm.xs, hm.ys, indexing="ij")
    near = (np.hypot(X - p[0], Y - p[1]) < radius) & hm.valid
    if not near.any():
        return p
    k = np.nanargmax(np.where(near, hm.Hs, -np.inf))
    return np.array([X.ravel()[k], Y.ravel()[k]])


def _patch(jaw: Mesh, q: np.ndarray, tip: np.ndarray, prev_plane, next_plane, top: float,
           depth: float, radius: float = 6.5) -> Mesh:
    """Crown surface of one tooth: between its two contact planes, around its tip."""
    xy = q[:, :2]
    keep = (q[:, 2] > top - depth) & (np.linalg.norm(xy - tip, axis=1) < radius)
    n, u = prev_plane
    keep &= (xy - n) @ u > 0
    if next_plane is not None:
        n, u = next_plane
        keep &= (xy - n) @ u < 0
    f = jaw.faces[keep[jaw.faces].all(axis=1)]
    used, inv = np.unique(f, return_inverse=True)
    return Mesh(jaw.vertices[used], inv.reshape(-1, 3))


def analyze_neighbors(jaw: Mesh | list[Mesh], margin: np.ndarray, axis=(0.0, 0.0, 1.0), *,
                      tooth: int | None = None, resolution: float = 0.25,
                      search_radius: float = 45.0) -> NeighborAnalysis:
    """Find adjacent teeth, the mesiodistal space and the contralateral tooth."""
    from .anatomy import tooth_type_for_fdi

    if isinstance(jaw, list):
        offs = np.cumsum([0] + [len(m.vertices) for m in jaw[:-1]])
        jaw = Mesh(np.concatenate([m.vertices for m in jaw]),
                   np.concatenate([m.faces + o for m, o in zip(jaw, offs)]))
    frame0 = make_frame(margin.mean(axis=0), axis)
    m_loc = frame0.to_local(margin)
    r_prep = float(np.hypot(m_loc[:, 0], m_loc[:, 1]).max())
    hm = _HeightMap(jaw, frame0, search_radius, resolution)
    q = frame0.to_local(jaw.vertices)
    gum = m_loc[:, 2].mean() + 2.0  # crowns stand at least this far above the margin
    peaks = _peaks(hm, gum)
    # a peak on the preparation itself is not a neighbour
    peaks = peaks[np.linalg.norm(peaks, axis=1) > r_prep * 0.8] if len(peaks) else peaks

    origin = np.zeros(2)
    side_a = _chain(peaks, origin, None, max_step=r_prep + 9.0)[:1]
    chains: list[list[int]] = [[], []]
    if side_a:
        d_a = peaks[side_a[0]] / np.linalg.norm(peaks[side_a[0]])
        chains[0] = side_a + _chain(peaks, peaks[side_a[0]], d_a, max_step=15.0)
        chains[0] = list(dict.fromkeys(chains[0]))
        others = [i for i in range(len(peaks)) if i not in chains[0]]
        sub = peaks[others]
        side_b = _chain(sub, origin, -d_a, max_step=r_prep + 9.0, max_turn=60)[:1]
        if side_b:
            b0 = others[side_b[0]]
            d_b = peaks[b0] / np.linalg.norm(peaks[b0])
            rest = [i for i in others if i != b0]
            chains[1] = [b0] + [rest[i] for i in _chain(peaks[rest], peaks[b0], d_b, max_step=15.0)]

    crown_depth = tooth_type_for_fdi(tooth).height + 1.5 if tooth else 10.0
    contact_band = 0.65 * (crown_depth - 1.5)  # canine contacts sit ~5 mm below the tip
    segs: list[list[ToothSegment]] = [[], []]
    for side, ch in enumerate(chains):
        if not ch:
            continue
        path = _Path.through(np.vstack([origin, peaks[ch]]))
        found, t_end = _teeth_along(hm, path, r_prep * 0.6, gum)
        # tooth tips: highest point near each run's peak on the path
        tips = []
        for t_lo, t_hi, t_pk, top in found:
            tips.append(_local_max(hm, path.at(t_pk), 2.5))
        for i, (t_lo, t_hi, t_pk, top) in enumerate(found):
            # separating planes through the contacts, normal to the tip-to-tip line
            if i == 0:
                n_prev, u_prev = path.at(t_lo), tips[0] / max(np.linalg.norm(tips[0]), 1e-6)
            else:
                n_prev = path.at(0.5 * (found[i - 1][1] + t_lo))
                u_prev = (tips[i] - tips[i - 1]) / np.linalg.norm(tips[i] - tips[i - 1])
            if i + 1 < len(found):
                n_next = path.at(0.5 * (t_hi + found[i + 1][0]))
                u_next = (tips[i + 1] - tips[i]) / np.linalg.norm(tips[i + 1] - tips[i])
            else:
                n_next, u_next = None, u_prev
            patch = _patch(jaw, q, tips[i], (n_prev, u_prev), (n_next, u_next) if n_next is not None else None,
                           top, crown_depth)
            along = u_prev + u_next
            along /= np.linalg.norm(along)
            crown = frame0.to_local(patch.vertices)
            crown = crown[crown[:, 2] > top - contact_band]  # contacts lie in the upper crown
            if len(crown) < 20:
                continue
            a = (crown[:, :2] - tips[i]) @ along
            segs[side].append(ToothSegment(tips[i], along, float(np.percentile(a, 1)),
                                           float(np.percentile(a, 99)), top, patch))

    # mesial side = the chain long enough to reach the contralateral tooth
    pos = tooth % 10 if tooth else None
    mesial = 0
    if pos:
        need = 2 * pos - 1
        if len(segs[1]) >= need > len(segs[0]) or (len(segs[1]) > len(segs[0]) and pos > 1):
            mesial = 1
    sm, sd = segs[mesial], segs[1 - mesial]
    contra = sm[2 * pos - 2] if pos and len(sm) >= 2 * pos - 1 else None

    neighbors = [s_[0] for s_ in (sm, sd) if s_ and np.linalg.norm(s_[0].near) < r_prep + 4.0]
    near_m = sm[0].near if sm and any(n is sm[0] for n in neighbors) else None
    near_d = sd[0].near if sd and any(n is sd[0] for n in neighbors) else None
    if sm:
        mes_dir = sm[0].along  # arch direction at the adjacent tooth
    elif sd:
        mes_dir = -sd[0].along
    else:
        mes_dir = np.array([1.0, 0.0])
    space = None
    if near_m is not None and near_d is not None:
        space = float(np.linalg.norm(near_m - near_d))
        center = 0.5 * (near_m + near_d)
        mes_dir = (near_m - near_d) / max(space, 1e-6)
    elif near_m is not None and contra is not None:
        center = (near_m @ mes_dir - 0.5 * contra.md_width) * mes_dir
    elif near_d is not None and contra is not None:
        center = (near_d @ mes_dir + 0.5 * contra.md_width) * mes_dir
    else:
        center = np.zeros(2)
    # buccolingually the crown stays centred over the preparation (the root)
    center = (center @ mes_dir) * mes_dir

    frame = make_frame(frame0.origin, axis, frame0.vector_to_world(np.array([mes_dir[0], mes_dir[1], 0.0])))
    template = None
    if contra is not None:
        width = space if space is not None else contra.md_width
        template = _mirror(contra, frame0, center, mes_dir, width)
        # keep the crown only: the scan below the preparation's margin level is gingiva
        t_loc = frame0.to_local(template.vertices)
        keep = t_loc[:, 2] > m_loc[:, 2].min() - 0.5
        f = template.faces[keep[template.faces].all(axis=1)]
        used, inv = np.unique(f, return_inverse=True)
        template = Mesh(template.vertices[used], inv.reshape(-1, 3))
    return NeighborAnalysis(frame, frame.x, frame0.to_world(np.array([center[0], center[1], 0.0])),
                            space, neighbors, contra, template, (len(sm), len(sd)))


def _mirror(tooth: ToothSegment, frame0: ToothFrame, center: np.ndarray, mesial: np.ndarray,
            width: float) -> Mesh:
    """Mirror image of the contralateral ``tooth`` placed at ``center``.

    Arch symmetry maps a point (along, left, z) of the contralateral tooth -
    ``along`` pointing away from the preparation along the chain - to
    (-along, left, z) in the preparation's frame whose ``along`` points
    mesially towards the chain.
    """
    q = frame0.to_local(tooth.mesh.vertices)
    a = tooth.along
    left = np.array([-a[1], a[0]])  # same handedness as ``left0`` below
    rel = q[:, :2] - tooth.center
    along, lat = rel @ a, rel @ left
    scale = float(np.clip(width / max(tooth.md_width, 1e-6), 0.85, 1.15))
    left0 = np.array([-mesial[1], mesial[0]])
    xy = center + np.outer(-along * scale, mesial) + np.outer(lat, left0)
    placed = frame0.to_world(np.column_stack([xy, q[:, 2]]))
    return Mesh(placed, tooth.mesh.faces[:, ::-1].copy())  # a mirror image flips orientation
