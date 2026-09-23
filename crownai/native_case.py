"""Locate a prepared tooth's margin on an un-segmented arch scan.

Some exocad exports never produce a dedicated preparation/die STL or a
margin-line file - only the jaw scans (``*upperjaw*.stl`` / ``*lowerjaw*.stl``)
plus the tooth numbers from the project XML. A prepared tooth has no
cusp/incisal bump, so it shows up as a gap in the chain of crown peaks along
the arch. This module finds that gap with the same occlusal-height-map
technique as :mod:`crownai.arch`, crops the scan down to just that tooth and
runs the normal margin detector (:func:`crownai.margin.detect_margin`) on the
crop - which is meaningful once the region is local to one tooth.

This is a geometric heuristic, not a segmentation model: it needs at least
one visible tooth on each side of the gap and works best when the restored
teeth in a case are contiguous. Always sanity-check ``anatomy_source`` /
``warnings`` in the report and the preview PNG before trusting the result.
"""

from __future__ import annotations

import numpy as np

from .anatomy import tooth_type_for_fdi
from .arch import _HeightMap, _peaks
from .margin import detect_margin, make_frame
from .mesh import Mesh


def _order_along_arch(peaks: np.ndarray) -> list[int]:
    """Greedy nearest-neighbour walk through every peak, starting at one arch end.

    The two mutually farthest peaks are taken as the arch ends (robust to a
    few stray points); walking from one of them visits the rest in arch order.
    """
    n = len(peaks)
    if n <= 1:
        return list(range(n))
    d = np.linalg.norm(peaks[:, None, :] - peaks[None, :, :], axis=-1)
    start = int(np.unravel_index(np.argmax(d), d.shape)[0])
    order = [start]
    remaining = set(range(n)) - {start}
    cur = start
    while remaining:
        rem = list(remaining)
        nxt = rem[int(np.argmin(np.linalg.norm(peaks[rem] - peaks[cur], axis=1)))]
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return order


def locate_prep(jaw: Mesh, tooth: int, other_teeth: list[int] = (), axis=(0.0, 0.0, 1.0),
                md_direction=(1.0, 0.0, 0.0), search_radius: float = 45.0,
                resolution: float = 0.3) -> np.ndarray:
    """Estimate the margin line of ``tooth`` on an unsegmented arch scan.

    ``other_teeth``: FDI numbers of the other teeth restored in the same case
    (also missing from the peak chain), so a run of several missing peaks can
    be split between them by expected width instead of assigned whole to one.
    Raises ``ValueError`` when the gap cannot be found confidently - callers
    should fall back to asking for an explicit margin file rather than trust
    a guess.
    """
    v = jaw.vertices
    frame0 = make_frame(0.5 * (v.min(0) + v.max(0)), axis, md_direction)
    hm = _HeightMap(jaw, frame0, search_radius, resolution)
    gum = float(np.nanpercentile(hm.H, 25)) + 2.0
    peaks = _peaks(hm, gum)
    if len(peaks) < 2:
        raise ValueError("not enough visible teeth on the scan to locate the preparation")

    order = _order_along_arch(peaks)
    path_xy = peaks[order]
    heights = hm.sample(path_xy)
    # A prepared stump still stands out of the gum and can register as a "peak",
    # but it is clearly lower than the crowns either side: take it out of the chain.
    low = np.zeros(len(path_xy), bool)
    for i in range(1, len(path_xy) - 1):
        if np.nanmean([heights[i - 1], heights[i + 1]]) - heights[i] > 1.5:
            low[i] = True
    stumps = path_xy[low]
    path_xy = path_xy[~low]
    if len(path_xy) < 2:
        raise ValueError("not enough visible teeth on the scan to locate the preparation")
    gaps = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    # Compare each gap with its neighbours, not a global median: molars are ~11 mm
    # apart peak to peak, incisors ~5.5 mm - a missing tooth doubles the local spacing.
    rel = np.empty(len(gaps))
    for i in range(len(gaps)):
        nb = [gaps[j] for j in (i - 1, i + 1) if 0 <= j < len(gaps)]
        rel[i] = gaps[i] / max(float(np.mean(nb)) if nb else 9.0, 1e-6)
    k = int(np.argmax(rel))
    if rel[k] < 1.4:
        raise ValueError("no clear gap in the tooth chain - preparation not found automatically")

    missing = sorted({int(tooth), *(int(t) for t in other_teeth)})
    pos = [t % 10 for t in missing]
    if pos != sorted(pos) and pos != sorted(pos, reverse=True):
        raise ValueError("restored teeth are not contiguous - cannot split the gap between them")
    ascending = pos == sorted(pos)
    ordered = missing if ascending else list(reversed(missing))
    widths = [max(tooth_type_for_fdi(t).mesiodistal, 1.0) for t in ordered]
    total_w = sum(widths)

    p_before, p_after = path_xy[k], path_xy[k + 1]
    gap_vec = p_after - p_before
    gap_len = float(np.linalg.norm(gap_vec))
    unit = gap_vec / max(gap_len, 1e-6)

    cum = 0.0
    center = radius = None
    for t, w in zip(ordered, widths):
        frac = (cum + w / 2) / total_w
        if t == int(tooth):
            center = p_before + unit * frac * gap_len
            radius = w / 2 + 1.5
        cum += w
    if center is None:
        raise ValueError(f"tooth {tooth} is not among the case's restored teeth")
    # A visible stump in the gap is a better centre than the width-based split
    # (neighbouring crowns differ in width, so the midpoint of their peaks drifts).
    if len(stumps):
        along = (stumps - p_before) @ unit
        across = np.abs((stumps - p_before) @ np.array([-unit[1], unit[0]]))
        inside = (along > 0) & (along < gap_len) & (across < 4.0)
        if inside.sum() == len(ordered):
            center = stumps[inside][np.argsort(along[inside])][ordered.index(int(tooth))]

    # Only the stump: the part inside the gap that stands out of the local gum level
    # (the whole crop would put the "widest point", i.e. the margin, on its border).
    q = frame0.to_local(v)
    r_xy = np.linalg.norm(q[:, :2] - center, axis=1)
    ring = (r_xy > radius) & (r_xy < radius + 2.5)
    if not ring.any():
        raise ValueError("too little geometry around the estimated preparation")
    gum_level = float(np.percentile(q[ring, 2], 30))
    keep_v = (r_xy < radius) & (q[:, 2] > gum_level + 0.3)
    faces = jaw.faces[keep_v[jaw.faces].all(axis=1)]
    if len(faces) < 20:
        raise ValueError("no preparation stump found in the gap")
    used, inv = np.unique(faces, return_inverse=True)
    patch = Mesh(jaw.vertices[used], inv.reshape(-1, 3))
    # keep the piece of surface the stump belongs to, not the flanks of the neighbours
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = patch.faces
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = len(patch.vertices)
    _, labels = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)),
                                     directed=False)
    pq = frame0.to_local(patch.vertices)
    top = int(np.argmin(np.linalg.norm(pq[:, :2] - center, axis=1) - 0.05 * pq[:, 2]))
    keep = labels == labels[top]
    f = f[keep[f].all(axis=1)]
    used, inv = np.unique(f, return_inverse=True)
    return detect_margin(Mesh(patch.vertices[used], inv.reshape(-1, 3)), axis=axis)
