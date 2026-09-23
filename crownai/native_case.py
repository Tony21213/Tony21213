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

    path_xy = peaks[_order_along_arch(peaks)]
    gaps = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    typical = float(np.median(gaps[gaps < 12.0])) if np.any(gaps < 12.0) else 9.0
    k = int(np.argmax(gaps))
    if gaps[k] < 1.4 * typical:
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

    z = hm.sample(center[None, :])[0]
    if not np.isfinite(z):
        z = float(np.nanmean(hm.H))
    q = frame0.to_local(v)
    keep_v = np.linalg.norm(q[:, :2] - center, axis=1) < radius + 1.5
    faces = jaw.faces[keep_v[jaw.faces].all(axis=1)]
    if len(faces) < 20:
        raise ValueError("too little geometry around the estimated preparation")
    used, inv = np.unique(faces, return_inverse=True)
    patch = Mesh(jaw.vertices[used], inv.reshape(-1, 3))
    return detect_margin(patch, axis=axis)
