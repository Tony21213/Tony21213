"""Locate prepared teeth's margins on an un-segmented arch scan.

Some exocad exports never produce a dedicated preparation/die STL or a
margin-line file - only the jaw scans (``*upperjaw*.stl`` / ``*lowerjaw*.stl``)
plus the tooth numbers from the project XML. A prepared tooth has no
cusp/incisal bump, so it shows up as a gap in the chain of crown peaks along
the arch. This module finds that gap with the same occlusal-height-map
technique as :mod:`crownai.arch`, crops the scan down to just that tooth and
runs the normal margin detector (:func:`crownai.margin.detect_margin`) on the
crop - which is meaningful once the region is local to one tooth.

This is a geometric heuristic, not a segmentation model: it needs at least
one visible tooth on each side of every gap. **Always resolve every tooth of
one jaw in a single :func:`locate_preps` call, never one at a time** - a call
that only sees one requested tooth has no way to tell whether the single
most obvious gap in the whole arch actually belongs to that tooth, and will
confidently mislabel every additional tooth with the same gap (this shipped
once: two different quadrants' teeth came back as bit-identical crowns).
``locate_preps`` guards against exactly that: it never returns two different
teeth at the same spot - it either places every requested tooth correctly or
raises, rather than guess.

Known limitation: when one quadrant has *more than one* missing tooth that
are not immediate neighbours (e.g. 34 and 36, with 35 still present between
them), telling its gaps apart from an equally-strong gap belonging to the
*other* requested quadrant is not always reliable - nothing in a symmetric
jaw's geometry says which physical side is which quadrant, so this is
resolved by signal strength, and a real prep's strength can occasionally lose
to an unrelated coincidence elsewhere in the arch. The collision check still
catches the case where this goes wrong (it raises instead of silently
swapping two teeth), but it may also raise on a case a careful human would
have resolved correctly. If that happens, resolve the quadrants separately
across two calls, each time also passing every *other* real tooth known to
be missing in ``other_teeth``/``teeth`` so the gap count still lines up.
Always sanity-check ``anatomy_source`` / ``warnings`` in the report and the
preview PNG before trusting any result from this module.
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


def _arch_chain(jaw: Mesh, frame0, search_radius: float, resolution: float):
    """Peak chain along the arch, with likely prepared stumps split out separately."""
    hm = _HeightMap(jaw, frame0, search_radius, resolution)
    gum = float(np.nanpercentile(hm.H, 25)) + 2.0
    peaks = _peaks(hm, gum)
    if len(peaks) < 2:
        raise ValueError("not enough visible teeth on the scan to locate the preparation(s)")
    path_xy = peaks[_order_along_arch(peaks)]
    heights = hm.sample(path_xy)
    # A prepared stump still stands out of the gum and can register as a "peak",
    # but it is clearly lower than the crowns either side: take it out of the chain.
    low = np.zeros(len(path_xy), bool)
    for i in range(1, len(path_xy) - 1):
        if np.nanmean([heights[i - 1], heights[i + 1]]) - heights[i] > 1.5:
            low[i] = True
    stumps, path_xy = path_xy[low], path_xy[~low]
    if len(path_xy) < 2:
        raise ValueError("not enough visible teeth on the scan to locate the preparation(s)")
    gaps = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    # A gap's own immediate neighbours are not a safe reference: when two missing
    # teeth sit close together (one real tooth apart), each one's neighbour gap is
    # itself already inflated by the other, which dilutes both below any fixed
    # threshold. A plain global median has the opposite failure mode (molars sit
    # ~11 mm apart peak to peak vs ~5.5 mm for incisors even with nothing missing,
    # so it drifts every time the arch's tooth-size mix changes). The median of
    # just the smaller ~70% of gaps stays close to "normal local spacing" even
    # with several genuine gaps present, as long as they are a minority.
    sorted_gaps = np.sort(gaps)
    reference = float(np.median(sorted_gaps[: max(3, int(len(gaps) * 0.7))]))
    rel = gaps / max(reference, 1e-6)
    return hm, path_xy, stumps, gaps, rel


def _resolve_side(path_xy: np.ndarray, gaps: np.ndarray, rel: np.ndarray, candidate_idx: list[int],
                  teeth: list[int], toward_midline: int) -> tuple[dict[int, tuple[np.ndarray, float]], float]:
    """Match significant gaps on one side of the midline to this quadrant's requested teeth.

    Returns ``({tooth: (centre_xy, radius)}, confidence)`` - confidence is the
    weakest ``rel`` among the gaps used, so callers can prefer a stronger match
    when both sides of the midline happen to produce *some* match (see
    :func:`locate_preps`). ``teeth`` must already be sorted ascending by FDI
    position (closest to the midline first). ``candidate_idx`` is walked from
    the midline outward regardless of ``toward_midline`` (+1 if increasing
    index in ``path_xy`` moves outward, i.e. this is the side after the
    midline in the arch chain; -1 if increasing index moves *toward* the
    midline, i.e. this is the side before it) - the two halves of one chain
    run in opposite directions relative to "distance from the midline", and
    getting this backwards silently swaps which tooth a gap is assigned to.
    """
    ordered = candidate_idx if toward_midline > 0 else list(reversed(candidate_idx))
    significant = [i for i in ordered if rel[i] >= 1.4]
    if not significant:
        raise ValueError(f"no clear gap found for teeth {teeth} - preparation(s) not found automatically")
    if len(significant) > len(teeth) > 1:
        # more candidates than requested teeth: keep the strongest len(teeth) of
        # them (a spurious gap next to some unrelated merged peak elsewhere in
        # this same candidate list can coexist with the real one(s)), then
        # restore midline order for the tooth-position matching below.
        significant = sorted(sorted(significant, key=lambda i: rel[i], reverse=True)[: len(teeth)],
                             key=ordered.index)
    elif len(significant) > 1 and len(teeth) == 1:
        significant = [max(significant, key=lambda i: rel[i])]
    confidence = float(min(rel[i] for i in significant))

    if len(significant) == len(teeth):
        # one gap per tooth: match by order along the arch, both already ascending
        # from the midline outward.
        out = {}
        for k, tooth in zip(significant, teeth):
            p_before, p_after = path_xy[k], path_xy[k + 1]
            w = max(tooth_type_for_fdi(tooth).mesiodistal, 1.0)
            out[tooth] = (0.5 * (p_before + p_after), w / 2 + 1.5)
        return out, confidence

    if len(significant) == 1 and len(teeth) > 1:
        # a contiguous run of missing teeth merges into one gap - split it by
        # each tooth's expected width, same as when only one tooth is missing.
        pos = [t % 10 for t in teeth]
        if pos != sorted(pos):
            raise ValueError(f"teeth {teeth} are not contiguous but only one merged gap was found")
        k = significant[0]
        # walk from the midline-side end of the gap outward, so teeth (already
        # sorted midline-first) get assigned in the right physical order.
        near, far = (path_xy[k], path_xy[k + 1]) if toward_midline > 0 else (path_xy[k + 1], path_xy[k])
        gap_vec = far - near
        gap_len = float(np.linalg.norm(gap_vec))
        unit = gap_vec / max(gap_len, 1e-6)
        widths = [max(tooth_type_for_fdi(t).mesiodistal, 1.0) for t in teeth]
        total_w = sum(widths)
        out, cum = {}, 0.0
        for tooth, w in zip(teeth, widths):
            frac = (cum + w / 2) / total_w
            out[tooth] = (near + unit * frac * gap_len, w / 2 + 1.5)
            cum += w
        return out, confidence

    raise ValueError(f"found {len(significant)} candidate gap(s) for {len(teeth)} requested tooth/teeth "
                     f"{teeth} - cannot match them up without guessing")


def _extract_margin(jaw: Mesh, frame0, stumps: np.ndarray, center: np.ndarray, radius: float,
                    axis) -> np.ndarray:
    """Crop the stump at ``center`` out of the jaw and detect its margin line."""
    # A visible stump in the gap is a better centre than the width-based split
    # (neighbouring crowns differ in width, so the midpoint drifts).
    if len(stumps):
        d = np.linalg.norm(stumps - center, axis=1)
        j = int(np.argmin(d))
        if d[j] < radius:
            center = stumps[j]

    q = frame0.to_local(jaw.vertices)
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

    # keep only the connected piece of surface the stump belongs to, not the flanks
    # of the neighbours that a purely radial crop can also catch.
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


def locate_preps(jaw: Mesh, teeth: list[int], axis=(0.0, 0.0, 1.0), md_direction=(1.0, 0.0, 0.0),
                 search_radius: float = 45.0, resolution: float = 0.3) -> dict[int, np.ndarray]:
    """Estimate the margin line of every tooth in ``teeth`` on one unsegmented arch scan.

    ``teeth`` may span at most two FDI quadrants (i.e. one jaw's worth). Each
    quadrant's teeth are matched against the strongest gap(s) anywhere in the
    chain independently (there is no reliable way to tell from geometry alone
    which physical side of a symmetric jaw is which quadrant), and then - the
    actual bug this function exists to catch - every quadrant's placement is
    checked against every other's: two different quadrants landing on the
    same physical spot is anatomically impossible and means the match is
    wrong, not that both teeth really are there.
    """
    teeth = [int(t) for t in teeth]
    quadrants: dict[int, list[int]] = {}
    for t in teeth:
        quadrants.setdefault(t // 10, []).append(t)
    for q in quadrants:
        quadrants[q] = sorted(quadrants[q], key=lambda t: t % 10)
    if len(quadrants) > 2:
        raise ValueError("locate_preps only supports teeth from at most two quadrants (one jaw) at a time")

    v = jaw.vertices
    frame0 = make_frame(0.5 * (v.min(0) + v.max(0)), axis, md_direction)
    _, path_xy, stumps, gaps, rel = _arch_chain(jaw, frame0, search_radius, resolution)
    # a tooth's own expected width, in arch-chain units: two placements this
    # close together cannot belong to two different quadrants.
    min_sep = min(tooth_type_for_fdi(t).mesiodistal for t in teeth) * 0.75

    def _best(qteeth: list[int], candidates: list[int]) -> tuple[dict[int, tuple[np.ndarray, float]], float] | None:
        found = []
        for direction in (1, -1):
            try:
                found.append(_resolve_side(path_xy, gaps, rel, candidates, qteeth, direction))
            except ValueError:
                continue
        return max(found, key=lambda r: r[1]) if found else None

    placements: dict[int, tuple[np.ndarray, float]] = {}
    if len(quadrants) == 1:
        # a single quadrant's teeth can be anywhere in the chain - there is no
        # second quadrant to disambiguate against, so search the whole thing
        # rather than guessing a midline split that could cut its real gap in
        # half. Try both chain directions and keep the stronger fit.
        (qteeth,), = [quadrants.values()]
        best = _best(qteeth, list(range(len(gaps))))
        if best is None:
            raise ValueError(f"no gap found for teeth {qteeth}")
        placements.update(best[0])
    else:
        # two quadrants always sit on opposite sides of the midline - the
        # smallest peak-to-peak spacing in the whole arch, since the two
        # central incisors are the narrowest teeth. Rather than letting each
        # quadrant pick its own best-scoring side independently (which lets an
        # unrelated, coincidentally-strong gap in the wrong half outscore the
        # real gap and steal both quadrants onto the same side), try both ways
        # of PAIRING the two quadrants to the two halves and keep whichever
        # pairing is the one where both sides actually resolve.
        midline = int(np.argmin(gaps))
        left, right = list(range(0, midline + 1)), list(range(midline + 1, len(gaps)))
        (qa, teeth_a), (qb, teeth_b) = quadrants.items()
        pairings = []
        for side_a, side_b in ((left, right), (right, left)):
            best_a, best_b = _best(teeth_a, side_a), _best(teeth_b, side_b)
            if best_a is not None and best_b is not None:
                pairings.append((min(best_a[1], best_b[1]), best_a[0], best_b[0]))
        if not pairings:
            raise ValueError(f"could not find gaps for both {teeth_a} and {teeth_b} on opposite sides "
                             f"of the midline")
        pairings.sort(key=lambda p: p[0], reverse=True)
        if len(pairings) > 1 and pairings[0][0] < 1.3 * pairings[1][0]:
            raise ValueError(f"teeth {teeth_a} and {teeth_b} match either side of the midline about "
                             f"equally well - ambiguous")
        placements.update(pairings[0][1])
        placements.update(pairings[0][2])

    for tooth, (center, _radius) in placements.items():
        for other_tooth, (other_center, _) in placements.items():
            if other_tooth != tooth and np.linalg.norm(center - other_center) < min_sep:
                raise ValueError(f"tooth {tooth} and tooth {other_tooth} resolved to the same spot "
                                 f"on the arch - the gap detector cannot tell them apart here")

    return {tooth: _extract_margin(jaw, frame0, stumps, center, radius, axis)
           for tooth, (center, radius) in placements.items()}


def locate_prep(jaw: Mesh, tooth: int, other_teeth: list[int] = (), axis=(0.0, 0.0, 1.0),
                md_direction=(1.0, 0.0, 0.0), search_radius: float = 45.0,
                resolution: float = 0.3) -> np.ndarray:
    """Estimate the margin line of a single ``tooth``.

    Convenience wrapper around :func:`locate_preps` for the common single- or
    same-quadrant case. If other teeth are missing **anywhere else on the same
    jaw** (including a different quadrant) and not listed in ``other_teeth``,
    prefer calling :func:`locate_preps` directly with the full list - this
    wrapper has no way to notice a gap it finds actually belongs to one of
    those other teeth instead.
    """
    teeth = sorted({int(tooth), *(int(t) for t in other_teeth)})
    return locate_preps(jaw, teeth, axis=axis, md_direction=md_direction,
                        search_radius=search_radius, resolution=resolution)[int(tooth)]
