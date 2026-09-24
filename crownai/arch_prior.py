"""A statistical prior of arch shape, derived from exocad's bundled tooth libraries.

``crownai/data/arch_prior.json`` is *derived statistics* computed once (see
``git log`` for the extraction script), not a copy of the library geometry
itself: for every FDI tooth position, across ~900-920 different
professionally designed tooth-shape libraries bundled with exocad DentalCAD
3.3 (``library/metadata/*/upperjaw.xml`` and ``lowerjaw.xml``), it aggregates

- the distance between adjacent tooth *centres* (``adjacent_pair_center_distance_mm``) -
  unambiguous, since it needs no assumption about where one tooth ends and
  the next begins;
- how far each tooth's occlusal axis tilts from the arch's own mean "up"
  direction (``occlusal_axis_tilt``).

What this data does **not** give: a single tooth's own mesiodistal width.
The library XML only records each tooth's centre point, not its margin or
contact boundaries, so a width cannot be derived without guessing how an
adjacent-centre gap splits between the two teeth - guessing that risked
exactly the crown-oversizing bug this session already found and fixed
elsewhere (a real gap must come from actual contact-point geometry, e.g.
:func:`crownai.arch.analyze_neighbors` on a real scan). Do not use this
module to size a crown.

The tilt data's ``median_lateral_component`` also is not sign-verified
against a fixed buccal/lingual convention (the arch-chain traversal direction
differs between the left and right side of the mouth), so it is exposed for
inspection only and is not wired into :func:`crownai.design.design_crown` by
default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "arch_prior.json"


@dataclass(frozen=True)
class AxisTilt:
    """Occlusal-axis orientation of one FDI tooth, relative to the arch's own mean "up"."""

    median_tilt_from_arch_up_deg: float
    median_tangent_component: float
    median_lateral_component: float
    n: int


_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _cache


def adjacent_pair_distance_mm(a: int, b: int) -> dict | None:
    """Median/p10/p90 distance (mm) between the centres of two adjacent FDI teeth ``a``, ``b``.

    Order does not matter; returns ``None`` if this pair is not adjacent in
    either arch chain (e.g. across the midline gap on the wrong side, or two
    non-neighbouring teeth).
    """
    table = _load()["adjacent_pair_center_distance_mm"]
    return table.get(f"{a}-{b}") or table.get(f"{b}-{a}")


def axis_tilt_for_fdi(fdi: int) -> AxisTilt | None:
    """Typical occlusal-axis tilt of tooth ``fdi``, from ~900 exocad library arches."""
    entry = _load()["occlusal_axis_tilt"].get(str(fdi))
    return AxisTilt(**entry) if entry else None
