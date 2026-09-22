"""Functional occlusion: Slavicek's sequential guidance and bite correction.

Sequential guidance (R. Slavicek)
---------------------------------
In excursive movements the teeth guide the mandible in a sequence: the
steepness of the guiding surfaces increases from the last molar towards the
canine, every tooth a few degrees steeper than the one distal to it, and all
of them steeper than the condylar path.  The posterior teeth therefore
separate one after the other and protect the joints, while centric stops
keep the jaw supported in maximal intercuspation.

For a crown this means:

* centric: the cusp tips reach the antagonist (point contacts);
* protrusion, laterotrusion and mediotrusion: moving the antagonist along
  the path this tooth position allows (the "occlusal compass" directions,
  opening at the sequential guidance angle), the crown must not collide -
  its surfaces are carved to be no steeper than that angle.

The movement directions at the tooth come from the arch orientation
(``anterior`` and ``buccal``), the angles from the condylar inclination and
the tooth position; measured values from axiography (e.g. CADIAX) can be
entered in place of the mean values.

Bite correction
---------------
Intraoral scans often come with the jaws slightly interpenetrating or apart.
:func:`fix_bite` moves the antagonist rigidly along the occlusal axis and
tilts it a little until the jaws touch without penetrating, supported on
contacts spread over the arch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, raycast

# degrees of rotation of the mesial direction towards buccal that give the
# anterior (protrusive) direction at each tooth position, when the midline
# is unknown: incisors move labially, molars mesially.
_ANTERIOR_TURN = {1: 85.0, 2: 70.0, 3: 45.0, 4: 25.0, 5: 15.0, 6: 8.0, 7: 5.0, 8: 3.0}


@dataclass
class SlavicekConcept:
    """Sequential guidance parameters (mean values unless measured)."""

    condylar_inclination: float = 35.0  # deg, sagittal condylar path vs occlusal plane
    bennett_angle: float = 10.0  # deg, medial component of the non-working condyle path
    fischer_angle: float = 5.0  # deg, non-working path steeper than the protrusive path
    sequence_step: float = 5.0  # deg steeper per tooth from the second molar to the canine
    excursion: float = 3.0  # mm of movement simulated
    steps: int = 8
    centric_contacts: bool = True
    patient_guidance: bool = True  # use the relief-guided path from the scans when available

    def guidance_angle(self, fdi: int | None, movement: str) -> float:
        """Opening angle allowed at this tooth for a movement (deg vs occlusal plane)."""
        pos = min(fdi % 10 if fdi else 6, 7)
        seq = self.sequence_step * (7 - max(pos, 3))  # canine is the steepest guide
        if pos < 3:  # incisors guide protrusion steeper still
            seq += self.sequence_step * (3 - pos) * 0.5
        base = self.condylar_inclination + seq
        if movement == "mediotrusion":
            return base + self.fischer_angle
        return base

    def movements(self, fdi: int | None, anterior: np.ndarray, buccal: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        """(name, horizontal direction of the mandible at this tooth, opening angle)."""
        medial = -buccal
        b = np.radians(40.0 + self.bennett_angle)
        nonwork = anterior * np.cos(b) + medial * np.sin(b)
        return [
            ("protrusion", anterior, self.guidance_angle(fdi, "protrusion")),
            ("laterotrusion", buccal, self.guidance_angle(fdi, "laterotrusion")),
            ("mediotrusion", nonwork / np.linalg.norm(nonwork), self.guidance_angle(fdi, "mediotrusion")),
        ]


def arch_directions(md_direction: np.ndarray, axis: np.ndarray, buccal_hint: np.ndarray | None,
                    fdi: int | None, midline_normal: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Anterior and buccal unit vectors in the occlusal plane at a tooth."""
    z = axis / np.linalg.norm(axis)
    x = md_direction - (md_direction @ z) * z
    x /= np.linalg.norm(x)
    left = np.cross(z, x)
    buccal = left if buccal_hint is None or buccal_hint @ left >= 0 else -left
    if midline_normal is not None:
        ant = np.cross(z, midline_normal)
        ant -= (ant @ z) * z
        ant /= np.linalg.norm(ant)
        if ant @ (x + buccal) < 0:
            ant = -ant
        return ant, buccal
    turn = np.radians(_ANTERIOR_TURN.get(fdi % 10 if fdi else 6, 10.0))
    ant = x * np.cos(turn) + buccal * np.sin(turn)
    return ant / np.linalg.norm(ant), buccal


def _gaps(points: np.ndarray, antagonist: Mesh, z: np.ndarray, shift: np.ndarray, back: float = 30.0) -> np.ndarray:
    """Distance from each point up (along ``z``) to the antagonist moved by ``shift``."""
    origins = points - shift - z * back
    t = raycast(origins, np.broadcast_to(z, points.shape), antagonist)
    return t - back


def raise_to_centric_contacts(outer: np.ndarray, antagonist: Mesh, z: np.ndarray, region: np.ndarray,
                              clearance: float, reach: float = 1.0, spread: float = 0.35) -> tuple[np.ndarray, int]:
    """Bring the crown's closest cusp regions up into centric contact.

    Only the points already within ``reach`` of the antagonist move; the
    closest ones come fully into contact and the rest follow less, so the
    contacts stay small (cusp tip to fossa), not flattened plateaus.
    """
    pts = outer.reshape(-1, 3)
    g = _gaps(pts, antagonist, z, np.zeros(3)).reshape(outer.shape[:2])
    g = np.where(region & np.isfinite(g), g, np.inf)
    if not np.isfinite(g).any():
        return outer, 0
    gmin = float(np.min(g))
    if gmin <= clearance or gmin > reach:
        return outer, int((g <= clearance + 0.02).sum())
    lift = (gmin - clearance) * np.exp(-((g - gmin) / spread) ** 2)
    lift = np.where(np.isfinite(g), lift, 0.0)
    out = outer + lift[..., None] * z
    g2 = _gaps(out.reshape(-1, 3), antagonist, z, np.zeros(3)).reshape(outer.shape[:2])
    return out, int(((g2 <= clearance + 0.05) & region).sum())


def carve_excursions(outer: np.ndarray, antagonist: Mesh, z: np.ndarray, movements, *, crown_on_mandible: bool,
                     clearance: float, excursion: float, steps: int, region: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove every interference along the simulated excursive paths.

    The antagonist is swept along each movement relative to the crown
    (horizontally opposite to the crown's jaw, opening at the movement's
    guidance angle); crown points are lowered below the swept envelope.
    """
    pts = outer.reshape(-1, 3)
    lower = np.zeros(len(pts))
    report = {}
    for mv in movements:
        name, horiz, angle = mv[:3]
        path = mv[3] if len(mv) > 3 else None  # GuidedPath: the patient's own guidance
        rel = -horiz if crown_on_mandible else horiz
        worst = 0.0
        for s in np.linspace(excursion / steps, excursion, steps):
            opening = s * np.tan(np.radians(angle)) if path is None else float(np.interp(s, path.s, path.lift))
            shift = rel * s + z * opening
            g = _gaps(pts, antagonist, z, shift)
            need = np.where(np.isfinite(g) & region.reshape(-1), clearance - g, 0.0)
            need = np.clip(need, 0.0, None)
            worst = max(worst, float(need.max(initial=0.0)))
            lower = np.maximum(lower, need)
        report[name] = {"guidance_angle_deg": round(float(angle if path is None else path.angle()), 1),
                        "guidance": "patient (relief-guided)" if path is not None else "concept",
                        "interference_removed_mm": round(worst, 3)}
    return outer - (lower.reshape(outer.shape[:2]))[..., None] * z, report


def excursion_interference(points: np.ndarray, antagonist: Mesh, z: np.ndarray, movements, *,
                           crown_on_mandible: bool, clearance: float, excursion: float, steps: int) -> float:
    """Deepest remaining collision along the simulated movements (mm, 0 = none)."""
    worst = 0.0
    for mv in movements:
        _, horiz, angle = mv[:3]
        path = mv[3] if len(mv) > 3 else None
        rel = -horiz if crown_on_mandible else horiz
        for s in np.linspace(excursion / steps, excursion, steps):
            opening = s * np.tan(np.radians(angle)) if path is None else float(np.interp(s, path.s, path.lift))
            g = _gaps(points, antagonist, z, rel * s + z * opening)
            g = g[np.isfinite(g)]
            if len(g):
                worst = max(worst, float(clearance - g.min()))
    return max(worst, 0.0)


# --------------------------------------------------------------------------
# Bite correction
# --------------------------------------------------------------------------

def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def fix_bite(jaw: Mesh, antagonist: Mesh, axis=(0.0, 0.0, 1.0), *, contact: float = 0.0,
             max_tilt_deg: float = 2.0, sample: int = 6000, seed: int = 0) -> tuple[Mesh, dict]:
    """Rigidly move ``antagonist`` so the jaws meet without penetrating.

    ``axis`` points from ``jaw`` towards ``antagonist``.  A small tilt search
    (about two horizontal axes through the contact area) maximises the
    number of well-spread contacts; then the antagonist is translated along
    the axis so its deepest point touches at ``contact`` mm.
    """
    z = np.asarray(axis, dtype=float)
    z /= np.linalg.norm(z)
    rng = np.random.default_rng(seed)
    pts = jaw.vertices[rng.choice(len(jaw.vertices), min(sample, len(jaw.vertices)), replace=False)]
    g0 = _gaps(pts, antagonist, z, np.zeros(3))
    ok = np.isfinite(g0)
    if ok.sum() < 20:
        raise ValueError("the jaws do not overlap in occlusal view")
    pts, g0 = pts[ok], g0[ok]
    near = pts[g0 < np.percentile(g0, 20)]  # occluding area
    pivot = near.mean(0)
    x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    def evaluate(R):
        moved = Mesh((antagonist.vertices - pivot) @ R.T + pivot, antagonist.faces)
        g = _gaps(pts, moved, z, np.zeros(3))
        g = g[np.isfinite(g)]
        shift = contact - g.min()  # translate along z so the closest point touches
        g = g + shift
        close = g < contact + 0.1
        return shift, int(close.sum()), float(np.mean(g[g < np.percentile(g, 30)]))

    best = None
    angles = np.radians(np.linspace(-max_tilt_deg, max_tilt_deg, 5))
    for ax_ in angles:
        for ay in angles:
            R = _rotation(x, ax_) @ _rotation(y, ay)
            shift, n_contacts, spread = evaluate(R)
            key = (n_contacts, -spread)
            if best is None or key > best[0]:
                best = (key, R, shift)
    n_c, R, shift = best[0][0], best[1], best[2]
    verts = (antagonist.vertices - pivot) @ R.T + pivot + z * shift
    fixed = Mesh(verts, antagonist.faces.copy())
    initial_pen = float(max(0.0, -g0.min()))
    return fixed, {"initial_penetration_mm": round(initial_pen, 3),
                   "initial_gap_mm": round(float(max(0.0, g0.min())), 3),
                   "moved_along_axis_mm": round(float(shift), 3),
                   "tilt_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
                   "contact_points": int(n_c)}


# --------------------------------------------------------------------------
# Relief-guided jaw motion (patient-specific guidance from the scans alone)
# --------------------------------------------------------------------------

@dataclass
class GuidedPath:
    """Mandible path along one movement, guided by the existing teeth.

    ``lift[k]`` is how far the mandible must open (along the occlusal axis)
    after moving ``s[k]`` mm horizontally so that no tooth penetrates - the
    path the dentition itself dictates (canine guidance, incisal guidance or
    group function), found by collision detection as in occlusal fingerprint
    analysis.  ``guides[k]`` is the world point that carries the contact.
    """

    name: str
    direction: np.ndarray  # horizontal mandible direction (world)
    s: np.ndarray
    lift: np.ndarray
    guides: np.ndarray

    def angle(self, upto: float | None = None) -> float:
        """Mean guidance angle (deg vs occlusal plane) over the first ``upto`` mm."""
        k = len(self.s) if upto is None else max(1, int(np.searchsorted(self.s, upto, side="right")))
        return float(np.degrees(np.arctan2(self.lift[k - 1], self.s[k - 1])))

    def summary(self) -> dict:
        return {"guidance_angle_deg": round(self.angle(), 1),
                "initial_angle_deg": round(self.angle(1.0), 1),
                "opening_mm": round(float(self.lift[-1]), 2)}


def guided_path(mandible: Mesh, maxilla: Mesh, z: np.ndarray, direction: np.ndarray, name: str = "",
                excursion: float = 3.0, steps: int = 12, contact: float = 0.0,
                sample: int = 8000, seed: int = 0) -> GuidedPath:
    """Slide the mandible along ``direction``; open it just enough to stay out of the maxilla.

    ``z`` points from the mandible to the maxilla (occlusal direction of the
    lower teeth).  Only the mandible's occluding surface matters, so a random
    sample of its vertices facing the maxilla is used.
    """
    z = np.asarray(z, float) / np.linalg.norm(z)
    rng = np.random.default_rng(seed)
    pts = mandible.vertices
    g0 = _gaps(pts, maxilla, z, np.zeros(3))
    near = np.flatnonzero(np.isfinite(g0) & (g0 < 3.0))  # the occluding surface
    if len(near) == 0:
        raise ValueError("the jaws do not occlude: no mandibular surface within 3 mm of the maxilla")
    pts = pts[rng.choice(near, min(sample, len(near)), replace=False)]
    s_all = np.linspace(excursion / steps, excursion, steps)
    lifts, guides = [], []
    base = np.nanmin(_gaps(pts, maxilla, z, np.zeros(3)))
    for s in s_all:
        # the maxilla moves the other way relative to the mandible
        g = _gaps(pts, maxilla, z, -direction * s)
        ok = np.isfinite(g)
        if not ok.any():
            lifts.append(lifts[-1] if lifts else 0.0)
            guides.append(guides[-1] if guides else pts.mean(0))
            continue
        k = int(np.argmin(np.where(ok, g, np.inf)))
        lifts.append(max(0.0, min(base, contact) - g[k]))
        guides.append(pts[k] + direction * s)
    lift = np.maximum.accumulate(np.array(lifts))  # the jaw does not close again mid-excursion
    return GuidedPath(name, direction, s_all, lift, np.array(guides))


def patient_guidance(mandible: Mesh, maxilla: Mesh, z: np.ndarray, movements, *,
                     excursion: float = 3.0, steps: int = 12) -> list[GuidedPath]:
    """Relief-guided paths for the (name, direction, _) movements of a concept."""
    return [guided_path(mandible, maxilla, z, d, name, excursion, steps) for name, d, _ in movements]
