"""A crown library built from the lab's own finished crowns.

Rules and averages give correct but bland teeth: the median of eighty first
molars is no first molar anybody would make - every technician puts the
secondary grooves somewhere else, and averaging cancels them out.  A
technician's own crown, on the other hand, is a complete, lively tooth.

So, like a tooth library in exocad but made of this lab's work, every
finished crown in the archive is stored in the tooth's own coordinates
(``u`` mesial, ``w`` buccal, ``z`` up from the cervical line - the same
frame :class:`crownai.anatomy_model.PlacedAnatomy` places a model in) at
0.1 mm detail:

* the occlusal surface as a height map;
* the side walls as the outline radius around the tooth axis at every
  height (heights of contour, contacts, emergence).

For a new case :meth:`CrownLibrary.choose` picks the crown of the same tooth
type (either side - the frame is side-independent) closest in proportions,
and :class:`TemplateCrown` scales it to the case's space and height.  The
design then seats it whole into contact with the antagonist.  ``variant``
steps through the next-closest crowns, like browsing library variants.

The library holds crown geometry from patient cases: keep it on the lab's
machine (it is not meant for the repository); the index stores hashed case
ids only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .anatomy_model import PlacedAnatomy
from .margin import make_frame
from .mesh import Mesh, raycast
from .metrics import surface_samples
from .posterior_model import _textbook_posterior, profile_key

GRID_R = 7.0  # height map half-size (mm)
GRID_RES = 0.1
N_THETA = 180
Z_STEP = 0.2
LIBRARY_VERSION = 1


@dataclass
class LibraryCrown:
    id: str
    key: str  # tooth type, e.g. "lower_6"
    tooth: int
    md: float
    bl: float
    height: float
    top: np.ndarray = field(repr=False)  # (n, n) occlusal height over u, w in [-GRID_R, GRID_R]; nan = none
    outline: np.ndarray = field(repr=False)  # (n_z, N_THETA) radius around the tooth axis
    z0: float = 0.0  # height of outline row 0
    cerv_md: float = 0.0  # crown size at the cervical line (what the finish line fixes)
    cerv_bl: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)  # where the crown stood relative to its case's axis (u, w)

    def save(self, folder: Path) -> None:
        np.savez_compressed(folder / f"{self.id.replace('/', '_')}.npz", top=self.top.astype(np.float16),
                            outline=self.outline.astype(np.float16), z0=self.z0)

    @classmethod
    def load(cls, folder: Path, entry: dict) -> "LibraryCrown":
        d = np.load(folder / entry["file"])
        return cls(entry["id"], entry["key"], entry["tooth"], entry["md"], entry["bl"], entry["height"],
                   d["top"].astype(float), d["outline"].astype(float), float(d["z0"]),
                   entry.get("cerv_md", 0.0), entry.get("cerv_bl", 0.0), tuple(entry.get("offset", (0.0, 0.0))))


def canonical_crown(crown: Mesh, margin: np.ndarray, axis, md_direction, tooth: int, case_id: str = "") -> LibraryCrown:
    """A finished crown in its tooth's own coordinates, at library detail."""
    from .rules_tuning import _outline_measures, buccal_from_fdi, isolate_crown

    base = _textbook_posterior(tooth)
    crown = isolate_crown(crown, margin, axis, md_direction, tooth)
    frame = make_frame(np.asarray(margin).mean(axis=0), axis, md_direction)
    m_loc = frame.to_local(margin)
    labial = 1.0 if frame.to_local(frame.origin + buccal_from_fdi(tooth, axis, md_direction))[1] >= 0 else -1.0
    placed = PlacedAnatomy.on_margin(base, labial, np.zeros(2), m_loc, fade=0.55 * base.height)

    q = placed.to_model(frame.to_local(surface_samples(crown, 8)))
    q = q[q[:, 2] > -1.0]
    if len(q) < 500:
        raise ValueError("too little crown surface above the margin")
    H = float(np.percentile(q[:, 2], 99.8))
    meas = _outline_measures(q[q[:, 2] > 0.3], H)
    # centre the crown on itself: where it stands relative to this case's preparation
    # or implant is the case's business - the design places it by its own rules
    cu, cw = float(meas["su"]), float(meas["sw"])
    q = q - [cu, cw, 0.0]

    # occlusal height map
    g = np.arange(-GRID_R, GRID_R + 1e-9, GRID_RES)
    GU, GW = np.meshgrid(g, g, indexing="ij")
    top_z = frame.to_local(crown.vertices)[:, 2].max() + 5.0
    xy = np.column_stack([GU.ravel() + cu, labial * (GW.ravel() + cw)])
    t = raycast(frame.to_world(np.column_stack([xy, np.full(len(xy), top_z)])),
                np.broadcast_to(-frame.z, (len(xy), 3)), crown)
    top = np.full(len(xy), np.nan)
    hit = np.isfinite(t)
    top[hit] = placed.to_model(np.column_stack([xy[hit], top_z - t[hit]]))[:, 2]
    top = top.reshape(GU.shape)
    top = _fill_holes(top)  # screw-access channels of implant crowns, scan holes

    # side walls: outline radius around the axis at every height
    z0 = -1.0
    zs = np.arange(z0, H + Z_STEP, Z_STEP)
    r = np.hypot(q[:, 0], q[:, 1])
    th = np.arctan2(q[:, 1], q[:, 0])
    iz = np.clip(np.round((q[:, 2] - z0) / Z_STEP).astype(int), 0, len(zs) - 1)
    it = np.round((th + np.pi) / (2 * np.pi) * N_THETA).astype(int) % N_THETA
    outline = np.full((len(zs), N_THETA), np.nan)
    np.fmax.at(outline, (iz, it), r)
    if np.isnan(outline[len(zs) // 3]).mean() > 0.5:
        raise ValueError("crown does not surround its own axis (offset or partial)")
    ang = np.arange(N_THETA)
    for i in range(len(zs)):  # fill gaps around the circumference, then up and down
        row = outline[i]
        ok = np.isfinite(row)
        if ok.sum() >= 6:
            outline[i] = np.interp(ang, np.concatenate([ang[ok] - N_THETA, ang[ok], ang[ok] + N_THETA]),
                                   np.tile(row[ok], 3))
    for j in range(N_THETA):
        col = outline[:, j]
        ok = np.isfinite(col)
        if ok.any():
            outline[:, j] = np.interp(np.arange(len(zs)), np.flatnonzero(ok), col[ok])
    outline = np.nan_to_num(outline, nan=float(np.nanmedian(outline)))
    cerv = q[(q[:, 2] > 0.0) & (q[:, 2] < 0.8)]
    cerv_md = float(np.percentile(cerv[:, 0], 99) - np.percentile(cerv[:, 0], 1)) if len(cerv) > 50 else 0.0
    cerv_bl = float(np.percentile(cerv[:, 1], 99) - np.percentile(cerv[:, 1], 1)) if len(cerv) > 50 else 0.0
    return LibraryCrown(case_id, profile_key(tooth), int(tooth), float(meas["md"]), float(meas["bl"]), H,
                        top, outline, z0, cerv_md, cerv_bl, (cu, cw))


def _disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _fill_holes(top: np.ndarray, iters: int = 600) -> np.ndarray:
    """Fill gaps enclosed by the occlusal surface (not the area around the crown).

    A screw-access channel is not empty in the height map - looking down it the
    rays meet its walls and the ti-base seat far below.  It shows as a pit that
    is both deep and wide; fissures are deep but narrow and stay.
    """
    from scipy import ndimage

    filled = np.where(np.isnan(top), np.nanmin(top), top)
    closed = ndimage.grey_closing(filled, footprint=_disk(int(round(1.5 / GRID_RES))))
    pit = (closed - filled > 1.2) & ~np.isnan(top)
    pit = ndimage.binary_opening(pit, structure=_disk(int(round(0.6 / GRID_RES))))
    if pit.any():
        top = np.where(ndimage.binary_dilation(pit, iterations=6), np.nan, top)  # and its chamfered rim
    missing = np.isnan(top)
    outside = ndimage.label(missing)[0]
    border = set(np.unique(np.concatenate([outside[0], outside[-1], outside[:, 0], outside[:, -1]])))
    holes = missing & ~np.isin(outside, list(border - {0}))
    if not holes.any():
        return top
    out = np.where(holes, np.nanmean(top), top)
    for _ in range(iters):  # harmonic in-painting from the rim of each hole
        avg = 0.25 * (np.roll(out, 1, 0) + np.roll(out, -1, 0) + np.roll(out, 1, 1) + np.roll(out, -1, 1))
        out = np.where(holes, np.nan_to_num(avg, nan=np.nanmean(top)), out)
    return out


@dataclass
class TemplateCrown:
    """A library crown scaled to a case - an anatomy model for PlacedAnatomy."""

    src: LibraryCrown
    su: float = 1.0
    sw: float = 1.0
    sz: float = 1.0
    tip_u: float = 0.0
    tip_w: float = 0.0
    tilt: float = 0.0

    @property
    def height(self) -> float:
        return self.src.height * self.sz

    @property
    def md(self) -> float:
        return self.src.md * self.su

    @property
    def bl(self) -> float:
        return self.src.bl * self.sw

    def top_at(self, u, w):
        """Occlusal height at model (u, w); nan off the crown."""
        s = self.src
        n = s.top.shape[0]
        fu = (np.asarray(u) / self.su + GRID_R) / GRID_RES
        fw = (np.asarray(w) / self.sw + GRID_R) / GRID_RES
        i0 = np.clip(np.floor(fu).astype(int), 0, n - 2)
        j0 = np.clip(np.floor(fw).astype(int), 0, n - 2)
        tu, tw = np.clip(fu - i0, 0, 1), np.clip(fw - j0, 0, 1)
        T = s.top
        v = (T[i0, j0] * (1 - tu) * (1 - tw) + T[i0 + 1, j0] * tu * (1 - tw)
             + T[i0, j0 + 1] * (1 - tu) * tw + T[i0 + 1, j0 + 1] * tu * tw)
        inside = (fu >= 0) & (fu <= n - 1) & (fw >= 0) & (fw <= n - 1)
        return np.where(inside, v * self.sz, np.nan)

    def inside(self, p: np.ndarray) -> np.ndarray:
        s = self.src
        u, w, z = p[..., 0] / self.su, p[..., 1] / self.sw, p[..., 2] / self.sz
        r = np.hypot(u, w)
        ft = (np.arctan2(w, u) + np.pi) / (2 * np.pi) * N_THETA
        t0 = np.floor(ft).astype(int)
        tt = ft - t0
        fz = np.clip((z - s.z0) / Z_STEP, 0, s.outline.shape[0] - 1)
        z0 = np.clip(np.floor(fz).astype(int), 0, s.outline.shape[0] - 2)
        tz = fz - z0
        O = s.outline
        a, b = t0 % N_THETA, (t0 + 1) % N_THETA
        R = (O[z0, a] * (1 - tt) * (1 - tz) + O[z0, b] * tt * (1 - tz)
             + O[z0 + 1, a] * (1 - tt) * tz + O[z0 + 1, b] * tt * tz)
        top = self.top_at(p[..., 0], p[..., 1])
        return (r <= R) & (z >= -3.0) & np.isfinite(top) & (p[..., 2] <= np.nan_to_num(top, nan=-1e9))


class CrownLibrary:
    """The lab's crowns, indexed by tooth type (loaded lazily)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        index = self.folder / "index.json"
        self.entries = json.loads(index.read_text()) if index.exists() else []
        self._cache: dict[str, LibraryCrown] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def ids(self) -> set[str]:
        return {e["id"] for e in self.entries}

    def add(self, crown: LibraryCrown) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        crown.save(self.folder)
        self.entries = [e for e in self.entries if e["id"] != crown.id]
        self.entries.append({"id": crown.id, "key": crown.key, "tooth": crown.tooth, "md": round(crown.md, 3),
                             "bl": round(crown.bl, 3), "height": round(crown.height, 3),
                             "cerv_md": round(crown.cerv_md, 3), "cerv_bl": round(crown.cerv_bl, 3),
                             "offset": [round(float(v), 3) for v in crown.offset],
                             "file": f"{crown.id.replace('/', '_')}.npz", "version": LIBRARY_VERSION})
        (self.folder / "index.json").write_text(json.dumps(self.entries, indent=0))

    def get(self, entry: dict) -> LibraryCrown:
        if entry["id"] not in self._cache:
            self._cache[entry["id"]] = LibraryCrown.load(self.folder, entry)
        return self._cache[entry["id"]]

    def candidates(self, key: str, md: float | None, bl: float | None, height: float | None = None,
                   exclude=(), cervix: tuple[float, float] | None = None) -> list[dict]:
        """Library crowns of this tooth type, closest first.

        Compared by what the case fixes: the space (``md``, ``bl``), the
        neighbours' level (``height``) and the finish line (``cervix`` = its
        mesiodistal and buccolingual size), each when known.
        """
        def cost(e):
            c = 0.0
            if md:
                c += 2.0 * abs(np.log(e["md"] / md))
            if bl:
                c += abs(np.log(e["bl"] / bl))
            if height:
                c += 0.5 * abs(np.log(e["height"] / height))
            if cervix and e.get("cerv_md", 0) > 0 and e.get("cerv_bl", 0) > 0:
                c += 1.5 * abs(np.log(e["cerv_md"] / cervix[0])) + abs(np.log(e["cerv_bl"] / cervix[1]))
            return c
        pool = [e for e in self.entries if e["key"] == key and e["id"] not in set(exclude)]
        return sorted(pool, key=cost)

    def choose(self, key: str, md: float | None, bl: float | None, height: float | None = None, *,
               variant: int = 0, exclude=(), cervix: tuple[float, float] | None = None) -> LibraryCrown | None:
        c = self.candidates(key, md, bl, height, exclude, cervix)
        return self.get(c[min(variant, len(c) - 1)]) if c else None


def build_library(root, out, *, limit: int | None = None, log=print) -> CrownLibrary:
    """Put every finished premolar/molar crown under ``root`` into the library at ``out`` (resumable)."""
    import random

    from .exocad import crop_to_margin
    from .mesh import load_mesh
    from .rules_tuning import _case_id, _safe_error, archive_crowns

    lib = CrownLibrary(out)
    have = {e["id"] for e in lib.entries if e.get("version") == LIBRARY_VERSION}
    items = list(archive_crowns(root))
    if limit:
        random.Random(0).shuffle(items)
        items = items[:limit]
    log(f"{len(items)} finished premolar/molar crowns found ({len(have)} already in the library)")
    added = failed = 0
    for folder, case, tooth, ref in items:
        cid = _case_id(folder, tooth)
        if cid in have:
            continue
        info = case.teeth[tooth]
        try:
            crown = crop_to_margin(load_mesh(ref), info.margin, info.axis, radial_pad=2.5, depth=1.0)
            lib.add(canonical_crown(crown, info.margin, info.axis, info.md_direction, tooth, cid))
            added += 1
        except Exception as exc:
            failed += 1
            log(f"  skipped {cid}: {_safe_error(exc)}")
        if (added + failed) % 10 == 0:
            log(f"  {added} added, {failed} skipped")
    counts = {}
    for e in lib.entries:
        counts[e["key"]] = counts.get(e["key"], 0) + 1
    log(f"library: {len(lib)} crowns {dict(sorted(counts.items()))}")
    return lib


def check_library(root, library, *, n: int = 12, rules_profile: dict | None = None, log=print) -> list[dict]:
    """Design archive crowns with the library (never with their own crown) and compare.

    Each crown is also designed by the rules (and the lab profile, if given);
    the distance to the technician's crown is reported for all of them.
    """
    import random

    from .design import CrownParameters
    from .exocad import crop_to_margin
    from .exocad_project import design_construction_case
    from .mesh import load_mesh
    from .metrics import compare_to_reference
    from .rules_tuning import _case_id, _safe_error, archive_crowns, isolate_crown

    lib = library if isinstance(library, CrownLibrary) else CrownLibrary(library)
    items = list(archive_crowns(root))
    random.Random(1).shuffle(items)
    variants = {"rules": lambda cid: CrownParameters()}
    if rules_profile:
        variants["rules+profile"] = lambda cid: CrownParameters(rules_profile=rules_profile)
    variants["library"] = lambda cid: CrownParameters(crown_library=lib, library_exclude=(cid,))
    rows = []
    for folder, case, tooth, ref_path in items:
        if len(rows) >= n:
            break
        cid = _case_id(folder, tooth)
        info = case.teeth[tooth]
        row = {"case_id": cid, "tooth": tooth}
        try:
            ref = isolate_crown(crop_to_margin(load_mesh(ref_path), info.margin, info.axis, radial_pad=2.5, depth=1.0),
                                info.margin, info.axis, info.md_direction, tooth)
            for name, make in variants.items():
                res, _, _ = design_construction_case(folder, tooth, params=make(cid), case=case,
                                                     compare_reference=False)
                top = res.frame.to_local(res.margin)[:, 2].max()
                row[name] = compare_to_reference(res.outer[:, 4:].reshape(-1, 3), res.crown, ref, res.frame,
                                                 top)["reference_to_crown_mean_mm"]
        except Exception as exc:
            row["error"] = _safe_error(exc)
        rows.append(row)
        log(f"  {row}")
    ok = [r for r in rows if all(r.get(k) is not None for k in variants)]
    if ok:
        log("mean distance to the technician's crown over %d crowns: " % len(ok)
            + ", ".join(f"{k} {np.mean([r[k] for r in ok]):.3f} mm" for k in variants))
    return rows
