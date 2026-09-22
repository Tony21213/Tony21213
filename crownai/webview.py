"""Import meshes from an exocad webview HTML export.

exocad's "webview" HTML embeds the whole scene as base64 in
``DentalWebGL.m_Data``: a small binary scene description (lights, views,
annotations) followed by one OpenCTM-compressed mesh per scene object, each
with its world matrix and its name in the exocad object tree ("tree paths").

Password-protected exports are encrypted; export the case without a
password to use it here.
"""

from __future__ import annotations

import base64
import lzma
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import Mesh, merge_vertices


@dataclass
class WebviewObject:
    name: str  # full tree path, e.g. "Scans / Lower jaw"
    mesh: Mesh
    color: tuple[int, int, int]
    visible: bool = True
    tree: list[str] = field(default_factory=list)


class _Reader:
    def __init__(self, data: bytes):
        self.b = data
        self.o = 0

    def int(self) -> int:
        (v,) = struct.unpack_from("<i", self.b, self.o)
        self.o += 4
        return v

    def uint(self) -> int:
        (v,) = struct.unpack_from("<I", self.b, self.o)
        self.o += 4
        return v

    def float(self) -> float:
        (v,) = struct.unpack_from("<f", self.b, self.o)
        self.o += 4
        return v

    def bool(self) -> bool:
        return self.int() == 1

    def bytes(self, n: int, pad: bool = True) -> bytes:
        out = self.b[self.o:self.o + n]
        self.o += 4 * ((n + 3) // 4) if pad else n
        return out

    def string(self, pad: bool = True) -> str:
        n = self.int()
        return self.bytes(n, pad).decode("utf-8", errors="replace")

    def vec3(self):
        v = struct.unpack_from("<3f", self.b, self.o)
        self.o += 12
        return v

    def color(self):
        c = tuple(self.b[self.o:self.o + 3])
        self.o += 4
        return c

    def matrix(self) -> np.ndarray:
        m = np.frombuffer(self.b, dtype="<f4", count=16, offset=self.o).reshape(4, 4).T  # column-major
        self.o += 64
        return m.astype(np.float64)

    def image(self) -> None:
        n = self.int()
        if n > 0:
            self.string()
            self.bytes(n)

    def tree_paths(self) -> list[tuple[str, tuple]]:
        return [(self.string(), self.color()) for _ in range(self.int())]


def extract_scene_bytes(html: str) -> bytes:
    m = re.search(r'DentalWebGL\.m_Data = \{"data": "([A-Za-z0-9+/=]+)"\}', html)
    if not m:
        raise ValueError("not an exocad webview HTML file (no embedded scene)")
    return base64.b64decode(m.group(1))


def load_webview(path: str | Path) -> list[WebviewObject]:
    """All mesh objects of an exocad webview HTML, in world coordinates (mm)."""
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_scene(extract_scene_bytes(html))


def parse_scene(data: bytes) -> list[WebviewObject]:
    r = _Reader(data)
    version = r.int()
    if version > 1:
        r.string()  # UI language
    if version > 5:
        r.bool()
        r.string()
        r.string()  # design ids
    if r.string():
        raise ValueError("webview is password protected: export it from exocad without a password")
    if version > 1:
        r.string()  # engine build
    r.string()
    # light
    for _ in range(3):
        r.float()
    r.vec3(); r.bool(); r.vec3(); r.int()
    for _ in range(4):
        r.color()
    for _ in range(3):
        r.float()
    r.image()
    for _ in range(2):  # custom and default views
        for _ in range(r.int()):
            r.string()
            r.matrix()
    for _ in range(r.int()):  # annotations
        if version > 3:
            r.bytes(r.int())
        r.string(); r.vec3(); r.vec3(); r.color(); r.tree_paths()
        if version > 2:
            r.bool()

    objects = []
    for _ in range(r.int()):
        r.bool(); r.bool(); r.bool()  # flat shading, vertex colours, texture
        r.color()
        diffuse = r.color()
        r.color(); r.color()
        r.float(); r.float(); r.float()
        r.color()
        r.float()
        world = r.matrix()
        ctm = r.bytes(r.int())
        r.image()
        r.float()
        paths = r.tree_paths()
        visible = r.bool() if version > 2 else True
        if version > 4:
            r.bool(); r.bool()
        verts, tris = decode_ctm(ctm)
        verts = verts @ world[:3, :3].T + world[:3, 3]
        names = [p[0] for p in paths]
        objects.append(WebviewObject(" / ".join(n for n in names if n), merge_vertices(verts, tris),
                                     diffuse, visible, names))
    return objects


# --------------------------------------------------------------------------
# OpenCTM decoder (RAW / MG1 / MG2), mesh geometry only
# --------------------------------------------------------------------------

def _lzma_packed(r: _Reader, count: int, size: int, signed: bool = False) -> np.ndarray:
    packed = r.uint()
    props = r.bytes(5, pad=False)
    raw = r.bytes(packed, pad=False)
    d = props[0]
    lc, d = d % 9, d // 9
    lp, pb = d % 5, d // 5
    dict_size = struct.unpack("<I", props[1:5])[0]
    dec = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA1, "dict_size": dict_size, "lc": lc, "lp": lp, "pb": pb}])
    n = count * size * 4
    buf = dec.decompress(raw, max_length=n)
    if len(buf) < n:
        raise ValueError("truncated OpenCTM stream")
    planes = np.frombuffer(buf, dtype=np.uint8).reshape(4, size, count)  # byte plane, component, element
    vals = ((planes[0].astype(np.uint32) << 24) | (planes[1].astype(np.uint32) << 16)
            | (planes[2].astype(np.uint32) << 8) | planes[3].astype(np.uint32)).T  # (count, size)
    if signed:
        v = vals.astype(np.int64)
        return np.where(v & 1, -((v + 1) >> 1), v >> 1)
    return vals


def _restore_indices(idx: np.ndarray) -> np.ndarray:
    t = idx.astype(np.int64).copy()
    if len(t):
        t[0, 1] += t[0, 0]
        t[0, 2] += t[0, 0]
    for i in range(1, len(t)):
        t[i, 0] += t[i - 1, 0]
        t[i, 1] += t[i - 1, 1] if t[i, 0] == t[i - 1, 0] else t[i, 0]
        t[i, 2] += t[i, 0]
    return t & 0xFFFFFFFF  # OpenCTM does this arithmetic in uint32


def decode_ctm(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    r = _Reader(data)
    if r.bytes(4, pad=False) != b"OCTM":
        raise ValueError("not an OpenCTM stream")
    r.int()  # format version
    method = r.bytes(4, pad=False)
    nv, nt = r.int(), r.int()
    r.int(); r.int(); r.int()  # uv maps, attribute maps, flags
    r.string(pad=False)  # comment
    if method == b"RAW\x00":
        r.int()
        tris = np.frombuffer(r.bytes(nt * 12, pad=False), "<u4").reshape(nt, 3)
        r.int()
        verts = np.frombuffer(r.bytes(nv * 12, pad=False), "<f4").reshape(nv, 3)
        return verts.astype(np.float64), tris.astype(np.int64)
    if method == b"MG1\x00":
        r.int()
        tris = _restore_indices(_lzma_packed(r, nt, 3))
        r.int()
        verts = _lzma_packed(r, nv * 3, 1).reshape(nv, 3).astype("<u4").view("<f4")
        return verts.astype(np.float64), tris
    if method == b"MG2\x00":
        r.int()  # "MG2H"
        prec = r.float()
        r.float()  # normal precision
        lo = np.array([r.float(), r.float(), r.float()])
        hi = np.array([r.float(), r.float(), r.float()])
        div = np.array([r.int(), r.int(), r.int()])
        size = (hi - lo) / div
        r.int()  # "VERT"
        deltas = _lzma_packed(r, nv, 3).astype(np.int64)
        r.int()  # "GIDX"
        grid = np.cumsum(_lzma_packed(r, nv, 1)[:, 0].astype(np.int64))
        r.int()  # "INDX"
        tris = _restore_indices(_lzma_packed(r, nt, 3))
        gz = grid // (div[0] * div[1])
        rem = grid - gz * div[0] * div[1]
        gy = rem // div[0]
        gx = rem - gy * div[0]
        # x deltas accumulate within a run of vertices in the same grid cell
        dx = deltas[:, 0].copy()
        for i in range(1, nv):
            if grid[i] == grid[i - 1]:
                dx[i] += dx[i - 1]
        verts = np.column_stack([lo[0] + gx * size[0] + prec * dx,
                                 lo[1] + gy * size[1] + prec * deltas[:, 1],
                                 lo[2] + gz * size[2] + prec * deltas[:, 2]])
        return verts, tris
    raise ValueError(f"unsupported OpenCTM method {method!r}")


# --------------------------------------------------------------------------
# Making sense of an exocad scene
# --------------------------------------------------------------------------

_KINDS = (  # first match wins; lower-case substrings of an object's tree path
    ("image", ("2d-", "2d ", ".jpg", ".png", "img_", "bild")),
    ("gingiva", ("десн", "gingiva", "zahnfleisch", " gum")),
    ("prep", ("препарир", "prep", "stumpf", "die")),
    ("waxup", ("ваксап", "wax", "анатом", "anatom")),
    ("antagonist", ("антагонист", "antagonist", "gegenkiefer")),
    ("jaw", ("челюст", "jaw", "kiefer", "скан", "scan")),
)


def classify(obj: WebviewObject) -> dict:
    """Kind of object, FDI teeth it belongs to and the jaw, from its tree path."""
    name = obj.name.lower()
    kind = next((k for k, keys in _KINDS if any(key in name for key in keys)), "other")
    leaf = obj.tree[-1] if obj.tree else obj.name
    own = re.match(r"\s*(\d\d)(?:-(\d\d))?\s*:", leaf)  # "33: ..." / "24-25: ..."
    teeth = []
    if own:
        a = int(own.group(1))
        b = int(own.group(2)) if own.group(2) else a
        teeth = list(range(min(a, b), max(a, b) + 1)) if a // 10 == b // 10 else [a, b]
    else:
        teeth = [int(t) for t in re.findall(r"(?<!\d)([1-4][1-8])(?!\d)", leaf)]
    teeth = [t for t in teeth if 1 <= t // 10 <= 4 and 1 <= t % 10 <= 8]
    jaw = None
    if any(k in name for k in ("верх", "upper", "oberkiefer", "maxill")):
        jaw = "upper"
    elif any(k in name for k in ("ниж", "lower", "unterkiefer", "mandib")):
        jaw = "lower"
    elif teeth:
        jaw = "upper" if teeth[0] // 10 in (1, 2) else "lower"
    return {"kind": kind, "teeth": teeth, "jaw": jaw, "tooth_specific": bool(own)}


def _merge(meshes: list[Mesh]) -> Mesh | None:
    if not meshes:
        return None
    offs = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return Mesh(np.concatenate([m.vertices for m in meshes]),
                np.concatenate([m.faces + o for m, o in zip(meshes, offs)]))


@dataclass
class WebviewCase:
    """The pieces of an exocad scene needed to design one tooth."""

    tooth: int
    axis: tuple[float, float, float]
    prep: Mesh
    jaw: Mesh | None  # same-jaw scan (neighbours)
    antagonist: Mesh | None
    reference: Mesh | None  # the technician's single-tooth wax-up, if any
    used: dict  # what was picked, by kind (no patient-identifying names)


def case_for_tooth(objects: list[WebviewObject], tooth: int) -> WebviewCase:
    info = [classify(o) for o in objects]
    side = "upper" if tooth // 10 in (1, 2) else "lower"
    other = "lower" if side == "upper" else "upper"
    axis = (0.0, 0.0, -1.0) if side == "upper" else (0.0, 0.0, 1.0)

    preps = [o for o, c in zip(objects, info) if c["kind"] == "prep" and tooth in c["teeth"]]
    if not preps:
        available = sorted({t for c in info if c["kind"] == "prep" for t in c["teeth"]})
        raise ValueError(f"no preparation for tooth {tooth} in this webview (preparations: {available})")
    jaws = [o for o, c in zip(objects, info)
            if c["kind"] == "jaw" and not c["tooth_specific"] and c["jaw"] in (side, None)]
    if not jaws:  # fall back to tooth scans of the same jaw
        jaws = [o for o, c in zip(objects, info) if c["kind"] == "jaw" and c["jaw"] == side]
    ants = [o for o, c in zip(objects, info)
            if (c["kind"] == "antagonist") or (c["kind"] in ("jaw", "waxup", "prep") and c["jaw"] == other)]
    refs = [o for o, c in zip(objects, info) if c["kind"] == "waxup" and c["teeth"] == [tooth]]
    used = {"prep": len(preps), "jaw_scans": len(jaws), "antagonist_objects": len(ants),
            "reference_waxup": bool(refs)}
    return WebviewCase(tooth, axis, _merge([o.mesh for o in preps]), _merge([o.mesh for o in jaws]),
                       _merge([o.mesh for o in ants]), refs[0].mesh if refs else None, used)


def design_from_webview(path: str | Path, tooth: int, *, learner=None, params=None,
                        use_neighbors: bool = True, occlusion=None, fix_bite_first: bool = False):
    """Design ``tooth`` from an exocad webview: preparation, neighbours, antagonist.

    Returns ``(CrownResult, WebviewCase, NeighborAnalysis | None)``; when the
    scene holds the technician's wax-up of the tooth, the report gains a
    ``reference`` block comparing the two.
    """
    from .arch import analyze_neighbors
    from .design import design_crown
    from .margin import detect_margin
    from .metrics import compare_to_reference

    case = case_for_tooth(load_webview(path), tooth)
    margin = detect_margin(case.prep, axis=case.axis)
    na = None
    if use_neighbors and case.jaw is not None:
        na = analyze_neighbors(case.jaw, margin, case.axis, tooth=tooth)
    antagonist = case.antagonist
    bite = None
    if fix_bite_first and case.jaw is not None and antagonist is not None:
        from .occlusion import fix_bite

        antagonist, bite = fix_bite(case.jaw, antagonist, case.axis)
    res = design_crown(case.prep, tooth=tooth, margin=margin, antagonist=antagonist,
                       axis=case.axis, neighbors=na, learner=learner, occlusion=occlusion, params=params)
    res.report["inputs"] = case.used
    if bite is not None:
        res.report["bite_correction"] = bite
    if case.reference is not None:
        top = res.frame.to_local(margin)[:, 2].max()
        res.report["reference"] = compare_to_reference(res.outer[:, 4:].reshape(-1, 3), res.crown,
                                                       case.reference, res.frame, top)
    return res, case, na


def learn_from_webview(path: str | Path, learner, *, teeth: list[int] | None = None) -> list[dict]:
    """Add every tooth that has both a preparation and a single-tooth wax-up to ``learner``."""
    import hashlib

    from .arch import analyze_neighbors
    from .learning import learn_from_crown
    from .margin import detect_margin

    objects = load_webview(path)
    case_key = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]  # no patient name in ids
    candidates = sorted({t for o in objects for t in classify(o)["teeth"]
                         if classify(o)["kind"] == "waxup" and classify(o)["teeth"] == [t]})
    out = []
    for tooth in candidates:
        if teeth and tooth not in teeth:
            continue
        try:
            case = case_for_tooth(objects, tooth)
        except ValueError:
            continue  # wax-up without a preparation (e.g. pontic)
        margin = detect_margin(case.prep, axis=case.axis)
        md = (1.0, 0.0, 0.0)
        if case.jaw is not None:
            md = analyze_neighbors(case.jaw, margin, case.axis, tooth=tooth).md_direction
        info = learn_from_crown(learner, case.reference, prep=case.prep, margin=margin, tooth=tooth,
                                axis=case.axis, md_direction=md, case_id=f"{case_key}/{tooth}")
        out.append({"tooth": tooth, **info})
    return out
