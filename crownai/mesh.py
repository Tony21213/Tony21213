"""Triangle mesh container, STL I/O and basic geometry queries (numpy only)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float64, millimetres
    faces: np.ndarray  # (F, 3) int64, counter-clockwise seen from outside

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    def face_normals(self) -> np.ndarray:
        t = self.triangles
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def volume(self) -> float:
        """Signed volume; positive for a closed, outward-oriented mesh."""
        t = self.triangles
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def is_watertight(self) -> bool:
        """Every directed edge must be matched by exactly one opposite edge."""
        f = self.faces
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        directed = {tuple(e) for e in edges.tolist()}
        if len(directed) != len(edges):
            return False  # duplicated directed edge -> non-manifold or flipped face
        return all((b, a) in directed for a, b in directed)

    def flipped(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces[:, ::-1].copy())

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        """Apply a 4x4 homogeneous transform."""
        v = self.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        return Mesh(v, self.faces.copy())


def merge_vertices(vertices: np.ndarray, faces: np.ndarray, decimals: int = 6) -> Mesh:
    """Weld coincident vertices (STL stores every triangle corner separately)."""
    key = np.round(vertices, decimals)
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return Mesh(unique, faces[keep])


def load_stl(path: str | Path) -> Mesh:
    data = Path(path).read_bytes()
    if len(data) >= 84:
        (count,) = struct.unpack_from("<I", data, 80)
        if 84 + count * 50 == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([
                ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")
            ]), count=count, offset=84)
            verts = rec["v"].reshape(-1, 3).astype(np.float64)
            return merge_vertices(verts, np.arange(len(verts)).reshape(-1, 3))
    text = data.decode("ascii", errors="ignore")
    coords = [line.split()[1:4] for line in text.splitlines() if line.strip().startswith("vertex")]
    if not coords:
        raise ValueError(f"{path}: not a valid STL file")
    verts = np.array(coords, dtype=np.float64)
    return merge_vertices(verts, np.arange(len(verts)).reshape(-1, 3))


_PLY_TYPE = {  # PLY property type name -> (numpy dtype, size in bytes)
    "char": ("i1", 1), "int8": ("i1", 1), "uchar": ("u1", 1), "uint8": ("u1", 1), "b1": ("u1", 1),
    "short": ("i2", 2), "int16": ("i2", 2), "ushort": ("u2", 2), "uint16": ("u2", 2),
    "int": ("i4", 4), "int32": ("i4", 4), "uint": ("u4", 4), "uint32": ("u4", 4),
    "float": ("f4", 4), "float32": ("f4", 4), "double": ("f8", 8), "float64": ("f8", 8),
}


def load_ply(path: str | Path) -> Mesh:
    """Load a binary or ASCII PLY scan (common intraoral-scanner export format)."""
    data = Path(path).read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="ignore").splitlines()
    if not header or header[0].strip() != "ply":
        raise ValueError(f"{path}: not a valid PLY file")
    fmt = next((ln.split()[1] for ln in header if ln.startswith("format")), "ascii")

    elements: list[tuple[str, int, list[tuple[str, str, str | None]]]] = []  # (name, count, props)
    cur_name, cur_count, cur_props = None, 0, []
    for ln in header[1:]:
        parts = ln.split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "element":
            if cur_name is not None:
                elements.append((cur_name, cur_count, cur_props))
            cur_name, cur_count, cur_props = parts[1], int(parts[2]), []
        elif parts[0] == "property":
            if parts[1] == "list":
                cur_props.append((parts[4], parts[2], parts[3]))  # (name, count_type, item_type)
            else:
                cur_props.append((parts[2], parts[1], None))
    if cur_name is not None:
        elements.append((cur_name, cur_count, cur_props))

    if fmt == "ascii":
        return _load_ply_ascii(data[header_end:].decode("ascii", errors="ignore"), elements)
    endian = "<" if "little" in fmt else ">"
    return _load_ply_binary(data, header_end, elements, endian)


def _load_ply_ascii(text: str, elements) -> Mesh:
    lines = iter(text.splitlines())
    verts = faces = None
    for name, count, props in elements:
        if name == "vertex":
            xi, yi, zi = (next(i for i, p in enumerate(props) if p[0] == c) for c in ("x", "y", "z"))
            verts = np.empty((count, 3), dtype=np.float64)
            for i in range(count):
                tok = next(lines).split()
                verts[i] = [float(tok[xi]), float(tok[yi]), float(tok[zi])]
        elif name == "face":
            rows = []
            for _ in range(count):
                tok = [int(float(t)) for t in next(lines).split()]
                n = tok[0]
                for k in range(1, n - 1):
                    rows.append((tok[1], tok[1 + k], tok[2 + k]))
                if not rows or n < 3:
                    continue
            faces = np.array(rows, dtype=np.int64) if rows else np.zeros((0, 3), dtype=np.int64)
        else:
            for _ in range(count):
                next(lines)
    if verts is None:
        raise ValueError("PLY file has no vertex element")
    return merge_vertices(verts, faces if faces is not None else np.zeros((0, 3), dtype=np.int64))


def _load_ply_binary(data: bytes, offset: int, elements, endian: str) -> Mesh:
    verts = faces = None
    for name, count, props in elements:
        if all(item_type is None for _, _, item_type in props):
            dtype = np.dtype([(pname, endian + _PLY_TYPE[ptype][0]) for pname, ptype, _ in props])
            rec = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
            offset += dtype.itemsize * count
            if name == "vertex":
                verts = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
        else:
            # a record with at least one list property (faces): every property must be walked in
            # declaration order to keep the byte offset aligned, scalar or list, even ones we
            # don't need - a face record commonly has a second list ("texcoord") right after
            # "vertex_indices", and skipping it silently desyncs every record after the first.
            rows = []
            for _ in range(count):
                idx = None
                for pname, ptype, item_type in props:
                    if item_type is None:
                        offset += _PLY_TYPE[ptype][1]
                        continue
                    ct_dtype, ct_size = _PLY_TYPE[ptype]
                    it_dtype, it_size = _PLY_TYPE[item_type]
                    (n,) = np.frombuffer(data, dtype=endian + ct_dtype, count=1, offset=offset)
                    offset += ct_size
                    n = int(n)
                    if pname == "vertex_indices":
                        idx = np.frombuffer(data, dtype=endian + it_dtype, count=n, offset=offset)
                    offset += it_size * n
                if name == "face" and idx is not None:
                    for k in range(1, len(idx) - 1):
                        rows.append((idx[0], idx[k], idx[k + 1]))
            if name == "face":
                faces = np.array(rows, dtype=np.int64) if rows else np.zeros((0, 3), dtype=np.int64)
    if verts is None:
        raise ValueError("PLY file has no vertex element")
    return merge_vertices(verts, faces if faces is not None else np.zeros((0, 3), dtype=np.int64))


def load_mesh(path: str | Path) -> Mesh:
    """Load an STL or PLY scan, picked by file extension."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return load_ply(path)
    return load_stl(path)


def save_stl(mesh: Mesh, path: str | Path) -> None:
    """Write a binary STL."""
    tris = mesh.triangles.astype(np.float32)
    rec = np.zeros(len(tris), dtype=np.dtype([
        ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")
    ]))
    rec["normal"] = mesh.face_normals().astype(np.float32)
    rec["v"] = tris
    header = b"crownai binary STL".ljust(80, b" ")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(tris)))
        fh.write(rec.tobytes())


def raycast(origins: np.ndarray, directions: np.ndarray, mesh: Mesh, farthest: bool = False) -> np.ndarray:
    """Distance along each ray to the nearest (or ``farthest``) hit; inf on miss.

    Brute force over triangles in ray chunks: fine for single-tooth scans
    (tens of thousands of triangles).
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if len(o) == 1 and len(d) > 1:
        o = np.repeat(o, len(d), axis=0)
    tris = mesh.triangles
    out = np.full(len(d), np.inf)
    parallel = len(d) > 1 and np.allclose(d, d[0])
    if parallel:
        # Parallel rays: cull triangles by their footprint on the plane normal
        # to the rays, chunking spatially sorted rays.
        u_ax = np.cross(d[0], [1.0, 0.0, 0.0] if abs(d[0, 0]) < 0.9 else [0.0, 1.0, 0.0])
        u_ax /= np.linalg.norm(u_ax)
        w_ax = np.cross(d[0], u_ax)
        ray_uv = np.stack([o @ u_ax, o @ w_ax], 1)
        tri_uv = np.stack([tris @ u_ax, tris @ w_ax], -1)
        tri_lo, tri_hi = tri_uv.min(1), tri_uv.max(1)
        order = np.lexsort((ray_uv[:, 1], np.round(ray_uv[:, 0], 0)))
        chunk = 256
        for s in range(0, len(order), chunk):
            sel = order[s:s + chunk]
            lo, hi = ray_uv[sel].min(0), ray_uv[sel].max(0)
            keep = np.all((tri_hi >= lo - 1e-9) & (tri_lo <= hi + 1e-9), axis=1)
            if keep.any():
                out[sel] = _moller_trumbore(o[sel], d[sel], tris[keep], farthest)
        return out
    chunk = max(1, int(1_500_000 // max(len(tris), 1)))
    for s in range(0, len(d), chunk):
        out[s:s + chunk] = _moller_trumbore(o[s:s + chunk], d[s:s + chunk], tris, farthest)
    return out


def _moller_trumbore(o: np.ndarray, d: np.ndarray, t: np.ndarray, farthest: bool = False) -> np.ndarray:
    v0, e1, e2 = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    oc, dc = o[:, None, :], d[:, None, :]
    pvec = np.cross(dc, e2[None])
    det = np.einsum("kij,ij->ki", pvec, e1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = oc - v0[None]
    u = np.einsum("kij,kij->ki", tvec, pvec) * inv
    qvec = np.cross(tvec, e1[None])
    v = np.einsum("kij,kij->ki", np.broadcast_to(dc, qvec.shape), qvec) * inv
    dist = np.einsum("kij,ij->ki", qvec, e2) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (dist > 1e-9)
    if farthest:
        far = np.where(hit, dist, -np.inf).max(axis=1)
        return np.where(np.isfinite(far), far, np.inf)
    return np.where(hit, dist, np.inf).min(axis=1)


def nearest_distance(points: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to its nearest target point, and that target's index."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        dist, idx = cKDTree(targets).query(points)
        return np.asarray(dist), np.asarray(idx, dtype=np.int64)
    dist = np.empty(len(points))
    idx = np.empty(len(points), dtype=np.int64)
    chunk = max(1, int(4_000_000 // max(len(targets), 1)))
    for s in range(0, len(points), chunk):
        d2 = ((points[s:s + chunk, None, :] - targets[None]) ** 2).sum(-1)
        j = d2.argmin(axis=1)
        idx[s:s + chunk] = j
        dist[s:s + chunk] = np.sqrt(d2[np.arange(len(j)), j])
    return dist, idx
