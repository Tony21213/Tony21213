"""Teeth with FDI numbers and gingiva on an intraoral scan - DentalModelSeg, no GPU.

Taken unchanged in substance from the sister project KStom Case Designer
(``custom-case-designer/casedesigner/scan_seg.py``, same owner); its original
Russian notes follow.  The network is DentalModelSeg v3.0 (DCBIA-OrthoLab,
UNC / University of Michigan; Apache-2.0; 2nd place MICCAI 3DTeethSeg 2022),
converted to ONNX by that project and published in this repository's release
``photo-v1`` (``dentalmodelseg-v3.0.onnx`` with ``LICENSE-DentalModelSeg.txt``).
Download it with :func:`crownai.models.ensure_dentalmodelseg`.

Зубы с номерами FDI и десна на внутриротовом скане — нейросеть DentalModelSeg без КТ.

DentalModelSeg (DCBIA-OrthoLab: Leclercq M., Prieto J. C., Cevidanes L., Styner M. и др.; Apache-2.0; 2-е место
MICCAI 3DTeethSeg'22) смотрит на скан с 42 сторон: скан вписан в единичную сферу, камера — в вершинах
подразделённого икосаэдра на радиусе 1,35, поле зрения 60°, снимок 320×320 — цвет по нормалям вершин и глубина;
2D U-Net (MONAI) даёт 34 класса на пиксель (0 — фон, 1–32 — зубы по Universal, 33 — десна), вероятности
складываются по треугольникам. Обучена со случайными поворотами скана — ориентация не нужна.

Отрисовка — своя, на numpy, по соглашениям PyTorch3D, на которых сеть обучали (`look_at_rotation`, вид: x — влево,
y — вверх, z — вперёд; пиксель (0, 0) — левый верхний; вид не зеркальный; глубина — по оси вида, фон — белый, глубина
фона −1); сеть — ONNX (tools/prepare_scan_seg.py). Чистка — как у авторов: островки меньше порога получают номер
соседей. Челюсть скана известна — номера другой челюсти не берутся.

Проверка на 6 кейсах с КТ (12 сканов, номера по КТ — вершины у коронок КТ): см. docs/models.md.
"""

import os

import numpy as np

MODEL_DIR = "dentalmodelseg"
IMAGE = 320
RADIUS = 1.35
FOV_DEG = 60.0
LEVEL = 2  # подразделение икосаэдра: 42 вида (как у авторов при распознавании)
BATCH = 6
GUM = 33
UNIVERSAL_TO_FDI = np.array([0, 18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
                             38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48, 0])
JAW_CLASSES = {"upper": range(1, 17), "lower": range(17, 33)}  # Universal: 1–16 — верхние, 17–32 — нижние
ISLAND_TOOTH, ISLAND_GUM = 200, 500  # у авторов, вершин — на скан ~100 тыс. вершин (масштабируется)


def available(models_dir: str | None) -> bool:
    return bool(models_dir) and os.path.isfile(os.path.join(models_dir, MODEL_DIR, "model.onnx"))


def view_positions(radius: float = RADIUS, level: int = LEVEL) -> np.ndarray:
    """Камеры: вершины икосаэдра и середины рёбер (level=2), на сфере radius."""
    a, b = 0.5257311121191336, 0.85065080835204
    ico = np.array([[0, a, b], [0, a, -b], [0, -a, b], [0, -a, -b], [a, b, 0], [a, -b, 0], [-a, b, 0], [-a, -b, 0],
                    [b, 0, a], [b, 0, -a], [-b, 0, a], [-b, 0, -a]])
    pts = [ico]
    if level >= 2:
        d = np.linalg.norm(ico[:, None] - ico[None], axis=-1)
        i, j = np.nonzero(np.triu(np.isclose(d, d[d > 0].min(), atol=1e-6)))
        pts.append((ico[i] + ico[j]) / 2)
    p = np.vstack(pts)
    return radius * p / np.linalg.norm(p, axis=1, keepdims=True)


def _look_at(c: np.ndarray) -> np.ndarray:
    """Оси вида (строки x, y, z) камеры в c, смотрящей в начало координат: PyTorch3D, up = +Y (у полюса — +Z)."""
    z = -c / np.linalg.norm(c)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.999 else np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def unit(vertices: np.ndarray):
    """Скан — в единичную сферу, как у авторов: центр рамки, угол рамки — на расстоянии 1."""
    v = np.asarray(vertices, float)
    lo, hi = v.min(0), v.max(0)
    mean = (lo + hi) / 2
    return (v - mean) / np.linalg.norm(hi - mean)


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Нормали вершин — сумма единичных нормалей граней (vtkPolyDataNormals без разбиения)."""
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def render(v: np.ndarray, f: np.ndarray, colors: np.ndarray, camera: np.ndarray, size: int = IMAGE):
    """Вид скана: (картинка 4×size×size — цвет и глубина, номер грани в пикселе или −1)."""
    R = _look_at(camera)
    q = (v - camera) @ R.T
    zv = q[:, 2]
    s = 1.0 / np.tan(np.radians(FOV_DEG) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        col = (1 - s * q[:, 0] / zv) * size / 2 - 0.5
        row = (1 - s * q[:, 1] / zv) * size / 2 - 0.5
    P = np.c_[col, row]
    t = P[f]  # (M, 3, 2)
    zf = zv[f]
    lo = np.ceil(t.min(1)).astype(np.int64)
    hi = np.floor(t.max(1)).astype(np.int64)
    lo, hi = np.maximum(lo, 0), np.minimum(hi, size - 1)
    w, h = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    ok = (w > 0) & (h > 0) & (zf > 1e-6).all(1)
    faces = np.flatnonzero(ok)
    pix, face, bary = [], [], []
    if len(faces):
        tw, th, tlo, tt = w[faces], h[faces], lo[faces], t[faces]
        x0, y0 = tt[:, 0, 0], tt[:, 0, 1]
        e1, e2 = tt[:, 1] - tt[:, 0], tt[:, 2] - tt[:, 0]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        good = np.abs(det) > 1e-12
        for dy in range(int(th.max())):
            for dx in range(int(tw.max())):
                k = np.flatnonzero((tw > dx) & (th > dy) & good)
                if not len(k):
                    continue
                px, py = tlo[k, 0] + dx, tlo[k, 1] + dy
                rx, ry = px - x0[k], py - y0[k]
                b1 = (rx * e2[k, 1] - ry * e2[k, 0]) / det[k]
                b2 = (e1[k, 0] * ry - e1[k, 1] * rx) / det[k]
                b0 = 1 - b1 - b2
                inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
                if inside.any():
                    pix.append(py[inside] * size + px[inside])
                    face.append(faces[k[inside]])
                    bary.append(np.c_[b0[inside], b1[inside], b2[inside]])
    img = np.ones((4, size, size), np.float32)
    img[3] = -1.0
    pf = np.full(size * size, -1, np.int64)
    if pix:
        pix, face, bary = np.concatenate(pix), np.concatenate(face), np.vstack(bary)
        inv = bary / zf[face]
        depth = 1.0 / inv.sum(1)
        order = np.lexsort((depth, pix))
        first = np.r_[True, pix[order][1:] != pix[order][:-1]]
        sel = order[first]
        p, fc = pix[sel], face[sel]
        bc = inv[sel] * depth[sel, None]  # барицентрические с учётом перспективы
        rgb = np.einsum("ij,ijk->ik", bc, colors[f[fc]])
        img[:3].reshape(3, -1)[:, p] = rgb.T
        img[3].reshape(-1)[p] = depth[sel]
        pf[p] = fc
    return img, pf.reshape(size, size)


class DentalModelSeg:
    def __init__(self, models_dir: str):
        self.path = os.path.join(models_dir, MODEL_DIR, "model.onnx")
        self._session = None

    def _run(self, images: np.ndarray) -> np.ndarray:
        if self._session is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            self._session = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        return self._session.run(None, {"image": images.astype(np.float32)})[0]

    def face_votes(self, vertices, faces, progress=None) -> np.ndarray:
        """Сумма вероятностей классов по граням со всех видов: (граней × 34)."""
        v = unit(vertices)
        f = np.asarray(faces, np.int64)
        n = vertex_normals(v, f)
        if np.einsum("ij,ij->", n, v - v.mean(0)) < 0:  # нормали внутрь — развернуть (сеть видела наружу)
            n = -n
        colors = np.floor((n * 0.5 + 0.5) * 255.0) / 255.0
        votes = np.zeros((len(f), 34))
        cams = view_positions()
        for i0 in range(0, len(cams), BATCH):
            batch = [render(v, f, colors, c) for c in cams[i0:i0 + BATCH]]
            logits = self._run(np.stack([b[0] for b in batch]))
            for (_img, pf), lg in zip(batch, logits):
                on = pf >= 0
                e = np.exp(lg[:, on] - lg[:, on].max(0))
                prob = e / e.sum(0)
                np.add.at(votes, pf[on], prob.T)
            if progress:
                progress(min(1.0, (i0 + BATCH) / len(cams)), "Зубы на скане: виды")
        return votes

    def labels(self, vertices, faces, jaw: str | None = None, progress=None) -> np.ndarray:
        """Номер FDI у каждой вершины скана: 11–48 — зуб, 0 — десна (и то, что сеть не разобрала). jaw —
        "upper" | "lower", если челюсть скана известна: номера другой челюсти не берутся (сеть их иногда ставит
        на кусочки скана — кейсы 3, 6)."""
        f = np.asarray(faces, np.int64)
        votes = self.face_votes(vertices, f, progress)
        if jaw in JAW_CLASSES:
            other = [c for c in range(1, 33) if c not in JAW_CLASSES[jaw]]
            votes[:, other] = 0.0
        vv = np.zeros((len(vertices), 34))
        for k in range(3):
            np.add.at(vv, f[:, k], votes)
        lab = np.where(vv.sum(1) > 0, vv.argmax(1), -1)
        lab = _fill_unknown(np.asarray(vertices, float), lab)
        lab = clean_islands(f, lab, len(vertices))
        return UNIVERSAL_TO_FDI[lab]


def _fill_unknown(v: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Вершины без голосов (не видны ни с одной стороны) — номер ближайшей с голосом."""
    from scipy.spatial import cKDTree

    unknown = lab < 0
    if unknown.any() and (~unknown).any():
        _d, j = cKDTree(v[~unknown]).query(v[unknown])
        lab = lab.copy()
        lab[unknown] = lab[~unknown][j]
    return np.maximum(lab, 0)


def clean_islands(f: np.ndarray, lab: np.ndarray, n: int, scale: float | None = None) -> np.ndarray:
    """Островки номера меньше порога (как у авторов: зуб — 200, десна — 500 вершин на ~100 тыс.) — номер соседей."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    scale = max(1.0, n / 100_000) if scale is None else scale
    lab = lab.copy()
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    for _ in range(3):
        same = lab[e[:, 0]] == lab[e[:, 1]]
        g = coo_matrix((np.ones(int(same.sum())), (e[same, 0], e[same, 1])), shape=(n, n))
        ncomp, comp = connected_components(g, directed=False)
        sizes = np.bincount(comp, minlength=ncomp)
        comp_lab = np.zeros(ncomp, np.int64)
        comp_lab[comp] = lab
        limit = np.where(comp_lab == GUM, ISLAND_GUM, ISLAND_TOOTH) * scale
        small = sizes < limit
        if not small[comp].any():
            break
        # номер соседей: по рёбрам из маленького островка в большие (соседние маленькие — на следующем круге)
        cross = ~same & small[comp[e[:, 0]]] & ~small[comp[e[:, 1]]]
        if not cross.any():
            break
        src, dst = comp[e[cross, 0]], lab[e[cross, 1]]
        best = {}
        for c, d in zip(src.tolist(), dst.tolist()):
            best.setdefault(c, {}).setdefault(d, 0)
            best[c][d] += 1
        new = comp_lab.copy()
        for c, cnt in best.items():
            new[c] = max(cnt, key=cnt.get)
        lab = new[comp]
    return lab
