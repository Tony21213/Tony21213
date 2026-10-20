"""Зубы на фото по скану и уточнение камеры фото по их контурам (шаг «Эстетика»).

* **Маски** — SAM 2.1 Hiera-Tiny (Meta, Apache-2.0; ONNX в ``models/sam2.1-tiny``: кодировщик и декодер для
  onnxruntime) по подсказке — одной точке внутри зуба. С отрицательными подсказками у дальних зубов SAM путается
  (проба на кейсе 7: маски — пятна), с одной точкой — уверенность 0,96.
* **Номера** — со скана: вершины на коронке (``Registration.deviation`` не nan; без него — не дальше REACH_MM от
  зуба КТ) получают номер ближайшего зуба КТ (сегментация ``teeth/tooth_FDI``). Вершины десны номера не получают:
  с допуском 1 мм коронки скана заходили на десну.
* **Проекция** — z-буфер точек на сетке ``grid`` px в окне вокруг зубов: точки, повёрнутые от камеры, не видны;
  загораживают — нижние зубы в прикусе (на фото с ретрактором зубы сомкнуты).
* **Уточнение** — «силуэтный ICP»: точки края каждого зуба скана в проекции ↔ ближайший край маски того же зуба на
  фото; по парам — поворот, сдвиг и фокус камеры (расстояние до резцовой точки — мягко у заданного: перспективу по
  контурам видно лишь отчасти), несколько итераций.

Проба на кейсе 7 (фото с ретрактором; 2 пары точек → уточнение): совпадение передних зубов 0,74–0,85 (IoU), 95 %
проекции зубов — в 8–10 px от камеры по 6 парам точек (было 23 px).
"""

import os
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

MODEL_DIR = "sam2.1-tiny"
ENCODER, DECODER = "sam2.1_hiera_tiny.encoder.onnx", "sam2.1_hiera_tiny.decoder.onnx"
SAM_SIZE = 1024
SAM_MEAN, SAM_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
MIN_SCORE = 0.7  # уверенность SAM ниже — маска зуба не берётся
REACH_MM = 0.5
MAX_POINTS = 250_000
DEPTH_TOL_MM = 0.3
WINDOW_MARGIN = (0.25, 0.6)  # окно вокруг зубов: доли ширины и высоты
MIN_CELLS = 30  # зуб меньше — не в расчёте
EDGE_REACH_PX = 50.0
ITERS = 6
EDGE_PAIRS = 6000
DISTANCE_SIGMA_MM = 300.0
GRIDS = (1, 2, 3, 4, 6, 8, 12)
FILL = 0.8
MIN_IOU = 0.4  # зуб в уточнении — если маска и проекция при исходной камере уже совпадают хоть так (иначе маска — не тот зуб)


def grid_for(camera, points, labels, window, normals=None) -> int:
    """Шаг сетки проекции, px фото: наименьший, при котором в зубах проекции не меньше FILL клеток — с точкой
    (иначе зуб — решето: совпадение и края — шум)."""
    for g in GRIDS:
        v = project(camera, points, labels, window, g, normals)
        teeth = v.labels > 0
        if teeth.sum() and (v.index >= 0).sum() >= FILL * teeth.sum():
            return g
    return GRIDS[-1]


class Sam2:
    """SAM 2.1 (ONNX): маска объекта на снимке по точке."""

    def __init__(self, models_dir: str):
        self.folder = os.path.join(models_dir, MODEL_DIR)
        self._enc = self._dec = None

    @staticmethod
    def available(models_dir: str | None) -> bool:
        return bool(models_dir) and all(os.path.isfile(os.path.join(models_dir, MODEL_DIR, f)) for f in (ENCODER, DECODER))

    def _sessions(self):
        if self._enc is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            self._enc = ort.InferenceSession(os.path.join(self.folder, ENCODER), so, providers=["CPUExecutionProvider"])
            self._dec = ort.InferenceSession(os.path.join(self.folder, DECODER), so, providers=["CPUExecutionProvider"])
        return self._enc, self._dec

    def encode(self, image: np.ndarray, box) -> dict:
        """Признаки окна box = (x0, y0, x1, y1) снимка (RGB): окно — в квадрат, квадрат — в 1024."""
        from PIL import Image

        enc, _dec = self._sessions()
        x0, y0, x1, y1 = (int(v) for v in box)
        side = max(x1 - x0, y1 - y0)
        square = np.zeros((side, side, 3), np.uint8)
        square[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1, :3]
        im = np.asarray(Image.fromarray(square).resize((SAM_SIZE, SAM_SIZE), Image.BILINEAR), np.float32) / 255.0
        im = ((im - SAM_MEAN) / SAM_STD).transpose(2, 0, 1)[None].astype(np.float32)
        f0, f1, emb = enc.run(None, {"image": im})
        return {"f0": f0, "f1": f1, "emb": emb, "origin": (x0, y0), "side": side, "size": (x1 - x0, y1 - y0)}

    def mask(self, state: dict, point_xy) -> tuple[np.ndarray, float]:
        """Маска (булева, пиксели окна) и уверенность по точке снимка внутри объекта."""
        from PIL import Image

        _enc, dec = self._sessions()
        s = SAM_SIZE / state["side"]
        p = np.array([[[(point_xy[0] - state["origin"][0]) * s, (point_xy[1] - state["origin"][1]) * s]]], np.float32)
        masks, iou = dec.run(None, {"image_embed": state["emb"], "high_res_feats_0": state["f0"],
                                    "high_res_feats_1": state["f1"], "point_coords": p,
                                    "point_labels": np.ones((1, 1), np.float32),
                                    "mask_input": np.zeros((1, 1, 256, 256), np.float32),
                                    "has_mask_input": np.zeros(1, np.float32)})
        k = int(np.argmax(iou[0]))
        side = state["side"]
        m = np.asarray(Image.fromarray((masks[0, k] > 0).astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 0
        w, h = state["size"]
        return m[:h, :w], float(iou[0, k])


def crown_labels(vertices, teeth: dict, crown=None, reach: float = REACH_MM) -> np.ndarray:
    """Номер зуба КТ у каждой вершины скана (0 — не коронка): teeth — {FDI: вершины зуба КТ}, координаты те же;
    crown — признак «на коронке КТ» (``Registration.deviation`` не nan)."""
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, float)
    out = np.zeros(len(v), np.int32)
    if not teeth:
        return out
    allv = np.vstack([np.asarray(teeth[k], float) for k in teeth])
    lab = np.concatenate([np.full(len(teeth[k]), int(k)) for k in teeth])
    limit = reach if crown is None else max(reach, 1.0)
    d, j = cKDTree(allv).query(v, distance_upper_bound=limit)
    ok = np.isfinite(d)
    if crown is not None:
        ok &= np.asarray(crown, bool)
    out[ok] = lab[j[ok]]
    return out


@dataclass
class View:
    """Проекция зубов скана на фото: номера и индекс видимой точки по клеткам окна."""

    labels: np.ndarray  # номер зуба в клетке; 0 — нет зуба, −1 — загорожено (нижние зубы)
    index: np.ndarray  # индекс точки в клетке; −1 — нет
    origin: tuple  # левый верхний угол окна, px фото
    grid: int

    def cell_centres(self, cy, cx):
        return np.c_[self.origin[0] + (cx + 0.5) * self.grid, self.origin[1] + (cy + 0.5) * self.grid]


def _zbuffer(camera, P, normals, window, grid):
    """Ближайшая точка в клетке окна: (индекс точки или −1, глубина) по клеткам."""
    x0, y0, x1, y1 = (int(v) for v in window)
    w3, h3 = max(1, (x1 - x0) // grid), max(1, (y1 - y0) // grid)
    c = camera.centre
    depth_all = (P - c) @ camera.R[2]
    visible = depth_all > 1.0
    if normals is not None:
        n = np.asarray(normals, float)
        visible &= (np.einsum("ij,ij->i", n, c - P) > 0) | (np.abs(n).sum(1) == 0)
    q = camera.project(P) - [x0, y0]
    ij = np.floor(q / grid).astype(int)
    ok = visible & (ij[:, 0] >= 0) & (ij[:, 0] < w3) & (ij[:, 1] >= 0) & (ij[:, 1] < h3)
    flat = ij[ok, 1] * w3 + ij[ok, 0]
    depth = np.full(w3 * h3, np.inf)
    np.minimum.at(depth, flat, depth_all[ok])
    sel = np.flatnonzero(ok)
    order = np.argsort(-depth_all[sel])  # дальние — первыми: ближайшая запись — последней
    index = np.full(w3 * h3, -1)
    index[flat[order]] = sel[order]
    return index.reshape(h3, w3), depth.reshape(h3, w3)


def project(camera, points, labels, window, grid: int, normals=None, occluders=None, occluder_normals=None) -> View:
    """Проекция зубов скана (точки в координатах кейса, номера labels) в окне window = (x0, y0, x1, y1) снимка:
    z-буфер по клеткам grid px, точки, повёрнутые от камеры, не видны; дырки между точками и внутри зуба — по
    соседям. occluders — точки, что загораживают (нижние зубы; −1): видны там, где ближе зубов (глубина зубов в
    пустых клетках — по соседним, иначе нижние зубы «просвечивают» сквозь промежутки между точками)."""
    P = np.asarray(points, float)
    lab = np.asarray(labels, int)
    index, depth = _zbuffer(camera, P, normals, window, grid)
    labels_img = np.where(index >= 0, lab[np.maximum(index, 0)], 0)
    near = np.where(np.isfinite(depth), depth, np.inf)
    near = np.minimum(near, ndimage.grey_erosion(near, size=3))
    teeth = labels_img > 0
    if occluders is not None and len(occluders):
        on, od = _zbuffer(camera, np.asarray(occluders, float),
                          None if occluder_normals is None else np.asarray(occluder_normals, float), window, grid)
        hide = (on >= 0) & (od < near - DEPTH_TOL_MM)
        labels_img = np.where(hide, -1, labels_img)
        index = np.where(hide, -1, index)
        teeth &= ~hide
    # дырки: между точками (соседи) и внутри зуба (залить по ближайшей клетке зуба)
    empty = labels_img == 0
    grown = ndimage.grey_dilation(np.where(labels_img > 0, labels_img, 0), size=3)
    labels_img = np.where(empty & ~ndimage.binary_dilation(labels_img < 0), grown, labels_img)
    inner = ndimage.binary_fill_holes(labels_img > 0) & (labels_img <= 0)
    if inner.any():
        _d, (iy, ix) = ndimage.distance_transform_edt(~(labels_img > 0), return_indices=True)
        labels_img = np.where(inner, labels_img[iy, ix], labels_img)
    index = np.where((index >= 0) & (labels_img > 0) & (lab[np.maximum(index, 0)] == labels_img), index, -1)
    return View(labels_img, index, (int(window[0]), int(window[1])), grid)


def teeth_window(camera, points, size, margin=WINDOW_MARGIN):
    """Окно снимка вокруг зубов скана (по камере)."""
    q = camera.project(np.asarray(points, float))
    w, h = size
    ok = (q[:, 0] > -w) & (q[:, 0] < 2 * w) & (q[:, 1] > -h) & (q[:, 1] < 2 * h)
    if not ok.any():
        raise ValueError("зубы скана не попадают на фото — проверьте пары точек")
    lo, hi = np.percentile(q[ok], 1, axis=0), np.percentile(q[ok], 99, axis=0)
    mx, my = (hi - lo) * margin
    x0, y0 = max(0, int(lo[0] - mx)), max(0, int(lo[1] - my))
    x1, y1 = min(w, int(hi[0] + mx)), min(h, int(hi[1] + my))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError("зубы скана на фото — за кадром")
    return x0, y0, x1, y1


def tooth_masks(sam: Sam2, image, view: View, window) -> dict:
    """Маски зубов на фото (клетки окна проекции) по подсказке — самой глубокой клетке зуба проекции:
    {FDI: (маска, уверенность)}."""
    state = sam.encode(image, window)
    out = {}
    for n in sorted(set(view.labels[view.labels > 0].tolist())):
        region = view.labels == n
        if region.sum() < MIN_CELLS:
            continue
        dt = ndimage.distance_transform_edt(region)
        cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        m, score = sam.mask(state, view.cell_centres(np.array([cy]), np.array([cx]))[0])
        if score < MIN_SCORE:
            continue
        g = view.grid
        h3, w3 = view.labels.shape
        cells = m[g // 2::g, g // 2::g][:h3, :w3]
        mask = np.zeros((h3, w3), bool)
        mask[: cells.shape[0], : cells.shape[1]] = cells
        if mask.sum() >= MIN_CELLS:
            out[n] = (mask, score)
    return out


def _boundary(mask):
    return mask & ~ndimage.binary_erosion(mask)


def iou(view: View, masks: dict) -> dict:
    out = {}
    for n, (m, _s) in masks.items():
        r = view.labels == n
        out[n] = float((r & m).sum() / max((r | m).sum(), 1))
    return out


@dataclass
class Refined:
    camera: object
    iou_before: dict = field(default_factory=dict)
    iou_after: dict = field(default_factory=dict)
    edge_px: float | None = None  # медиана расхождения краёв после уточнения
    teeth: list = field(default_factory=list)  # зубы в уточнении
    masks: dict = field(default_factory=dict)  # маски зубов на фото (клетки окна)
    window: tuple = ()
    grid: int = 1


def _sdf(mask):
    """Расстояние до края маски со знаком, клетки: снаружи — плюс, внутри — минус."""
    return ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)


def refine(camera, points, labels, masks: dict, window, grid: int, anchor, distance_mm: float, normals=None,
           occluders=None, occluder_normals=None, iters: int = ITERS) -> Refined:
    """Уточнить камеру так, чтобы край каждого зуба скана лёг на край маски того же зуба (маски — клетки окна):
    точки края проекции зуба (3D) — в поле «расстояние до края маски со знаком», по нему — поворот, сдвиг и фокус
    (край скользит вдоль контура — сходится за несколько шагов); расстояние до anchor — мягко у distance_mm."""
    from dataclasses import replace

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    P_all = np.asarray(points, float)
    anchor = np.asarray(anchor, float)
    start = project(camera, P_all, labels, window, grid, normals, occluders, occluder_normals)
    agree = iou(start, masks)
    fields = {n: _sdf(m) for n, (m, _s) in masks.items() if agree.get(n, 0.0) >= MIN_IOU}
    reach = EDGE_REACH_PX / grid
    rng = np.random.default_rng(0)
    origin = np.asarray(window[:2], float)
    cam, first, edge = camera, None, None
    for _ in range(iters):
        view = project(cam, P_all, labels, window, grid, normals, occluders, occluder_normals)
        if first is None:
            first = view
        src, tooth = [], []
        for n, f in fields.items():
            r = view.labels == n
            if r.sum() < MIN_CELLS:
                continue
            by, bx = np.nonzero(_boundary(r))
            vi = view.index[by, bx]
            ok = (vi >= 0) & (np.abs(f[by, bx]) < reach)
            src.append(P_all[vi[ok]])
            tooth.append(np.full(int(ok.sum()), n))
        if not src or sum(len(x) for x in src) < 20:
            break
        S, Nn = np.vstack(src), np.concatenate(tooth)
        if len(S) > EDGE_PAIRS:
            k = rng.choice(len(S), EDGE_PAIRS, replace=False)
            S, Nn = S[k], Nn[k]
        groups = [(n, np.flatnonzero(Nn == n)) for n in fields if (Nn == n).any()]
        R0, t0, f0 = cam.R.copy(), cam.t.copy(), float(cam.K[0, 0])

        def make(x):
            R = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
            K = cam.K.copy()
            K[0, 0] = K[1, 1] = f0 * np.exp(x[6])
            return replace(cam, K=K, R=R, t=t0 + x[3:6])

        def distances(c):
            cells = (c.project(S) - origin) / grid - 0.5
            out = np.empty(len(S))
            for n, idx in groups:
                out[idx] = ndimage.map_coordinates(fields[n], [cells[idx, 1], cells[idx, 0]], order=1, mode="nearest")
            return out * grid  # px

        def residual(x):
            c = make(x)
            r = np.clip(distances(c), -EDGE_REACH_PX, EDGE_REACH_PX)
            depth = float((anchor - c.centre) @ c.R[2])
            return np.r_[r, np.sqrt(len(S)) * (depth - distance_mm) / DISTANCE_SIGMA_MM]

        sol = least_squares(residual, np.zeros(7), x_scale=[1e-3] * 3 + [0.5] * 3 + [1e-3])
        cam = make(sol.x)
        new_edge = float(np.median(np.abs(distances(cam))))
        done = edge is not None and abs(edge - new_edge) < 0.05
        edge = new_edge
        if done:
            break
    last = project(cam, P_all, labels, window, grid, normals, occluders, occluder_normals)
    return Refined(cam, agree, iou(last, masks), edge, sorted(fields), masks, tuple(window), grid)


def align(sam: Sam2, image, camera, points, labels, anchor, distance_mm: float, normals=None, occluders=None,
          occluder_normals=None) -> Refined:
    """Маски зубов по камере camera и уточнение камеры по ним."""
    size = (image.shape[1], image.shape[0])
    on = np.asarray(labels) > 0
    if on.sum() < 200:
        raise ValueError("на скане нет зубов с номерами — нужна сегментация зубов на КТ")
    pts, lab = np.asarray(points, float)[on], np.asarray(labels)[on]
    nrm = None if normals is None else np.asarray(normals, float)[on]
    if len(pts) > MAX_POINTS:
        k = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, lab = pts[k], lab[k]
        nrm = None if nrm is None else nrm[k]
    window = teeth_window(camera, pts, size)
    grid = grid_for(camera, pts, lab, window, nrm)
    view = project(camera, pts, lab, window, grid, nrm, occluders, occluder_normals)
    masks = tooth_masks(sam, image, view, window)
    if len(masks) < 2:
        raise ValueError("нейросеть не нашла зубы на фото — проверьте пары точек")
    return refine(camera, pts, lab, masks, window, grid, anchor, distance_mm, nrm, occluders, occluder_normals)


def _polylines(region: np.ndarray, step: int):
    """Контуры области (клетки): список массивов (строка, столбец) в долях клетки."""
    from skimage import measure

    out = []
    for c in measure.find_contours(np.pad(region.astype(float), 1), 0.5):
        c = c - 1.0
        if len(c) >= 8:
            out.append(c[:: max(1, step)])
    return out


def scan_contours(view: View, points, step: int = 2, max_points: int = 6000) -> list:
    """Контуры зубов скана в проекции view — точками скана (3D, координаты кейса): [{"tooth", "points" N×3}] — окно
    перепроецирует их само, когда скан двигают мышью."""
    P = np.asarray(points, float)
    out, total = [], 0
    for n in sorted(set(view.labels[view.labels > 0].tolist())):
        region = view.labels == n
        if region.sum() < MIN_CELLS:
            continue
        border = _boundary(region) & (view.index >= 0)
        if not border.any():
            continue
        by, bx = np.nonzero(border)
        from scipy.spatial import cKDTree

        tree = cKDTree(np.c_[by, bx])
        for line in _polylines(region, step):
            _d, k = tree.query(line)
            idx = view.index[by[k], bx[k]]
            pts = P[idx]
            pts = pts[np.r_[True, idx[1:] != idx[:-1]]]
            if len(pts) >= 4:
                out.append({"tooth": int(n), "points": pts})
                total += len(pts)
    if total > max_points:  # проредить равномерно
        k = int(np.ceil(total / max_points))
        out = [{"tooth": c["tooth"], "points": c["points"][::k]} for c in out if len(c["points"][::k]) >= 3]
    return out


def mask_contours(masks: dict, window, grid: int, step: int = 2) -> list:
    """Контуры масок зубов на фото: [{"tooth", "px" N×2}] — пиксели снимка."""
    out = []
    for n, (m, _s) in sorted(masks.items()):
        for line in _polylines(m, step):
            out.append({"tooth": int(n), "px": np.c_[window[0] + (line[:, 1] + 0.5) * grid,
                                                      window[1] + (line[:, 0] + 0.5) * grid]})
    return out
