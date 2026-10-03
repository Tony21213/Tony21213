"""Сегментация КТ моделями ONNX и построение поверхностей структур.

Модель — папка с двумя файлами:

* ``model.onnx`` — сеть: вход (1, 1, z, y, x), выход (1, классы, z, y, x);
* ``model.json`` — как готовить снимок и что значат выходы::

    {
      "scheme": "anatomy",            # набор классов из structures.SCHEMES
      "onnx": "model.onnx",
      "spacing": 0.3,                 # шаг сетки, на которой обучалась сеть, мм
      "patch": [160, 320, 320],       # размер окна (z, y, x), в вокселях
      "overlap": 0.5,                 # перекрытие окон
      "normalization": {"clip": [-1000, 3500], "mean": 650.0, "std": 1075.0},
      "extra_classes": ["ignore"]     # выходы после классов схемы, в STL не идут
    }

Снимок переводится в ориентацию LPS и шаг сетки модели, окна идут с
перекрытием и весами Гаусса (края окна весят меньше центра). Поверхность
структуры строится не по маске, а по разнице выходов сети: её ноль — граница
argmax, но положение внутри вокселя берётся из значений, а не округляется.
"""

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy import sparse
from skimage import measure

from .structures import SCHEMES, TEETH_MARGIN_MM, TEETH_SOURCE, Structure
from .volume import Volume


@dataclass
class Model:
    folder: str
    scheme: str
    classes: tuple[Structure, ...]
    spacing: float
    patch: tuple[int, int, int]
    overlap: float
    clip: tuple[float, float]
    mean: float
    std: float
    extra_classes: tuple[str, ...] = ()
    onnx: str = "model.onnx"

    @classmethod
    def load(cls, folder: str) -> "Model":
        path = os.path.join(folder, "model.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{folder}: нет model.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("scheme") not in SCHEMES:
            raise ValueError(f"{path}: неизвестный набор классов {d.get('scheme')!r}")
        n = d["normalization"]
        model = cls(folder, d["scheme"], SCHEMES[d["scheme"]], float(d["spacing"]), tuple(d["patch"]),
                    float(d.get("overlap", 0.5)), tuple(n["clip"]), float(n["mean"]), float(n["std"]),
                    tuple(d.get("extra_classes", ())), d.get("onnx", "model.onnx"))
        if not os.path.isfile(os.path.join(folder, model.onnx)):
            raise FileNotFoundError(f"{folder}: нет {model.onnx}")
        return model

    @property
    def n_outputs(self) -> int:
        return 1 + len(self.classes) + len(self.extra_classes)

    def session(self, device: str = "auto"):
        import onnxruntime as ort

        available = ort.get_available_providers()
        wanted = {"auto": ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                  "gpu": ["DmlExecutionProvider", "CUDAExecutionProvider"],
                  "cpu": ["CPUExecutionProvider"]}[device]
        providers = [p for p in wanted if p in available]
        if not providers:
            raise RuntimeError(f"нет подходящего устройства для расчёта ({device}); доступно: {available}")
        return ort.InferenceSession(os.path.join(self.folder, self.onnx), providers=providers)


@dataclass
class Mesh:
    vertices: np.ndarray  # мм пациента
    faces: np.ndarray


@dataclass
class SegmentationResult:
    meshes: dict[str, Mesh] = field(default_factory=dict)  # ключ — путь файла без расширения
    labels: dict[str, tuple[Volume, np.ndarray]] = field(default_factory=dict)  # проход → (сетка, метки)


def to_grid(vol: Volume, spacing: float, roi=None) -> Volume:
    """Снимок на сетке модели: ориентация LPS, изотропный шаг spacing, при желании — только roi."""
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    img = sitk.DICOMOrient(img, "LPS")
    out = Volume(np.zeros((1, 1, 1), np.float32), np.full(3, spacing), np.array(img.GetOrigin()),
                 np.array(img.GetDirection()).reshape(3, 3))
    extent = np.array(img.GetSize()) * np.array(img.GetSpacing())
    lo, hi = np.zeros(3), extent / spacing
    if roi is not None:
        corners = np.array([[x, y, z] for x in (roi[0][0], roi[1][0]) for y in (roi[0][1], roi[1][1])
                            for z in (roi[0][2], roi[1][2])])
        idx = out.to_index(corners)
        lo = np.maximum(np.floor(idx.min(axis=0)), lo)
        hi = np.minimum(np.ceil(idx.max(axis=0)) + 1, hi)
        if np.any(hi - lo < 2):
            raise ValueError("область расчёта вне снимка")
    size = np.maximum(np.ceil(hi - lo).astype(int), 1)
    out.origin = out.to_world(lo[None])[0]
    res = sitk.Resample(img, size.tolist(), sitk.Transform(), sitk.sitkLinear, out.origin.tolist(),
                        [spacing] * 3, out.direction.ravel().tolist(), float(vol.data.min()), sitk.sitkFloat32)
    out.data = sitk.GetArrayFromImage(res)
    return out


def _gaussian(patch) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(p) - (p - 1) / 2 for p in patch], indexing="ij")
    g = np.exp(-sum(x ** 2 / (2 * (p / 8) ** 2) for x, p in zip(grids, patch)))
    g /= g.max()
    return np.maximum(g, 1e-3).astype(np.float32)


def _starts(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    n = math.ceil((length - patch) / (patch * (1 - overlap))) + 1
    return sorted({round(i * (length - patch) / (n - 1)) for i in range(n)})


def predict(session, model: Model, image: np.ndarray, progress=None) -> np.ndarray:
    """Выходы сети по всему снимку (классы, z, y, x) скользящим окном."""
    x = np.clip(image, *model.clip)
    x = ((x - model.mean) / model.std).astype(np.float32)
    shape = np.array(x.shape)
    patch = np.array(model.patch)
    # Снимок меньше окна дополняется до размера окна: сеть обучалась на окнах этого размера.
    pad = np.maximum(patch - shape, 0)
    x = np.pad(x, [(0, p) for p in pad], constant_values=float(x.min()))
    weights = _gaussian(patch)
    inp = session.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    out = np.zeros((model.n_outputs, *x.shape), np.float32)
    norm = np.zeros(x.shape, np.float32)
    tiles = [(z, y, xx) for z in _starts(x.shape[0], patch[0], model.overlap)
             for y in _starts(x.shape[1], patch[1], model.overlap)
             for xx in _starts(x.shape[2], patch[2], model.overlap)]
    for n, (z, y, xx) in enumerate(tiles, 1):
        sl = (slice(z, z + patch[0]), slice(y, y + patch[1]), slice(xx, xx + patch[2]))
        logits = session.run(None, {inp.name: x[sl][None, None].astype(dtype)})[0][0].astype(np.float32)
        if logits.shape[0] != model.n_outputs:
            raise ValueError(f"сеть выдаёт {logits.shape[0]} классов, а model.json описывает {model.n_outputs}")
        out[(slice(None),) + sl] += logits * weights
        norm[sl] += weights
        if progress:
            progress(n, len(tiles))
    out /= norm
    return out[:, :shape[0], :shape[1], :shape[2]]


def taubin(vertices: np.ndarray, faces: np.ndarray, iterations: int = 10, lam: float = 0.5, mu: float = -0.53):
    """Сглаживание Таубина: убирает ступеньки, не сжимая поверхность (в отличие от Лапласа)."""
    if iterations <= 0 or not len(faces):
        return vertices
    n = len(vertices)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    adj = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    avg = sparse.diags(1.0 / np.maximum(deg, 1)) @ adj
    v = vertices.copy()
    for _ in range(iterations):
        v = v + lam * (avg @ v - v)
        v = v + mu * (avg @ v - v)
    return v


def class_meshes(logits: np.ndarray, grid: Volume, classes, smooth: int = 10, prefix: str = ""):
    """Поверхность каждого найденного класса в мм пациента и карта меток."""
    labels = np.argmax(logits, axis=0)
    meshes = {}
    for c, structure in enumerate(classes, start=1):
        mask = labels == c
        if mask.sum() < 8:
            continue
        lo = np.maximum(np.argwhere(mask).min(axis=0) - 2, 0)
        hi = np.minimum(np.argwhere(mask).max(axis=0) + 3, labels.shape)
        sub = logits[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        others = np.delete(sub, c, axis=0).max(axis=0)
        field = sub[c] - others  # > 0 внутри структуры
        field = np.pad(field, 1, constant_values=min(float(field.min()), -1.0))
        verts, faces, _n, _v = measure.marching_cubes(field, 0.0, allow_degenerate=False)
        verts = verts - 1 + lo
        world = grid.to_world(verts[:, ::-1])
        world = taubin(world, faces, smooth)
        meshes[prefix + structure.key] = Mesh(world, faces.astype(np.int64))
    return meshes, labels


class Segmenter:
    """Два прохода: анатомия целиком, затем отдельные зубы в области зубных рядов."""

    def __init__(self, anatomy_dir: str, teeth_dir: str | None = None, device: str = "auto"):
        self.anatomy = Model.load(anatomy_dir)
        if self.anatomy.scheme != "anatomy":
            raise ValueError(f"{anatomy_dir}: это модель {self.anatomy.scheme}, а нужна anatomy")
        self.teeth = Model.load(teeth_dir) if teeth_dir else None
        if self.teeth and self.teeth.scheme != "teeth":
            raise ValueError(f"{teeth_dir}: это модель {self.teeth.scheme}, а нужна teeth")
        self.device = device

    @classmethod
    def from_folder(cls, folder: str, device: str = "auto") -> "Segmenter":
        """Папка моделей: anatomy/ обязательно, teeth/ — если есть."""
        teeth = os.path.join(folder, "teeth")
        return cls(os.path.join(folder, "anatomy"), teeth if os.path.isdir(teeth) else None, device)

    def run(self, vol: Volume, roi=None, teeth: bool = True, smooth: int = 10, progress=None) -> SegmentationResult:
        result = SegmentationResult()
        grid = to_grid(vol, self.anatomy.spacing, roi)
        logits = predict(self.anatomy.session(self.device), self.anatomy, grid.data, progress)
        meshes, labels = class_meshes(logits, grid, self.anatomy.classes, smooth)
        result.meshes.update(meshes)
        result.labels["anatomy"] = (grid, labels)
        if not (teeth and self.teeth):
            return result

        keys = [s.key for s in self.anatomy.classes]
        teeth_mask = np.isin(labels, [keys.index(k) + 1 for k in TEETH_SOURCE])
        if not teeth_mask.any():
            return result
        idx = np.argwhere(teeth_mask)[:, ::-1]
        corners = grid.to_world(np.vstack([idx.min(axis=0), idx.max(axis=0)]))
        box = (corners.min(axis=0) - TEETH_MARGIN_MM, corners.max(axis=0) + TEETH_MARGIN_MM)
        tgrid = to_grid(vol, self.teeth.spacing, box)
        tlogits = predict(self.teeth.session(self.device), self.teeth, tgrid.data, progress)
        tmeshes, tlabels = class_meshes(tlogits, tgrid, self.teeth.classes, smooth, prefix="teeth/")
        result.meshes.update(tmeshes)
        result.labels["teeth"] = (tgrid, tlabels)
        return result
