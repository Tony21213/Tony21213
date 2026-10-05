"""Сегментация КТ моделями ONNX и построение поверхностей структур.

Модели лежат в одной папке, каждая — в своей подпапке с двумя файлами:

* ``model.onnx`` — сеть: вход (1, 1, z, y, x), выход (1, метки, z, y, x);
* ``model.json`` — как готовить снимок и что значат выходы (пишет
  tools/export_nnunet_onnx.py)::

    {
      "name": "teeth",
      "onnx": "model.onnx",
      "orientation": "RAS",                 # оси сетки, на которой обучалась сеть
      "spacing": [0.5, 0.5, 0.5],           # шаг сетки по осям массива (z, y, x), мм
      "patch": [64, 160, 160],              # окно (z, y, x), вокселей
      "overlap": 0.5,
      "normalization": {"scheme": "zscore"} # или {"scheme": "ct", "clip": [...], "mean": ..., "std": ...}
      "region": {"around": "teeth", "margin_mm": [12, 12, 30]},  # или {"around": "whole"}
      "priority": 2,                        # структура берётся из модели с меньшим priority
      "labels": ["background", "lower_jawbone", ...],            # выходы сети по порядку
      "outputs": {"mandible": ["lower_jawbone"], ...},           # структура ← метки сети
      "license": "...", "attribution": "..."
    }

Снимок переводится на сетку модели, окна идут с перекрытием и весами Гаусса
(края окна весят меньше центра). Поверхность структуры строится не по маске,
а по разнице выходов сети: ноль этой разницы — граница argmax, но положение
внутри вокселя берётся из значений, а не округляется.
"""

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, sparse
from skimage import measure

from .volume import Volume

# Метки зубов у моделей, которые считают весь снимок (craniofacial): по ним ставится область модели
# зубов. По одной плотности на КЛКТ с большим полем область раздувается на всю голову, и модель
# зубов (обучена на зубных рядах) находит каналы и зубы в черепе.
TEETH_LABELS = ("teeth_upper", "teeth_lower")
# Плюс вся нижняя челюсть той же модели: нижнечелюстной канал начинается на ветви, позади зубов.
JAW_LABELS = ("mandible",)
JAW_MARGIN_MM = 3.0
# Структура лежит внутри другой: её куски вне той (с запасом) — ложные находки.
INSIDE = {"mandibular_canal": "mandible", "incisive_canal": "mandible", "lingual_canal": "mandible"}
INSIDE_MARGIN_MM = 2.0


@dataclass
class Model:
    folder: str
    name: str
    title: str
    onnx: str
    orientation: str
    spacing: tuple[float, float, float]  # (z, y, x)
    patch: tuple[int, int, int]  # (z, y, x)
    overlap: float
    normalization: dict
    labels: list[str]
    outputs: dict[str, list[str]]
    region: dict
    priority: int = 0
    license: str | None = None
    attribution: str | None = None

    @classmethod
    def load(cls, folder: str) -> "Model":
        path = os.path.join(folder, "model.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{folder}: нет model.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        spacing = d["spacing"]
        spacing = tuple(float(s) for s in (spacing if isinstance(spacing, list) else [spacing] * 3))
        model = cls(folder, d["name"], d.get("title", d["name"]), d.get("onnx", "model.onnx"),
                    d.get("orientation", "LPS"), spacing, tuple(int(p) for p in d["patch"]),
                    float(d.get("overlap", 0.5)), d["normalization"], list(d["labels"]), dict(d["outputs"]),
                    d.get("region", {"around": "whole"}), int(d.get("priority", 0)), d.get("license"),
                    d.get("attribution"))
        if model.normalization.get("scheme") not in ("ct", "zscore"):
            raise ValueError(f"{path}: неизвестная нормировка {model.normalization.get('scheme')!r}")
        unknown = sorted({lab for labs in model.outputs.values() for lab in labs} - set(model.labels))
        if unknown:
            raise ValueError(f"{path}: в labels нет меток {', '.join(unknown)}")
        if not os.path.isfile(os.path.join(folder, model.onnx)):
            raise FileNotFoundError(f"{folder}: нет {model.onnx}")
        return model

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
    labels: dict[str, tuple[Volume, np.ndarray]] = field(default_factory=dict)  # модель → (сетка, номер метки)
    models: list[Model] = field(default_factory=list)


def to_grid(vol: Volume, spacing_zyx, orientation: str = "LPS", roi=None) -> Volume:
    """Снимок на сетке модели: ориентация осей, шаг по осям (z, y, x), при желании — только roi (мм пациента)."""
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    img = sitk.DICOMOrient(img, orientation)
    spacing = np.array(spacing_zyx, float)[::-1]  # по осям индекса x, y, z
    out = Volume(np.zeros((1, 1, 1), np.float32), spacing, np.array(img.GetOrigin()),
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
                        spacing.tolist(), out.direction.ravel().tolist(), float(vol.data.min()), sitk.sitkFloat32)
    out.data = sitk.GetArrayFromImage(res)
    return out


def normalize(image: np.ndarray, normalization: dict) -> np.ndarray:
    """Яркость к виду, на котором обучалась сеть."""
    if normalization["scheme"] == "zscore":  # по самому снимку (области расчёта)
        return ((image - image.mean()) / max(float(image.std()), 1e-8)).astype(np.float32)
    x = np.clip(image, *normalization["clip"])
    return ((x - normalization["mean"]) / normalization["std"]).astype(np.float32)


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
    """Выходы сети по всему снимку (метки, z, y, x) скользящим окном."""
    x = normalize(image, model.normalization)
    shape = np.array(x.shape)
    patch = np.array(model.patch)
    # Снимок меньше окна дополняется до размера окна: сеть обучалась на окнах этого размера.
    pad = np.maximum(patch - shape, 0)
    x = np.pad(x, [(0, p) for p in pad], constant_values=float(x.min()))
    weights = _gaussian(patch)
    inp = session.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    out = np.zeros((len(model.labels), *x.shape), np.float32)
    norm = np.zeros(x.shape, np.float32)
    tiles = [(z, y, xx) for z in _starts(x.shape[0], patch[0], model.overlap)
             for y in _starts(x.shape[1], patch[1], model.overlap)
             for xx in _starts(x.shape[2], patch[2], model.overlap)]
    for n, (z, y, xx) in enumerate(tiles, 1):
        sl = (slice(z, z + patch[0]), slice(y, y + patch[1]), slice(xx, xx + patch[2]))
        logits = session.run(None, {inp.name: x[sl][None, None].astype(dtype)})[0][0].astype(np.float32)
        if logits.shape[0] != len(model.labels):
            raise ValueError(f"сеть выдаёт {logits.shape[0]} меток, а model.json описывает {len(model.labels)}")
        out[(slice(None),) + sl] += logits * weights
        norm[sl] += weights
        if progress:
            progress(model.name, n, len(tiles))
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


def output_meshes(logits: np.ndarray, grid: Volume, model: Model, smooth: int = 10, keys=None):
    """Поверхность каждой найденной структуры (или только keys) в мм пациента и карта меток."""
    labels = np.argmax(logits, axis=0)
    index = {name: i for i, name in enumerate(model.labels)}
    meshes = {}
    for key, members in model.outputs.items():
        if keys is not None and key not in keys:
            continue
        idx = [index[m] for m in members]
        mask = np.isin(labels, idx)
        if mask.sum() < 8:
            continue
        where = np.argwhere(mask)
        lo = np.maximum(where.min(axis=0) - 2, 0)
        hi = np.minimum(where.max(axis=0) + 3, labels.shape)
        sub = logits[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        field = sub[idx].max(axis=0) - np.delete(sub, idx, axis=0).max(axis=0)  # > 0 внутри структуры
        field = np.pad(field, 1, constant_values=min(float(field.min()), -1.0))
        verts, faces, _n, _v = measure.marching_cubes(field, 0.0, allow_degenerate=False)
        world = grid.to_world((verts - 1 + lo)[:, ::-1])
        meshes[key] = Mesh(taubin(world, faces, smooth), faces.astype(np.int64))
    return meshes, labels


def labels_box(grid: Volume, labels: np.ndarray, members, margin_mm) -> tuple[np.ndarray, np.ndarray] | None:
    """Область по меткам другой модели (мм пациента) плюс запас; None — меток почти нет."""
    where = np.argwhere(np.isin(labels, members))
    if len(where) < 200:
        return None
    pts = grid.to_world(where[:, ::-1])
    lo, hi = np.percentile(pts, [0.5, 99.5], axis=0)  # без одиночных ложных вокселей
    margin = np.array(margin_mm, float)
    return lo - margin, hi + margin


def keep_inside(mesh: Mesh, grid: Volume, mask: np.ndarray, share: float = 0.5) -> Mesh | None:
    """Куски сетки (связные части), у которых не меньше share вершин внутри маски на сетке grid."""
    faces = np.asarray(mesh.faces)
    if not len(faces):
        return None
    import trimesh

    part = trimesh.graph.connected_component_labels(
        trimesh.Trimesh(mesh.vertices, faces, process=False).face_adjacency, node_count=len(faces))
    idx = np.round(grid.to_index(mesh.vertices)).astype(int)[:, ::-1]  # (z, y, x)
    ok = np.all((idx >= 0) & (idx < mask.shape), axis=1)
    inside = np.zeros(len(mesh.vertices), bool)
    inside[ok] = mask[tuple(idx[ok].T)]
    vertex_part = np.full(len(mesh.vertices), -1)
    vertex_part[faces.ravel()] = np.repeat(part, 3)
    used = vertex_part >= 0
    total = np.bincount(vertex_part[used], minlength=part.max() + 1)
    within = np.bincount(vertex_part[used], weights=inside[used], minlength=part.max() + 1)
    keep_faces = faces[(within / np.maximum(total, 1) >= share)[part]]
    if not len(keep_faces):
        return None
    kept = np.unique(keep_faces)
    remap = np.full(len(mesh.vertices), -1)
    remap[kept] = np.arange(len(kept))
    return Mesh(np.asarray(mesh.vertices)[kept], remap[keep_faces])


def teeth_box(vol: Volume, margin_mm) -> tuple[np.ndarray, np.ndarray]:
    """Область зубных рядов в мм пациента: коронки по плотности плюс запас (корни, кость вокруг)."""
    from .teeth import Levels, crown_surface

    crowns = crown_surface(vol, Levels.of(vol), step=2)
    if len(crowns.points) < 100:
        raise ValueError("в КТ не найдены зубы — область для модели зубов не определить")
    lo, hi = np.percentile(crowns.points, [0.5, 99.5], axis=0)  # без одиночных выбросов (металл, шум)
    margin = np.array(margin_mm, float)
    return lo - margin, hi + margin


def find_models(folder: str) -> list[Model]:
    """Все модели в папке (подпапки с model.json) по priority, затем по имени."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{folder}: нет папки моделей")
    models = [Model.load(os.path.join(folder, d)) for d in sorted(os.listdir(folder))
              if os.path.isfile(os.path.join(folder, d, "model.json"))]
    if not models:
        raise FileNotFoundError(f"{folder}: нет ни одной модели (подпапок с model.json)")
    return sorted(models, key=lambda m: (m.priority, m.name))


class Segmenter:
    """Все модели из папки; одна и та же структура берётся из модели с меньшим priority."""

    def __init__(self, folder: str, device: str = "auto", only: list[str] | None = None):
        self.models = find_models(folder)
        if only:
            missing = set(only) - {m.name for m in self.models}
            if missing:
                raise ValueError(f"нет моделей: {', '.join(sorted(missing))}")
            self.models = [m for m in self.models if m.name in only]
        self.device = device

    def plan(self, want=None) -> list[Model]:
        """Какие модели считать: дающие нужные структуры (want(key) → bool; None — все) и модели на весь
        снимок с метками зубов, если нужна модель зубов — по ним ставится её область."""
        need = [m for m in self.models if want is None or any(want(k) for k in m.outputs)]
        if any(m.region.get("around") == "teeth" for m in need):
            need += [m for m in self.models if m not in need and m.region.get("around", "whole") == "whole"
                     and set(TEETH_LABELS) & set(m.labels)]
        return sorted(need, key=lambda m: (m.priority, m.name))

    def run(self, vol: Volume, smooth: int = 10, progress=None, want=None) -> SegmentationResult:
        result = SegmentationResult(models=self.plan(want))
        found = {}
        # Сначала модели на весь снимок: по их меткам зубов ставится область модели зубов.
        for model in sorted(result.models, key=lambda m: m.region.get("around", "whole") != "whole"):
            region = model.region.get("around", "whole")
            roi = None
            if region == "teeth":
                roi = self._teeth_region(vol, result, tuple(model.region.get("margin_mm", (10, 10, 10))))
            elif region != "whole":
                raise ValueError(f"{model.name}: неизвестная область {region!r}")
            grid = to_grid(vol, model.spacing, model.orientation, roi)
            logits = predict(model.session(self.device), model, grid.data, progress)
            keys = None if want is None else [k for k in model.outputs if want(k)]
            found[model.name], labels = output_meshes(logits, grid, model, smooth, keys)
            del logits
            result.labels[model.name] = (grid, labels)
        for model in result.models:  # одна и та же структура — из модели с меньшим priority
            for key, mesh in found[model.name].items():
                result.meshes.setdefault(key, mesh)
        self._keep_inside(result)
        return result

    def _teeth_region(self, vol: Volume, result: SegmentationResult, margin) -> tuple[np.ndarray, np.ndarray]:
        for model in result.models:
            members = [model.labels.index(lab) for lab in TEETH_LABELS if lab in model.labels]
            if members and model.name in result.labels:
                box = labels_box(*result.labels[model.name], members, margin)
                if box is not None:
                    jaw = [model.labels.index(lab) for lab in JAW_LABELS if lab in model.labels]
                    jaw_box = labels_box(*result.labels[model.name], jaw, [JAW_MARGIN_MM] * 3) if jaw else None
                    if jaw_box is not None:
                        box = (np.minimum(box[0], jaw_box[0]), np.maximum(box[1], jaw_box[1]))
                    return box
        return teeth_box(vol, margin)  # без такой модели — по плотности

    def _keep_inside(self, result: SegmentationResult):
        """Куски структуры вне структуры-вместилища (каналы вне нижней челюсти) — ложные находки."""
        for key, container in INSIDE.items():
            source = next((m for m in result.models if container in m.outputs and m.name in result.labels), None)
            if key not in result.meshes or source is None:
                continue
            grid, labels = result.labels[source.name]
            mask = np.isin(labels, [source.labels.index(lab) for lab in source.outputs[container]])
            if not mask.any():
                continue
            steps = int(np.ceil(INSIDE_MARGIN_MM / grid.spacing.min()))
            where = np.argwhere(mask)  # запас — только в рамке вместилища
            lo = np.maximum(where.min(axis=0) - steps - 1, 0)
            hi = np.minimum(where.max(axis=0) + steps + 2, mask.shape)
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            mask[box] = ndimage.binary_dilation(mask[box], iterations=steps)
            kept = keep_inside(result.meshes[key], grid, mask)
            if kept is None:
                del result.meshes[key]
            else:
                result.meshes[key] = kept
