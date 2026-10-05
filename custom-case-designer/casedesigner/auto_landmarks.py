"""Автоматические цефалометрические точки на КЛКТ: ALI-CBCT, переведённый в ONNX.

ALI-CBCT (Gillot M. и др., «Automatic landmark identification in cone-beam
computed tomography», Orthod Craniofac Res 2023; DCBIA-OrthoLab,
SlicerAutomatedDentalTools) ищет каждую точку своим «агентом». Агент стоит
в вокселе, видит вокруг себя куб 64³ и на каждом шаге сеть (DenseNet) говорит,
куда шагнуть: вверх, вниз, вперёд, назад, влево или вправо. Когда агент
возвращается туда, где недавно был, точка найдена на этом масштабе. Поиск
идёт сначала по КТ с вокселем 1 мм, затем уточняется на 0,3 мм; в конце
шесть «проб» из соседних вокселей сходятся к точке, и их среднее — результат.

Здесь — тот же поиск на numpy и ONNX Runtime, без PyTorch и MONAI:
подготовка КТ (обрезка яркостей по гистограмме, пересчёт вокселя, края по 33
вокселя нулями), окно агента и нормировка в [-1, 1], шаги, память позиций,
перезапуск из случайной точки, проба на зацикливание и итоговые пробы —
как в исходном коде. Перевод весов в ONNX: tools/prepare_landmarks.py.

Точки ALI-CBCT переименованы в наши (landmarks.LANDMARKS). Резцового сосочка
в ALI-CBCT нет: вместо него резцовое отверстие (IF) — костная точка под
сосочком, поэтому IP помечается как приближение. Крыловидно-челюстных
вырезок нет — их ставит врач.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk

from .volume import Volume

SCALES_MM = (1.0, 0.3)  # воксель грубого и точного поиска
SCALE_KEYS = ("1", "0-3")  # так названы сети каждой точки
FOV = 64  # сторона куба, который видит агент, вокселей
PAD = FOV // 2 + 1  # поля вокруг КТ, чтобы куб у края не выходил за данные
MOVES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)  # (z, y, x)
SHORT_MEMORY = 10  # «был здесь недавно» — среди стольких последних позиций
MAX_STEPS = 1000  # шагов на одну точку; сходящийся поиск укладывается в 30–200
STALL_STEPS = 64  # столько шагов подряд без новой позиции — агент ходит по кругу
MAX_RESTARTS = 2  # перезапусков из случайной точки, дальше — точка не найдена
SPAWN_RADIUS = 10  # перезапуск на точном масштабе — в этом радиусе от найденного грубо
FOCUS_RADIUS = 4  # итоговые пробы стартуют в стольких вокселях от точки
HIST_BINS = 1000
HIST_LOW, HIST_HIGH = 0.01, 0.99  # доли гистограммы, по которым обрезаются яркости
HU_LOW, HU_HIGH = -1500, 4000

# Наша точка → точка ALI-CBCT.
POINTS = {
    "Po_R": "RPo", "Po_L": "LPo", "Or_R": "ROr", "Or_L": "LOr", "Co_R": "RCo", "Co_L": "LCo",
    "N": "N", "S": "S", "ANS": "ANS", "PNS": "PNS", "Ba": "Ba", "IP": "IF",
}
APPROXIMATE = {"IP": "по резцовому отверстию: сосочек лежит на несколько мм ниже, на слизистой"}
LICENSE_NOTE = ("ALI-CBCT (Gillot M. et al., DCBIA-OrthoLab), веса из SlicerAutomatedDentalTools "
                "(лицензия 3D Slicer, BSD-подобная)")


@dataclass
class Found:
    key: str  # наша точка
    point: np.ndarray | None  # мм пациента (LPS) или None — не найдена
    steps: int = 0
    note: str = ""


@dataclass
class _Scale:
    image: np.ndarray  # int16 (z, y, x) с полями PAD
    size: np.ndarray  # без полей, (z, y, x)
    spacing: float
    sitk_image: sitk.Image = field(repr=False, default=None)


def _sitk_image(vol: Volume) -> sitk.Image:
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.data))
    img.SetSpacing([float(s) for s in vol.spacing])
    img.SetOrigin([float(o) for o in vol.origin])
    img.SetDirection([float(d) for d in np.asarray(vol.direction, float).ravel()])
    return img


def correct_histogram(a: np.ndarray) -> np.ndarray:
    """Обрезка яркостей по 1% и 99% гистограммы (в пределах -1500…4000), int16 — как CorrectHisto."""
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    span = hi - lo
    hist = np.histogram(a, HIST_BINS)[0]
    cum = np.cumsum(hist).astype(np.float64)
    cum -= cum.min()
    cum /= max(cum.max(), 1e-12)
    top = int(np.argmax(cum > HIST_HIGH)) * span / HIST_BINS + lo
    bottom = int(np.argmax(cum > HIST_LOW)) * span / HIST_BINS + lo
    bottom, top = max(bottom, HU_LOW), min(top, HU_HIGH)
    return np.clip(a, bottom, top).astype(np.int16)  # отбрасывание дробной части, как sitk.Cast


def _resample(img: sitk.Image, spacing: float) -> sitk.Image:
    """Изотропный воксель spacing мм: размер и начало — как SetSpacing в ALI-CBCT."""
    old = np.array(img.GetSpacing())
    size = np.array(img.GetSize())
    if np.array_equal(old, [spacing] * 3):
        return img
    new_size = (size * (old / spacing)).astype(int)
    origin = np.array(img.GetOrigin()) - (new_size * spacing - size * old) / 2.0
    return sitk.Resample(img, [int(s) for s in new_size], sitk.Transform(), sitk.sitkLinear, origin.tolist(),
                         [spacing] * 3, img.GetDirection(), 0, sitk.sitkInt16)


def prepare(vol: Volume) -> list[_Scale]:
    """КТ для агентов: оси LPS, яркости по гистограмме, воксель 1 и 0,3 мм, поля нулями."""
    img = sitk.DICOMOrient(_sitk_image(vol), "LPS")
    fixed = sitk.GetImageFromArray(correct_histogram(sitk.GetArrayFromImage(img)))
    fixed.CopyInformation(img)
    out = []
    for sp in SCALES_MM:
        r = _resample(fixed, sp)
        a = sitk.GetArrayFromImage(r)
        out.append(_Scale(np.pad(a, PAD), np.array(a.shape), sp, r))
    return out


def zone(scale: _Scale, centre) -> np.ndarray:
    """Куб 64³ вокруг позиции агента, яркости в [-1, 1] — как SpatialCrop + ScaleIntensity."""
    start = [max(int(c + PAD) - FOV // 2, 0) for c in centre]
    z, y, x = start
    crop = scale.image[z:z + FOV, y:y + FOV, x:x + FOV].astype(np.float32)
    lo, hi = crop.min(), crop.max()
    if lo == hi:
        return crop * -1.0
    return (crop - lo) / (hi - lo) * 2.0 - 1.0


class _Agent:
    """Поиск одной точки: два масштаба, память позиций, перезапуски, итоговые пробы."""

    def __init__(self, scales: list[_Scale], nets, rng: np.random.Generator):
        self.scales, self.nets, self.rng = scales, nets, rng
        self.level = 0
        self.pos = np.zeros(3)
        self.start = np.zeros(3)
        self.memory = [[] for _ in scales]
        self.restarts = 0
        self.ground, self.stalled = set(), 0

    def _act(self) -> int:
        q = self.nets[self.level](zone(self.scales[self.level], self.pos)[None, None])
        return int(np.argmax(q))

    def _visited(self) -> bool:
        return any(np.array_equal(self.pos, p) for p in self.memory[self.level][-SHORT_MEMORY:])

    def _remember(self):
        self.memory[self.level].append(self.pos.copy())

    def _respawn(self):
        self.memory[self.level].clear()
        self.ground, self.stalled = set(), 0
        size = self.scales[self.level].size
        if self.level == 0:
            p = self.rng.integers(1, size).astype(np.int16)
            self.start = p
        else:
            p = self.start + self.rng.integers([1, 1, 1], SPAWN_RADIUS * 2) - SPAWN_RADIUS
            p = np.where(p < 0, 0, p).astype(np.int16)
        self.pos = p.astype(float)

    def _move(self, action: int):
        new = self.pos + MOVES[action]
        if np.all(new != 0) and np.all(new < self.scales[self.level].size):
            self.pos = new
        else:  # вышел за КТ — заново из случайной точки
            self._respawn()
            self.restarts += 1

    def _cycling(self) -> bool:
        key = (self.level,) + tuple(int(c) for c in self.pos)
        if key in self.ground:
            self.stalled += 1
        else:
            self.ground.add(key)
            self.stalled = 0
        return self.stalled >= STALL_STEPS

    def _go_to(self, level: int):
        ratio = self.scales[self.level].spacing / self.scales[level].spacing
        self.pos = (self.pos * ratio).astype(np.int16).astype(float)
        self.level = level
        self.restarts = 0

    def search(self) -> tuple[np.ndarray | None, int, str]:
        self.level, self.restarts = 0, 0
        self.pos = self.scales[0].size / 2
        self._remember()
        steps = 0
        while steps < MAX_STEPS:
            steps += 1
            self._move(self._act())
            found = self._visited()
            self._remember()
            if not found and self._cycling():
                self.memory[self.level].clear()
                self._respawn()
                self.restarts += 1
            if found:
                if self.level < len(self.scales) - 1:
                    self._go_to(self.level + 1)
                    self.start = self.pos.copy()
                else:
                    return self._focus(), steps, ""
            if self.restarts > MAX_RESTARTS:
                return None, steps, "агент не сошёлся: трижды уходил за край КТ или ходил по кругу"
        return None, steps, f"не сошёлся за {MAX_STEPS} шагов"

    def _focus(self) -> np.ndarray:
        """Шесть проб из соседних вокселей; их среднее — точка (индекс точного масштаба)."""
        centre = self.pos.copy()
        total = np.zeros(3)
        for d in MOVES:
            self.memory[self.level].clear()
            self.pos = centre + FOCUS_RADIUS * d
            for _ in range(MAX_STEPS):
                self._move(self._act())
                done = self._visited()
                self._remember()
                if done:
                    break
            total += self.pos
        return total / len(MOVES)


def _index_to_lps(scale: _Scale, zyx: np.ndarray) -> np.ndarray:
    return np.array(scale.sitk_image.TransformContinuousIndexToPhysicalPoint([float(v) for v in zyx[::-1]]))


class Models:
    """Сети точек из папки: <точка ALI>/<масштаб>.onnx (1.onnx и 0-3.onnx)."""

    def __init__(self, folder: str, device: str = "cpu"):
        self.folder, self.device = folder, device

    def available(self) -> list[str]:
        return [k for k, ali in POINTS.items()
                if all(os.path.isfile(os.path.join(self.folder, ali, f"{s}.onnx")) for s in SCALE_KEYS)]

    def load(self, ali: str, threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        wanted = {"cpu": ["CPUExecutionProvider"],
                  "auto": ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]}[self.device]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        nets = []
        for s in SCALE_KEYS:
            sess = ort.InferenceSession(os.path.join(self.folder, ali, f"{s}.onnx"), opts, providers=providers)
            nets.append(lambda x, sess=sess: sess.run(None, {"zone": x})[0])
        return nets


def find(vol: Volume, models: Models, keys=None, progress=None, seed: int = 0, workers: int = 2,
         cancel: threading.Event | None = None) -> dict[str, Found]:
    """Найти точки keys (по умолчанию все, для которых есть сети) на КТ vol.

    progress(доля, сообщение). Точки ищутся параллельно в workers потоках;
    каждая точка — со своим генератором случайных чисел (seed + номер), поэтому
    результат не зависит от числа потоков.
    """
    keys = [k for k in (keys or POINTS) if k in models.available()]
    if not keys:
        return {}
    scales = prepare(vol)
    threads = max(1, (os.cpu_count() or 2) // max(workers, 1))
    done, lock = [0], threading.Lock()

    def one(i_key):
        i, key = i_key
        if cancel is not None and cancel.is_set():
            return Found(key, None, 0, "остановлено")
        agent = _Agent(scales, models.load(POINTS[key], threads), np.random.default_rng(seed + i))
        zyx, steps, why = agent.search()
        point = None if zyx is None else _index_to_lps(scales[-1], zyx)
        with lock:
            done[0] += 1
            if progress:
                progress(done[0] / len(keys), f"Точки: {done[0]} из {len(keys)}")
        return Found(key, point, steps, why or APPROXIMATE.get(key, ""))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = list(pool.map(one, enumerate(keys)))
    return {f.key: f for f in found}


def model_info(folder: str) -> dict:
    """Что за сети лежат в папке (manifest.json, если есть)."""
    path = os.path.join(folder, "manifest.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}
