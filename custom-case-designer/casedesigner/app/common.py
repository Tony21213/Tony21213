"""Общее для сессии и шагов (casedesigner/app/steps): импорты, константы, помощники. Разделено из session.py."""

import dataclasses
import io
import os
import threading
import uuid

import numpy as np
import trimesh

from .. import articulators as arts
from .. import condyles as cond
from .. import guidance as gd
from . import errorlog
from .. import exocad_project
from .. import model_store
from .. import landmarks as lmk
from .. import mount_planes as mplanes
from .. import motion as mo
from ..fusion import (BITE, SHARED_SHARE, CaseCT, Registration, Scan, deviation_colors, export_case, in_occlusion,
                      is_bite_name, jaw_by_name, place_bites, shared_share, split_by_jaw)
from ..jawcase import JawCase, Mesh as JawMesh
from ..learning import AlignmentMemory
from ..register import apply, rigid
from ..segment import Segmenter
from ..structures import jaw_of, structure
from ..volume import load_volume

# Оси срезов: какие оси DICOM идут по горизонтали и вертикали картинки и где верх.
# Радиологическая раскладка: правая сторона пациента — слева на экране, перед — сверху,
# голова — сверху; на сагиттальном срезе лицо смотрит влево.
SLICES = {
    "axial": {"normal": 2, "u": 0, "v": 1, "v_down": True},
    "coronal": {"normal": 1, "u": 0, "v": 2, "v_down": False},
    "sagittal": {"normal": 0, "u": 1, "v": 2, "v_down": False},
}
SLICE_PIXELS = 512
RECENT = 8  # недавних кейсов на стартовом экране
CASE_EXT = ".ccdcase"
MOUNT_PLANE_SETTING = "aesthetic_hinge"  # плоскость монтажа по умолчанию — решение владельца
DEV_COLORS = ("#28aa46", "#e6be1e", "#d23228", "#8a8f99")  # ≤ 0.1, ≤ 0.2, > 0.2 мм, не коронки (как карта в 3D)
MAX_SLICE_PIXELS = 2048  # сторона картинки видимой части среза
UPPER_BONES, LOWER_BONES = ("skull", "maxilla"), ("mandible",)  # в exocad — bone_upper.stl / bone_lower.stl
EXOCAD_SUBFOLDER = "KStomCaseDesigner"  # куда в папке проекта exocad кладётся экспорт
SCAN_COLORS = ["#7fb2ff", "#ffb86b", "#b48cff", "#6be0c1"]
# Скан совмещён до сегментации — только по плотности (см. CaseCT.use_teeth).
UNGUIDED = ("КТ не сегментировано: зубы найдены только по плотности, и на снимке с большим полем скан может сесть "
            "со сдвигом вдоль дуги на несколько миллиметров — метрики этого не покажут. Сегментируйте КТ: скан "
            "совместится заново по зубам.")
# Что сегментировать (настройка): часть → ключи структур; ключ на «/» или «_» — начало ключа.
SEGMENT_PARTS = [
    ("teeth", "Зубы с номерами FDI", ("teeth/tooth_", "upper_teeth", "lower_teeth")),
    ("pulp", "Пульпа зубов", ("pulp/",)),
    ("jaws", "Верхняя и нижняя челюсть", ("mandible", "maxilla")),
    ("canals", "Каналы нижней челюсти", ("mandibular_canal", "incisive_canal", "lingual_canal")),
    ("prosthetics", "Импланты, коронки, мосты", ("teeth/implant", "teeth/crown", "teeth/bridge")),
    ("sinuses", "Гайморовы и лобные пазухи", ("maxillary_sinus", "frontal_sinus")),
    ("airway", "Полость носа, нёбо, глотка", ("nasal_cavity", "pharynx", "nasopharynx", "oropharynx", "hypopharynx",
                                              "soft_palate", "hard_palate")),
    ("skull", "Череп", ("skull",)),
    ("ears", "Слуховые проходы", ("auditory_canal_",)),
]
# Зубы из сегментации нужны совмещению (CaseCT.use_teeth), даже если их не показывать.
GUIDE_KEYS = ("upper_teeth", "lower_teeth")


def part_of(key: str) -> str | None:
    for part, _title, keys in SEGMENT_PARTS:
        if any(key.startswith(k) if k.endswith(("/", "_")) else key == k for k in keys):
            return part
    return None


GROUPS = [
    ("Кости", ("mandible", "maxilla", "skull", "hard_palate")),
    ("Зубы", ("upper_teeth", "lower_teeth", "teeth/", "pulp/")),
    ("Каналы", ("mandibular_canal", "incisive_canal", "lingual_canal")),
    ("Пазухи и дыхательные пути", ("maxillary_sinus", "frontal_sinus", "nasal_cavity", "pharynx", "nasopharynx",
                                   "oropharynx", "hypopharynx", "soft_palate", "auditory_canal_right",
                                   "auditory_canal_left")),
]


def _axis_aligned(direction: np.ndarray) -> bool:
    """Оси КТ идут по осям пациента (с точностью до знака и перестановки) — так почти у всех КЛКТ."""
    a = np.abs(np.asarray(direction, float))
    return bool(np.allclose(a.max(axis=0), 1, atol=1e-4) and np.allclose(a.sum(axis=0), 1, atol=1e-4))


class Sectioner:
    """Быстрое сечение сетки плоскостью, перпендикулярной оси пациента.

    Треугольники заранее упорядочены по нижнему краю вдоль каждой оси: для
    плоскости берутся только те, что её пересекают, и отрезки считаются
    сразу для всех. Сетка сканов с сотнями тысяч треугольников режется за
    миллисекунды, а не за десятки.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.v = np.asarray(vertices, np.float64)
        self.f = np.asarray(faces, np.int64)
        self.axes = {}

    def _axis(self, n: int):
        if n not in self.axes:
            z = self.v[self.f, n]  # (M, 3)
            lo, hi = z.min(1), z.max(1)
            order = np.argsort(lo, kind="stable")
            self.axes[n] = (order, lo[order], hi)
        return self.axes[n]

    def cut(self, n: int, pos: float, uv: list[int], with_faces: bool = False):
        """Отрезки сечения плоскостью «координата n = pos»: плоский список u1, v1, u2, v2, …
        (with_faces — и номера треугольников, по отрезку на треугольник)."""
        empty = ([], np.zeros(0, np.int64)) if with_faces else []
        if not len(self.f):
            return empty
        order, lo_sorted, hi = self._axis(n)
        cand = order[: np.searchsorted(lo_sorted, pos, side="right")]
        cand = cand[hi[cand] >= pos]
        if not len(cand):
            return empty
        tri = self.v[self.f[cand]]  # (K, 3, 3)
        d = tri[..., n] - pos
        d = np.where(d == 0, 1e-9, d)  # вершина ровно на плоскости — чуть выше: у треугольника ровно 2 пересечения
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cross = (d[:, a] > 0) != (d[:, b] > 0)
            t = d[:, a] / np.where(cross, d[:, a] - d[:, b], 1.0)
            p = tri[:, a] + (tri[:, b] - tri[:, a]) * t[:, None]
            pts.append((cross, p[:, uv]))
        (c0, p0), (c1, p1), (c2, p2) = pts
        first = np.where(c0[:, None], p0, p1)
        second = np.where((c0 & c1)[:, None], p1, p2)
        keep = (c0.astype(int) + c1 + c2) == 2
        seg = np.concatenate([first[keep], second[keep]], axis=1)
        flat = np.round(seg.reshape(-1), 3).tolist()
        return (flat, cand[keep]) if with_faces else flat


def group_of(key: str) -> str:
    if key in ("teeth/implant", "teeth/crown", "teeth/bridge"):
        return "Ортопедия и импланты"
    for title, keys in GROUPS:
        if any(key == k or (k.endswith("/") and key.startswith(k)) for k in keys):
            return title
    return "Другое"


def mesh_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Сетка для интерфейса: число вершин и граней (uint32), вершины (float32), грани (uint32)."""
    v = np.ascontiguousarray(vertices, np.float32)
    f = np.ascontiguousarray(faces, np.uint32)
    return np.array([len(v), len(f)], np.uint32).tobytes() + v.tobytes() + f.tobytes()



def open_folder(folder: str):
    """Показать папку в проводнике (Windows) или файловом менеджере."""
    import subprocess
    import sys

    if sys.platform.startswith("win"):
        os.startfile(folder)  # noqa: S606 — своя папка программы
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])

JAWS_APART = ("Без КТ: сканы челюстей не в прикусе (выгружены по отдельности), а сканов прикуса нет — нижний "
              "стоит как в файле. Добавьте сканы прикуса или выгрузите сканы одной сессией сканера.")


def scanner_registration(scan: Scan, jaw: str) -> Registration:
    """Скан без КТ: «совмещён» как есть — координаты файла (сканера), без метрик (об этом — пояснение в шаге)."""
    return Registration(scan, jaw, np.eye(4), np.full(len(scan.vertices), np.nan))
