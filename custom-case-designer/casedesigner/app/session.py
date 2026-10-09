"""Состояние открытого кейса и операции над ним — то, что вызывает интерфейс.

Всё в координатах пациента из DICOM (мм, LPS): КТ, структуры, сканы (через
текущую матрицу «скан → КТ»). Интерфейс двигает скан, присылает новую
матрицу, а сессия оценивает точность и при желании уточняет положение.
"""

import dataclasses
import io
import os
import threading
import uuid

import numpy as np
import trimesh

from .. import articulators as arts
from . import errorlog
from .. import exocad_project
from .. import model_store
from .. import landmarks as lmk
from ..fusion import (BITE, SHARED_SHARE, CaseCT, Registration, Scan, deviation_colors, export_case, in_occlusion,
                      is_bite_name, jaw_by_name, place_bites, shared_share, split_by_jaw)
from ..jawcase import JawCase, Mesh as JawMesh
from ..learning import AlignmentMemory
from ..register import apply
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
DEV_COLORS = ("#28aa46", "#e6be1e", "#d23228", "#8a8f99")  # ≤ 0.1, ≤ 0.2, > 0.2 мм, не коронки (как карта в 3D)
MAX_SLICE_PIXELS = 2048  # сторона картинки видимой части среза
UPPER_BONES, LOWER_BONES = ("skull", "maxilla"), ("mandible",)  # в exocad — бюгельными каркасами своей челюсти
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


class Session:
    def __init__(self, memory_path: str | None = None, models_dir: str | None = None):
        self.lock = threading.RLock()
        self.memory = AlignmentMemory(memory_path or os.path.join(os.path.expanduser("~"), ".casedesigner",
                                                                    "memory.jsonl"))
        self.models_dir = models_dir
        self.ct_path = None
        self.vol = None
        self.case: CaseCT | None = None
        self.window = (400.0, 3000.0)
        self.scans: dict[str, dict] = {}
        # Прикус сканов в окне: нижняя челюсть со всеми её структурами — в прикусе сканов (врача), а не как на КТ.
        self.bite_view = False
        self.bite_motion = None  # движение нижней челюсти с КТ в прикус сканов (мм КТ → мм КТ); None — прикуса нет
        self._bite_meshes = {}
        self.structures: dict[str, trimesh.Trimesh] = {}
        self._sections = {}
        self.landmarks: dict[str, np.ndarray] = {}
        self.suggested: set[str] = set()  # предложены программой и ещё не подтверждены врачом
        self.articulators_path = os.path.join(os.path.dirname(self.memory.path), "articulators.json")
        self.settings_path = os.path.join(os.path.dirname(self.memory.path), "settings.json")
        self.settings = self._load_settings()
        self.case_path: str | None = None  # файл кейса, куда сохранять
        self.guide_teeth: dict = {}  # зубы из сегментации — опора совмещения (сохраняются с кейсом)
        self.jaw = JawCase()  # артикуляция: монтаж, суставы, движения, контакты (интерфейс — позже)
        self._articulation_ready = False

    # --- настройки (рядом с памятью совмещений) -----------------------------
    def _load_settings(self) -> dict:
        import json

        known = [p for p, _t, _k in SEGMENT_PARTS]
        settings = {"segment_parts": known, "recent": [], "incognito": False}
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                saved = json.load(f)
            settings["segment_parts"] = [p for p in known if p in saved.get("segment_parts", [])]
            settings["recent"] = [r for r in saved.get("recent", []) if isinstance(r, str)][:RECENT]
            settings["incognito"] = saved.get("incognito") is True
        except (OSError, ValueError, AttributeError):  # нет файла или он испорчен — всё по умолчанию
            pass
        return settings

    def set_segment_parts(self, parts: list[str]) -> dict:
        known = [p for p, _t, _k in SEGMENT_PARTS]
        unknown = sorted(set(parts) - set(known))
        if unknown:
            raise ValueError(f"нет таких частей: {', '.join(unknown)}")
        self.settings["segment_parts"] = [p for p in known if p in parts]
        self._save_settings()
        return self.segment_parts_info()

    def _save_settings(self):
        import json

        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=1)

    def set_incognito(self, on: bool) -> dict:
        """Инкогнито: интерфейс не показывает имён пациентов и путей (для показа программы). Помнится между
        запусками — программа, открытая на демонстрации, сразу скрывает недавние кейсы."""
        self.settings["incognito"] = bool(on)
        self._save_settings()
        return {"incognito": self.settings["incognito"]}

    def segment_parts_info(self) -> dict:
        return {"parts": [{"id": p, "title": t} for p, t, _k in SEGMENT_PARTS],
                "selected": list(self.settings["segment_parts"])}

    # --- КТ -----------------------------------------------------------------
    def ct_series(self, path: str) -> list[dict]:
        """Серии DICOM в папке или архиве — чтобы выбрать, какую открыть (у файла КТ — пусто)."""
        from ..volume import series_of

        errorlog.private(path, "КТ")
        return series_of(path)

    def load_ct(self, path: str, progress=None, series: str | None = None) -> dict:
        errorlog.private(path, "КТ")
        vol = load_volume(path, series)
        if progress:
            progress(0.5, "Ищу коронки зубов")
        prior = self.memory.prior(vol.device)
        case = CaseCT(vol, *prior)
        with self.lock:
            self.ct_path, self.vol, self.case, self.ct_series_id = path, vol, case, series
            self.case_path, self.guide_teeth = None, {}
            lo, hi = case.levels.hard, case.levels.dense
            self.window = ((lo + hi) / 2, max(hi - lo, 1.0) * 2.5)
            self.structures, self._sections = {}, {}
            self.bite_view, self.bite_motion, self._bite_meshes = False, None, {}
            self.landmarks, self.suggested = {}, set()
            for item in self.scans.values():
                item.update(reg=None, auto=None, transform=None, guided=False, lower_bite=None)
        return self.ct_info()

    def close_ct(self):
        """Без КТ: только сканы (кейс, сохранённый без КТ)."""
        with self.lock:
            self.ct_path, self.vol, self.case, self.ct_series_id = None, None, None, None
            self.case_path, self.guide_teeth = None, {}
            self.structures, self._sections = {}, {}
            self.bite_view, self.bite_motion, self._bite_meshes = False, None, {}
            self.landmarks, self.suggested = {}, set()
            for item in self.scans.values():
                item.update(reg=None, auto=None, transform=None, guided=False, lower_bite=None)

    def ct_info(self) -> dict | None:
        if self.vol is None:
            return None
        lo, hi = self.bounds()
        prior = self.memory.prior(self.vol.device)
        crowns = np.vstack([t.points for t in self.case.coarse.values()])
        return {"path": self.ct_path, "name": os.path.basename(os.path.normpath(self.ct_path)),
                "focus": crowns.mean(axis=0).round(2).tolist(),  # срезы открываются на зубах
                "shape": list(self.vol.data.shape[::-1]), "spacing": self.vol.spacing.round(4).tolist(),
                "device": self.vol.device, "bounds": [lo.tolist(), hi.tolist()],
                "window": [round(self.window[0]), round(self.window[1])],
                "levels": {"hard": round(float(self.case.levels.hard)), "dense": round(float(self.case.levels.dense)),
                           "min": round(float(self._cval))},
                "learned_edge_shift_mm": round(prior[0], 3) if prior[1] else None}

    def bounds(self):
        n = np.array(self.vol.data.shape[::-1]) - 1
        corners = np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])], float)
        w = self.vol.to_world(corners)
        return w.min(axis=0), w.max(axis=0)

    def slice_geometry(self, axis: str) -> dict:
        """Сетка картинки среза: мм пациента пикселя (i, j) = (u0 + i·du, v0 + j·dv) по осям u, v."""
        s = SLICES[axis]
        lo, hi = self.bounds()
        step = max(hi[s["u"]] - lo[s["u"]], hi[s["v"]] - lo[s["v"]]) / SLICE_PIXELS
        width = int((hi[s["u"]] - lo[s["u"]]) / step) + 1
        height = int((hi[s["v"]] - lo[s["v"]]) / step) + 1
        v0, dv = (float(lo[s["v"]]), step) if s["v_down"] else (float(lo[s["v"]] + (height - 1) * step), -step)
        return {"axis": axis, "u_axis": s["u"], "v_axis": s["v"], "normal_axis": s["normal"],
                "u0": float(lo[s["u"]]), "du": step, "v0": v0, "dv": dv, "width": width, "height": height,
                "range": [float(lo[s["normal"]]), float(hi[s["normal"]])], "step": step}

    def slice_values(self, axis: str, pos: float, region=None, size=None) -> np.ndarray:
        """Яркость среза (строки × столбцы); region=(ua, ub, va, vb) — только эта часть (мм по осям среза,
        края слева-направо и сверху-вниз), size=(cols, rows) — в таком разрешении."""
        g = self.slice_geometry(axis)
        s = SLICES[axis]
        if region is None:
            us, vs = g["u0"] + g["du"] * np.arange(g["width"]), g["v0"] + g["dv"] * np.arange(g["height"])
        else:
            (ua, ub, va, vb), (cols, rows) = region, (int(np.clip(n, 1, MAX_SLICE_PIXELS)) for n in size)
            us, vs = ua + (np.arange(cols) + 0.5) * (ub - ua) / cols, va + (np.arange(rows) + 0.5) * (vb - va) / rows
        if _axis_aligned(self.vol.direction):
            return self._plane(s, pos, us, vs)
        uu, vv = np.meshgrid(us, vs)  # КТ повёрнут относительно осей пациента — общий путь
        pts = np.zeros((uu.size, 3))
        pts[:, s["u"]], pts[:, s["v"]], pts[:, s["normal"]] = uu.ravel(), vv.ravel(), pos
        return self.vol.sample(pts).reshape(uu.shape)

    def _plane(self, s: dict, pos: float, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """Срез КТ, лежащего по осям пациента: плоскость берётся из массива (линейно между соседними
        слоями), в плоскости — билинейно по отдельности вдоль строк и столбцов. В десятки раз быстрее
        трёхмерной интерполяции и даёт то же самое."""
        vol = self.vol
        data = vol.data
        cval = self._cval
        to_index = vol.to_index
        k = {a: int(np.argmax(np.abs(vol.direction[a]))) for a in range(3)}  # ось пациента → ось индекса x,y,z

        def along(a, values):  # индекс вдоль оси пациента a для значений values (остальные — любые)
            pts = np.zeros((len(values), 3))
            pts[:, s["normal"]] = pos
            pts[:, a] = values
            return to_index(pts)[:, k[a]]

        def axis_of(index_axis):  # ось массива data ([z, y, x]) для оси индекса x/y/z
            return 2 - index_axis

        n_idx = float(along(s["normal"], np.array([pos]))[0])
        n0 = int(np.floor(n_idx))
        t = n_idx - n0
        n_axis = axis_of(k[s["normal"]])
        size_n = data.shape[n_axis]

        def layer(i):
            if 0 <= i < size_n:
                return np.take(data, i, axis=n_axis).astype(np.float32)
            return None

        a, b = layer(n0), layer(n0 + 1)
        if a is None and b is None:
            return np.full((len(vs), len(us)), cval, np.float32)
        a = a if a is not None else np.full_like(b, cval)
        b = b if b is not None else np.full_like(a, cval)
        plane = a * (1 - t) + b * t if t > 1e-6 else a
        rest = [ax for ax in (0, 1, 2) if ax != n_axis]  # оси массива, оставшиеся в плоскости
        u_ax, v_ax = rest.index(axis_of(k[s["u"]])), rest.index(axis_of(k[s["v"]]))
        plane = plane if (v_ax, u_ax) == (0, 1) else plane.T  # строки — v, столбцы — u

        def weights(idx, n):
            i0 = np.floor(idx).astype(np.int64)
            f = (idx - i0).astype(np.float32)
            ok0, ok1 = (i0 >= 0) & (i0 < n), (i0 + 1 >= 0) & (i0 + 1 < n)
            return np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1), f, ok0, ok1

        iu0, iu1, fu, ou0, ou1 = weights(along(s["u"], us), plane.shape[1])
        iv0, iv1, fv, ov0, ov1 = weights(along(s["v"], vs), plane.shape[0])
        p = np.where(ov0[:, None], plane[iv0], cval)  # строки
        q = np.where(ov1[:, None], plane[iv1], cval)
        rows = p * (1 - fv)[:, None] + q * fv[:, None]
        left = np.where(ou0[None], rows[:, iu0], cval)
        right = np.where(ou1[None], rows[:, iu1], cval)
        return left * (1 - fu)[None] + right * fu[None]

    @property
    def _cval(self) -> float:
        if getattr(self, "_cval_for", None) is not self.vol:
            self._cval_for, self._cval_value = self.vol, float(self.vol.data.min())
        return self._cval_value

    def _windowed(self, values: np.ndarray, level, width) -> np.ndarray:
        level = self.window[0] if level is None else level
        width = self.window[1] if width is None else width
        return np.clip((values - (level - width / 2)) * (255.0 / width), 0, 255).astype(np.uint8)

    def slice_png(self, axis: str, pos: float, level: float | None = None, width: float | None = None,
                  region=None, size=None) -> bytes:
        """Срез картинкой PNG (оттенки серого в окне level/width)."""
        from PIL import Image

        img = self._windowed(self.slice_values(axis, pos, region, size), level, width)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    def slice_raw(self, axis: str, pos: float, level: float | None = None, width: float | None = None,
                  region=None, size=None) -> bytes:
        """Срез без сжатия — для интерфейса: столбцы и строки (uint32), затем байты серого построчно.
        Сжатие PNG и его разбор в браузере дольше, чем передать байты по локальному соединению."""
        img = self._windowed(self.slice_values(axis, pos, region, size), level, width)
        return np.array([img.shape[1], img.shape[0]], np.uint32).tobytes() + np.ascontiguousarray(img).tobytes()

    # --- сечения сеток плоскостью среза --------------------------------------
    def _sectioner(self, key: str, vertices: np.ndarray, faces: np.ndarray, version) -> "Sectioner":
        cache = self.__dict__.setdefault("_sectioners", {})
        item = cache.get(key)
        if item is None or item[0] != version:
            item = (version, Sectioner(vertices, faces))
            cache[key] = item
        return item[1]

    def overlays(self, axis: str, pos: float, visible: list[str], heat: bool = False) -> list[dict]:
        out = []
        n = SLICES[axis]["normal"]
        uv = [SLICES[axis]["u"], SLICES[axis]["v"]]
        with self.lock:
            for sid, item in self.scans.items():
                if item["transform"] is None or sid not in visible:
                    continue
                T = self.shown_transform(item)
                sec = self._sectioner(f"scan:{sid}", apply(T, item["scan"].vertices), item["scan"].faces,
                                      (id(item["scan"]), T.tobytes()))
                if not heat or item["reg"] is None:
                    out.append({"id": sid, "color": item["color"], "width": 1.6, "segments": sec.cut(n, pos, uv)})
                    continue
                # Контур по цвету отклонения от КТ: где скан лёг на эмаль, а где нет.
                segs, faces = sec.cut(n, pos, uv, with_faces=True)
                if not len(faces):
                    continue
                dev = np.abs(item["reg"].deviation)
                d = np.nanmax(np.where(np.isnan(dev[item["scan"].faces[faces]]), -1, dev[item["scan"].faces[faces]]), 1)
                segs = np.asarray(segs).reshape(-1, 4)
                for color, mask in ((DEV_COLORS[0], (d >= 0) & (d <= 0.1)), (DEV_COLORS[1], (d > 0.1) & (d <= 0.2)),
                                    (DEV_COLORS[2], d > 0.2), (DEV_COLORS[3], d < 0)):
                    if mask.any():
                        out.append({"id": sid, "color": color, "width": 1.6 if color != DEV_COLORS[3] else 1.0,
                                    "segments": segs[mask].reshape(-1).tolist()})
            for key, mesh in self.structures.items():
                if key in visible:
                    sec = self._sectioner(f"structure:{key}", *self._shown_structure(key))
                    out.append({"id": key, "color": structure(key).color, "width": 1.2,
                                "segments": sec.cut(n, pos, uv)})
        return [o for o in out if o["segments"]]

    # --- сканы -------------------------------------------------------------
    def add_scan(self, path: str) -> dict:
        errorlog.private(path, "скан")
        scan = Scan.load(path)
        errorlog.private(scan.name, "скан")
        sid = uuid.uuid4().hex[:8]
        with self.lock:
            color = SCAN_COLORS[len(self.scans) % len(SCAN_COLORS)]
            self.scans[sid] = {"scan": scan, "path": path, "color": color, "jaw": None,
                               "reg": None, "auto": None, "transform": None, "accepted": False, "guided": False,
                               "role": BITE if is_bite_name(scan.name) else "jaw", "lower_bite": None}
        return self.scan_info(sid)

    def remove_scan(self, sid: str):
        with self.lock:
            self.scans.pop(sid, None)
        self._place_bites()

    # --- сканы прикуса: стоят на сканах челюстей и задают прикус врача ---------------------
    def _jaw_items(self, skip: str | None = None):
        """Совмещённые сканы челюстей (не прикуса): верхний и нижний, первые по порядку."""
        found = {}
        for sid, item in self.scans.items():
            if sid != skip and item["role"] != BITE and item["reg"] is not None:
                found.setdefault(item["reg"].jaw, item)
        return found.get("upper"), found.get("lower")

    def _on_both_jaws(self, sid: str) -> bool:
        """Скан лежит в координатах файла и на верхнем, и на нижнем скане — значит, это скан прикуса."""
        up, lo = self._jaw_items(skip=sid)
        if up is None or lo is None:
            return False
        scan = self.scans[sid]["scan"]
        return shared_share(scan, up["scan"]) >= SHARED_SHARE and shared_share(scan, lo["scan"]) >= SHARED_SHARE

    def _place_bites(self):
        """Сканы прикуса — на сканы челюстей; нижний скан — в прикус по ним, если челюсти не в прикусе."""
        # Любое изменение совмещения требует заново передать модели в окно
        # артикулятора, иначе просмотр мог бы использовать старую гипсовку.
        self._articulation_ready = False
        try:
            self._place_bite_scans()
        finally:
            self._update_bite_motion()

    def _update_bite_motion(self):
        """Как переезжает нижняя челюсть с КТ в прикус сканов: по сканам прикуса или прикус со сканера."""
        up, lo = self._jaw_items()
        motion = None
        # Без КТ: сканы челюстей не в прикусе и сканов прикуса нет — нижний стоит как в файле (подсказка у скана).
        self.jaws_apart = bool(self.case is None and up is not None and lo is not None and lo.get("lower_bite") is None
                               and not (in_occlusion(up["scan"], lo["scan"]) or in_occlusion(lo["scan"], up["scan"])))
        if up is not None and lo is not None and self.case is not None:
            if lo.get("lower_bite") is not None:
                motion = lo["lower_bite"] @ np.linalg.inv(lo["reg"].transform)
            elif in_occlusion(up["scan"], lo["scan"]) or in_occlusion(lo["scan"], up["scan"]):
                motion = up["reg"].transform @ np.linalg.inv(lo["reg"].transform)  # общие координаты сканера
        self.bite_motion, self._bite_meshes = motion, {}
        if motion is None:
            self.bite_view = False

    def _lower_moves(self, item) -> bool:
        """Скан нижней челюсти, который в прикусе сканов стоит не как на КТ."""
        return bool(self.bite_view and self.bite_motion is not None and item["role"] != BITE
                    and item["reg"] is not None and item["reg"].jaw == "lower")

    def shown_transform(self, item):
        """Где скан показан: как на КТ или, с включённым прикусом, нижние — в прикусе сканов."""
        T = item["transform"]
        if self.case is None and T is not None and item.get("lower_bite") is not None:
            return item["lower_bite"]  # без КТ прикус один — по сканам прикуса
        return self.bite_motion @ T if T is not None and self._lower_moves(item) else T

    def _shown_structure(self, key: str):
        """Сетка структуры, как показана: с включённым прикусом структуры нижней челюсти — за ней,
        у смешанных (импланты, коронки, мосты) — их нижняя часть."""
        mesh = self.structures[key]
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        jaw = jaw_of(key)
        if not self.bite_view or self.bite_motion is None or jaw == "upper":
            return v, f, id(mesh)
        cached = self._bite_meshes.get(key)
        if cached is None:
            from ..segment import Mesh

            if jaw == "lower":
                v = apply(self.bite_motion, v)
            else:
                vs, fs, offset = [], [], 0
                for part, m in split_by_jaw(Mesh(v, f), self.case).items():
                    vs.append(apply(self.bite_motion, m.vertices) if part == "lower" else m.vertices)
                    fs.append(m.faces + offset)
                    offset += len(m.vertices)
                v, f = np.vstack(vs), np.vstack(fs)
            cached = self._bite_meshes[key] = (v, f, ("bite", id(mesh), self.bite_motion.tobytes()))
        return cached

    def set_bite_view(self, on: bool) -> dict:
        if on and self.bite_motion is None:
            raise ValueError("прикуса сканов нет: сканы челюстей выгружены не в прикусе, а сканов прикуса нет")
        self.bite_view, self._bite_meshes = bool(on), {}
        return self.bite_info()

    def bite_info(self) -> dict:
        info = {"on": self.bite_view, "available": self.bite_motion is not None, "shift_mm": None}
        if self.bite_motion is not None:
            _up, lo = self._jaw_items()
            at = apply(lo["reg"].transform, lo["scan"].vertices[lo["scan"].crowns])
            info["shift_mm"] = round(float(np.linalg.norm(apply(self.bite_motion, at) - at, axis=1).mean()), 2)
        return info

    def _place_bite_scans(self):
        up, lo = self._jaw_items()
        for item in self.scans.values():
            item["lower_bite"] = None
        bites = [item for item in self.scans.values() if item["role"] == BITE]
        if not bites:
            return
        if up is None and lo is None:
            for item in bites:
                item.update(reg=None, auto=None, transform=None, note=None)
            return
        result = place_bites([item["scan"] for item in bites], up and up["reg"], lo and lo["reg"], self.case,
                             on_ct=self.case is not None)
        with self.lock:
            for k, item in enumerate(bites):
                reg = result.regs[k]
                item.update(reg=reg, auto=reg, transform=None if reg is None else reg.transform, guided=True,
                            note=result.failed.get(k))
            if lo is not None:
                lo["lower_bite"] = result.lower

    def scan_info(self, sid: str) -> dict:
        item = self.scans[sid]
        reg: Registration | None = item["reg"]
        return {"id": sid, "name": item["scan"].name, "path": item["path"], "color": item["color"],
                "vertices": len(item["scan"].vertices),
                "jaw": BITE if item["role"] == BITE else (reg.jaw if reg else item["jaw"]), "role": item["role"],
                "bite_from_scans": item.get("lower_bite") is not None,
                "transform": None if item["transform"] is None else self.shown_transform(item).tolist(),
                "auto_transform": None if item["auto"] is None else item["auto"].transform.tolist(),
                "registered": reg is not None, "accepted": item["accepted"],
                "stats": reg.stats if reg else None, "edge_shift_mm": round(reg.edge_shift, 3) if reg else None,
                "warnings": ((([] if item["guided"] or item["role"] == BITE else [UNGUIDED]) + reg.warnings
                              + ([JAWS_APART] if self._apart(item) else [])) if reg
                             else [item["note"]] if item.get("note") else []),
                "segments": [vars(s) for s in reg.segments] if reg else [],
                "corrected_mm": self._corrected(sid)}

    def _apart(self, item) -> bool:
        return bool(getattr(self, "jaws_apart", False) and item["reg"] is not None and item["role"] != BITE
                    and item["reg"].jaw == "lower")

    def _corrected(self, sid: str) -> float | None:
        item = self.scans[sid]
        if item["auto"] is None or item["transform"] is None:
            return None
        v = item["scan"].vertices
        return round(float(np.linalg.norm(apply(item["transform"], v) - apply(item["auto"].transform, v), axis=1).max()), 3)

    def scan_mesh(self, sid: str) -> bytes:
        s = self.scans[sid]["scan"]
        return mesh_bytes(s.vertices, s.faces)

    def scan_colors(self, sid: str) -> bytes:
        reg = self.scans[sid]["reg"]
        if reg is None:
            return b""
        return np.ascontiguousarray(deviation_colors(reg.deviation)[:, :3], np.uint8).tobytes()

    def _require_ct(self):
        if self.case is None:
            raise ValueError("сначала откройте КТ")

    def register(self, sid: str, jaw: str | None = None, pairs=None, start=None, progress=None) -> dict:
        """Совместить скан с КТ; без КТ — поставить в координатах сканера (как в файле)."""
        if self.case is None and (start is not None or pairs is not None):
            raise ValueError("без КТ скан стоит как в файле: корректировать не по чему")
        item = self.scans[sid]
        if jaw is not None:  # челюсть указал пользователь (значок у скана)
            self.set_jaw(sid, jaw)
            jaw = None if jaw == BITE else jaw
        if item["role"] != BITE and jaw is None and start is None and pairs is None and self._on_both_jaws(sid):
            item["role"] = BITE  # лежит и на верхнем, и на нижнем скане — это скан прикуса
        if self._lower_moves(item) and (start is not None or pairs is not None):
            raise ValueError("выключите «Прикус», чтобы корректировать нижний скан: коррекция — по КТ")
        if item["role"] == BITE:  # с КТ не совмещается: стоит на сканах челюстей
            self._place_bites()
            if item["reg"] is None and not item.get("note"):
                raise ValueError("скан прикуса ставится на сканы челюстей: сначала добавьте и совместите их")
            return self.scan_info(sid)
        jaw = jaw or item["jaw"]
        if self.case is None:
            return self._place_without_ct(sid, jaw)
        reg = self.case.register(item["scan"], jaw=jaw, pairs=pairs,
                                 start=None if start is None else np.asarray(start, float))
        with self.lock:
            item["reg"], item["transform"], item["accepted"] = reg, reg.transform, False
            item["guided"] = reg.jaw in self.case.guided
            if item["auto"] is None or (start is None and pairs is None):
                item["auto"] = reg
        self._place_bites()
        return self.scan_info(sid)

    def _place_without_ct(self, sid: str, jaw: str | None) -> dict:
        """Без КТ: скан челюсти — в координатах своего файла (у выгрузки сканера они общие и в прикусе);
        челюсть — по имени файла, иначе противоположная уже поставленной."""
        item = self.scans[sid]
        if jaw is None:
            jaw = jaw_by_name(item["scan"].name)
        if jaw is None:
            taken = {it["reg"].jaw for s, it in self.scans.items()
                     if s != sid and it["reg"] is not None and it["role"] != BITE}
            jaw = {frozenset({"upper"}): "lower", frozenset({"lower"}): "upper"}.get(frozenset(taken))
        if jaw is None:
            raise ValueError(f"{item['scan'].name}: без КТ челюсть по имени файла не понять — укажите её "
                             "(щёлкните по значку челюсти у скана)")
        reg = scanner_registration(item["scan"], jaw)
        with self.lock:
            item.update(reg=reg, auto=reg, transform=reg.transform, accepted=False, guided=True, jaw=jaw)
        self._place_bites()
        return self.scan_info(sid)

    def evaluate(self, sid: str, transform) -> dict:
        self._require_ct()
        item = self.scans[sid]
        if item["role"] == BITE:  # скан прикуса двигается вместе со сканами челюстей
            self._place_bites()
            return self.scan_info(sid)
        if self._lower_moves(item):
            raise ValueError("выключите «Прикус», чтобы корректировать нижний скан: коррекция — по КТ")
        reg = self.case.evaluate(item["scan"], np.asarray(transform, float), jaw=item["reg"].jaw if item["reg"] else None)
        with self.lock:
            item["reg"], item["transform"], item["accepted"] = reg, reg.transform, False
            item["guided"] = reg.jaw in self.case.guided
        self._place_bites()
        return self.scan_info(sid)

    def set_jaw(self, sid: str, jaw: str | None):
        """Челюсть скана (upper/lower) или «скан прикуса» (bite)."""
        item = self.scans[sid]
        item["role"] = BITE if jaw == BITE else "jaw"
        item["jaw"] = None if jaw == BITE else (jaw or None)

    def accept(self, sid: str) -> dict:
        item = self.scans[sid]
        if item["reg"] is None:
            raise ValueError("скан ещё не совмещён")
        if item["role"] == BITE:
            raise ValueError("скан прикуса стоит на сканах челюстей — принимать нужно их")
        self._require_ct()
        from ..learning import MIN_CASES

        rec = self.memory.record(self.vol.device, item["auto"] or item["reg"], item["reg"])
        item["accepted"] = True
        # выученное — сразу в работу: следующие сканы этого кейса совмещаются уже с новой поправкой
        self.case.edge_prior, self.case.prior_weight = self.memory.prior(self.vol.device)
        return {"scan": self.scan_info(sid),
                "record": {k: rec[k] for k in ("corrected_mm", "edge_shift_mm")} | {"learned": self.memory.usable(rec)},
                "memory": self.memory.summary().get(self.vol.device or "unknown"), "min_cases": MIN_CASES}

    # --- модели сегментации --------------------------------------------------
    def models_status(self) -> dict:
        return model_store.status(self.models_dir or model_store.default_dir())

    def download_models(self, progress=None) -> dict:
        """Скачать недостающие модели сегментации (докачка, проверка SHA-256); после — сегментация доступна."""
        folder = self.models_dir or model_store.default_dir()
        self._download_cancel = threading.Event()
        result = model_store.download(folder, progress, self._download_cancel)
        self.models_dir = folder
        return result

    def cancel_download(self):
        cancel = getattr(self, "_download_cancel", None)
        if cancel is not None:
            cancel.set()

    # --- структуры ----------------------------------------------------------
    def segment(self, models_dir: str | None = None, device: str = "auto", progress=None) -> dict:
        """Сегментировать выбранные в настройке части КТ (segment_parts); зубы для совмещения — всегда."""
        self._require_ct()
        models_dir = models_dir or self.models_dir
        if not models_dir:
            raise ValueError("укажите папку моделей сегментации")
        parts = set(self.settings["segment_parts"])
        if not parts:
            raise ValueError("не выбрано, что сегментировать: откройте «Что сегментировать…»")
        shown = lambda key: part_of(key) in parts or part_of(key) is None  # noqa: E731
        segmenter = Segmenter(models_dir, device=device)
        names = {m.name: m.title for m in segmenter.models}
        planned = {m.name for m in segmenter.plan(lambda k: shown(k) or k in GUIDE_KEYS)}
        done = {}

        def report(model, n, total):
            done[model] = n / total
            if progress:
                progress(sum(done.values()) / len(planned), f"{names[model]}: окно {n}/{total}")

        result = segmenter.run(self.vol, progress=report, want=lambda k: shown(k) or k in GUIDE_KEYS)
        with self.lock:
            self.models_dir = models_dir
            self.structures = {k: trimesh.Trimesh(m.vertices, m.faces, process=False)
                               for k, m in result.meshes.items() if shown(k)}
        self.guide_teeth = {jaw: result.meshes.get(f"{jaw}_teeth") for jaw in ("upper", "lower")}
        self.case.use_teeth(self.guide_teeth)
        # Сканы, совмещённые до сегментации по одной плотности, — заново по зубам.
        # Принятые и поправленные вручную остаются как есть.
        for sid, item in list(self.scans.items()):
            if item["reg"] is not None and not item["guided"] and not item["accepted"] and item["reg"] is item["auto"]:
                if progress:
                    progress(1.0, f"Совмещаю заново по зубам: {item['scan'].name}")
                try:
                    self.register(sid, item["jaw"])
                except (ValueError, RuntimeError):  # не вышло — скан остаётся где был, с подсказкой
                    pass
        self._place_bites()  # челюсти могли переехать — сканы прикуса за ними
        return {**self.structures_info(), "scans": [self.scan_info(s) for s in self.scans]}

    def structures_info(self) -> dict:
        def order(key):  # сначала целые структуры, затем зубы и пульпа по номеру FDI
            kind = 1 if key.startswith("teeth/tooth_") else 2 if key.startswith("pulp/") else 0
            digits = "".join(ch for ch in key if ch.isdigit())
            return kind, int(digits) if kind and digits else 0, key

        items = [{"key": k, "name": structure(k).name, "color": structure(k).color, "group": group_of(k),
                  "jaw": structure(k).jaw} for k in sorted(self.structures, key=order)]
        return {"structures": items}

    def structure_mesh(self, key: str) -> bytes:
        v, f, _version = self._shown_structure(key)
        return mesh_bytes(v, f)

    # --- предварительная артикуляция ---------------------------------------
    def _prepare_jawcase(self):
        """Передать совмещённые сканы и кости в ядро артикулятора.

        Монтаж всегда выполняется в положении прикуса сканов, если такой
        прикус найден. Это даёт пользователю ту же картину, которую он увидит
        при экспорте, ещё до создания файлов exocad.
        """
        up, lo = self._jaw_items()
        if up is None or lo is None or up.get("reg") is None or lo.get("reg") is None:
            raise ValueError("сначала поставьте верхний и нижний сканы")

        def vertices(item, bite=True):
            T = item["reg"].transform
            if bite and item["reg"].jaw == "lower" and self.bite_motion is not None:
                T = self.bite_motion @ T
            return apply(T, item["scan"].vertices)

        self.jaw.set_scans(JawMesh(vertices(up), up["scan"].faces, "скан верхней челюсти"),
                           JawMesh(vertices(lo), lo["scan"].faces, "скан нижней челюсти"))
        if self.case is not None:
            for key, attr in (("mandible", "mandible"), ("skull", "skull")):
                source = self.structures.get(key)
                if source is not None:
                    setattr(self.jaw, attr, JawMesh(np.asarray(source.vertices), np.asarray(source.faces), key))
        self._articulation_ready = True

    def articulation_prepare(self) -> dict:
        """Подготовить окно визуальной проверки артикуляции."""
        self._prepare_jawcase()
        return self.jaw.state()

    def articulation_mount(self, method: str = "auto") -> dict:
        """Смонтировать модели для предварительного просмотра."""
        if not self._articulation_ready:
            self._prepare_jawcase()
        return self.jaw.mount(method)

    def articulation_generate(self, travel: float = 6.0) -> dict:
        """Построить виртуальные движения для просмотра до экспорта."""
        if not self._articulation_ready:
            self._prepare_jawcase()
        self.jaw.generate(travel=float(travel), guided=True)
        return self.jaw.state()

    def articulation_analysis(self) -> dict:
        """Рассчитать сводку суставных путей и контактов."""
        if not self._articulation_ready:
            self._prepare_jawcase()
        return self.jaw.analysis()

    def articulation_motion(self, mid: str) -> dict:
        """Отдать кадры одного движения для визуального проигрывания."""
        if mid not in self.jaw.movements:
            raise ValueError("движение не найдено")
        rec = self.jaw.movements[mid].recording
        # Для интерактивного окна достаточно 61 кадров, исходная запись
        # остаётся полной и экспортируется отдельно.
        ids = np.linspace(0, len(rec.transforms) - 1, min(61, len(rec.transforms))).round().astype(int)
        return {"id": mid, "name": rec.name, "frames": rec.transforms[ids].round(6).tolist(),
                "duration_s": round(rec.duration, 2) if rec.timed else None}

    def ct_surface(self) -> bytes:
        """Зубы по плотности — для 3D до сегментации."""
        self._require_ct()
        mesh = self.case.surfaces(step=2)["ct_teeth"]
        return mesh_bytes(mesh.vertices, mesh.faces)

    # --- ориентиры и плоскости ----------------------------------------------
    def landmarks_info(self) -> dict:
        planes = []
        for key, p in lmk.PLANES.items():
            missing = lmk.missing(self.landmarks, key)
            planes.append({"key": key, "name": p["name"], "ready": not missing, "missing": missing})
        return {
            "landmarks": [{"key": l.key, "name": l.name, "hint": l.hint,
                           "point": self.landmarks[l.key].round(2).tolist() if l.key in self.landmarks else None,
                           "suggested": l.key in self.suggested} for l in lmk.LANDMARKS],
            "planes": planes, "angles": lmk.plane_angles(self.landmarks),
            "articulators": [{"key": a.key, "name": a.name, "maker": a.maker, "plane": a.plane,
                              "calibrated": a.calibrated} for a in arts.load(self.articulators_path)],
        }

    def set_landmark(self, key: str, point) -> dict:
        if key not in lmk.BY_KEY:
            raise ValueError(f"нет такого ориентира: {key}")
        with self.lock:
            if point is None:
                self.landmarks.pop(key, None)
            else:
                self.landmarks[key] = np.asarray(point, float)
            self.suggested.discard(key)
        return self.landmarks_info()

    def _landmark_folder(self) -> str | None:
        """Папка моделей ориентиров: рядом с моделями сегментации или в папке пользователя."""
        from ..auto_landmarks import Models

        places = [model_store.landmarks_dir(d) for d in (self.models_dir, model_store.default_dir()) if d]
        usable = [p for p in places if os.path.isdir(p) and Models(p).available()]
        return (usable or places or [None])[0]

    def landmark_models_status(self) -> dict:
        from ..auto_landmarks import Models

        folder = self._landmark_folder()
        st = model_store.landmarks_status(os.path.dirname(folder) if folder else None)
        st["available"] = Models(folder).available() if folder and os.path.isdir(folder) else []
        st["usable"] = bool(st["available"])
        return st

    def download_landmark_models(self, progress=None) -> dict:
        """Скачать модели ориентиров (докачка, проверка SHA-256)."""
        folder = self.models_dir or model_store.default_dir()
        self._download_cancel = threading.Event()
        model_store.download_landmarks(folder, progress, self._download_cancel)
        return self.landmark_models_status()

    def auto_landmarks(self, progress=None, keys=None) -> dict:
        """Найти ориентиры на КТ нейросетью (ALI-CBCT). Поставленное врачом не трогается; найденное — «проверьте»."""
        from .. import auto_landmarks as al

        self._require_ct()
        folder = self._landmark_folder()
        models = al.Models(folder) if folder and os.path.isdir(folder) else None
        if models is None or not models.available():
            raise ValueError("нет моделей ориентиров: скачайте их в программе")
        self._landmark_cancel = threading.Event()
        found = al.find(self.vol, models, keys, progress, cancel=self._landmark_cancel)
        with self.lock:
            for key, f in found.items():
                if f.point is not None and key in lmk.BY_KEY and (key not in self.landmarks or key in self.suggested):
                    self.landmarks[key] = np.asarray(f.point, float)
                    self.suggested.add(key)
        info = self.landmarks_info()
        info["auto"] = {k: {"found": f.point is not None, "note": f.note} for k, f in found.items()}
        return info

    def suggest_landmarks(self) -> dict:
        """Предложить мыщелки и порионы по сегментации (не трогая поставленные врачом)."""
        found = {}
        if "mandible" in self.structures:
            found.update(lmk.suggest_condyles(self.structures["mandible"].vertices))
        mid_x = float(np.median(self.structures["mandible"].vertices[:, 0])) if "mandible" in self.structures else 0.0
        for side, key in (("right", "Po_R"), ("left", "Po_L")):
            canal = self.structures.get(f"auditory_canal_{side}")
            if canal is not None:
                found.update(lmk.suggest_porion(canal.vertices, key, mid_x))
        if not found:
            raise ValueError("нечего предложить: сначала сегментируйте КТ (нужны нижняя челюсть и слуховые проходы)")
        with self.lock:
            for key, point in found.items():
                if key not in self.landmarks or key in self.suggested:
                    self.landmarks[key] = np.asarray(point, float)
                    self.suggested.add(key)
        return self.landmarks_info()

    def reference(self, kind: str) -> tuple[np.ndarray, str]:
        """Матрица «КТ → система» и её название: plane:<плоскость> или articulator:<ключ>."""
        what, _, key = kind.partition(":")
        if what == "plane":
            return lmk.reference_frame(self.landmarks, key), lmk.PLANES[key]["name"]
        if what == "articulator":
            art = next((a for a in arts.load(self.articulators_path) if a.key == key), None)
            if art is None:
                raise ValueError(f"нет такого артикулятора: {key}")
            return arts.articulator_frame(self.landmarks, art), f"{art.maker} {art.name}"
        raise ValueError(f"неизвестная система координат: {kind}")

    # --- экспорт ------------------------------------------------------------
    def export(self, out_dir: str, bite: str = "scan", frame: str = "exocad", include: list[str] | None = None,
               reference: str | None = None) -> dict:
        errorlog.private(out_dir, "папка экспорта")
        if self.case is None and not any(item["reg"] is not None for item in self.scans.values()):
            raise ValueError("нечего экспортировать: откройте КТ или добавьте сканы")
        if self.case is None:  # только сканы: координаты сканера, прикус сканов
            frame, bite, reference = "exocad", "scan", None
        regs = []
        for item in self.scans.values():
            reg = item["reg"]
            if reg is not None and item.get("lower_bite") is not None:  # нижний — в прикусе по сканам прикуса
                reg = dataclasses.replace(reg, bite=item["lower_bite"])
            if reg is not None:
                regs.append(reg)
        meshes = {}
        from ..segment import Mesh

        for key, mesh in self.structures.items():
            if include is None or key in include:
                meshes[key] = Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        project = exocad_project.find(out_dir)
        if not regs:  # без сканов — только КТ: структуры в координатах КТ (DICOM)
            if project is not None:
                raise ValueError("для проекта exocad нужны совмещённые сканы: структуры КТ ставятся к ним")
            if not meshes:
                raise ValueError("нечего экспортировать: сегментируйте КТ или совместите сканы")
            return self._export_ct_only(out_dir, meshes)
        if not self.structures and self.case is not None:  # без сегментации — зубы из КТ по плотности (как в 3D)
            meshes["ct_teeth"] = self.case.surfaces(step=1)["ct_teeth"]
        if project is not None:
            return self._export_to_exocad(project, regs, meshes, frame, bite)
        matrix, name = self.reference(reference) if frame == "reference" else (None, "")
        self.last_export = out_dir
        report = export_case(out_dir, regs, meshes, bite=bite, frame=frame, ct=self.case, reference=matrix,
                             reference_name=name, without_ct=self.case is None)
        return {"out_dir": out_dir, "files": sorted(report["files"]), "notes": report["notes"],
                "bite": report["ct_bite_vs_scans"], "frame": report["frame"]}

    def open_export_folder(self):
        """Показать папку последнего экспорта в проводнике (только её — путь не приходит снаружи)."""
        folder = getattr(self, "last_export", None)
        if not folder or not os.path.isdir(folder):
            raise ValueError("экспорта ещё не было")
        open_folder(folder)

    @staticmethod
    def open_log_folder():
        """Показать папку журнала ошибок."""
        folder = errorlog.folder()
        if not folder or not os.path.isdir(folder):
            raise ValueError("журнал ошибок не ведётся")
        open_folder(folder)

    def _export_to_exocad(self, project, regs, meshes: dict, frame: str, bite: str = "scan") -> dict:
        """Папка проекта exocad: всё — в подпапку проекта, в координатах его сцены (прикус — bite).

        Файлы exocad не меняются. Копии сканов там же — по ним видно в exocad, что всё встало на место.
        """
        jaws = [r for r in regs if r.jaw != BITE]
        if not jaws:
            raise ValueError("нет совмещённых сканов челюстей: скан прикуса ставится на них")
        ref = next((r for r in jaws if r.jaw == "upper"), jaws[0])  # опорный скан — как в export_case
        matched = project.match(ref.scan.vertices)
        scene = project.to_scene(matched) if matched else project.default
        out_dir = os.path.join(project.folder, EXOCAD_SUBFOLDER)
        self.last_export = out_dir
        report = export_case(out_dir, regs, meshes, bite=bite, frame="exocad", ct=self.case, scene=scene,
                             without_ct=self.case is None)
        notes = [f"Проект exocad: файлы в подпапке {EXOCAD_SUBFOLDER}, в координатах сцены проекта (матрица — "
                 f"{project.source}). В exocad: «Load mesh as …»; копии сканов там же должны совпасть со сканами проекта."]
        bone_files, bone_notes = self._bones_as_frameworks(project, jaws)
        if bone_files:
            notes.append("Кости — бюгельными каркасами проекта: верхний — череп и верхняя челюсть (с ВЧ), нижний — "
                         "нижняя челюсть (с НЧ); exocad подгрузит их сам и будет двигать с челюстями в артикуляторе.")
        notes += bone_notes
        if frame == "dicom" and self.case is not None:
            notes.append("Для проекта exocad выгрузка всегда в координатах его сцены, а не КТ.")
        if matched is None:
            notes.append(f"Скан {ref.scan.name} не найден среди сканов проекта — положение в exocad не гарантировано "
                         "(взята общая матрица проекта). Загрузите в программу сканы из папки этого проекта.")
        return {"out_dir": out_dir, "files": sorted(report["files"]), "notes": notes + report["notes"],
                "bite": report["ct_bite_vs_scans"], "frame": report["frame"], "frameworks": bone_files}

    def _bones_as_frameworks(self, project, jaws) -> tuple[list, list]:
        """Кости КТ — в координаты файла скана своей челюсти и бюгельными каркасами в папку проекта."""
        bones, notes = {}, []
        for jaw, keys in (("upper", UPPER_BONES), ("lower", LOWER_BONES)):
            reg = next((r for r in jaws if r.jaw == jaw), None)
            parts = [self.structures[k] for k in keys if k in self.structures]
            if reg is None or not parts:
                continue
            if project.match(reg.scan.vertices) is None:
                notes.append(f"{'Верхний' if jaw == 'upper' else 'Нижний'} скан не из этого проекта — кость "
                             "каркасом не записана.")
                continue
            mesh = trimesh.util.concatenate([trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.faces), process=False)
                                             for m in parts])
            mesh.apply_transform(np.linalg.inv(reg.transform))  # КТ → координаты файла скана этой челюсти
            bones[jaw] = mesh
        files, more = exocad_project.write_bone_frameworks(project, bones)
        return files, notes + more

    def _export_ct_only(self, out_dir: str, meshes: dict) -> dict:
        import json

        self.last_export = out_dir
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for key, m in meshes.items():
            path = os.path.join(out_dir, *key.split("/")) + ".stl"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            trimesh.Trimesh(m.vertices, m.faces, process=False).export(path)
            files.append(key + ".stl")
        with open(os.path.join(out_dir, "case.json"), "w", encoding="utf-8") as f:
            json.dump({"frame": "dicom", "note": "координаты пациента из DICOM (мм)", "files": sorted(files)}, f,
                      ensure_ascii=False, indent=1)
        return {"out_dir": out_dir, "files": sorted(files + ["case.json"]),
                "notes": ["сканов нет — структуры КТ в координатах КТ (DICOM)"], "bite": None, "frame": "dicom"}

    # --- кейс: сохранить и открыть ----------------------------------------------
    def save_case(self, path: str | None = None) -> dict:
        """Кейс в один файл: путь к КТ, сканы (сетки — внутри, чтобы не зависеть от исходных файлов) с
        положениями, автоматические положения (для обучения), структуры сегментации, окно КТ."""
        import json

        if self.case is None and not self.scans:
            raise ValueError("нечего сохранять: откройте КТ или добавьте сканы")
        path = path or self.case_path
        errorlog.private(path, "кейс")
        if not path:
            raise ValueError("укажите, куда сохранить кейс")
        if not path.lower().endswith(CASE_EXT):
            path += CASE_EXT
        arrays = {}
        meta = {"version": 1, "ct_path": self.ct_path, "ct_series": getattr(self, "ct_series_id", None),
                "landmarks": {k: np.asarray(v, float).tolist() for k, v in self.landmarks.items()},
                "suggested": sorted(self.suggested),
                "window": list(self.window), "scans": [], "structures": [],
                "guide": []}
        for i, (sid, item) in enumerate(self.scans.items()):
            scan = item["scan"]
            arrays[f"scan{i}_v"], arrays[f"scan{i}_f"] = scan.vertices, scan.faces
            auto = item["auto"]
            meta["scans"].append({
                "name": scan.name, "path": item["path"], "color": item["color"], "jaw": item["jaw"],
                "reg_jaw": item["reg"].jaw if item["reg"] is not None else None,
                "transform": None if item["transform"] is None else np.asarray(item["transform"]).tolist(),
                "auto": None if auto is None else np.asarray(auto.transform).tolist(), "accepted": item["accepted"],
                "role": item["role"]})
        for k, (key, mesh) in enumerate(self.structures.items()):
            arrays[f"st{k}_v"], arrays[f"st{k}_f"] = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            meta["structures"].append(key)
        for jaw, mesh in self.guide_teeth.items():
            if mesh is not None and len(mesh.faces):
                arrays[f"guide_{jaw}_v"], arrays[f"guide_{jaw}_f"] = np.asarray(mesh.vertices), np.asarray(mesh.faces)
                meta["guide"].append(jaw)
        arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)  # прерванная запись не портит прежний файл
        self.case_path = path
        self._remember(path)
        return {"path": path, "name": os.path.basename(path)}

    def open_case(self, path: str, progress=None, ct_path: str | None = None) -> dict:
        """Открыть сохранённый кейс: КТ по пути из кейса (или ct_path, если КТ переносили), сканы на своих местах,
        метрики пересчитываются, структуры и опора совмещения — из файла, без повторной сегментации."""
        import json
        from types import SimpleNamespace

        from ..segment import Mesh

        errorlog.private(path, "кейс")
        errorlog.private(ct_path, "КТ")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        errorlog.private(meta.get("ct_path"), "КТ")
        for sm in meta.get("scans", []):
            errorlog.private(sm.get("path"), "скан")
            errorlog.private(sm.get("name"), "скан")
        ct = ct_path or meta["ct_path"]
        if ct is None:  # кейс только со сканами
            self.close_ct()
        elif not os.path.exists(ct):
            raise FileNotFoundError(f"КТ этого кейса не найден: {ct} — его переместили или удалили")
        else:
            self.load_ct(ct, progress, meta.get("ct_series"))
        with self.lock:
            self.window = tuple(meta.get("window", self.window))
            self.structures = {key: trimesh.Trimesh(arrays[f"st{k}_v"], arrays[f"st{k}_f"], process=False)
                               for k, key in enumerate(meta["structures"])}
            self.scans = {}
        self.guide_teeth = {jaw: Mesh(arrays[f"guide_{jaw}_v"], arrays[f"guide_{jaw}_f"]) for jaw in meta["guide"]}
        with self.lock:
            self.landmarks = {k: np.asarray(v, float) for k, v in meta.get("landmarks", {}).items() if k in lmk.BY_KEY}
            self.suggested = set(meta.get("suggested", [])) & set(self.landmarks)
        if self.guide_teeth and self.case is not None:
            self.case.use_teeth(self.guide_teeth)
        for i, sm in enumerate(meta["scans"]):
            if progress:
                progress(0.6 + 0.4 * i / max(len(meta["scans"]), 1), f"Сканы: {sm['name']}")
            scan = Scan(sm["name"], arrays[f"scan{i}_v"], arrays[f"scan{i}_f"])
            sid = uuid.uuid4().hex[:8]
            role = sm.get("role") or (BITE if is_bite_name(scan.name) else "jaw")
            item = {"scan": scan, "path": sm["path"], "color": sm["color"], "jaw": sm["jaw"], "reg": None, "auto": None,
                    "role": role, "lower_bite": None,
                    "transform": None, "accepted": False, "guided": False}
            if sm["transform"] is not None and role != BITE and self.case is None:  # без КТ — как в файле
                reg = scanner_registration(scan, sm["reg_jaw"])
                item.update(reg=reg, auto=reg, transform=reg.transform, guided=True)
            elif sm["transform"] is not None and role != BITE:  # сканы прикуса ставятся заново на челюсти
                reg = self.case.evaluate(scan, np.asarray(sm["transform"], float), jaw=sm["reg_jaw"])
                item.update(reg=reg, transform=reg.transform, guided=reg.jaw in self.case.guided)
                auto = np.asarray(sm["auto"], float) if sm["auto"] is not None else reg.transform
                item["auto"] = reg if np.allclose(auto, reg.transform) else SimpleNamespace(transform=auto, jaw=reg.jaw)
                item["accepted"] = sm["accepted"]
            with self.lock:
                self.scans[sid] = item
        self._place_bites()  # сканы прикуса — на сканы челюстей, как при совмещении
        self.case_path = path
        self._remember(path)
        return self.state()

    def _remember(self, path: str):
        recent = [p for p in self.settings.get("recent", []) if os.path.normcase(p) != os.path.normcase(path)]
        self.settings["recent"] = [path] + recent[: RECENT - 1]
        self._save_settings()

    def recent_cases(self) -> list[dict]:
        return [{"path": p, "name": os.path.splitext(os.path.basename(p))[0], "exists": os.path.isfile(p)}
                for p in self.settings.get("recent", [])]

    def state(self) -> dict:
        from .. import __version__

        return {"version": __version__, "ct": self.ct_info(), "scans": [self.scan_info(s) for s in self.scans],
                "case": None if not self.case_path else {"path": self.case_path,
                                                         "name": os.path.splitext(os.path.basename(self.case_path))[0]},
                "recent": self.recent_cases(),
                "models_dir": self.models_dir, "models": self.models_status(), **self.structures_info(),
                "segment_parts": self.segment_parts_info(), "bite_view": self.bite_info(),
                "incognito": self.settings["incognito"],
                **self.landmarks_info(),
                "articulation": self.jaw.state()}
