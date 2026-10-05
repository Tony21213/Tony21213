"""Состояние открытого кейса и операции над ним — то, что вызывает интерфейс.

Всё в координатах пациента из DICOM (мм, LPS): КТ, структуры, сканы (через
текущую матрицу «скан → КТ»). Интерфейс двигает скан, присылает новую
матрицу, а сессия оценивает точность и при желании уточняет положение.
"""

import io
import os
import threading
import uuid

import numpy as np
import trimesh

from .. import articulators as arts
from . import errorlog
from .. import model_store
from .. import landmarks as lmk
from ..fusion import CaseCT, Registration, Scan, deviation_colors, export_case
from ..jawcase import JawCase
from ..learning import AlignmentMemory
from ..register import apply
from ..segment import Segmenter
from ..structures import structure
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

    # --- настройки (рядом с памятью совмещений) -----------------------------
    def _load_settings(self) -> dict:
        import json

        known = [p for p, _t, _k in SEGMENT_PARTS]
        settings = {"segment_parts": known, "recent": []}
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                saved = json.load(f)
            settings["segment_parts"] = [p for p in known if p in saved.get("segment_parts", [])]
            settings["recent"] = [r for r in saved.get("recent", []) if isinstance(r, str)][:RECENT]
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
            self.landmarks, self.suggested = {}, set()
            for item in self.scans.values():
                item.update(reg=None, auto=None, transform=None, guided=False)
        return self.ct_info()

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
                T = item["transform"]
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
                    sec = self._sectioner(f"structure:{key}", np.asarray(mesh.vertices), np.asarray(mesh.faces),
                                          id(mesh))
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
                               "reg": None, "auto": None, "transform": None, "accepted": False, "guided": False}
        return self.scan_info(sid)

    def remove_scan(self, sid: str):
        with self.lock:
            self.scans.pop(sid, None)

    def scan_info(self, sid: str) -> dict:
        item = self.scans[sid]
        reg: Registration | None = item["reg"]
        return {"id": sid, "name": item["scan"].name, "path": item["path"], "color": item["color"],
                "vertices": len(item["scan"].vertices),
                "jaw": reg.jaw if reg else item["jaw"],
                "transform": None if item["transform"] is None else item["transform"].tolist(),
                "auto_transform": None if item["auto"] is None else item["auto"].transform.tolist(),
                "registered": reg is not None, "accepted": item["accepted"],
                "stats": reg.stats if reg else None, "edge_shift_mm": round(reg.edge_shift, 3) if reg else None,
                "warnings": (([] if item["guided"] else [UNGUIDED]) + reg.warnings) if reg else [],
                "segments": [vars(s) for s in reg.segments] if reg else [],
                "corrected_mm": self._corrected(sid)}

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
        self._require_ct()
        item = self.scans[sid]
        jaw = jaw or item["jaw"]
        reg = self.case.register(item["scan"], jaw=jaw, pairs=pairs,
                                 start=None if start is None else np.asarray(start, float))
        with self.lock:
            item["reg"], item["transform"], item["accepted"] = reg, reg.transform, False
            item["guided"] = reg.jaw in self.case.guided
            if item["auto"] is None or (start is None and pairs is None):
                item["auto"] = reg
        return self.scan_info(sid)

    def evaluate(self, sid: str, transform) -> dict:
        self._require_ct()
        item = self.scans[sid]
        reg = self.case.evaluate(item["scan"], np.asarray(transform, float), jaw=item["reg"].jaw if item["reg"] else None)
        with self.lock:
            item["reg"], item["transform"], item["accepted"] = reg, reg.transform, False
            item["guided"] = reg.jaw in self.case.guided
        return self.scan_info(sid)

    def set_jaw(self, sid: str, jaw: str | None):
        self.scans[sid]["jaw"] = jaw or None

    def accept(self, sid: str) -> dict:
        item = self.scans[sid]
        if item["reg"] is None:
            raise ValueError("скан ещё не совмещён")
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
        m = self.structures[key]
        return mesh_bytes(m.vertices, m.faces)

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
        self._require_ct()
        errorlog.private(out_dir, "папка экспорта")
        regs = [item["reg"] for item in self.scans.values() if item["reg"] is not None]
        meshes = {}
        from ..segment import Mesh

        for key, mesh in self.structures.items():
            if include is None or key in include:
                meshes[key] = Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        if not regs:  # без сканов — только КТ: структуры в координатах КТ (DICOM)
            if not meshes:
                raise ValueError("нечего экспортировать: сегментируйте КТ или совместите сканы")
            return self._export_ct_only(out_dir, meshes)
        if not self.structures:  # без сегментации — хотя бы зубы из КТ по плотности (как в 3D)
            meshes["ct_teeth"] = self.case.surfaces(step=1)["ct_teeth"]
        matrix, name = self.reference(reference) if frame == "reference" else (None, "")
        self.last_export = out_dir
        report = export_case(out_dir, regs, meshes, bite=bite, frame=frame, ct=self.case, reference=matrix,
                             reference_name=name)
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

        self._require_ct()
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
                "auto": None if auto is None else np.asarray(auto.transform).tolist(), "accepted": item["accepted"]})
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
        if not os.path.exists(ct):
            raise FileNotFoundError(f"КТ этого кейса не найден: {ct} — его переместили или удалили")
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
        if self.guide_teeth:
            self.case.use_teeth(self.guide_teeth)
        for i, sm in enumerate(meta["scans"]):
            if progress:
                progress(0.6 + 0.4 * i / max(len(meta["scans"]), 1), f"Сканы: {sm['name']}")
            scan = Scan(sm["name"], arrays[f"scan{i}_v"], arrays[f"scan{i}_f"])
            sid = uuid.uuid4().hex[:8]
            item = {"scan": scan, "path": sm["path"], "color": sm["color"], "jaw": sm["jaw"], "reg": None, "auto": None,
                    "transform": None, "accepted": False, "guided": False}
            if sm["transform"] is not None:
                reg = self.case.evaluate(scan, np.asarray(sm["transform"], float), jaw=sm["reg_jaw"])
                item.update(reg=reg, transform=reg.transform, guided=reg.jaw in self.case.guided)
                auto = np.asarray(sm["auto"], float) if sm["auto"] is not None else reg.transform
                item["auto"] = reg if np.allclose(auto, reg.transform) else SimpleNamespace(transform=auto, jaw=reg.jaw)
                item["accepted"] = sm["accepted"]
            with self.lock:
                self.scans[sid] = item
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
                "segment_parts": self.segment_parts_info(),
                **self.landmarks_info(),
                "articulation": self.jaw.state()}
