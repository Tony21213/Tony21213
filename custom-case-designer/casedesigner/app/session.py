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
        self.jaw = JawCase()  # артикуляция: монтаж, суставы, движения, контакты (интерфейс — позже)

    # --- настройки (рядом с памятью совмещений) -----------------------------
    def _load_settings(self) -> dict:
        import json

        known = [p for p, _t, _k in SEGMENT_PARTS]
        settings = {"segment_parts": known}
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                saved = json.load(f)
            settings["segment_parts"] = [p for p in known if p in saved.get("segment_parts", [])]
        except (OSError, ValueError, AttributeError):  # нет файла или он испорчен — всё по умолчанию
            pass
        return settings

    def set_segment_parts(self, parts: list[str]) -> dict:
        import json

        known = [p for p, _t, _k in SEGMENT_PARTS]
        unknown = sorted(set(parts) - set(known))
        if unknown:
            raise ValueError(f"нет таких частей: {', '.join(unknown)}")
        self.settings["segment_parts"] = [p for p in known if p in parts]
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=1)
        return self.segment_parts_info()

    def segment_parts_info(self) -> dict:
        return {"parts": [{"id": p, "title": t} for p, t, _k in SEGMENT_PARTS],
                "selected": list(self.settings["segment_parts"])}

    # --- КТ -----------------------------------------------------------------
    def load_ct(self, path: str, progress=None) -> dict:
        vol = load_volume(path)
        if progress:
            progress(0.5, "Ищу коронки зубов")
        prior = self.memory.prior(vol.device)
        case = CaseCT(vol, *prior)
        with self.lock:
            self.ct_path, self.vol, self.case = path, vol, case
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

    def slice_png(self, axis: str, pos: float, level: float | None = None, width: float | None = None,
                  region=None, size=None) -> bytes:
        """Срез картинкой; region=(ua, ub, va, vb) — только эта часть (мм по осям среза, края слева-направо и
        сверху-вниз), size=(cols, rows) — в таком разрешении: при увеличении — резко, в разрешении экрана."""
        from PIL import Image

        g = self.slice_geometry(axis)
        s = SLICES[axis]
        if region is None:
            us, vs = g["u0"] + g["du"] * np.arange(g["width"]), g["v0"] + g["dv"] * np.arange(g["height"])
        else:
            (ua, ub, va, vb), (cols, rows) = region, (int(np.clip(n, 1, MAX_SLICE_PIXELS)) for n in size)
            us, vs = ua + (np.arange(cols) + 0.5) * (ub - ua) / cols, va + (np.arange(rows) + 0.5) * (vb - va) / rows
        uu, vv = np.meshgrid(us, vs)
        pts = np.zeros((uu.size, 3))
        pts[:, s["u"]], pts[:, s["v"]], pts[:, s["normal"]] = uu.ravel(), vv.ravel(), pos
        values = self.vol.sample(pts).reshape(uu.shape)
        level = self.window[0] if level is None else level
        width = self.window[1] if width is None else width
        img = np.clip((values - (level - width / 2)) / width * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    # --- сечения сеток плоскостью среза --------------------------------------
    def _section(self, mesh: trimesh.Trimesh, axis: str, pos: float) -> list[float]:
        s = SLICES[axis]
        normal = np.zeros(3)
        normal[s["normal"]] = 1
        origin = np.zeros(3)
        origin[s["normal"]] = pos
        lo, hi = mesh.bounds
        if not lo[s["normal"]] <= pos <= hi[s["normal"]]:
            return []
        lines = trimesh.intersections.mesh_plane(mesh, normal, origin)
        if not len(lines):
            return []
        return np.round(lines[:, :, [s["u"], s["v"]]].reshape(-1), 3).tolist()

    def overlays(self, axis: str, pos: float, visible: list[str]) -> list[dict]:
        out = []
        with self.lock:
            for sid, item in self.scans.items():
                if item["transform"] is None or sid not in visible:
                    continue
                mesh = trimesh.Trimesh(apply(item["transform"], item["scan"].vertices), item["scan"].faces,
                                       process=False)
                out.append({"id": sid, "color": item["color"], "width": 1.6, "segments": self._section(mesh, axis, pos)})
            for key, mesh in self.structures.items():
                if key in visible:
                    out.append({"id": key, "color": structure(key).color, "width": 1.2,
                                "segments": self._section(mesh, axis, pos)})
        return [o for o in out if o["segments"]]

    # --- сканы -------------------------------------------------------------
    def add_scan(self, path: str) -> dict:
        scan = Scan.load(path)
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
        rec = self.memory.record(self.vol.device, item["auto"] or item["reg"], item["reg"])
        item["accepted"] = True
        return {"scan": self.scan_info(sid), "record": {k: rec[k] for k in ("corrected_mm", "edge_shift_mm")},
                "memory": self.memory.summary().get(self.vol.device or "unknown")}

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
        self.case.use_teeth({jaw: result.meshes.get(f"{jaw}_teeth") for jaw in ("upper", "lower")})
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
        report = export_case(out_dir, regs, meshes, bite=bite, frame=frame, ct=self.case, reference=matrix,
                             reference_name=name)
        return {"out_dir": out_dir, "files": sorted(report["files"]), "notes": report["notes"],
                "bite": report["ct_bite_vs_scans"], "frame": report["frame"]}

    def _export_ct_only(self, out_dir: str, meshes: dict) -> dict:
        import json

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

    def state(self) -> dict:
        from .. import __version__

        return {"version": __version__, "ct": self.ct_info(), "scans": [self.scan_info(s) for s in self.scans],
                "models_dir": self.models_dir, "models": self.models_status(), **self.structures_info(),
                "segment_parts": self.segment_parts_info(),
                **self.landmarks_info(),
                "articulation": self.jaw.state()}
