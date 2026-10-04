"""Кейс артикуляции: сканы в прикусе, монтаж, суставы, движения и результаты — то, что вызывает интерфейс.

Один объект держит всё, что нужно для работы с движениями челюсти:

* сканы обеих челюстей в прикусе (файлы, сцена exocad или переданные сетки),
  кость и череп из КТ, если есть;
* монтаж (Anatomy): средний артикулятор, КТ, запись движения или
  эстетическая система;
* настройки суставов (Settings) с источником каждого числа;
* движения: записанные (P-ART и др.) и построенные виртуальным артикулятором;
* результаты: анализ, контакты по участкам и зубам, FGP, сверка записи со
  сканами, посадка прикуса.

Каждое изменение можно отменить (undo/redo), кейс сохраняется в один файл
(.ccdjaw — архив numpy со сведениями в JSON). Всё в координатах кейса
(координаты сканов); state() — сводка для интерфейса, только JSON-типы.
"""

import copy
import json
import uuid
from dataclasses import dataclass

import numpy as np

from . import kinematics as kin
from . import motion as mo
from .register import apply

UNDO_DEPTH = 50
MOUNTINGS = {"average": "средний артикулятор", "ct": "по КТ", "motion": "по записи движения",
             "aesthetic": "по эстетической плоскости"}


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    source: str = ""


@dataclass
class Movement:
    """Движение кейса: записанное или построенное артикулятором."""

    id: str
    recording: mo.Recording
    recorded: bool  # True — запись пациента, False — виртуальный артикулятор


class JawCase:
    def __init__(self):
        self.upper: Mesh | None = None
        self.lower: Mesh | None = None
        self.mandible: Mesh | None = None
        self.skull: Mesh | None = None
        self.designs: dict = {}  # номер зуба → Mesh (моделировки из сцены exocad: масштаб дуги)
        self.anatomy: mo.Anatomy | None = None
        self.mounting: str | None = None
        self.settings = kin.Settings()
        self.lower_fix = np.eye(4)  # посадка прикуса: «нижний скан → посаженный»
        self.movements: dict[str, Movement] = {}
        self.motion_case: mo.MotionCase | None = None  # последняя прочитанная выгрузка записей
        self.notes: list[str] = []
        self._undo: list = []
        self._redo: list = []
        self._occlusion = None

    # --- отмена -------------------------------------------------------------
    def _snapshot(self) -> dict:
        return {"anatomy": copy.deepcopy(self.anatomy), "mounting": self.mounting,
                "settings": self.settings.copy(), "lower_fix": self.lower_fix.copy(),
                "movements": dict(self.movements), "notes": list(self.notes)}

    def _restore(self, snap: dict):
        self.anatomy, self.mounting = snap["anatomy"], snap["mounting"]
        self.settings, self.lower_fix = snap["settings"], snap["lower_fix"]
        self.movements, self.notes = snap["movements"], snap["notes"]
        self._occlusion = None

    def _edit(self, what: str):
        """Перед изменением: запомнить состояние для отмены."""
        self._undo.append((what, self._snapshot()))
        del self._undo[:-UNDO_DEPTH]
        self._redo.clear()
        self._occlusion = None

    def undo(self) -> str | None:
        if not self._undo:
            return None
        what, snap = self._undo.pop()
        self._redo.append((what, self._snapshot()))
        self._restore(snap)
        return what

    def redo(self) -> str | None:
        if not self._redo:
            return None
        what, snap = self._redo.pop()
        self._undo.append((what, self._snapshot()))
        self._restore(snap)
        return what

    # --- данные -------------------------------------------------------------
    def set_scans(self, upper: Mesh, lower: Mesh):
        """Сканы обеих челюстей в прикусе, в общих координатах (внутриротовые или модели)."""
        self._edit("сканы")
        self.upper, self.lower = upper, lower
        self.lower_fix = np.eye(4)
        self.anatomy, self.mounting = None, None
        self.movements = {}

    def load_scans(self, upper_path: str, lower_path: str):
        from .fusion import Scan

        u, low = Scan.load(upper_path), Scan.load(lower_path)
        self.set_scans(Mesh(u.vertices, u.faces, upper_path), Mesh(low.vertices, low.faces, lower_path))

    def load_scene(self, path: str):
        """Сцена exocad (HTML-экспорт): сканы, кость и череп из КТ, моделировки с номерами зубов."""
        from . import exocad_webview as ew

        p = ew.load(path).parts()
        if p["upper_scan"] is None or p["lower_scan"] is None:
            raise ValueError("в сцене нужны сканы обеих челюстей (или скан и антагонист)")

        def mesh(o):
            return None if o is None else Mesh(o.vertices, o.faces, o.label())

        self.set_scans(mesh(p["upper_scan"]), mesh(p["lower_scan"]))
        self.mandible, self.skull = mesh(p["mandible"]), mesh(p["skull"])
        self.designs = {t: mesh(o) for t, o in p["designs"].items()}

    @property
    def lower_vertices(self) -> np.ndarray:
        return apply(self.lower_fix, self.lower.vertices)

    def _require(self, anatomy: bool = True):
        if self.upper is None or self.lower is None:
            raise ValueError("сначала загрузите сканы обеих челюстей")
        if anatomy and self.anatomy is None:
            raise ValueError("сначала смонтируйте модели (mount)")

    @property
    def occlusion(self) -> kin.Occlusion:
        """Верхние зубы как препятствие для нижних — только сканы челюстей."""
        self._require()
        if self._occlusion is None:
            self._occlusion = kin.Occlusion(self.upper.vertices, self.upper.faces, self.lower_vertices,
                                            self.anatomy.frame)
        return self._occlusion

    # --- монтаж и суставы ---------------------------------------------------
    def mount(self, method: str = "auto", icd: float = mo.ICD_MM) -> dict:
        """Смонтировать модели: average, ct (кость в кейсе), motion (записи движения) или auto.

        auto: по КТ, если есть кость нижней челюсти; иначе по записи, если она
        прочитана; иначе средний артикулятор. Настройки суставов берутся тем же
        способом (бугорок по КТ, запись), чего нет — остаются прежними.
        """
        from . import exocad_webview as ew

        self._require(anatomy=False)
        if method == "auto":
            method = "ct" if self.mandible is not None else "motion" if self._recorded() else "average"
        if method not in ("average", "ct", "motion"):
            raise ValueError(f"монтаж: {', '.join(MOUNTINGS)}")
        self._edit(f"монтаж: {MOUNTINGS[method]}")
        notes = []
        if method == "motion":
            case = mo.MotionCase("кейс", [m.recording for m in self.movements.values() if m.recorded])
            if not case.recordings:
                raise ValueError("записей движения нет — загрузите выгрузку (load_motion)")
            self.anatomy = mo.estimate_anatomy(case, lower=self.lower_vertices, icd=icd)
            analysis = mo.analyze_case(case, self.anatomy)
            self.settings = kin.settings_from_analysis(analysis, self.settings, self.anatomy.frame)
        else:
            if method == "ct" and self.mandible is None:
                raise ValueError("кости нижней челюсти в кейсе нет — монтаж по КТ невозможен")
            anatomy, settings, eminence, notes = ew.mount(self.upper.vertices, self.lower_vertices,
                                                          self.mandible if method == "ct" else None,
                                                          self.skull if method == "ct" else None)
            self.anatomy = anatomy
            base = self.settings.copy()
            for key, value in vars(settings).items():
                if key in settings.sources:
                    setattr(base, key, value)
                    base.sources[key] = settings.sources[key]
            if not any(v == "запись движения" for v in base.sources.values()):
                base.frame = None  # углы — от горизонтали нового монтажа; углы записи — от своей
            self.settings = base
        self.mounting = method
        self.notes = notes
        return self.state()

    def mount_aesthetic(self, aesthetic) -> dict:
        """Монтаж по эстетической системе (aesthetic.aesthetic_frame): мыщелки и углы суставов не меняются —
        углы остаются от прежней горизонтали, модели стоят по эстетической плоскости."""
        self._require()
        self._edit("монтаж: по эстетической плоскости")
        if self.settings.frame is None:
            self.settings.frame = self.anatomy.frame.copy()
        self.anatomy = mo.Anatomy(np.asarray(aesthetic.frame, float), dict(self.anatomy.points),
                                  f"эстетическая плоскость (крен {aesthetic.roll_deg:+.1f}°); "
                                  f"суставы — от прежней горизонтали", self.anatomy.hinge_rms_mm)
        self.mounting = "aesthetic"
        return self.state()

    def set_settings(self, **values) -> dict:
        """Изменить настройки суставов вручную: sagittal_right_deg=…, bennett_left_deg=… и т.д."""
        self._edit("настройки суставов")
        s = self.settings.copy()
        for key, value in values.items():
            if key in ("sources", "frame") or not hasattr(s, key):
                raise ValueError(f"нет такой настройки: {key}")
            setattr(s, key, float(value))
            s.sources[key] = "вручную"
        self.settings = s
        return self.settings_info()

    def settings_info(self) -> dict:
        s = self.settings
        return {"values": {k: v for k, v in vars(s).items() if k not in ("sources", "frame")},
                "sources": dict(s.sources), "own_horizontal": s.frame is not None}

    def seat_bite(self) -> dict:
        """Посадить прикус сканов на шарнирной оси до касания без проникновения (отменяется undo)."""
        self._require()
        M, info = kin.seat_bite(self.upper.vertices, self.upper.faces, self.lower_vertices, self.anatomy)
        self._edit("посадка прикуса")
        self.lower_fix = M @ self.lower_fix
        return info

    # --- движения -----------------------------------------------------------
    def _recorded(self) -> list[Movement]:
        return [m for m in self.movements.values() if m.recorded]

    def _add(self, rec: mo.Recording, recorded: bool) -> str:
        mid = uuid.uuid4().hex[:8]
        self.movements[mid] = Movement(mid, rec, recorded)
        return mid

    def load_motion(self, path: str, split: bool = False, smoothing: bool = False) -> dict:
        """Записи движений пациента (P-ART, Zebris, CADIAX…) — в кейс; со сканами сверяются сразу."""
        case = mo.read_case(path, split=split, smoothing=smoothing)
        if not case.recordings:
            raise ValueError("положений челюсти в выгрузке нет: " + "; ".join(case.notes or ["пусто"]))
        self._edit("записи движения")
        self.motion_case = case
        for rec in case.recordings:
            self._add(rec, recorded=True)
        out = {"recordings": len(case.recordings), "systems": case.systems, "notes": case.notes}
        if self.upper is not None and self.lower is not None:
            out["scans_check"] = self.check_scans()
        return out

    def generate(self, travel: float = 6.0, guided: bool = True, chewing: bool = False) -> list[str]:
        """Движения виртуального артикулятора: протрузия и латеротрузии (и жевание); прежние построенные
        заменяются. guided — передние зубы ведут по сканам."""
        self._require()
        occ = self.occlusion if guided else None
        self._edit("движения артикулятора")
        self.movements = {k: m for k, m in self.movements.items() if m.recorded}
        recs = [kin.protrusion(self.anatomy, self.settings, travel, occ),
                kin.laterotrusion(self.anatomy, self.settings, "right", travel, occ),
                kin.laterotrusion(self.anatomy, self.settings, "left", travel, occ)]
        if chewing:
            recs.append(kin.chewing(self.anatomy, self.settings))
        return [self._add(r, recorded=False) for r in recs]

    def remove_movement(self, mid: str):
        self._edit("удаление движения")
        self.movements.pop(mid)

    def _recs(self, ids=None) -> list[mo.Recording]:
        ids = list(self.movements) if ids is None else ids
        return [self.movements[i].recording for i in ids]

    # --- результаты ---------------------------------------------------------
    def analysis(self, ids=None) -> dict:
        """Анализ движений (углы суставных путей, ведение зубами, Беннетт) в системе монтажа."""
        self._require()
        return mo.analyze_case(mo.MotionCase("кейс", self._recs(ids)), self.anatomy)

    def contacts(self, mid: str) -> dict:
        """Где касаются зубы при движении: участки дуги и пары зубов, с долей пути."""
        from .arch_teeth import contacts_by_tooth
        from .exocad_webview import _arch_lines

        self._require()
        rec = self.movements[mid].recording
        upper = _SceneLike(self.upper.vertices)
        lower = _SceneLike(self.lower_vertices)
        designs = {t: _SceneLike(m.vertices, jaw="upper" if int(t) < 30 else "lower") for t, m in self.designs.items()}
        arches = _arch_lines(upper, lower, designs, self.anatomy)
        return {"sectors": kin.contact_sectors(rec, self.anatomy, self.occlusion),
                "teeth": contacts_by_tooth(rec, self.anatomy, self.occlusion, arches["upper"], arches["lower"])}

    def check_scans(self) -> dict:
        """Сходятся ли записи пациента со сканами (kinematics.check_against_scans)."""
        self._require(anatomy=False)
        recs = [m.recording for m in self._recorded()]
        if not recs:
            raise ValueError("записей движения нет")
        ref = mo.reference_pose(mo.MotionCase("кейс", recs))
        return kin.check_against_scans(recs, self.upper.vertices, self.upper.faces, self.lower_vertices, ref,
                                       self.anatomy)

    def fgp(self, for_jaw: str = "upper", cell: float | None = None, ids=None) -> Mesh:
        """Функционально сформированный антагонист по движениям кейса (по умолчанию — по всем)."""
        from . import fgp as fg

        self._require()
        recs = self._recs(ids) if self.movements else None
        v, f, _recs = fg.fgp(self.upper.vertices, self.upper.faces, self.lower_vertices, self.anatomy,
                             self.settings, for_jaw, cell or fg.CELL_MM, recs)
        return Mesh(v, f, f"FGP для {'верхней' if for_jaw == 'upper' else 'нижней'} челюсти")

    # --- сводка, сохранение -------------------------------------------------
    def state(self) -> dict:
        a = self.anatomy
        return {
            "scans": {k: None if m is None else {"vertices": len(m.vertices), "source": m.source}
                      for k, m in (("upper", self.upper), ("lower", self.lower))},
            "ct": {"mandible": self.mandible is not None, "skull": self.skull is not None},
            "designs": sorted(self.designs),
            "mounting": None if a is None else {
                "method": self.mounting, "name": MOUNTINGS.get(self.mounting, self.mounting), "source": a.source,
                "points": {k: [round(float(c), 2) for c in apply(a.frame, p[None])[0]] for k, p in a.points.items()},
                "intercondylar_mm": round(float(np.linalg.norm(a.points["condyle_right"] - a.points["condyle_left"])),
                                          1)},
            "settings": self.settings_info(),
            "bite_seated": not np.allclose(self.lower_fix, np.eye(4)),
            "movements": [{"id": m.id, "name": m.recording.name, "recorded": m.recorded,
                           "frames": len(m.recording.transforms),
                           "duration_s": round(m.recording.duration, 2) if m.recording.timed else None}
                          for m in self.movements.values()],
            "undo": self._undo[-1][0] if self._undo else None,
            "redo": self._redo[-1][0] if self._redo else None,
            "notes": list(self.notes),
        }

    def save(self, path: str):
        """Кейс в один файл: сетки, монтаж, настройки, посадка прикуса, движения (без истории отмен)."""
        arrays, meta = {}, {"version": 1, "mounting": self.mounting, "notes": self.notes, "meshes": {},
                            "designs": sorted(self.designs), "movements": []}
        for key, m in (("upper", self.upper), ("lower", self.lower), ("mandible", self.mandible),
                       ("skull", self.skull), *((f"design_{t}", m) for t, m in self.designs.items())):
            if m is not None:
                arrays[f"{key}_v"], arrays[f"{key}_f"] = m.vertices, m.faces
                meta["meshes"][key] = m.source
        if self.anatomy is not None:
            arrays["anatomy_frame"] = self.anatomy.frame
            meta["anatomy"] = {"points": {k: v.tolist() for k, v in self.anatomy.points.items()},
                               "source": self.anatomy.source, "hinge_rms_mm": self.anatomy.hinge_rms_mm}
        s = self.settings
        meta["settings"] = {k: v for k, v in vars(s).items() if k not in ("frame",)}
        if s.frame is not None:
            arrays["settings_frame"] = s.frame
        arrays["lower_fix"] = self.lower_fix
        for i, m in enumerate(self.movements.values()):
            r = m.recording
            arrays[f"mov{i}_t"], arrays[f"mov{i}_T"] = r.times, r.transforms
            meta["movements"].append({"id": m.id, "name": r.name, "source": r.source, "timed": r.timed,
                                      "recorded": m.recorded})
        arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)

    @classmethod
    def load(cls, path: str) -> "JawCase":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            jc = cls()

            def mesh(key):
                return Mesh(z[f"{key}_v"], z[f"{key}_f"], meta["meshes"][key]) if key in meta["meshes"] else None

            jc.upper, jc.lower, jc.mandible, jc.skull = (mesh(k) for k in ("upper", "lower", "mandible", "skull"))
            jc.designs = {t: mesh(f"design_{t}") for t in meta["designs"]}
            if "anatomy" in meta:
                a = meta["anatomy"]
                jc.anatomy = mo.Anatomy(z["anatomy_frame"], {k: np.array(v) for k, v in a["points"].items()},
                                        a["source"], a["hinge_rms_mm"])
            jc.mounting, jc.notes = meta["mounting"], meta["notes"]
            sv = dict(meta["settings"])
            jc.settings = kin.Settings(**{k: v for k, v in sv.items() if k != "sources"}, sources=sv["sources"],
                                       frame=z["settings_frame"] if "settings_frame" in z else None)
            jc.lower_fix = z["lower_fix"]
            for i, m in enumerate(meta["movements"]):
                rec = mo.Recording(m["name"], z[f"mov{i}_t"], z[f"mov{i}_T"], m["source"], m["timed"])
                jc.movements[m["id"]] = Movement(m["id"], rec, m["recorded"])
        return jc


@dataclass
class _SceneLike:
    """Сетка в виде объекта сцены для _arch_lines (нужны только вершины и челюсть)."""

    vertices: np.ndarray
    jaw: str | None = None
