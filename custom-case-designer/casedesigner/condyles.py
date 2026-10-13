"""Точки мыщелков Co_R и Co_L в шаге «Артикулятор»: приоритет, система координат, проверки, состояние.

Контракт (подробно — docs/handoff/archive/condyle-adjustment.md):

* **Хранение.** Точки — ориентиры ``Co_R`` / ``Co_L`` сессии (``Session.landmarks``): миллиметры пациента КТ
  (DICOM LPS: x — влево, y — назад, z — вверх), нижняя челюсть — как на снимке. Это та же точка, что
  «Мыщелок правый/левый» в ориентирах: верхушка головки нижней челюсти. В ``.ccdcase`` — ``meta["landmarks"]``,
  ``meta["suggested"]`` и ``meta["landmark_sources"]``.
* **Происхождение и приоритет.** У точки есть источник (``SOURCES``) и признак подтверждения: подтверждённая
  врачом точка (ключа нет в ``Session.suggested``) выше любой автоматической; автоматическая
  (ALI-CBCT или по сетке нижней челюсти) — «проверьте», её заменяет следующий автоматический поиск;
  поставленную или подтверждённую врачом автоматика не трогает никогда. «По нижней челюсти» — верхушка
  головки, отрезанной от челюсти выше вырезки (``landmarks.condylar_heads``: венечный отросток не путается);
  точка ALI дальше ``ALI_HEAD_MAX_MM`` от неё заменяется головкой (``prefer_head``), с заметкой.
* **Применение.** Модели монтируются в прикусе сканов, если он найден (``Session.bite_motion``), иначе — как
  на КТ. Мыщелок принадлежит нижней челюсти и едет с ней: в прикусе сканов шарнирная точка —
  ``bite_motion · p``, в прикусе КТ — ``p``. ССП по суставному бугорку — всегда в соотношении КТ: точка как
  есть и череп КТ (кость черепа с нижней челюстью не едет).
* **Проверки** — ``checks``: межмыщелковое расстояние, стороны, точка на сетке нижней челюсти, асимметрия.
  Уровни ``ok`` / ``warn`` / ``error`` / ``unknown`` (проверить нечем). С ``error`` монтаж по точкам не идёт.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point

from .register import apply

SIDES = {"right": "Co_R", "left": "Co_L"}
SOURCE_USER, SOURCE_ALI, SOURCE_MANDIBLE = "user", "ali-cbct", "mandible"
SOURCES = (SOURCE_USER, SOURCE_ALI, SOURCE_MANDIBLE)
SOURCE_LABELS = {SOURCE_USER: "врач", SOURCE_ALI: "нейросеть ALI-CBCT", SOURCE_MANDIBLE: "по нижней челюсти"}

ICD_WARN_MM = (80.0, 125.0)  # межмыщелковое вне — предупреждение
ICD_ERROR_MM = (60.0, 140.0)  # вне — ошибка: точка явно не на мыщелке
ON_MANDIBLE_WARN_MM, ON_MANDIBLE_ERROR_MM = 1.5, 5.0  # точка от поверхности нижней челюсти
ASYMMETRY_WARN_MM = 5.0  # разница правого и левого по высоте и по глубине (система монтажа)
ALI_HEAD_MAX_MM = 4.0  # точка ALI дальше от верхушки головки по сегментации — автоматической берётся головка
LEVELS = ("ok", "warn", "error", "unknown")


def prefer_head(ali_point, head_top) -> tuple[np.ndarray, str, str | None]:
    """Автоматическая точка мыщелка: ALI-CBCT или верхушка головки, отрезанной от нижней челюсти
    (landmarks.condylar_heads). Шарнир программы — центр головки на 4 мм ниже её верхушки, поэтому при
    расхождении больше ALI_HEAD_MAX_MM берётся головка (кейс 4: ALI слева — на стенке ямки, в 5–6 мм).
    Возвращает (точка, источник, заметка или None)."""
    ali = np.asarray(ali_point, float)
    if head_top is None:
        return ali, SOURCE_ALI, None
    head = np.asarray(head_top, float)
    d = float(np.linalg.norm(ali - head))
    if d <= ALI_HEAD_MAX_MM:
        return ali, SOURCE_ALI, None
    return head, SOURCE_MANDIBLE, f"ALI-CBCT в {d:.1f} мм от верхушки головки по сегментации — взята головка"


@dataclass
class CondylePoint:
    side: str  # "right" | "left"
    point: np.ndarray | None  # мм КТ (LPS), нижняя челюсть как на КТ; None — точки нет
    source: str | None  # SOURCES или None
    confirmed: bool  # подтверждена врачом: поставлена вручную или принята

    @property
    def key(self) -> str:
        return SIDES[self.side]


@dataclass
class Check:
    id: str  # "icd", "sides", "on_mandible_right", "on_mandible_left", "asymmetry"
    level: str  # LEVELS
    value: float | None
    text: str

    def to_json(self) -> dict:
        return {"id": self.id, "level": self.level,
                "value": None if self.value is None else round(float(self.value), 2), "text": self.text}


def resolve(landmarks: dict, suggested: set, sources: dict) -> dict[str, CondylePoint]:
    """Текущие точки по сторонам из ориентиров сессии.

    В сессии на сторону хранится одна точка: подтверждённая врачом заменяет автоматическую, автоматическая не
    заменяет подтверждённую (это обеспечивают set/confirm/reset сессии и автопоиск). Источник без записи:
    подтверждённая — ``user``, предложенная — ``mandible`` (так ставила точки прежняя версия программы)."""
    out = {}
    for side, key in SIDES.items():
        p = landmarks.get(key)
        if p is None:
            out[side] = CondylePoint(side, None, None, False)
            continue
        confirmed = key not in suggested
        source = sources.get(key) or (SOURCE_USER if confirmed else SOURCE_MANDIBLE)
        out[side] = CondylePoint(side, np.asarray(p, float), source, confirmed)
    return out


def to_view(point, bite_motion) -> np.ndarray:
    """Из мм КТ — в координаты монтажа (окно «Артикулятор»): в прикусе сканов точка едет с нижней челюстью."""
    p = np.asarray(point, float)
    return p.copy() if bite_motion is None else apply(np.asarray(bite_motion, float), p[None])[0]


def from_view(point, bite_motion) -> np.ndarray:
    """Обратно: точка, поставленная в окне «Артикулятор», — в мм КТ (нижняя челюсть как на КТ)."""
    p = np.asarray(point, float)
    return p.copy() if bite_motion is None else apply(np.linalg.inv(np.asarray(bite_motion, float)), p[None])[0]


def intercondylar(points: dict[str, CondylePoint]) -> float | None:
    r, l = points["right"].point, points["left"].point
    return None if r is None or l is None else float(np.linalg.norm(r - l))


def _check_icd(points) -> Check:
    icd = intercondylar(points)
    if icd is None:
        return Check("icd", "unknown", None, "нет одной из точек")
    if not ICD_ERROR_MM[0] <= icd <= ICD_ERROR_MM[1]:
        return Check("icd", "error", icd, f"межмыщелковое {icd:.1f} мм — точки не на мыщелках")
    if not ICD_WARN_MM[0] <= icd <= ICD_WARN_MM[1]:
        return Check("icd", "warn", icd, f"межмыщелковое {icd:.1f} мм — необычно, проверьте точки")
    return Check("icd", "ok", icd, f"межмыщелковое {icd:.1f} мм")


def _check_sides(points) -> Check:
    """Правый мыщелок — справа пациента: в LPS у него меньше x, чем у левого."""
    r, l = points["right"].point, points["left"].point
    if r is None or l is None:
        return Check("sides", "unknown", None, "нет одной из точек")
    if r[0] >= l[0]:
        return Check("sides", "error", float(r[0] - l[0]), "правый и левый мыщелки перепутаны")
    return Check("sides", "ok", None, "стороны верны")


class MandibleSurface:
    """Дерево вершин и соседние грани: строятся один раз на неизменную сетку."""

    def __init__(self, vertices, faces=None):
        self.vertices = np.asarray(vertices, float).reshape(-1, 3)
        self.tree = cKDTree(self.vertices)
        self.faces = np.empty((0, 3), int) if faces is None else np.asarray(faces, int).reshape(-1, 3)
        # Для каждой вершины храним диапазон индексов прилегающих граней.
        flat = self.faces.ravel()
        self.face_ids = np.argsort(flat, kind="stable") // 3
        self.offsets = np.r_[0, np.cumsum(np.bincount(flat, minlength=len(self.vertices)))]

    def distance(self, point) -> float:
        if not len(self.vertices):
            raise ValueError("пустая сетка нижней челюсти")
        distance, nearest = self.tree.query(np.asarray(point, float), k=min(16, len(self.vertices)))
        best = float(np.min(np.atleast_1d(distance)))
        if not len(self.faces):
            return best
        ids = np.unique(np.concatenate([self.face_ids[self.offsets[v]:self.offsets[v + 1]]
                                        for v in np.atleast_1d(nearest)]))
        if not len(ids):
            return best
        triangles = self.vertices[self.faces[ids]]
        query = np.repeat(np.asarray(point, float)[None], len(triangles), axis=0)
        projected = closest_point(triangles, query)
        return min(best, float(np.linalg.norm(projected - query, axis=1).min()))


def cached_surface(mesh) -> MandibleSurface:
    """Кэш Trimesh сбрасывается при замене или изменении вершин и граней."""
    surface = mesh._cache["condyle_surface"]
    if surface is None:
        surface = MandibleSurface(mesh.vertices, mesh.faces)
        mesh._cache["condyle_surface"] = surface
    return surface


def _surface_distance(point, vertices, faces=None) -> float:
    """Ближайшие вершины и проекции на соседние грани, без rtree."""
    surface = vertices if isinstance(vertices, MandibleSurface) else MandibleSurface(vertices, faces)
    return surface.distance(point)


def _check_on_mandible(point: CondylePoint, mandible_vertices, mandible_faces=None) -> Check:
    """Точка на поверхности нижней челюсти (как на КТ)."""
    cid = f"on_mandible_{point.side}"
    if point.point is None:
        return Check(cid, "unknown", None, "точки нет")
    if mandible_vertices is None:
        return Check(cid, "unknown", None, "нет сегментации нижней челюсти — проверить нечем")
    try:
        distance = _surface_distance(point.point, mandible_vertices, mandible_faces)
    except ValueError:
        return Check(cid, "unknown", None, "сетка нижней челюсти пуста — проверить нечем")
    side = "правого" if point.side == "right" else "левого"
    if distance > ON_MANDIBLE_ERROR_MM:
        return Check(cid, "error", distance,
                     f"точка {side} мыщелка в {distance:.1f} мм от нижней челюсти")
    if distance > ON_MANDIBLE_WARN_MM:
        return Check(cid, "warn", distance,
                     f"точка {side} мыщелка в {distance:.1f} мм от поверхности — проверьте")
    return Check(cid, "ok", distance, f"точка {side} мыщелка на нижней челюсти")


def _check_asymmetry(points, frame) -> Check:
    """Асимметрия правого и левого по высоте и глубине в системе монтажа (frame: мм КТ → x вправо, y вперёд,
    z вверх). Ведомому: warn больше ASYMMETRY_WARN_MM; текст — какая сторона выше/глубже и на сколько."""
    if points["right"].point is None or points["left"].point is None:
        return Check("asymmetry", "unknown", None, "нет одной из точек")
    if frame is None:
        return Check("asymmetry", "unknown", None, "нет системы монтажа")
    right = apply(np.asarray(frame, float), points["right"].point[None])[0]
    left = apply(np.asarray(frame, float), points["left"].point[None])[0]
    depth, height = float(right[1] - left[1]), float(right[2] - left[2])
    value = max(abs(depth), abs(height))
    details = []
    if abs(height) > ASYMMETRY_WARN_MM:
        details.append(f"{'правый' if height > 0 else 'левый'} выше на {abs(height):.1f} мм")
    if abs(depth) > ASYMMETRY_WARN_MM:
        # В системе монтажа +y направлена вперёд, поэтому меньший y расположен глубже.
        details.append(f"{'левый' if depth > 0 else 'правый'} глубже на {abs(depth):.1f} мм")
    if details:
        return Check("asymmetry", "warn", value, "; ".join(details) + " — проверьте точки")
    return Check("asymmetry", "ok", value,
                 f"разница по высоте {abs(height):.1f} мм, по глубине {abs(depth):.1f} мм")


def checks(points: dict[str, CondylePoint], mandible_vertices=None, frame=None, mandible_faces=None) -> list[Check]:
    """Все проверки точек в мм КТ; грани нужны для точного расстояния до редкой сетки."""
    return [_check_icd(points), _check_sides(points),
            _check_on_mandible(points["right"], mandible_vertices, mandible_faces),
            _check_on_mandible(points["left"], mandible_vertices, mandible_faces),
            _check_asymmetry(points, frame)]


def ready(points: dict[str, CondylePoint], found: list[Check]) -> bool:
    """Монтаж по точкам возможен: обе точки есть и ни одной ошибки."""
    return all(p.point is not None for p in points.values()) and not any(c.level == "error" for c in found)


def state(points: dict[str, CondylePoint], found: list[Check], *, available: bool, bite_motion=None,
          applied: bool = False, mandible: bool = False, skull: bool = False) -> dict:
    """Состояние для интерфейса — только JSON-типы (контракт — docs/handoff/archive/condyle-adjustment.md)."""

    def one(p: CondylePoint) -> dict:
        has = p.point is not None
        return {"key": p.key, "ct": p.point.round(3).tolist() if has else None,
                "view": to_view(p.point, bite_motion).round(3).tolist() if has else None,
                "source": p.source, "source_label": SOURCE_LABELS.get(p.source), "confirmed": p.confirmed}

    icd = intercondylar(points)
    return {"available": available, "bite": "ct" if bite_motion is None else "scan",
            "points": {side: one(p) for side, p in points.items()},
            "icd_mm": None if icd is None else round(icd, 2),
            "checks": [c.to_json() for c in found],
            "ready": ready(points, found), "confirmed": all(p.confirmed and p.point is not None for p in points.values()),
            "applied": applied, "segmentation": {"mandible": mandible, "skull": skull}}
