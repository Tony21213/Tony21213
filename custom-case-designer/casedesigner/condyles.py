"""Точки мыщелков Co_R и Co_L в шаге «Артикулятор»: приоритет, система координат, проверки, состояние.

Контракт (подробно — docs/handoff.md, «Ручная проверка и поправка мыщелков»):

* **Хранение.** Точки — ориентиры ``Co_R`` / ``Co_L`` сессии (``Session.landmarks``): миллиметры пациента КТ
  (DICOM LPS: x — влево, y — назад, z — вверх), нижняя челюсть — как на снимке. Это та же точка, что
  «Мыщелок правый/левый» в ориентирах: верхушка головки нижней челюсти. В ``.ccdcase`` — ``meta["landmarks"]``,
  ``meta["suggested"]`` и ``meta["landmark_sources"]``.
* **Происхождение и приоритет.** У точки есть источник (``SOURCES``) и признак подтверждения: подтверждённая
  врачом точка (ключа нет в ``Session.suggested``) выше любой автоматической; автоматическая
  (ALI-CBCT или по сетке нижней челюсти) — «проверьте», её заменяет следующий автоматический поиск;
  поставленную или подтверждённую врачом автоматика не трогает никогда.
* **Применение.** Модели монтируются в прикусе сканов, если он найден (``Session.bite_motion``), иначе — как
  на КТ. Мыщелок принадлежит нижней челюсти и едет с ней: в прикусе сканов шарнирная точка —
  ``bite_motion · p``, в прикусе КТ — ``p``. ССП по суставному бугорку — всегда в соотношении КТ: точка как
  есть и череп КТ (кость черепа с нижней челюстью не едет).
* **Проверки** — ``checks``: межмыщелковое расстояние, стороны, точка на сетке нижней челюсти, асимметрия.
  Уровни ``ok`` / ``warn`` / ``error`` / ``unknown`` (проверить нечем). С ``error`` монтаж по точкам не идёт.
"""

from dataclasses import dataclass

import numpy as np

from .register import apply

SIDES = {"right": "Co_R", "left": "Co_L"}
SOURCE_USER, SOURCE_ALI, SOURCE_MANDIBLE = "user", "ali-cbct", "mandible"
SOURCES = (SOURCE_USER, SOURCE_ALI, SOURCE_MANDIBLE)
SOURCE_LABELS = {SOURCE_USER: "врач", SOURCE_ALI: "нейросеть ALI-CBCT", SOURCE_MANDIBLE: "по нижней челюсти"}

ICD_WARN_MM = (80.0, 125.0)  # межмыщелковое вне — предупреждение
ICD_ERROR_MM = (60.0, 140.0)  # вне — ошибка: точка явно не на мыщелке
ON_MANDIBLE_WARN_MM, ON_MANDIBLE_ERROR_MM = 1.5, 5.0  # точка от поверхности нижней челюсти
ASYMMETRY_WARN_MM = 5.0  # разница правого и левого по высоте и по глубине (система монтажа)
LEVELS = ("ok", "warn", "error", "unknown")


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


def _check_on_mandible(point: CondylePoint, mandible_vertices) -> Check:
    """Точка на поверхности нижней челюсти (как на КТ). Ведомому: расстояние до сетки, пороги
    ON_MANDIBLE_WARN_MM / ON_MANDIBLE_ERROR_MM; точка у венечного отростка — warn."""
    cid = f"on_mandible_{point.side}"
    if point.point is None:
        return Check(cid, "unknown", None, "точки нет")
    if mandible_vertices is None:
        return Check(cid, "unknown", None, "нет сегментации нижней челюсти — проверить нечем")
    return Check(cid, "unknown", None, "проверка ещё не реализована")  # TODO(ведомый)


def _check_asymmetry(points, frame) -> Check:
    """Асимметрия правого и левого по высоте и глубине в системе монтажа (frame: мм КТ → x вправо, y вперёд,
    z вверх). Ведомому: warn больше ASYMMETRY_WARN_MM; текст — какая сторона выше/глубже и на сколько."""
    if points["right"].point is None or points["left"].point is None:
        return Check("asymmetry", "unknown", None, "нет одной из точек")
    if frame is None:
        return Check("asymmetry", "unknown", None, "нет системы монтажа")
    return Check("asymmetry", "unknown", None, "проверка ещё не реализована")  # TODO(ведомый)


def checks(points: dict[str, CondylePoint], mandible_vertices=None, frame=None) -> list[Check]:
    """Все проверки точек, в мм КТ (челюсть как на КТ). mandible_vertices — сетка нижней челюсти КТ."""
    return [_check_icd(points), _check_sides(points),
            _check_on_mandible(points["right"], mandible_vertices),
            _check_on_mandible(points["left"], mandible_vertices),
            _check_asymmetry(points, frame)]


def ready(points: dict[str, CondylePoint], found: list[Check]) -> bool:
    """Монтаж по точкам возможен: обе точки есть и ни одной ошибки."""
    return all(p.point is not None for p in points.values()) and not any(c.level == "error" for c in found)


def state(points: dict[str, CondylePoint], found: list[Check], *, available: bool, bite_motion=None,
          applied: bool = False, mandible: bool = False, skull: bool = False) -> dict:
    """Состояние для интерфейса — только JSON-типы (контракт в docs/handoff.md)."""

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
