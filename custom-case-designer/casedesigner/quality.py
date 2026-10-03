"""Проверка скана после совмещения: не искажена ли дуга.

Внутриротовой скан полной дуги склеивается из сотен кадров, и ошибка склейки
копится вдоль дуги: дальний участок может «уехать» на десятые доли миллиметра.
Такой скан целиком на КТ не ляжет — какой-то участок будет упорно
отклоняться. Проверка делит скан на участки вдоль дуги и подгоняет каждый к
КТ отдельно: у цельного скана участки остаются на месте, у искажённого
участок сдвигается относительно остальных. Это только предупреждение —
решение (пересканировать, поправить вручную, принять) за пользователем.
"""

from dataclasses import dataclass

import numpy as np

from .register import Target, apply, icp
from .teeth import OCCLUSAL

# Участок сдвинулся при отдельной подгонке больше этого — скан, вероятно, искажён.
SHIFT_WARN_MM = 0.1
# Доля точек участка на коронках меньше этой доли от типичной — участок «не ложится».
MATCH_WARN_RATIO = 0.5
# Общее отклонение: 90% точек дальше этого — совмещение в целом неточное.
P90_WARN_MM = 0.25
LOCAL_SCHEDULE = ((0.5, 10), (0.3, 15))


@dataclass
class Segment:
    where: str  # где участок у пациента: «справа сзади» и т. п.
    points: int
    matched_fraction: float  # доля точек участка, легших на коронки
    shift_mm: float  # насколько участок сдвигается, если подгонять его отдельно (среднее по точкам)


def _where(centre: np.ndarray, arch_centre: np.ndarray) -> str:
    """Положение участка у пациента по осям DICOM (LPS: +x — к левой стороне пациента, +y — назад)."""
    d = centre - arch_centre
    side = "слева" if d[0] > 3 else "справа" if d[0] < -3 else "по центру"
    depth = "сзади" if d[1] > 3 else "спереди" if d[1] < -3 else ""
    return f"{side} {depth}".strip()


def _segments_along_arch(points: np.ndarray, occlusal: np.ndarray, count: int) -> np.ndarray:
    """Номер участка для каждой точки: дуга режется на count частей поровну по числу точек."""
    u = np.cross(occlusal, [1.0, 0, 0] if abs(occlusal[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(occlusal, u)
    rel = points - points.mean(axis=0)
    angle = np.arctan2(rel @ v, rel @ u)
    # Разрез — в самом широком пустом промежутке углов (там, где дуга открыта).
    order = np.sort(angle)
    gaps = np.diff(np.append(order, order[0] + 2 * np.pi))
    cut = order[(np.argmax(gaps) + 1) % len(order)]
    unwrapped = np.mod(angle - cut, 2 * np.pi)
    edges = np.quantile(unwrapped, np.linspace(0, 1, count + 1)[1:-1])
    return np.searchsorted(edges, unwrapped)


def check_scan(scan_vertices: np.ndarray, transform: np.ndarray, deviation: np.ndarray, target: Target,
               jaw: str, segments: int = 6):
    """Участки скана и предупреждения для пользователя.

    target — коронки КТ, к которым подгонялся скан (с учётом сдвига границы
    эмали), deviation — отклонения вершин из совмещения.
    """
    placed = apply(transform, scan_vertices)
    occlusal = OCCLUSAL[jaw]
    label = _segments_along_arch(placed, occlusal, segments)
    matched = np.isfinite(deviation)
    arch_centre = placed[matched].mean(axis=0) if matched.any() else placed.mean(axis=0)
    typical = np.median([matched[label == k].mean() for k in range(segments)])

    result, warnings = [], []
    for k in range(segments):
        idx = np.flatnonzero(label == k)
        frac = float(matched[idx].mean())
        shift = 0.0
        mine = idx[matched[idx]]
        if len(mine) >= 50:
            local, _ = icp(scan_vertices[mine], target, transform, schedule=LOCAL_SCHEDULE, keep=0.9)
            shift = float(np.linalg.norm(apply(local, scan_vertices[mine]) - placed[mine], axis=1).mean())
        seg = Segment(_where(placed[idx].mean(axis=0), arch_centre), len(idx), round(frac, 3), round(shift, 3))
        result.append(seg)
        if shift > SHIFT_WARN_MM:
            warnings.append(f"Участок {seg.where} не совпадает с остальной дугой: при отдельной подгонке "
                            f"он сдвигается на {shift:.2f} мм. Вероятно, скан искажён при склейке — "
                            "проверьте его или пересканируйте этот участок.")
        elif typical > 0 and frac < MATCH_WARN_RATIO * typical:
            warnings.append(f"Участок {seg.where} почти не ложится на КТ ({100 * frac:.0f}% точек на коронках "
                            f"при обычных {100 * typical:.0f}%). Возможны искажение скана, металл или "
                            "изменения в полости рта между КТ и сканированием.")
    d = np.abs(deviation[matched])
    if len(d) and np.quantile(d, 0.9) > P90_WARN_MM:
        warnings.append(f"Совмещение в целом неточное: 10% точек коронок дальше {np.quantile(d, 0.9):.2f} мм от КТ.")
    return result, warnings
