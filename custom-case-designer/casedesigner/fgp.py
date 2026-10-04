"""Функционально сформированный путь (FGP): огибающая движений антагониста.

Для верхних реставраций важно, где во время движений проходят нижние зубы:
всё пространство, которое они заметают при протрузии, боковых движениях и
промежуточных направлениях — с ведением по реальным зубам (сканам). Верхняя
граница этого пространства — «функционально сформированный антагонист»:
коронка, не заходящая за неё, не мешает движениям пациента. Для нижних
реставраций — наоборот: нижняя граница верхних зубов в системе нижней челюсти.

Поверхность — карта высот над окклюзионной плоскостью (анатомическая система
Anatomy.frame, z — вверх) с шагом CELL_MM: в каждой клетке самое высокое
(для нижних реставраций — самое низкое) положение зубов антагониста за все
кадры движений. Клетка берёт максимум, поэтому на склонах поверхность выше
истинной на половину клетки × тангенс наклона (0.06 мм на 40°) — с запасом.
Результат — сетка в координатах кейса (exocad): её можно загрузить в exocad
как обычную модель антагониста.
"""

import numpy as np
from scipy.spatial import cKDTree

from . import kinematics as kin
from .motion import Anatomy
from .register import apply

CELL_MM = 0.15  # шаг карты высот
TRAVEL_MM = 6.0  # путь мыщелков в каждом движении
FRAMES = 31
BLENDS = (0.0, 1 / 3, 2 / 3, 1.0)  # от протрузии (0) к чистой латеротрузии (1)
REACH_MM = 4.0  # точки антагониста не дальше этого от зубов в прикусе — жевательная часть, без десны и нёба


def excursions(anatomy: Anatomy, settings: kin.Settings, occlusion: kin.Occlusion, travel: float = TRAVEL_MM,
               blends=BLENDS, frames: int = FRAMES) -> list:
    """Веер движений из прикуса: протрузия, боковые в обе стороны и промежуточные направления."""
    c0 = kin.condyles(anatomy)
    V = kin._to_mount(anatomy, settings)
    recs = []
    for side in ("right", "left"):
        for b in blends:
            if b == 0 and side == "left":
                continue  # протрузия — одна
            poses, theta = [], 0.0
            for t in np.linspace(0, travel, frames):
                pro = kin.protrusion_targets(c0, settings, t, V)
                lat, anchor = kin.laterotrusion_targets(c0, settings, side, t, V)
                c1 = {k: (1 - b) * pro[k] + b * lat[k] for k in c0}
                anchor = anchor if b == 1 else "mid"
                theta = occlusion.contact(c0, c1, anchor, theta)
                poses.append(kin.jaw_pose(c0, c1, theta, anchor))
            name = "протрузия" if b == 0 else f"{'вправо' if side == 'right' else 'влево'} {round(100 * b)}%"
            recs.append(kin._recording(name, anatomy, poses, 1.0))
    return recs


def _near(points: np.ndarray, other: np.ndarray, reach: float) -> np.ndarray:
    d, _ = cKDTree(other).query(points, distance_upper_bound=reach)
    return points[np.isfinite(d)]


def envelope(points: np.ndarray, poses: list, cell: float = CELL_MM, upward: bool = True):
    """Карта высот заметаемого пространства: (вершины, грани) в анатомической системе.

    points — точки антагониста (анатомическая система), poses — их положения
    по кадрам (4×4 в той же системе). upward — верхняя граница (для верхних
    реставраций), иначе нижняя.
    """
    sign = 1.0 if upward else -1.0
    lo = points[:, :2].min(0) - TRAVEL_MM * 2
    hi = points[:, :2].max(0) + TRAVEL_MM * 2
    nx, ny = (np.ceil((hi - lo) / cell).astype(int) + 1)
    best = np.full(nx * ny, -np.inf)
    for M in poses:
        q = apply(M, points)
        ij = np.floor((q[:, :2] - lo) / cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        np.maximum.at(best, ij[ok, 0] * ny + ij[ok, 1], sign * q[ok, 2])
    grid = best.reshape(nx, ny)
    have = np.isfinite(grid)
    index = np.full((nx, ny), -1)
    index[have] = np.arange(have.sum())
    gx, gy = np.nonzero(have)
    verts = np.c_[lo[0] + (gx + 0.5) * cell, lo[1] + (gy + 0.5) * cell, sign * grid[have]]
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
    quad = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d = a[quad], b[quad], c[quad], d[quad]
    faces = np.vstack([np.c_[a, b, d], np.c_[a, d, c]])
    if not upward:
        faces = faces[:, ::-1]  # нормали — к реставрации
    return verts, faces


def fgp(upper_vertices, upper_faces, lower_vertices, anatomy: Anatomy, settings: kin.Settings | None = None,
        for_jaw: str = "upper", cell: float = CELL_MM, recordings: list | None = None):
    """Функционально сформированный антагонист для реставраций на челюсти for_jaw.

    Сканы — в прикусе, в координатах кейса. Возвращает (вершины, грани) в
    координатах кейса и записи движений, по которым он построен.
    """
    settings = settings or kin.Settings()
    F = anatomy.frame
    back = np.linalg.inv(F)
    occlusion = kin.Occlusion(upper_vertices, upper_faces, lower_vertices, F)
    recs = recordings if recordings is not None else excursions(anatomy, settings, occlusion)
    up_a, low_a = apply(F, np.asarray(upper_vertices, float)), apply(F, np.asarray(lower_vertices, float))
    moves = [F @ T @ back for r in recs for T in r.transforms]  # положения нижней челюсти, анатомическая система
    if for_jaw == "upper":  # нижние зубы в системе верхней челюсти
        verts, faces = envelope(_near(low_a, up_a, REACH_MM), moves, cell, upward=True)
    elif for_jaw == "lower":  # верхние зубы в системе нижней челюсти
        verts, faces = envelope(_near(up_a, low_a, REACH_MM), [np.linalg.inv(M) for M in moves], cell, upward=False)
    else:
        raise ValueError("for_jaw: upper или lower")
    return apply(back, verts), faces, recs
