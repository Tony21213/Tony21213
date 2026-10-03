"""Поверхности твёрдых тканей из КТ и их уточнение до долей вокселя."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import filters, measure


@dataclass
class Surface:
    points: np.ndarray  # (n, 3) мм пациента
    normals: np.ndarray  # (n, 3) единичные, наружу (от плотной ткани)
    faces: np.ndarray | None = None  # (m, 3), если это сетка


def tissue_levels(vol, sample: int = 2_000_000, seed: int = 0):
    """Пороги «мягкие ткани / кость» и «кость / зубы».

    КЛКТ не откалиброван в HU, поэтому пороги берутся из самого снимка:
    четырёхуровневый Оцу делит воксели на воздух, мягкие ткани, кость и зубы
    (эмаль и дентин плотнее кости).
    """
    flat = vol.data.ravel()
    if flat.size > sample:
        flat = np.random.default_rng(seed).choice(flat, sample, replace=False)
    _air, hard, dense = filters.threshold_multiotsu(flat, classes=4)
    return float(hard), float(dense)


def extract_surface(vol, level: float, roi=None, step: int = 1) -> Surface:
    """Изоповерхность уровня level (marching cubes) в мм пациента.

    roi — (min_xyz, max_xyz) в мм пациента: считать только внутри этой рамки.
    step > 1 — грубая и быстрая поверхность.
    """
    lo_idx = np.zeros(3, int)
    hi_idx = np.array(vol.data.shape[::-1]) - 1
    if roi is not None:
        corners = np.array([[x, y, z] for x in (roi[0][0], roi[1][0]) for y in (roi[0][1], roi[1][1])
                            for z in (roi[0][2], roi[1][2])])
        idx = vol.to_index(corners)
        lo_idx = np.maximum(np.floor(idx.min(axis=0)).astype(int), 0)
        hi_idx = np.minimum(np.ceil(idx.max(axis=0)).astype(int), hi_idx)
        if np.any(hi_idx - lo_idx < 2):
            return Surface(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3), int))
    sub = vol.data[lo_idx[2]:hi_idx[2] + 1, lo_idx[1]:hi_idx[1] + 1, lo_idx[0]:hi_idx[0] + 1]
    if not (sub.min() < level < sub.max()):
        return Surface(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3), int))
    verts, faces, _n, _v = measure.marching_cubes(sub, level, step_size=step, allow_degenerate=False)
    points = vol.to_world(verts[:, ::-1] + lo_idx)
    return Surface(points, outward_normals(vol, points), faces.astype(np.int64))


def outward_normals(vol, points: np.ndarray) -> np.ndarray:
    """Нормали по градиенту яркости: смотрят туда, где ткань становится менее плотной."""
    g = -vol.gradient(points)
    norm = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(norm, 1e-12)


def refine_edges(vol, surface: Surface, reach: float = 0.8, step: float = 0.05, min_contrast: float = 0.0):
    """Сдвигает каждую точку вдоль нормали на максимум перепада яркости.

    Изоповерхность по порогу смещена относительно настоящей границы: где
    именно — зависит от плотностей по обе стороны. Граница же — это место
    самого резкого перепада, и его положение находится с точностью до
    долей вокселя. Возвращает уточнённую поверхность и маску точек, где
    граница выражена (между зубами в прикусе её нет — такие точки отбрасываются).
    """
    t = np.arange(-reach, reach + step / 2, step)
    pts, nrm = surface.points, surface.normals
    if not len(pts):
        return surface, np.zeros(0, bool)
    samples = pts[:, None, :] + t[None, :, None] * nrm[:, None, :]
    profile = _sample_cubic(vol, samples.reshape(-1, 3)).reshape(len(pts), len(t))
    drop = -np.gradient(profile, step, axis=1)  # падение яркости наружу, на мм
    k = np.argmax(drop, axis=1)
    peak = drop[np.arange(len(pts)), k]

    inner = (k > 0) & (k < len(t) - 1)
    kk = np.clip(k, 1, len(t) - 2)
    a = drop[np.arange(len(pts)), kk - 1]
    c = drop[np.arange(len(pts)), kk + 1]
    denom = a - 2 * peak + c
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0)
    offset = t[kk] + np.clip(shift, -0.5, 0.5) * step

    span = profile.max(axis=1) - profile.min(axis=1)
    valid = inner & (peak > 0) & (span > min_contrast)
    refined = Surface(pts + offset[:, None] * nrm, nrm, None)
    return refined, valid


def _sample_cubic(vol, points: np.ndarray) -> np.ndarray:
    """Яркость в точках с кубической интерполяцией.

    У трилинейной интерполяции производная ступенчатая, и максимум перепада
    «прилипает» к узлам сетки; кубический сплайн гладкий. Сплайн строится
    только по области вокруг точек, а не по всему объёму.
    """
    idx = vol.to_index(points)
    size = np.array(vol.data.shape[::-1])
    lo = np.clip(np.floor(idx.min(axis=0)).astype(int) - 3, 0, size - 1)
    hi = np.clip(np.ceil(idx.max(axis=0)).astype(int) + 4, 1, size)
    sub = vol.data[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    coef = ndimage.spline_filter(sub, order=3, output=np.float32)
    return ndimage.map_coordinates(coef, (idx - lo)[:, ::-1].T, order=3, prefilter=False, mode="nearest")
