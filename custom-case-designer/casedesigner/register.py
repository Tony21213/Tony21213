"""Жёсткое совмещение скана с поверхностью из КТ: по парам точек и ICP."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, float) @ T[:3, :3].T + T[:3, 3]


def rigid(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Поворот и сдвиг, переводящие src в dst с наименьшей суммой квадратов отклонений."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 3:
        raise ValueError("нужно минимум 3 пары точек")
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    cs, cd = w @ src, w @ dst
    H = ((src - cs) * w[:, None]).T @ (dst - cd)
    U, _s, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return rigid(R, cd - R @ cs)


@dataclass
class Target:
    """Поверхность, к которой прижимается скан: точки, нормали и дерево поиска."""

    points: np.ndarray
    normals: np.ndarray
    tree: cKDTree | None = None

    def __post_init__(self):
        if self.tree is None:
            self.tree = cKDTree(self.points)


def icp(src: np.ndarray, target: Target, T0: np.ndarray,
        schedule=((2.0, 15), (1.0, 15), (0.5, 15), (0.3, 20)), keep: float = 0.9, tol: float = 1e-6):
    """ICP «точка — плоскость» с отбрасыванием далёких и худших пар.

    schedule — пары (максимальное расстояние пары в мм, число итераций): сначала
    пары ищутся широко, потом допуск сужается. keep — доля лучших пар, которые
    участвуют в расчёте (остальное — десна, артефакты, то, чего нет в КТ).
    Возвращает итоговую матрицу и список пар последней итерации.
    """
    T = T0.copy()
    used = np.zeros(len(src), bool)
    for max_dist, iterations in schedule:
        for _ in range(iterations):
            p = apply(T, src)
            dist, j = target.tree.query(p, distance_upper_bound=max_dist)
            ok = np.isfinite(dist)
            if ok.sum() < 6:
                break
            idx = np.flatnonzero(ok)
            q, n = target.points[j[idx]], target.normals[j[idx]]
            r = np.einsum("ij,ij->i", p[idx] - q, n)
            if keep < 1:
                sel = np.abs(r) <= np.quantile(np.abs(r), keep)
                idx, q, n, r = idx[sel], q[sel], n[sel], r[sel]
            pp = p[idx]
            # Линеаризация малого поворота: r + (p × n)·ω + n·t → 0.
            A = np.hstack([np.cross(pp, n), n])
            x = np.linalg.lstsq(A, -r, rcond=None)[0]
            w, t = x[:3], x[3:]
            angle = np.linalg.norm(w)
            R = axis_angle(w, angle) if angle > 1e-12 else np.eye(3)
            T = rigid(R, t) @ T
            used = np.zeros(len(src), bool)
            used[idx] = True
            if angle < tol and np.linalg.norm(t) < tol:
                break
    return T, used


def fit_score(src: np.ndarray, target: Target, T: np.ndarray, tol: float = 0.5) -> float:
    """Доля точек скана, легших на поверхность ближе tol мм."""
    dist, _ = target.tree.query(apply(T, src), distance_upper_bound=tol)
    return float(np.isfinite(dist).mean())


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Кратчайший поворот, переводящий направление a в направление b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:  # противоположные направления: поворот на 180° вокруг любой перпендикулярной оси
        axis = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        return axis_angle(axis, np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def occlusal_init(src: np.ndarray, src_normals: np.ndarray, target: Target, occlusal: np.ndarray,
                  angle_step: float = 15.0, seed: int = 0):
    """Грубое совмещение скана с коронками одной челюсти без пар точек.

    Скан снят со стороны жевательных поверхностей, поэтому его средняя
    нормаль смотрит туда же, куда жевательные поверхности. Скан поворачивается
    этой стороной к челюсти в КТ (occlusal — направление её жевательных
    поверхностей), затем перебирается поворот вокруг этой оси, и каждый
    вариант уточняется коротким ICP по коронкам. Возвращает лучшую матрицу
    и долю точек скана, легших на коронки.
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(src), min(len(src), 4000), replace=False)
    sub = src[pick]
    view = src_normals.sum(axis=0)
    R0 = rotation_between(view, occlusal)
    src_top = src[src_normals @ (view / np.linalg.norm(view)) > 0.5].mean(axis=0)
    tgt_top = target.points[target.normals @ occlusal > 0.5].mean(axis=0)

    best = (-1.0, np.eye(4))
    for angle in np.arange(0.0, 360.0, angle_step):
        R = axis_angle(occlusal, np.radians(angle)) @ R0
        T0 = rigid(R, tgt_top - R @ src_top)
        T, _ = icp(sub, target, T0, schedule=((6.0, 10), (3.0, 10), (1.5, 10), (0.8, 10)), keep=0.7)
        score = fit_score(sub, target, T)
        if score > best[0]:
            best = (score, T)
    return best[1], best[0]
