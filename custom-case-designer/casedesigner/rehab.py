"""Полная реабилитация: подъём высоты прикуса по оси пациента (Л1) и шаблон окклюзионной плоскости (решение
владельца 13.10: «очень нравится» — цифровой «флаг Бродрика»; порядок работ, пп. 13–14).

* Подъём — поворот нижней челюсти вокруг шарнирной оси пациента (центры мыщелков, ``Anatomy.points``) так, чтобы
  резцовая точка опустилась по вертикали монтажа на заданные мм. Мыщелки на оси не сдвигаются; на молярах подъём
  меньше — во столько, во сколько они ближе к оси (сводка — справа и слева).
* Шаблон — сфера Монсона: радиус 4 дюйма (MONSON_MM, можно задать), через оба мыщелка и резцовую точку (после
  подъёма); центр — над окклюзионной плоскостью. Кривая Шпее — сагиттальное сечение сферы, кривая Уилсона —
  поперечное в области первых моляров. Поверхность — лоскут сферы над зубной дугой, линии — тонкими трубками.
* Всё — в системе кейса (мм КТ, нижняя челюсть — в прикусе монтажа, как у ``JawCase``); вертикаль, вперёд и
  вправо — строки ``Anatomy.frame``.
"""

import numpy as np

from .register import apply, rigid

MONSON_MM = 101.6  # 4 дюйма: радиус сферы Монсона (Бонвиль — сторона треугольника 4 дюйма)
MAX_RAISE_MM = 15.0
MOLAR_BACK_MM = (25.0, 35.0)  # область первых моляров: на столько мм позади резцовой точки
TEMPLATE_HALF_WIDTH = 30.0  # лоскут шаблона: ±мм вправо-влево от середины
TEMPLATE_BACK_MM, TEMPLATE_FRONT_MM = 50.0, 3.0  # назад от резцовой точки (до вторых моляров) и вперёд
CURVE_X_MM = 22.0  # линия бугров вправо-влево от середины — по ней кривые Шпее
WILSON_BACK_MM = 30.0  # кривая Уилсона — на уровне первых моляров
# Наклон окклюзионной плоскости (нормы: Downs — к Франкфурту 9,3 ± 3,8°, Steiner — к SN 14,5 ± 4°)
NORM_FH = (9.3, 3.8)
NORM_SN = (14.5, 4.0)


def _axes(frame):
    F = np.asarray(frame, float)
    return F[0, :3], F[1, :3], F[2, :3]  # вправо, вперёд, вверх


def hinge(anatomy):
    """Шарнирная ось: точка (середина мыщелков) и направление (от левого к правому)."""
    r, l = (np.asarray(anatomy.points[f"condyle_{s}"], float) for s in ("right", "left"))
    d = r - l
    return (r + l) / 2, d / np.linalg.norm(d)


def _rotation(axis_point, axis, deg) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(np.radians(deg) * axis).as_matrix()
    return rigid(R, axis_point - R @ axis_point)


def raise_transform(anatomy, incisal_mm: float) -> tuple[np.ndarray, float]:
    """Поворот нижней челюсти вокруг оси мыщелков: резцовая точка опускается по вертикали монтажа на incisal_mm.
    Возвращает (матрица 4×4 в системе кейса, угол, градусы)."""
    mm = float(incisal_mm)
    if not 0 <= mm <= MAX_RAISE_MM:
        raise ValueError(f"подъём по резцам: от 0 до {MAX_RAISE_MM:g} мм")
    if mm == 0:
        return np.eye(4), 0.0
    p, a = hinge(anatomy)
    _right, _ant, up = _axes(anatomy.frame)
    inc = np.asarray(anatomy.points["incisal"], float)
    drop = lambda deg: float((inc - apply(_rotation(p, a, deg), inc[None])[0]) @ up)  # noqa: E731
    sign = 1.0 if drop(1.0) > 0 else -1.0  # открывание — резцы вниз
    lo, hi = 0.0, 45.0
    if drop(sign * hi) < mm:
        raise ValueError("такой подъём не получить поворотом вокруг оси — проверьте монтаж")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if drop(sign * mid) < mm else (lo, mid)
    deg = sign * (lo + hi) / 2
    return _rotation(p, a, deg), abs(deg)


def raise_report(anatomy, T: np.ndarray, lower_vertices) -> dict:
    """Сколько подъём даёт по вертикали монтажа: на резцовой точке и в области первых моляров справа и слева
    (верхушки бугров нижнего скана в MOLAR_BACK_MM за резцами)."""
    right, ant, up = _axes(anatomy.frame)
    inc = np.asarray(anatomy.points["incisal"], float)
    out = {"incisal_mm": round(float((inc - apply(T, inc[None])[0]) @ up), 2), "molars": {}}
    v = np.asarray(lower_vertices, float)
    back = (inc - v) @ ant
    for side, sign in (("right", 1.0), ("left", -1.0)):
        zone = v[(back >= MOLAR_BACK_MM[0]) & (back <= MOLAR_BACK_MM[1]) & (sign * ((v - inc) @ right) > 5.0)]
        if len(zone) < 20:
            continue
        h = zone @ up
        cusp = zone[h >= np.quantile(h, 0.9)].mean(0)
        out["molars"][side] = round(float((cusp - apply(T, cusp[None])[0]) @ up), 2)
    return out


def monson_center(anatomy, incisal_point, radius: float = MONSON_MM) -> np.ndarray:
    """Центр сферы радиуса radius через оба мыщелка и резцовую точку — над окклюзионной плоскостью."""
    r, l = (np.asarray(anatomy.points[f"condyle_{s}"], float) for s in ("right", "left"))
    q = np.asarray(incisal_point, float)
    _right, _ant, up = _axes(anatomy.frame)
    a, b = r - q, l - q
    n = np.cross(a, b)
    n /= np.linalg.norm(n)
    # центр описанной окружности треугольника
    axb = np.cross(a, b)
    cc = q + np.cross(np.dot(a, a) * b - np.dot(b, b) * a, axb) / (2 * np.dot(axb, axb))
    rc = float(np.linalg.norm(cc - q))
    if radius < rc:
        raise ValueError(f"радиус сферы {radius:g} мм меньше расстояния от мыщелков до резцов — нужно не меньше {rc:.0f} мм")
    h = np.sqrt(radius ** 2 - rc ** 2)
    return cc + (h if n @ up > 0 else -h) * n


def _on_sphere(center, radius, x, y, inc, right, ant, up):
    """Точка лоскута: (x вправо, y вперёд от резцовой точки) — на нижней стороне сферы."""
    base = inc + x * right + y * ant
    d = base - center
    lateral = d - (d @ up) * up
    return center + lateral - np.sqrt(max(radius ** 2 - lateral @ lateral, 0.0)) * up


def template(anatomy, incisal_point, radius: float = MONSON_MM, landmarks: dict | None = None, step: float = 2.0) -> dict:
    """Шаблон окклюзионной плоскости: поверхность (лоскут сферы над дугой, две стороны) и линии (сетка через 5 мм,
    кривые Шпее справа и слева, кривая Уилсона на первых молярах — толще); сводка: центр и радиус, глубина кривой
    Шпее (от прямой «резцы — вторые моляры»), наклон плоскости шаблона к Франкфурту и SN (если есть точки)."""
    import trimesh

    right, ant, up = _axes(anatomy.frame)
    inc = np.asarray(incisal_point, float)
    c = monson_center(anatomy, inc, radius)
    xs = np.arange(-TEMPLATE_HALF_WIDTH, TEMPLATE_HALF_WIDTH + 1e-6, step)
    ys = np.arange(-TEMPLATE_BACK_MM, TEMPLATE_FRONT_MM + 1e-6, step)
    grid = np.array([[_on_sphere(c, radius, x, y, inc, right, ant, up) for x in xs] for y in ys])
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(ny, nx)
    quads = np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
    faces = np.r_[quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]
    surface = trimesh.Trimesh(grid.reshape(-1, 3), np.r_[faces, faces[:, ::-1]], process=False)  # две стороны

    def tube(points, r):
        parts = [trimesh.creation.cylinder(radius=r, segment=np.array([a, b]), sections=8)
                 for a, b in zip(points[:-1], points[1:]) if np.linalg.norm(b - a) > 1e-6]
        return trimesh.util.concatenate(parts)

    lines = []
    for x in np.arange(-TEMPLATE_HALF_WIDTH, TEMPLATE_HALF_WIDTH + 1e-6, 5.0):
        lines.append(tube([_on_sphere(c, radius, x, y, inc, right, ant, up) for y in ys], 0.12))
    for y in np.arange(-TEMPLATE_BACK_MM, TEMPLATE_FRONT_MM + 1e-6, 5.0):
        lines.append(tube([_on_sphere(c, radius, x, y, inc, right, ant, up) for x in xs], 0.12))
    spee = {side: [_on_sphere(c, radius, sign * CURVE_X_MM, y, inc, right, ant, up) for y in ys]
            for side, sign in (("right", 1.0), ("left", -1.0))}
    lines += [tube(spee["right"], 0.35), tube(spee["left"], 0.35)]
    lines.append(tube([_on_sphere(c, radius, x, -WILSON_BACK_MM, inc, right, ant, up) for x in xs], 0.35))
    curves = trimesh.util.concatenate(lines)

    # глубина кривой Шпее: наибольшее отклонение кривой от прямой между её концами (резцы — вторые моляры)
    depth = []
    for pts in spee.values():
        pts = np.asarray(pts)
        a, b = pts[-1], pts[0]
        d = (b - a) / np.linalg.norm(b - a)
        off = (pts - a) - np.outer((pts - a) @ d, d)
        depth.append(float(np.linalg.norm(off, axis=1).max()))
    # плоскость шаблона: резцовая точка и точки сферы на первых молярах справа и слева
    m_r = _on_sphere(c, radius, CURVE_X_MM, -WILSON_BACK_MM, inc, right, ant, up)
    m_l = _on_sphere(c, radius, -CURVE_X_MM, -WILSON_BACK_MM, inc, right, ant, up)
    out = {"surface": surface, "curves": curves, "center": c, "radius": float(radius),
           "spee_depth_mm": round(float(np.mean(depth)), 2), "tilt": tilt(inc, (m_r + m_l) / 2, landmarks)}
    return out


def tilt(incisal, molar_mid, landmarks: dict | None) -> dict:
    """Наклон окклюзионной плоскости шаблона (прямая «моляры — резцы» в срединной плоскости) к Франкфурту и SN,
    градусы, с нормами; нет точек — пусто."""
    from . import cephalometry as ce

    out = {}
    if not landmarks:
        return out
    try:
        plane = ce.midsagittal(landmarks)
    except ValueError:
        return out
    P = ce._points(landmarks)
    proj = lambda p: ce._project(np.asarray(p, float), plane)  # noqa: E731
    occ = proj(incisal) - proj(molar_mid)

    def angle(a, b):
        d = proj(b) - proj(a)
        x = ce._angle(occ, d)
        return min(x, 180.0 - x)

    if "Po" in P and "Or" in P:
        out["fh"] = {"deg": round(angle(P["Po"], P["Or"]), 1), "norm": NORM_FH}
    if "S" in P and "N" in P:
        out["sn"] = {"deg": round(angle(P["S"], P["N"]), 1), "norm": NORM_SN}
    return out
