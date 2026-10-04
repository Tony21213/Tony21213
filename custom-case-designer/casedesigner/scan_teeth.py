"""Коронки зубов на скане: что из скана участвует в совмещении.

Полезная для совмещения геометрия — коронки: они есть и на скане, и на КТ.
Десна и нёбо на скане бесполезны (на КТ их почти не видно), корни на КТ —
тоже (на скане их нет). На КТ коронки выделяются по плотности
(teeth.crown_surface), на скане — по форме:

1. скан снят со стороны прикуса, поэтому средняя нормаль смотрит в сторону
   жевательных поверхностей;
2. вершины бугров — самые высокие в этом направлении точки; через них
   проходит жевательная плоскость;
3. коронки не глубже своей высоты от этой плоскости. Нёбо и глубокая десна
   лежат глубже и отсекаются; десна у самых зубов остаётся, но её отсекает
   уже совмещение — на КТ под ней нет плотной поверхности.

После совмещения коронкой на скане окончательно считается то, что легло на
коронки КТ (Registration.deviation не nan).

Сканы гипсовых моделей с настольного сканера часто замкнуты (цоколь
отсканирован со всех сторон или дно закрыто): нормали такой сетки взаимно
гасятся, и пункт 1 не работает. Тогда направление — по форме модели
(model_direction): ось — та из главных осей облака, вдоль которой у края лежат
бугры, а не плоскость (дно, стенки цоколя). Дальше всё как у внутриротового
скана: цоколь лежит глубже коронок и отсекается.
"""

import numpy as np

# Высота коронки с запасом, мм: всё, что глубже от жевательной плоскости, — не коронка.
CROWN_HEIGHT_MM = 9.0
# Доля самых высоких точек, по которым строится жевательная плоскость (вершины бугров).
CUSP_SHARE = 0.03


# Средняя нормаль открытого скана (снят с одной стороны) не короче этой доли; у замкнутой модели — около нуля.
OPEN_SHARE = 0.15
# Доля крайних точек вдоль направления, по которой судят, бугры там или плоскость.
EDGE_SHARE = 0.03
# Край, где нормали в среднем совпадают с направлением сильнее, — плоскость (дно или стенка цоколя).
FLAT = 0.9


def is_closed(normals: np.ndarray) -> bool:
    """Сетка снята со всех сторон (замкнутая модель): нормали взаимно гасятся."""
    return bool(np.linalg.norm(normals.sum(axis=0)) < OPEN_SHARE * len(normals))


def model_direction(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Направление на жевательные поверхности замкнутой модели — по форме.

    Кандидаты — главные оси облака точек в обе стороны. На каждом краю берутся
    крайние EDGE_SHARE точек: у дна и стенок цоколя их нормали смотрят наружу
    почти точно вдоль оси (плоскость), у бугров — вразброс. Выбирается край
    с буграми; при равенстве — ось наименьшего размаха (высота модели).
    """
    rng = np.random.default_rng(0)
    pick = rng.choice(len(vertices), min(len(vertices), 200000), replace=False)
    v, n = vertices[pick], normals[pick]
    axes = np.linalg.svd(v - v.mean(0), full_matrices=False)[2]  # от большего размаха к меньшему
    best = None
    for rank, axis in enumerate(axes):
        for d in (axis, -axis):
            h = v @ d
            edge = h >= np.quantile(h, 1 - EDGE_SHARE)
            flat = float((n[edge] @ d).mean())
            score = (flat > FLAT, flat - 0.05 * rank)
            if best is None or score < best[0]:
                best = (score, d)
    return best[1] / np.linalg.norm(best[1])


def occlusal_direction(normals: np.ndarray, vertices: np.ndarray | None = None) -> np.ndarray:
    """Куда смотрят жевательные поверхности: средняя нормаль открытого скана, у замкнутой модели — по форме."""
    if vertices is not None and is_closed(normals):
        return model_direction(vertices, normals)
    view = normals.sum(axis=0)
    return view / np.linalg.norm(view)


def crown_candidates(vertices: np.ndarray, normals: np.ndarray, crown_height: float = CROWN_HEIGHT_MM) -> np.ndarray:
    """Маска вершин скана, которые могут быть коронками."""
    view = occlusal_direction(normals, vertices)
    u = np.cross(view, [1.0, 0, 0] if abs(view[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(view, u)
    h, a, b = vertices @ view, vertices @ u, vertices @ w
    cusps = h >= np.quantile(h, 1 - CUSP_SHARE)
    coef = np.linalg.lstsq(np.c_[a[cusps], b[cusps], np.ones(cusps.sum())], h[cusps], rcond=None)[0]
    plane = np.c_[a, b, np.ones(len(vertices))] @ coef
    return plane - h < crown_height
