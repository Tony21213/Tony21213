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
"""

import numpy as np

# Высота коронки с запасом, мм: всё, что глубже от жевательной плоскости, — не коронка.
CROWN_HEIGHT_MM = 9.0
# Доля самых высоких точек, по которым строится жевательная плоскость (вершины бугров).
CUSP_SHARE = 0.03


def occlusal_direction(normals: np.ndarray) -> np.ndarray:
    view = normals.sum(axis=0)
    return view / np.linalg.norm(view)


def crown_candidates(vertices: np.ndarray, normals: np.ndarray, crown_height: float = CROWN_HEIGHT_MM) -> np.ndarray:
    """Маска вершин скана, которые могут быть коронками."""
    view = occlusal_direction(normals)
    u = np.cross(view, [1.0, 0, 0] if abs(view[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(view, u)
    h, a, b = vertices @ view, vertices @ u, vertices @ w
    cusps = h >= np.quantile(h, 1 - CUSP_SHARE)
    coef = np.linalg.lstsq(np.c_[a[cusps], b[cusps], np.ones(cusps.sum())], h[cusps], rcond=None)[0]
    plane = np.c_[a, b, np.ones(len(vertices))] @ coef
    return plane - h < crown_height
