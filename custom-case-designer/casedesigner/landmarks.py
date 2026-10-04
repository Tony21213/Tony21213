"""Цефалометрические ориентиры, референтные плоскости и система координат для артикулятора.

Ориентиры — точки в мм пациента (DICOM, LPS). Часть программа предлагает
сама по сегментации (мыщелки, порион), остальные ставит врач кликом на
срезах; предложенные тоже проверяет врач.

Система координат плоскости (для выгрузки в артикулятор):
  * начало — середина шарнирной оси (между верхушками мыщелков);
  * Z — нормаль к выбранной горизонтальной плоскости, вверх;
  * X — вправо пациента, в плоскости, перпендикулярно срединно-сагиттальной;
  * Y — вперёд (X × Y = Z, правая тройка).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Landmark:
    key: str
    name: str
    hint: str


LANDMARKS = (
    Landmark("Po_R", "Порион правый", "верхняя точка наружного слухового прохода"),
    Landmark("Po_L", "Порион левый", "верхняя точка наружного слухового прохода"),
    Landmark("Or_R", "Орбиталь правая", "нижняя точка края глазницы"),
    Landmark("Or_L", "Орбиталь левая", "нижняя точка края глазницы"),
    Landmark("Co_R", "Мыщелок правый", "верхняя точка головки нижней челюсти"),
    Landmark("Co_L", "Мыщелок левый", "верхняя точка головки нижней челюсти"),
    Landmark("HN_R", "Крыловидно-челюстная вырезка правая", "между бугром верхней челюсти и крыловидным отростком"),
    Landmark("HN_L", "Крыловидно-челюстная вырезка левая", "между бугром верхней челюсти и крыловидным отростком"),
    Landmark("IP", "Резцовый сосочек", "на нёбе за центральными резцами (над резцовым отверстием)"),
    Landmark("Tr_R", "Козелок правый", "мягкие ткани, центр козелка уха"),
    Landmark("Tr_L", "Козелок левый", "мягкие ткани, центр козелка уха"),
    Landmark("Al_R", "Крыло носа правое", "мягкие ткани, нижний край крыла носа"),
    Landmark("Al_L", "Крыло носа левое", "мягкие ткани, нижний край крыла носа"),
    Landmark("N", "Назион", "лобно-носовой шов, срединная точка"),
    Landmark("ANS", "Передняя носовая ость", "срединная точка"),
    Landmark("PNS", "Задняя носовая ость", "срединная точка"),
    Landmark("Ba", "Базион", "передний край большого затылочного отверстия"),
)
BY_KEY = {lm.key: lm for lm in LANDMARKS}

# Горизонтальные плоскости: по каким точкам строятся (пары левый/правый усредняются, если нужно три точки).
PLANES = {
    "frankfurt": {"name": "Франкфуртская горизонталь", "points": (("Po_R",), ("Po_L",), ("Or_R", "Or_L"))},
    "hip": {"name": "HIP (вырезки — резцовый сосочек)", "points": (("HN_R",), ("HN_L",), ("IP",))},
    "camper": {"name": "Кемпера (козелок — крыло носа)", "points": (("Tr_R",), ("Tr_L",), ("Al_R", "Al_L"))},
}
# Срединные точки для срединно-сагиттальной плоскости.
MIDLINE = ("N", "ANS", "PNS", "Ba", "IP")
CORONOID_BAND_MM = 15.0  # мыщелок и венечный отросток — в верхних 15 мм ветви
NOTCH_GAP_MM = 4.0  # между ними вырезка нижней челюсти: пустой промежуток вперёд-назад шире 4 мм


def _point(landmarks: dict, keys) -> np.ndarray | None:
    pts = [np.asarray(landmarks[k], float) for k in keys if k in landmarks]
    return np.mean(pts, axis=0) if len(pts) == len(keys) else None


def plane(landmarks: dict, kind: str):
    """Плоскость (точка, единичная нормаль вверх) или None, если не хватает точек."""
    pts = [_point(landmarks, keys) for keys in PLANES[kind]["points"]]
    if any(p is None for p in pts):
        return None
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if np.linalg.norm(n) < 1e-6:
        return None
    n /= np.linalg.norm(n)
    if n[2] < 0:  # к голове пациента (+z в LPS)
        n = -n
    return np.mean(pts, axis=0), n


def missing(landmarks: dict, kind: str) -> list[str]:
    need = {k for keys in PLANES[kind]["points"] for k in keys} | {"Co_R", "Co_L"}
    return [BY_KEY[k].name for k in sorted(need) if k not in landmarks]


def reference_frame(landmarks: dict, kind: str) -> np.ndarray:
    """Матрица 4×4 «мм пациента → система плоскости» (см. описание модуля)."""
    p = plane(landmarks, kind)
    if p is None or "Co_R" not in landmarks or "Co_L" not in landmarks:
        raise ValueError(f"для плоскости «{PLANES[kind]['name']}» не хватает точек: {', '.join(missing(landmarks, kind))}")
    _, z = p
    co_r, co_l = np.asarray(landmarks["Co_R"], float), np.asarray(landmarks["Co_L"], float)
    origin = (co_r + co_l) / 2
    # Направление «вправо»: по срединным точкам, если их ≥ 2 (нормаль срединной плоскости),
    # иначе — по шарнирной оси.
    right = co_r - co_l
    mid = [np.asarray(landmarks[k], float) for k in MIDLINE if k in landmarks]
    if len(mid) >= 2:
        d = mid[-1] - mid[0]
        normal = np.cross(d, z)
        if np.linalg.norm(normal) > 1e-6:
            right = normal if normal @ right > 0 else -normal
    x = right - (right @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # вперёд
    R = np.vstack([x, y, z])  # строки — оси системы в мм пациента
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    return T


def plane_angles(landmarks: dict) -> dict:
    """Углы между построенными плоскостями, градусы (для отчёта)."""
    planes = {k: plane(landmarks, k) for k in PLANES}
    out = {}
    keys = [k for k, v in planes.items() if v is not None]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            cos = abs(float(planes[a][1] @ planes[b][1]))
            out[f"{a}/{b}"] = round(float(np.degrees(np.arccos(min(cos, 1.0)))), 2)
    return out


# --- предложения по сегментации -------------------------------------------------

def suggest_condyles(mandible_vertices: np.ndarray, up=(0.0, 0.0, 1.0), left=(1.0, 0.0, 0.0),
                     anterior=None) -> dict:
    """Верхушки мыщелков: самые верхние точки нижней челюсти справа и слева от середины.

    up — вверх, left — к левой стороне пациента; по умолчанию оси DICOM (LPS).
    Для сцен, где оси повёрнуты как угодно (exocad), их задают по зубам.
    Если задано anterior (вперёд), венечный отросток отбрасывается: в верхних
    CORONOID_BAND_MM ветви точки делятся по вырезке нижней челюсти на два
    скопления, и мыщелок — заднее. Без этого при наклонённой вертикали
    (по окклюзионной плоскости) самой высокой бывает верхушка венечного отростка.
    """
    v = np.asarray(mandible_vertices, float)
    up = np.asarray(up, float) / np.linalg.norm(up)
    left = np.asarray(left, float) - (np.asarray(left, float) @ up) * up
    left /= np.linalg.norm(left)
    lat, h = v @ left, v @ up
    mid = np.median(lat)
    out = {}
    for key, side in (("Co_R", lat < mid - 15), ("Co_L", lat > mid + 15)):
        if side.sum() < 50:
            continue
        part, hp = v[side], h[side]
        if anterior is not None:
            band = hp >= hp.max() - CORONOID_BAND_MM
            a = part[band] @ np.asarray(anterior, float)
            order = np.sort(a)
            gaps = np.diff(order)
            if len(gaps) and gaps.max() > NOTCH_GAP_MM:  # вырезка: заднее скопление — мыщелок
                cut = order[np.argmax(gaps)]
                part, hp = part[band][a <= cut], hp[band][a <= cut]
        out[key] = part[hp >= hp.max() - 1.5].mean(axis=0)  # верхушка головки, 1.5 мм
    return out


def suggest_porion(canal_vertices: np.ndarray, key: str, mid_x: float) -> dict:
    """Порион: верхняя точка наружного (латерального) отдела слухового прохода.

    mid_x — x срединной плоскости черепа (например, медиана нижней челюсти).
    """
    v = np.asarray(canal_vertices, float)
    if len(v) < 20:
        return {}
    away = np.abs(v[:, 0] - mid_x)
    lateral = v[away >= np.median(away)]
    top = lateral[lateral[:, 2] >= lateral[:, 2].max() - 0.5]
    return {key: top.mean(axis=0)}
