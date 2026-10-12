"""Плоскость монтажа: горизонталь, по которой модели ставятся в артикулятор и от которой считаются углы суставов.

Варианты (PLANES):

* **окклюзионная** — по сканам в прикусе (как было до выбора плоскости; ориентиры не нужны);
* **франкфуртская** — порионы и орбиталь (Po_R, Po_L, Or): классическая горизонталь черепа;
* **оси-орбитальная** — шарнирная ось через оба мыщелка и орбиталь (Co_R, Co_L, Or): как у кинематической лицевой
  дуги. Ось «вправо» — линия мыщелков, поэтому шарнир артикулятора проходит через оба мыщелка даже при их разной
  высоте; модели в артикуляторе при этом наклоняются вместе с осью пациента;
* **Кемпера** — козелки и крылья носа (Tr_R, Tr_L, Al): мягкие ткани, примерно параллельна окклюзионной;
* **HIP** — крыловидно-челюстные вырезки и резцовый сосочек (HN_R, HN_L, IP).

Ориентиры — landmarks, мм КТ. Порионы, орбитали, мыщелки ALI-CBCT находит сам, точки Кемпера и HIP ставит врач.
Точки черепа неподвижны вместе с верхней челюстью, поэтому годятся и в координатах кейса; мыщелки — в прикусе монтажа
(как шарнир, condyles в exocad_webview.mount).

Система монтажа (строки R — «вправо, вперёд, вверх» в координатах кейса): «вверх» — нормаль плоскости в сторону
окклюзионного «вверх»; «вперёд» — направление зубной дуги, спроецированное в плоскость (рыскание моделей как у
окклюзионной); у оси-орбитальной «вправо» — линия Co_L → Co_R. Углы ССП по бугорку считаются в этой же системе, то
есть от выбранной плоскости, — так их понимает и артикулятор exocad (его горизонталь — плоскость монтажа).
"""

import numpy as np

from . import landmarks as lmk

DEFAULT = "occlusal"
PLANES = {
    "occlusal": {"name": "Окклюзионная (по сканам)", "points": ()},
    "frankfurt": {"name": "Франкфуртская (порион — орбиталь)", "points": (("Po_R",), ("Po_L",), ("Or_R", "Or_L"))},
    "axis_orbital": {"name": "Оси-орбитальная (мыщелки — орбиталь)", "points": (("Co_R",), ("Co_L",), ("Or_R", "Or_L"))},
    "camper": {"name": "Кемпера (козелок — крыло носа)", "points": (("Tr_R",), ("Tr_L",), ("Al_R", "Al_L"))},
    "hip": {"name": "HIP (вырезки — резцовый сосочек)", "points": (("HN_R",), ("HN_L",), ("IP",))},
}


def _check(kind: str):
    if kind not in PLANES:
        raise ValueError(f"нет такой плоскости монтажа: {kind}; есть: {', '.join(PLANES)}")


def _points(kind: str, landmarks: dict, condyles: dict | None) -> list | None:
    """Три точки плоскости (пара слева/справа усредняется; для орбитали хватает одной стороны) или None."""
    have = dict(landmarks or {})
    if condyles is not None:  # шарнир — в прикусе монтажа
        have.update({"Co_R": condyles["right"], "Co_L": condyles["left"]})
    out = []
    for keys in PLANES[kind]["points"]:
        pts = [np.asarray(have[k], float) for k in keys if k in have]
        if not pts or (len(keys) == 2 and keys[0][:2] != "Or" and len(pts) < 2):
            return None
        out.append(np.mean(pts, axis=0))
    return out


def missing(kind: str, landmarks: dict | None, condyles: dict | None = None) -> list[str]:
    """Названия недостающих ориентиров плоскости (для орбитали — если нет ни одной стороны)."""
    _check(kind)
    have = set(landmarks or {}) | ({"Co_R", "Co_L"} if condyles is not None else set())
    out = []
    for keys in PLANES[kind]["points"]:
        if keys[0].startswith("Or"):
            if not any(k in have for k in keys):
                out.append(" или ".join(lmk.BY_KEY[k].name for k in keys))
            continue
        out += [lmk.BY_KEY[k].name for k in keys if k not in have]
    return out


def available(landmarks: dict | None, condyles: dict | None = None) -> dict:
    """Все плоскости: название, можно ли построить, чего не хватает."""
    return {k: {"name": p["name"], "available": not missing(k, landmarks, condyles),
                "missing": missing(k, landmarks, condyles)} for k, p in PLANES.items()}


def rotation(kind: str, up, anterior, landmarks: dict | None = None, condyles: dict | None = None) -> np.ndarray:
    """Оси монтажа (строки: вправо, вперёд, вверх) в координатах кейса для плоскости kind.

    up, anterior — окклюзионные «вверх» и «вперёд» по сканам (mo.occlusal_up, mo.arch_axes); condyles — шарнирные
    точки {"right", "left"} в координатах кейса (в прикусе монтажа)."""
    _check(kind)
    up, anterior = np.asarray(up, float), np.asarray(anterior, float)
    up /= np.linalg.norm(up)
    if kind == "occlusal":
        anterior = anterior - (anterior @ up) * up
        anterior /= np.linalg.norm(anterior)
        return np.array([np.cross(anterior, up), anterior, up])
    pts = _points(kind, landmarks, condyles)
    if pts is None:
        raise ValueError(f"для плоскости «{PLANES[kind]['name']}» не хватает точек: "
                         f"{', '.join(missing(kind, landmarks, condyles))}")
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if np.linalg.norm(n) < 1e-6:
        raise ValueError(f"точки плоскости «{PLANES[kind]['name']}» лежат на одной прямой")
    n /= np.linalg.norm(n)
    if n @ up < 0:
        n = -n
    if kind == "axis_orbital":
        right = np.asarray(condyles["right"], float) - np.asarray(condyles["left"], float)
        right -= (right @ n) * n  # лежит в плоскости и так; убираем погрешность
        right /= np.linalg.norm(right)
        forward = np.cross(n, right)
        if forward @ anterior < 0:
            raise ValueError("мыщелки перепутаны сторонами — проверьте точки Co_R и Co_L")
        return np.array([right, forward, n])
    forward = anterior - (anterior @ n) * n
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError(f"плоскость «{PLANES[kind]['name']}» почти вертикальна — проверьте точки")
    forward /= np.linalg.norm(forward)
    return np.array([np.cross(forward, n), forward, n])


def relative_angles(R: np.ndarray, up, anterior) -> dict:
    """Наклон плоскости монтажа к окклюзионной, градусы: pitch — передом вверх (+), roll — вправо вниз (+)."""
    occl = rotation("occlusal", up, anterior)
    local = occl @ np.asarray(R, float)[2]  # нормаль выбранной плоскости в окклюзионной системе
    return {"pitch_deg": round(float(np.degrees(np.arctan2(-local[1], local[2]))), 1),
            "roll_deg": round(float(np.degrees(np.arctan2(local[0], local[2]))), 1)}
