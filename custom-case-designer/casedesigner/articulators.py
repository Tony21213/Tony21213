"""Виртуальные артикуляторы: в какой системе координат выгружать модели.

Артикулятор задаётся тем, по какой плоскости монтируются модели и где в
его системе стоит шарнирная ось:

  артикулятор ← сдвиг(hinge_mm) · поворот вокруг оси X на tilt_deg ← система плоскости

Система плоскости — landmarks.reference_frame: начало в середине шарнирной
оси, Z — нормаль плоскости вверх, X — вправо пациента, Y — вперёд.

Числа hinge_mm и tilt_deg для каждого артикулятора в exocad нужно
откалибровать один раз по образцу (модель, выгруженная из exocad в этом
артикуляторе, вместе с её КТ). Пока калибровки нет, у артикулятора
calibrated=False, и выгрузка идёт в систему его монтажной плоскости с
началом на шарнирной оси.
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .landmarks import PLANES, reference_frame


@dataclass
class Articulator:
    key: str
    name: str
    maker: str
    plane: str = "frankfurt"  # монтажная плоскость: frankfurt (оси-орбитальная), camper, hip
    hinge_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # середина шарнирной оси в системе артикулятора
    tilt_deg: float = 0.0  # наклон горизонтали артикулятора относительно монтажной плоскости
    calibrated: bool = False
    note: str = ""


_BUILTIN = [
    ("artex_cr", "Artex CR", "Amann Girrbach"), ("artex_cn", "Artex CN / CT", "Amann Girrbach"),
    ("reference_sl", "Reference SL", "Gamma Dental"), ("reference_sr", "Reference SR", "Gamma Dental"),
    ("sam_2p", "SAM 2P", "SAM Präzisionstechnik"), ("sam_3", "SAM 3", "SAM Präzisionstechnik"),
    ("protarevo", "PROTARevo 5/7/9", "KaVo"), ("stratos_300", "Stratos 300", "Ivoclar"),
    ("stratos_200", "Stratos 200", "Ivoclar"), ("denar_mark2", "Denar Mark II", "Whip Mix"),
    ("denar_mark330", "Denar Mark 330", "Whip Mix"), ("panadent", "PCH / PSH", "Panadent"),
    ("hanau_modular", "Modular / Wide-Vue", "Hanau"), ("bioart_a7", "A7 Plus", "Bio-Art"),
    ("ps1", "PS1 (PlaneSystem)", "Zirkonzahn"), ("dentatus_arl", "ARL", "Dentatus"),
]
BUILTIN = [Articulator(k, n, m, note="нужна калибровка по образцу из exocad") for k, n, m in _BUILTIN]


def load(path: str | None = None) -> list[Articulator]:
    """Встроенный список, дополненный и уточнённый файлом калибровок (JSON-список артикуляторов)."""
    items = {a.key: a for a in BUILTIN}
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for d in json.load(f):
                d["hinge_mm"] = tuple(d.get("hinge_mm", (0, 0, 0)))
                items[d["key"]] = Articulator(**d)
    return list(items.values())


def save(path: str, articulators: list[Articulator]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articulators], f, ensure_ascii=False, indent=2)


def articulator_frame(landmarks: dict, art: Articulator) -> np.ndarray:
    """Матрица 4×4 «мм пациента → система артикулятора»."""
    if art.plane not in PLANES:
        raise ValueError(f"{art.name}: неизвестная монтажная плоскость {art.plane!r}")
    a = np.radians(art.tilt_deg)
    tilt = np.eye(4)
    tilt[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    shift = np.eye(4)
    shift[:3, 3] = art.hinge_mm
    return shift @ tilt @ reference_frame(landmarks, art.plane)


def calibrate(art: Articulator, landmarks: dict, ct_to_articulator: np.ndarray) -> Articulator:
    """Калибровка по образцу: известно, где в артикуляторе exocad стоит КТ этого пациента.

    ct_to_articulator — матрица «мм пациента → артикулятор exocad», найденная
    совмещением модели, выгруженной из exocad, с той же моделью в КТ. Из неё и
    системы монтажной плоскости находятся положение шарнирной оси и наклон.
    """
    plane_frame = reference_frame(landmarks, art.plane)
    rel = np.asarray(ct_to_articulator, float) @ np.linalg.inv(plane_frame)  # плоскость → артикулятор
    R = rel[:3, :3]
    tilt = float(np.degrees(np.arctan2(R[2, 1], R[1, 1])))
    off_axis = np.degrees(np.arccos(np.clip(R[0, 0], -1, 1)))
    if off_axis > 2.0:
        raise ValueError(f"образец повёрнут вокруг вертикали или сагиттальной оси на {off_axis:.1f}° — "
                         "проверьте точки и выбор монтажной плоскости")
    return Articulator(art.key, art.name, art.maker, art.plane, tuple(np.round(rel[:3, 3], 3).tolist()),
                       round(tilt, 3), True, "откалиброван по образцу")
