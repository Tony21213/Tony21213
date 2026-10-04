"""Настройки артикулятора по реальной записи и то, что артикулятор воспроизвести не может.

Артикулятор ведёт мыщелки по прямым: сагиттальный угол, угол Беннетта и
немедленный боковой сдвиг. Реальный путь мыщелка изогнут по скату бугорка,
поэтому даже лучшие настройки оставляют расхождение. Здесь:

1. углы подбираются по всему начальному участку пути (FIT_MM), а не по хорде:
   прямая через начало пути, ближайшая ко всем записанным точкам (МНК);
2. артикулятор с этими настройками повторяет каждое движение на тот же путь
   мыщелков, и сравниваются пути мыщелков и резцовой точки;
3. итог: насколько артикулятор отличается от пациента, мм. Если меньше
   допуска — артикулятора с этими настройками достаточно; если больше — для
   этой работы нужна сама запись (FGP и контакты по записанным движениям).
"""

import numpy as np

from . import kinematics as kin
from . import motion as mo

FIT_MM = 5.0  # углы — по этому начальному участку пути мыщелка
ENOUGH_MM = 0.3  # расхождение резцовой точки меньше — артикулятор воспроизводит движение


def _line_angle(d: np.ndarray) -> np.ndarray:
    """Направление прямой через начало координат, ближайшей к точкам d (МНК), — вперёд."""
    u = np.linalg.eigh(d.T @ d)[1][:, -1]
    return u if u[1] >= 0 else -u


def _section(c: np.ndarray, length: float = FIT_MM) -> np.ndarray:
    d = c - c[0]
    dist = np.linalg.norm(d, axis=1)
    end = int(np.argmax(dist >= length)) if (dist >= length).any() else len(d) - 1
    return d[: end + 1]


def _fit_protrusion(c: np.ndarray) -> tuple[float, float]:
    """Сагиттальный угол по участку пути и среднее отклонение точек пути от прямой, мм."""
    d = _section(c)
    u = _line_angle(d)
    off = d - np.outer(d @ u, u)
    return float(np.degrees(np.arctan2(-u[2], u[1]))), float(np.sqrt((off ** 2).sum(1).mean()))


def _fit_balancing(c: np.ndarray, sign: float) -> dict:
    """Балансирующий мыщелок: прямая после немедленного сдвига — сагиттальный угол и угол Беннетта."""
    d = _section(c)
    forward = d[:, 1]
    late = d[forward >= mo.ISS_FIT_MM]
    if len(late) < 3:
        late = d
    centre = late.mean(0)
    u = np.linalg.svd(late - centre, full_matrices=False)[2][0]
    u = u if u[1] >= 0 else -u
    medial = -sign * u[0]  # внутрь — к середине
    at_zero = centre - u * (centre[1] / u[1]) if abs(u[1]) > 1e-6 else centre  # где прямая пересекает y = 0
    off = (late - centre) - np.outer((late - centre) @ u, u)
    return {"sagittal_deg": float(np.degrees(np.arctan2(-u[2], u[1]))),
            "bennett_deg": float(np.degrees(np.arctan2(medial, u[1]))),
            "side_shift_mm": float(max(0.0, -sign * at_zero[0])),
            "rms_mm": float(np.sqrt((off ** 2).sum(1).mean()))}


def _deviation(recorded: np.ndarray, model: np.ndarray) -> float:
    """Наибольшее расстояние от записанного пути до пути артикулятора (ломаной), мм."""
    a, b = model[:-1], model[1:]
    ab = b - a
    t = np.clip(np.einsum("mij,ij->mi", recorded[:, None] - a[None], ab) / np.maximum((ab ** 2).sum(1), 1e-12),
                0, 1)
    nearest = a[None] + t[..., None] * ab[None]
    return float(np.linalg.norm(recorded[:, None] - nearest, axis=-1).min(1).max())


def fit(case: mo.MotionCase, anatomy: mo.Anatomy, base: kin.Settings | None = None,
        occlusion: kin.Occlusion | None = None) -> tuple[kin.Settings, dict]:
    """Настройки артикулятора по записям кейса и отчёт: расхождение артикулятора с записью.

    Углы — в системе anatomy (Settings.frame = anatomy.frame). occlusion —
    если записи сверены со сканами, артикулятор ведёт передние зубы по сканам,
    как и пациент; иначе — только по суставам.
    """
    ref = mo.reference_pose(case)
    s = (base or kin.Settings()).copy()
    s.frame = anatomy.frame.copy()
    recs = [(r, mo.analyze_recording(r, anatomy, ref)) for r in case.recordings]
    fitted, rms = {}, {}
    for side in kin.SIDES:
        pro = [_fit_protrusion(mo.paths(r, anatomy, ref)[f"condyle_{side}"]) for r, a in recs
               if a["kind"] == "protrusion"]
        if pro:
            fitted[f"sagittal_{side}_deg"] = float(np.mean([p[0] for p in pro]))
            rms[f"sagittal_{side}"] = float(np.max([p[1] for p in pro]))
        other = "left" if side == "right" else "right"
        bal = [_fit_balancing(mo.paths(r, anatomy, ref)[f"condyle_{side}"], 1.0 if side == "right" else -1.0)
               for r, a in recs if a["kind"] == f"laterotrusion_{other}"]
        if bal:
            fitted[f"bennett_{side}_deg"] = float(np.mean([b["bennett_deg"] for b in bal]))
            fitted[f"side_shift_{other}_mm"] = float(np.mean([b["side_shift_mm"] for b in bal]))
            rms[f"bennett_{side}"] = float(np.max([b["rms_mm"] for b in bal]))
    for key, value in fitted.items():
        setattr(s, key, round(value, 1 if key.endswith("deg") else 2))
        s.sources[key] = "подбор по записи"
    movements = []
    for rec, a in recs:
        kind = a["kind"]
        P = mo.paths(rec, anatomy, ref)
        if kind == "protrusion":
            travel = float(np.mean([np.linalg.norm(P[f"condyle_{x}"][-1] - P[f"condyle_{x}"][0]) for x in kin.SIDES]))
            model = kin.protrusion(anatomy, s, travel, occlusion)
        elif kind.startswith("laterotrusion"):
            working = kind.split("_")[1]
            balancing = "left" if working == "right" else "right"
            c = P[f"condyle_{balancing}"]
            travel = float(max(c[-1, 1] - c[0, 1], 0.1))
            model = kin.laterotrusion(anatomy, s, working, travel, occlusion)
        else:
            continue
        M = mo.paths(model, anatomy)
        dev = {k: round(_deviation(P[k] - P[k][0], M[k] - M[k][0]), 2) for k in mo.POINTS}
        movements.append({"name": rec.name, "kind": kind, "kind_name": a["kind_name"], "deviation_mm": dev})
    worst = max((m["deviation_mm"]["incisal"] for m in movements), default=None)
    notes = []
    if worst is None:
        notes.append("нет протрузии и боковых движений — подбирать нечего")
    elif worst <= ENOUGH_MM:
        notes.append(f"артикулятор с подобранными настройками повторяет движения пациента: резцовая точка "
                     f"расходится не больше чем на {worst:.2f} мм")
    else:
        m = max(movements, key=lambda m: m["deviation_mm"]["incisal"])
        notes.append(f"артикулятор не повторяет движения пациента: резцовая точка расходится до {worst:.2f} мм "
                     f"({m['kind_name']}) — для этой работы лучше сама запись (FGP и контакты по записанным "
                     "движениям)")
    return s, {"settings": {k: v for k, v in vars(s).items() if k not in ("sources", "frame")},
               "sources": dict(s.sources), "path_rms_mm": {k: round(v, 3) for k, v in rms.items()},
               "movements": movements, "worst_incisal_mm": None if worst is None else round(worst, 2),
               "enough": worst is not None and worst <= ENOUGH_MM, "notes": notes}
