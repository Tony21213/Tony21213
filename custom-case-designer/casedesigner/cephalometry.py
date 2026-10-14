"""Цефалометрия по КТ («Эстетика», часть 2; порядок работ владельца, п. 10): анализ — это данные (точки, измерения,
нормы), а не код, чтобы добавлять новые без переделки (решение владельца 13.10: «наверное, все»; первыми — Steiner
и Ricketts).

Контракт:

* Точки — ориентиры кейса в мм КТ (``landmarks``: ручные или ALI-CBCT): кроме точек плоскостей — CEPH_POINTS (A, B,
  Pog, Gn, Me, Go, резцы; сети ALI-CBCT — выпуск ``landmarks-v2``, или постановка на срезах). Двусторонние точки
  (Go_R/Go_L, Po_R/Po_L, Or_R/Or_L) — средним: нормы сняты на боковых ТРГ.
* Всё считается в проекции на срединную плоскость (``midsagittal``: нормаль — по парам двусторонних точек, иначе по
  срединным точкам), как на боковой ТРГ; 3D и асимметрия — позже.
* Измерение (``Measure``) — тип и точки: ``angle`` (угол в вершине: точки a, вершина, b), ``vectors`` (угол между
  направленными отрезками a→b и c→d), ``lines`` (острый угол между прямыми ab и cd), ``distance`` (расстояние от
  точки до прямой, мм: плюс — точка впереди прямой, то есть по другую сторону от S, а без неё — от Ba, PNS или Po), ``difference`` (разность двух
  измерений). Норма — среднее и отклонение; итог — значение, отклонение в сигмах и вывод.
* Нет точки — измерение пропускается с причиной («нет точек: A, B»).
"""

from dataclasses import dataclass, field

import numpy as np

# Точки цефалометрии сверх ориентиров плоскостей — в landmarks.LANDMARKS (названия и подсказки), ALI-CBCT — в выпуске
# landmarks-v2 (auto_landmarks.POINTS); резцы — правые центральные.
CEPH_POINTS = ("A", "B", "Pog", "Gn", "Me", "Go_R", "Go_L", "U1_tip", "U1_apex", "L1_tip", "L1_apex")
BILATERAL = ("Go", "Po", "Or", "Co")
MIDLINE = ("N", "S", "ANS", "PNS", "Ba", "A", "B", "Pog", "Me")


@dataclass
class Measure:
    key: str
    name: str
    kind: str  # angle / vectors / lines / distance / difference
    points: tuple  # для difference — ключи двух измерений
    norm: tuple  # (среднее, отклонение)
    unit: str = "°"
    high: str = "больше нормы"
    low: str = "меньше нормы"


@dataclass
class Analysis:
    key: str
    name: str
    measures: list
    note: str = ""


ANALYSES = {
    "steiner": Analysis("steiner", "Steiner", [
        Measure("SNA", "SNA", "angle", ("S", "N", "A"), (82.0, 2.0), high="верхняя челюсть впереди",
                low="верхняя челюсть позади"),
        Measure("SNB", "SNB", "angle", ("S", "N", "B"), (80.0, 2.0), high="нижняя челюсть впереди",
                low="нижняя челюсть позади"),
        Measure("ANB", "ANB", "difference", ("SNA", "SNB"), (2.0, 2.0), high="скелетный II класс",
                low="скелетный III класс"),
        Measure("SN_GoGn", "SN — GoGn (плоскость нижней челюсти)", "lines", ("S", "N", "Go", "Gn"), (32.0, 5.0),
                high="вертикальный тип", low="горизонтальный тип"),
        Measure("U1_NA_deg", "Верхний резец к NA", "vectors", ("U1_apex", "U1_tip", "N", "A"), (22.0, 5.0),
                high="протрузия верхнего резца", low="ретрузия верхнего резца"),
        Measure("U1_NA_mm", "Верхний резец до NA", "distance", ("U1_tip", "N", "A"), (4.0, 2.0), unit=" мм"),
        Measure("L1_NB_deg", "Нижний резец к NB", "lines", ("L1_apex", "L1_tip", "N", "B"), (25.0, 5.0),
                high="протрузия нижнего резца", low="ретрузия нижнего резца"),
        Measure("L1_NB_mm", "Нижний резец до NB", "distance", ("L1_tip", "N", "B"), (4.0, 2.0), unit=" мм"),
        Measure("Pog_NB_mm", "Погонион до NB", "distance", ("Pog", "N", "B"), (1.0, 2.0), unit=" мм"),
        Measure("interincisal", "Межрезцовый угол", "vectors", ("U1_tip", "U1_apex", "L1_tip", "L1_apex"),
                (131.0, 6.0), high="резцы ретрузированы", low="резцы протрузированы"),
    ]),
    "ricketts": Analysis("ricketts", "Ricketts (основные)", [
        Measure("facial_angle", "Лицевой угол (FH — NPog)", "vectors", ("Or", "Po", "N", "Pog"), (87.0, 3.0),
                high="подбородок впереди", low="подбородок позади"),
        Measure("mandibular_plane", "Угол плоскости нижней челюсти (FH — GoMe)", "lines", ("Po", "Or", "Go", "Me"),
                (26.0, 4.5), high="вертикальный тип", low="горизонтальный тип"),
        Measure("convexity", "Выпуклость (A до NPog)", "distance", ("A", "N", "Pog"), (2.0, 2.0), unit=" мм",
                high="выпуклый профиль (II класс)", low="вогнутый профиль (III класс)"),
        Measure("L1_APog_mm", "Нижний резец до APog", "distance", ("L1_tip", "A", "Pog"), (1.0, 2.0), unit=" мм",
                high="нижний резец впереди", low="нижний резец позади"),
        Measure("L1_APog_deg", "Нижний резец к APog", "lines", ("L1_apex", "L1_tip", "A", "Pog"), (22.0, 4.0),
                high="протрузия нижнего резца", low="ретрузия нижнего резца"),
    ], note="нормы Ricketts — для 9 лет: с возрастом лицевой угол растёт, а угол плоскости нижней челюсти и "
            "выпуклость уменьшаются"),
}


def _points(landmarks: dict) -> dict:
    """Точки кейса + средние двусторонних (Go = (Go_R + Go_L) / 2; если есть только одна — она)."""
    pts = {k: np.asarray(v, float) for k, v in (landmarks or {}).items() if v is not None}
    for base in BILATERAL:
        sides = [pts[f"{base}_{s}"] for s in ("R", "L") if f"{base}_{s}" in pts]
        if sides and base not in pts:
            pts[base] = np.mean(sides, axis=0)
    return pts


def midsagittal(landmarks: dict):
    """Срединная плоскость (начало, нормаль): нормаль — по парам двусторонних точек (правая минус левая), иначе — по
    срединным точкам (наименьшая ось разброса); начало — средняя срединных точек (или середины пар)."""
    pts = {k: np.asarray(v, float) for k, v in (landmarks or {}).items() if v is not None}
    pairs = [(pts[f"{b}_R"], pts[f"{b}_L"]) for b in BILATERAL if f"{b}_R" in pts and f"{b}_L" in pts]
    mid = [pts[k] for k in MIDLINE if k in pts]
    if pairs:
        normal = np.mean([(r - l) / np.linalg.norm(r - l) for r, l in pairs], axis=0)
        origin = np.mean(mid, axis=0) if mid else np.mean([(r + l) / 2 for r, l in pairs], axis=0)
    elif len(mid) >= 3:
        origin = np.mean(mid, axis=0)
        normal = np.linalg.svd(np.asarray(mid) - origin, full_matrices=False)[2][2]
    else:
        raise ValueError("срединная плоскость: нужны пары двусторонних точек или 3 срединные точки")
    return origin, normal / np.linalg.norm(normal)


def _project(p, plane):
    origin, n = plane
    return p - ((p - origin) @ n) * n


def _angle(u, v) -> float:
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))))


def _measure(m: Measure, P: dict, done: dict):
    if m.kind == "difference":
        a, b = (done.get(k) for k in m.points)
        return None if a is None or b is None else a - b
    p = [P[k] for k in m.points]
    if m.kind == "angle":
        return _angle(p[0] - p[1], p[2] - p[1])
    if m.kind == "vectors":
        return _angle(p[1] - p[0], p[3] - p[2])
    if m.kind == "lines":
        a = _angle(p[1] - p[0], p[3] - p[2])
        return min(a, 180.0 - a)
    if m.kind == "distance":
        q, a, b = p
        d = (b - a) / np.linalg.norm(b - a)
        off = (q - a) - ((q - a) @ d) * d
        dist = float(np.linalg.norm(off))
        ref = next((P[k] for k in ("S", "Ba", "PNS", "Po") if k in P), None)  # задняя точка — для знака
        if ref is not None:
            back = (ref - a) - ((ref - a) @ d) * d
            if off @ back > 0:  # с той же стороны, что S, — позади прямой
                dist = -dist
        return dist
    raise ValueError(f"неизвестный тип измерения: {m.kind}")


def analyze(landmarks: dict, analysis: str = "steiner") -> dict:
    """Анализ по точкам кейса: {"name", "note", "results": [{key, name, value, unit, norm, sd, z, text}],
    "skipped": [{key, name, missing}]}; всё — в проекции на срединную плоскость."""
    a = ANALYSES[analysis]
    raw = _points(landmarks)
    plane = midsagittal(landmarks)
    P = {k: _project(v, plane) for k, v in raw.items()}
    done, results, skipped = {}, [], []
    for m in a.measures:
        need = [k for k in (m.points if m.kind != "difference" else ()) if k not in P]
        if m.kind == "difference":
            need = [k for k in m.points if k not in done]
        if need:
            skipped.append({"key": m.key, "name": m.name, "missing": need})
            continue
        value = _measure(m, P, done)
        done[m.key] = value
        mean, sd = m.norm
        z = (value - mean) / sd
        text = "в норме" if abs(z) <= 1 else m.high if z > 0 else m.low
        results.append({"key": m.key, "name": m.name, "value": round(value, 1), "unit": m.unit, "norm": mean,
                        "sd": sd, "z": round(z, 2), "text": text})
    return {"name": a.name, "note": a.note, "results": results, "skipped": skipped}


def report(analyses: list[dict], case_label: str):
    """Отчёт цефалометрии на общем генераторе (reports): раздел на анализ — значение, норма, вывод; пропущенные
    измерения — заметками («нет точек: …»). analyses — результаты analyze."""
    from . import reports

    def num(v):
        return f"{v:.1f}".replace(".", ",")

    sections = []
    for res in analyses:
        rows = [(r["name"], f"{num(r['value'])}{r['unit']} (норма {num(r['norm'])} ± {num(r['sd'])}) — {r['text']}")
                for r in res["results"]]
        notes = ([res["note"]] if res.get("note") else []) + [
            f"{s['name']}: нет точек {', '.join(s['missing'])}" for s in res["skipped"]]
        level = "warn" if any(abs(r["z"]) > 2 for r in res["results"]) else "info"
        sections.append(reports.Section(res["name"], rows, notes, level))
    return reports.Report("Цефалометрия", case_label, sections)
