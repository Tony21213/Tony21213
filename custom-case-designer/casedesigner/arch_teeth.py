"""Номера зубов по положению на дуге — контакты «13/43» без сегментации зубов.

Точка скана относится к зубу по расстоянию вдоль зубной дуги от средней
линии. Дуга — парабола по жевательной части скана в окклюзионной плоскости
(анатомическая система: x — вправо, y — вперёд). Центры зубов вдоль дуги —
по медианам расстояний между центрами соседних зубов в ~900 библиотеках
формы зубов exocad DentalCAD 3.3 (статистика посчитана в DentalAI,
crownai/data/arch_prior.json). Зубы у пациентов разного размера, поэтому
масштаб уточняется по зубам с известными номерами (моделировки в сцене);
без них номер приблизительный — к молярам ошибка может доходить до половины зуба.
"""

import numpy as np

# Расстояния между центрами соседних зубов, мм (медианы по библиотекам exocad).
PAIRS = {
    "upper": {"right": [("11", "21", 8.64), ("12", "11", 7.46), ("13", "12", 7.10), ("14", "13", 7.57),
                        ("15", "14", 7.10), ("16", "15", 9.04), ("17", "16", 10.44), ("18", "17", 10.00)],
              "left": [("21", "11", 8.64), ("22", "21", 7.47), ("23", "22", 7.10), ("24", "23", 7.66),
                       ("25", "24", 7.16), ("26", "25", 9.00), ("27", "26", 10.32), ("28", "27", 9.97)]},
    "lower": {"right": [("41", "31", 5.58), ("42", "41", 5.55), ("43", "42", 6.08), ("44", "43", 7.25),
                        ("45", "44", 7.35), ("46", "45", 9.61), ("47", "46", 11.31), ("48", "47", 11.22)],
              "left": [("31", "41", 5.58), ("32", "31", 5.55), ("33", "32", 6.03), ("34", "33", 7.25),
                       ("35", "34", 7.37), ("36", "35", 9.58), ("37", "36", 11.37), ("38", "37", 11.15)]},
}


def centres(jaw: str) -> dict:
    """Положения центров зубов вдоль дуги от средней линии, мм: вправо — плюс, влево — минус."""
    out = {}
    for side, sign in (("right", 1.0), ("left", -1.0)):
        s = 0.0
        for k, (tooth, _prev, gap) in enumerate(PAIRS[jaw][side]):
            s += gap / 2 if k == 0 else gap  # центральный резец — половина промежутка между центральными
            out[int(tooth)] = sign * s
    return out


class ArchLine:
    """Зубная дуга челюсти и номера зубов на ней."""

    def __init__(self, points: np.ndarray, jaw: str, known: dict | None = None):
        """points — жевательная часть скана (анатомическая система); known — {номер FDI: точка} для масштаба."""
        if jaw not in PAIRS:
            raise ValueError("jaw: upper или lower")
        self.jaw = jaw
        x, y = points[:, 0], points[:, 1]
        self.coef = np.polyfit(x, y, 2)  # y = c x² + b x + a, c < 0: вершина — резцы
        c, b, _a = self.coef
        self.vertex = -b / (2 * c) if c else float(np.mean(x))
        self.grid = np.arange(x.min() - 15, x.max() + 15, 0.05)
        gy = np.polyval(self.coef, self.grid)
        step = np.hypot(np.diff(self.grid), np.diff(gy))
        length = np.r_[0.0, np.cumsum(step)]
        self.arc = length - np.interp(self.vertex, self.grid, length)  # от средней линии; вправо (x > 0) — плюс
        self.curve = np.c_[self.grid, gy]
        self.scale = 1.0
        if known:
            prior = centres(jaw)
            ratios = [self.position(np.asarray(p, float)[None])[0] / prior[t]
                      for t, p in known.items() if t in prior and abs(prior[t]) > 1]
            ratios = [r for r in ratios if 0.7 < r < 1.4]
            if ratios:
                self.scale = float(np.median(ratios))
        table = centres(jaw)
        self.numbers = np.array(list(table))
        self.positions = np.array([table[t] for t in self.numbers]) * self.scale

    def position(self, points: np.ndarray) -> np.ndarray:
        """Расстояние вдоль дуги от средней линии до проекции точек, мм (вправо — плюс)."""
        d = np.linalg.norm(points[:, None, :2] - self.curve[None], axis=2) if len(points) < 2000 else None
        if d is not None:
            k = d.argmin(1)
        else:
            from scipy.spatial import cKDTree
            k = cKDTree(self.curve).query(points[:, :2])[1]
        return self.arc[k]

    def teeth(self, points: np.ndarray) -> np.ndarray:
        """Номер FDI ближайшего по дуге зуба для каждой точки."""
        s = self.position(points)
        same_side = np.sign(self.positions)[None] == np.sign(s)[:, None]
        dist = np.abs(s[:, None] - self.positions[None]) + np.where(same_side, 0.0, 1e6)
        return self.numbers[dist.argmin(1)]


def contacts_by_tooth(rec, anatomy, occlusion, upper: ArchLine, lower: ArchLine, every: int = 4,
                      gap: float = 0.1) -> dict:
    """Какие пары зубов касаются и на какой части пути: {"13/43": [доля пути от, до]}.

    Номер нижнего зуба — по исходному положению точки (зуб едет вместе с
    челюстью), верхнего — по месту касания на верхней дуге.
    """
    from .register import apply

    F = anatomy.frame
    back = np.linalg.inv(F)
    lower_ids = lower.teeth(occlusion.lower)
    n = len(rec.transforms)
    out = {}
    for k in range(0, n, every):
        p = apply(F @ rec.transforms[k] @ back, occlusion.lower)
        d, i = occlusion.tree.query(p, distance_upper_bound=gap)
        hit = np.isfinite(d)
        if not hit.any():
            continue
        upper_ids = upper.teeth(occlusion.points[i[hit]])
        t = round(k / max(n - 1, 1), 2)
        for pair in {f"{a}/{b}" for a, b in zip(upper_ids.tolist(), lower_ids[hit].tolist())}:
            lo, hi = out.get(pair, (t, t))
            out[pair] = [min(lo, t), max(hi, t)]
    return dict(sorted(out.items()))
