"""Суставной путь в 3D по КТ (порядок работ, п. 16; запись в идеях 12.10 «настоящая головка мыщелка по ямке с
диском»).

Плоский расчёт (``guidance.condylar_path``) берёт одно сагиттальное сечение: головка — окружность под Co, Беннетт —
по формуле Ханау. Здесь — объёмно:

* головка — сетка мыщелкового отростка нижней челюсти (``landmarks.condylar_heads``); ямка — поверхность черепа
  вокруг мыщелка (крыша, задний скат бугорка, внутренняя стенка) точками с нормалями наружу из кости;
* диск — прослойка на кости: «постоянный» — толщиной щели в покое (как в плоском расчёте), «анатомический» — над
  скатом тоньше, промежуточная зона DISC_INTERMEDIATE_MM (допущение по анатомии: толщина диска на КТ не видна —
  отсюда два варианта и разница в градусах);
* протрузия — на каждом шаге вперёд головка прижимается вверх к поверхности «кость + диск» без проникновения: путь
  центра → ССП (по хорде 5 мм, как у записей) и путь для вставок (``exocad_facebow.fit_insert``);
* латеротрузия, балансирующая сторона — головка ещё и смещается внутрь, пока её не остановит внутренняя стенка ямки:
  немедленный боковой сдвиг и угол Беннетта — в той же форме, что у ``kinematics.laterotrusion_targets``
  (сдвиг × min(1, вперёд / ISS_MM) + вперёд × tg Беннетта). Стенки нет на КТ (тонкая, не сегментирована) —
  Беннетт не определяется (пусть будет формула Ханау).

Всё — в системе монтажа (``Anatomy.frame``: x — вправо, y — вперёд, z — вверх), мм; ответ — относительно верхушки
мыщелка (Co) на КТ.
"""

import numpy as np

from .register import apply

STEP_MM = 0.25
PATH_MM = 12.0  # протрузия: на столько мм вперёд
CHORD_MM = 5.0
LATERO_MM = 6.0  # латеротрузия: балансирующий мыщелок — на столько мм вперёд
ISS_MM = 1.0  # как kinematics.ISS_MM: немедленный сдвиг набирается на первом миллиметре
MEDIAL_MAX_MM = 4.0  # внутрь дальше — стенки нет (на КТ не видна)
BOX = {"x": 14.0, "y": (-10.0, 24.0), "z": (-16.0, 10.0)}  # ямка — в этой рамке вокруг Co (система монтажа)
HEAD_DEPTH_MM = 8.0  # головка — вершины отростка не ниже этого от верхушки
SAMPLE_MM = 0.3
NEAR_MM = 3.0  # дальше от поверхности — точка головки не касается
HEAD_REACH_MM = 6.0  # точки головки дальше от ямки в покое на пути не касаются — в расчёт не идут
HEAD_POINTS = 1500
DISC_INTERMEDIATE_MM = 1.0  # промежуточная зона диска — между головкой и скатом при выдвижении
DISC_BLEND_MM = (1.0, 4.0)  # от покоя к промежуточной зоне — на этом отрезке пути вперёд


class Joint:
    """Сустав одной стороны в системе монтажа, относительно Co: точки головки и поверхность ямки."""

    def __init__(self, head_points: np.ndarray, fossa_points: np.ndarray, fossa_normals: np.ndarray, medial: float):
        from scipy.spatial import cKDTree

        self.fossa, self.normals = np.asarray(fossa_points, float), np.asarray(fossa_normals, float)
        self.tree = cKDTree(self.fossa)
        head = np.asarray(head_points, float)
        near = self.tree.query(head, distance_upper_bound=HEAD_REACH_MM)[0]
        head = head[np.isfinite(near)]  # дальше от ямки — не коснётся на пути
        if len(head) > HEAD_POINTS:
            head = head[np.random.default_rng(0).choice(len(head), HEAD_POINTS, replace=False)]
        self.head = head
        self.medial = float(medial)  # направление «внутрь» по x: −1 у правого мыщелка, +1 у левого

    def clearance(self, shift: np.ndarray, disc: float = 0.0) -> float:
        """Наименьший зазор головки, сдвинутой на shift, до поверхности «кость + диск»; < 0 — проникновение. Диск
        лежит на крыше и скате (поверхность смотрит вниз), у внутренней стенки — нет: там щель настоящая (из неё —
        немедленный боковой сдвиг)."""
        p = self.head + shift
        d, j = self.tree.query(p, distance_upper_bound=NEAR_MM + disc)
        ok = np.isfinite(d)
        if not ok.any():
            return NEAR_MM
        n = self.normals[j[ok]]
        s = np.einsum("ij,ij->i", p[ok] - self.fossa[j[ok]], n) - disc * np.clip(-n[:, 2], 0.0, 1.0)
        return float(s.min())

    def advance(self, shift: np.ndarray, direction: np.ndarray, disc: float, limit: float = 3.0) -> np.ndarray:
        """Сдвинуть головку из shift вдоль direction, пока она не упрётся в «кость + диск» (не дальше limit): по каждой
        точке головки — на сколько она приближается к ближайшей поверхности за мм сдвига (нормаль), допустимый сдвиг —
        наименьший по точкам; несколько уточнений, проникновение — назад."""
        u = np.asarray(direction, float)
        shift = np.asarray(shift, float).copy()
        moved = 0.0
        for _ in range(6):
            p = self.head + shift
            d, j = self.tree.query(p, distance_upper_bound=NEAR_MM + disc)
            ok = np.isfinite(d)
            if not ok.any():
                step = limit - moved
            else:
                n = self.normals[j[ok]]
                s = np.einsum("ij,ij->i", p[ok] - self.fossa[j[ok]], n) - disc * np.clip(-n[:, 2], 0.0, 1.0)
                rate = n @ u  # зазор меняется на rate за мм сдвига вдоль u
                blocking = rate < -0.05
                step = float(np.min(s[blocking] / -rate[blocking])) if blocking.any() else limit - moved
                if s.min() < 0 and not blocking.any():  # проникла точка, которую этот сдвиг не вытащит
                    break
            step = min(step, limit - moved)
            shift = shift + step * u
            moved += step
            if abs(step) < 1e-3 or moved >= limit - 1e-6:
                break
        return shift

    def press_up(self, x: float, y: float, disc: float, z_from: float) -> float | None:
        """Наибольший z (головка прижата вверх) при сдвиге (x, y) без проникновения; поиск — от z_from."""
        start = np.array([x, y, z_from - 2.0])
        if self.clearance(start, disc) < 0:
            start = self.advance(start, np.array([0.0, 0.0, -1.0]), disc, limit=3.0)  # сначала вниз — из кости
            if self.clearance(start, disc) < -0.02:
                return None
        z = self.advance(start, np.array([0.0, 0.0, 1.0]), disc, limit=4.0)[2]
        return float(z) if self.clearance(np.array([x, y, z]), disc) >= -0.02 else None

def _disc(mode: str, rest: float, forward: float) -> float:
    if mode == "rest":
        return rest
    a, b = DISC_BLEND_MM
    w = float(np.clip((forward - a) / (b - a), 0.0, 1.0))
    thin = min(rest, DISC_INTERMEDIATE_MM)
    return (1 - w) * rest + w * thin


def joint(skull_vertices, skull_faces, head_vertices, frame: np.ndarray, condyle_top, side: str) -> Joint:
    """Сустав стороны side ("right" / "left"): head_vertices — вершины мыщелкового отростка (координаты КТ),
    condyle_top — Co (координаты КТ); всё переводится в систему монтажа относительно Co."""
    from . import gingiva as gg

    F = np.asarray(frame, float)
    c = apply(F, np.asarray(condyle_top, float)[None])[0]
    head = apply(F, np.asarray(head_vertices, float)) - c
    head = head[head[:, 2] >= -HEAD_DEPTH_MM]
    import trimesh

    skull = trimesh.Trimesh(apply(F, np.asarray(skull_vertices, float)) - c, np.asarray(skull_faces), process=False)
    lo = np.array([-BOX["x"], BOX["y"][0], BOX["z"][0]])
    hi = np.array([BOX["x"], BOX["y"][1], BOX["z"][1]])
    pts, normals, _ = gg._sample([skull], lo, hi, SAMPLE_MM)
    if pts is None or len(pts) < 200:
        raise ValueError("над мыщелком нет кости ямки — проверьте сегментацию черепа")
    keep = (np.abs(pts[:, 0]) <= BOX["x"]) & (pts[:, 1] >= BOX["y"][0]) & (pts[:, 1] <= BOX["y"][1]) \
        & (pts[:, 2] >= BOX["z"][0]) & (pts[:, 2] <= BOX["z"][1])
    pts, normals = pts[keep], normals[keep]
    # нормали — в сторону сустава: у поверхности над головкой — вниз (иначе сетка вывернута)
    roof = (np.abs(pts[:, 0]) < 4) & (np.abs(pts[:, 1]) < 3) & (pts[:, 2] > 0)
    if roof.any() and normals[roof, 2].mean() > 0:
        normals = -normals
    return Joint(head, pts, normals, -1.0 if side == "right" else 1.0)


def analyze(j: Joint, disc_mode: str = "rest") -> dict:
    """Путь головки при протрузии и при латеротрузии (балансирующая сторона): углы и пути."""
    rest = j.clearance(np.zeros(3))
    if rest < 0:
        rest = 0.0  # головка на КТ касается кости — диск нулевой (сегментация слиплась)
    out = {"rest_gap_mm": round(rest, 2), "disc": disc_mode, "notes": []}
    # протрузия
    path, z = [], 0.0
    for y in np.arange(0.0, PATH_MM + 1e-9, STEP_MM):
        zz = j.press_up(0.0, float(y), _disc(disc_mode, rest, float(y)), z)
        if zz is None:
            break
        z = zz
        path.append((float(y), z))
    path = np.array(path)
    if len(path) < int(CHORD_MM / STEP_MM) + 1:
        raise ValueError("головка не скользит по ямке — проверьте мыщелок и сегментацию сустава")
    path[:, 1] -= path[0, 1]
    d = np.hypot(path[:, 0] - path[0, 0], path[:, 1] - path[0, 1])
    k = int(np.argmax(d >= CHORD_MM)) if (d >= CHORD_MM).any() else len(d) - 1
    out["sagittal_deg"] = round(float(np.degrees(np.arctan2(path[0, 1] - path[k, 1], path[k, 0] - path[0, 0]))), 1)
    out["path"] = path
    # латеротрузия (балансирующая): вперёд по y, внутрь — пока пускает стенка (по очереди: внутрь, вверх)
    lat, z, m = [], 0.0, 0.0
    wall = False
    inward = np.array([j.medial, 0.0, 0.0])
    for y in np.arange(0.0, LATERO_MM + 1e-9, 0.5):
        disc = _disc(disc_mode, rest, float(y))
        zz = j.press_up(j.medial * m, float(y), disc, z)
        if zz is None:
            break
        z = zz
        for _ in range(3):
            moved = j.advance(np.array([j.medial * m, float(y), z]), inward, disc, limit=MEDIAL_MAX_MM - m)
            m_new = float(abs(moved[0]))
            if m_new < MEDIAL_MAX_MM - 1e-3:
                wall = True
            zz = j.press_up(j.medial * m_new, float(y), disc, z)
            if zz is None:
                break
            changed = abs(m_new - m) + abs(zz - z)
            m, z = m_new, zz
            if changed < 1e-3:
                break
        lat.append((float(y), m, z))
    lat = np.array(lat)
    out["balancing_path"] = lat
    if not wall or len(lat) < 4:
        out["bennett_deg"] = out["side_shift_mm"] = None
        out["notes"].append("внутренней стенки ямки на КТ нет (тонкая или не сегментирована) — угол Беннетта не "
                            "определяется")
        return out
    sel = lat[:, 0] >= ISS_MM
    y, m = lat[sel, 0], lat[sel, 1]
    A = np.c_[np.ones_like(y), y]
    (a, b), *_ = np.linalg.lstsq(A, m, rcond=None)
    out["side_shift_mm"] = round(float(max(a, 0.0)), 2)
    out["bennett_deg"] = round(float(np.degrees(np.arctan(max(b, 0.0)))), 1)
    return out


def analyze_both(j: Joint) -> dict:
    """Оба варианта диска: постоянный (основной) и анатомический — с разницей в градусах."""
    base = analyze(j, "rest")
    try:
        anat = analyze(j, "anatomic")
        base["anatomic"] = {k: anat.get(k) for k in ("sagittal_deg", "bennett_deg", "side_shift_mm")}
    except ValueError as e:
        base["anatomic"] = {"error": str(e)}
    return base
