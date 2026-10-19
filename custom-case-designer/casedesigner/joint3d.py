"""Суставной путь в 3D по КТ (порядок работ, п. 16; запись в идеях 12.10 «настоящая головка мыщелка по ямке с
диском»).

Плоский расчёт (``guidance.condylar_path``) берёт одно сагиттальное сечение: головка — окружность под Co, Беннетт —
по формуле Ханау. Здесь — объёмно:

* головка — сетка мыщелкового отростка нижней челюсти (``landmarks.condylar_heads``); ямка — поверхность черепа
  вокруг мыщелка (крыша, скаты, стенки) точками с нормалями из кости к суставу;
* диск — прослойка на крыше и скатах: «постоянный» — толщиной щели над головкой в покое, «анатомический» — над
  скатом тоньше, промежуточная зона DISC_INTERMEDIATE_MM (допущение по анатомии: толщина диска на КТ не видна —
  отсюда два варианта и разница в градусах);
* протрузия — на каждом шаге вперёд головка прижимается вверх к поверхности «кость + диск» без проникновения: путь
  → ССП (по хорде 5 мм, как у записей) и путь для вставок (``exocad_facebow.fit_insert``); в монтаж по КТ — ССП
  отсюда, Беннетт — по Ханау от него (``Session._apply_joint3d``);
* латеротрузия, балансирующая сторона — головка ещё и съезжает внутрь, пока её не остановит стенка ямки:
  немедленный боковой сдвиг и угол Беннетта — в той же форме, что у ``kinematics.laterotrusion_targets``
  (сдвиг × min(1, вперёд / ISS_MM) + вперёд × tg Беннетта). Справочно: на архиве КТ скачут от долей мм положения
  головки; стенки нет на КТ (тонкая, не сегментирована) — не определяются.

Покой — головка в прикусе (нижняя челюсть переезжает с КТ в прикус сканов: ``Session.bite_motion``), не как на
снимке: КТ бывает снят не в смыкании, а движения и вставки — от прикуса. Череп с нижней челюстью не едет.

Всё — в системе монтажа (``Anatomy.frame``: x — вправо, y — вперёд, z — вверх), мм; ответ — относительно верхушки
мыщелка (Co) в прикусе.
"""

import numpy as np

from .register import apply

STEP_MM = 0.25
PATH_MM = 12.0  # протрузия: на столько мм вперёд
CHORD_MM = 5.0
LATERO_MM = 6.0  # латеротрузия: балансирующий мыщелок — на столько мм вперёд
ISS_MM = 1.0  # как kinematics.ISS_MM: немедленный сдвиг набирается на первом миллиметре
MEDIAL_MAX_MM = 4.0  # внутрь дальше — стенки нет (на КТ не видна)
MEDIAL_STEP_MM = 0.1
WALL_DEG = 45.0  # поверхность над головкой опускается внутрь круче — стенка: головка по ней вниз-внутрь не съезжает
BOX = {"x": 14.0, "y": (-10.0, 24.0), "z": (-16.0, 10.0)}  # ямка — в этой рамке вокруг Co (система монтажа)
HEAD_DEPTH_MM = 8.0  # головка — вершины отростка не ниже этого от верхушки
SAMPLE_MM = 0.3
NEAR_MM = 3.0  # дальше от поверхности — точка головки не касается
HEAD_REACH_MM = 6.0  # точки головки дальше от ямки в покое на пути не касаются — в расчёт не идут
HEAD_POINTS = 1500
HEAD_MIN_POINTS = 300  # меньше точек головки у ямки — ямки над мыщелком нет (тонкая крыша не сегментирована)
PRESS_MM = 4.0  # прижатие вверх: дальше упора нет — над головкой нет кости
DISC_INTERMEDIATE_MM = 1.0  # промежуточная зона диска — между головкой и скатом при выдвижении
DISC_BLEND_MM = (1.0, 4.0)  # от покоя к промежуточной зоне — на этом отрезке пути вперёд
REST_INSIDE_WARN_MM = 0.5  # головка в покое глубже в кости — заметка: прикус или сегментация неточны


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
        # в покое головка бывает «в кости» на доли мм (прикус по сканам и сегментация не точнее): это положение —
        # начало пути, такой заход допускается и дальше не растёт
        d, j = self.tree.query(head, distance_upper_bound=NEAR_MM)
        ok = np.isfinite(d)
        inside = np.zeros(len(head))
        inside[ok] = -np.einsum("ij,ij->i", head[ok] - self.fossa[j[ok]], self.normals[j[ok]])
        self.allow = np.clip(inside, 0.0, None)
        self.medial = float(medial)  # направление «внутрь» по x: −1 у правого мыщелка, +1 у левого
        # диск лежит на крыше и скатах (поверхность смотрит вниз) — по вертикали на свою толщину; у внутренней стенки
        # нет (там щель настоящая — из неё немедленный боковой сдвиг), у задней стенки нет (позади диска — ткани)
        n = self.normals
        self.disc_w = np.clip(-n[:, 2], 0.0, 1.0) * np.clip((0.7 - n[:, 1]) / 0.4, 0.0, 1.0)

    def clearance(self, shift: np.ndarray, disc: float = 0.0) -> float:
        """Наименьший зазор головки, сдвинутой на shift, до поверхности «кость + диск»; < 0 — проникновение."""
        p = self.head + shift
        d, j = self.tree.query(p, distance_upper_bound=NEAR_MM + disc)
        ok = np.isfinite(d)
        if not ok.any():
            return NEAR_MM
        s = np.einsum("ij,ij->i", p[ok] - self.fossa[j[ok]], self.normals[j[ok]]) - disc * self.disc_w[j[ok]]
        return float((s + self.allow[ok]).min())

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
                s = np.einsum("ij,ij->i", p[ok] - self.fossa[j[ok]], n) - disc * self.disc_w[j[ok]] + self.allow[ok]
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

    def superior_gap(self) -> float:
        """Щель над головкой в покое — до поверхностей под диском (крыша, скаты): толщина «постоянного» диска,
        при ней головка в покое касается его, не проникая. Наименьший зазор по всем направлениям не годится: сзади
        и внутри головка на КТ бывает вплотную к кости."""
        k = 8
        d, j = self.tree.query(self.head, k=k, distance_upper_bound=NEAR_MM)
        ok = np.isfinite(d)
        if not ok.any():
            return 0.0
        jj = j[ok]
        h = np.broadcast_to(self.head[:, None, :], (len(self.head), k, 3))[ok]
        s = np.einsum("ij,ij->i", h - self.fossa[jj], self.normals[jj])             + np.broadcast_to(self.allow[:, None], (len(self.head), k))[ok]
        w = self.disc_w[jj]
        sel = w > 0.25
        return float(max(0.0, np.min(s[sel] / w[sel]))) if sel.any() else 0.0

    def press_up(self, x: float, y: float, disc: float, z_from: float) -> float | None:
        """Наибольший z (головка прижата вверх) при сдвиге (x, y) без проникновения; поиск — от z_from."""
        start = np.array([x, y, z_from - 2.0])
        if self.clearance(start, disc) < 0:
            start = self.advance(start, np.array([0.0, 0.0, -1.0]), disc, limit=3.0)  # сначала вниз — из кости
            if self.clearance(start, disc) < -0.02:
                return None
        z = self.advance(start, np.array([0.0, 0.0, 1.0]), disc, limit=PRESS_MM)[2]
        if z >= start[2] + PRESS_MM - 1e-6:
            return None  # упора нет
        return float(z) if self.clearance(np.array([x, y, z]), disc) >= -0.02 else None

def _disc(mode: str, rest: float, forward: float) -> float:
    if mode == "rest":
        return rest
    a, b = DISC_BLEND_MM
    w = float(np.clip((forward - a) / (b - a), 0.0, 1.0))
    thin = min(rest, DISC_INTERMEDIATE_MM)
    return (1 - w) * rest + w * thin


def joint(skull_vertices, skull_faces, head_vertices, frame: np.ndarray, condyle_top, side: str,
          bite: np.ndarray | None = None) -> Joint:
    """Сустав стороны side ("right" / "left"): head_vertices — вершины мыщелкового отростка (координаты КТ),
    condyle_top — Co (координаты КТ); bite — нижняя челюсть с КТ в прикус (``Session.bite_motion``; None — прикус
    как на КТ): головка и Co едут в прикус. Всё переводится в систему монтажа относительно Co."""
    from scipy.spatial import cKDTree

    from . import gingiva as gg

    F = np.asarray(frame, float)
    if bite is not None:
        B = np.asarray(bite, float)
        head_vertices, condyle_top = apply(B, np.asarray(head_vertices, float)), apply(B, np.asarray(condyle_top, float)[None])[0]
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
    near = cKDTree(pts).query(head, distance_upper_bound=HEAD_REACH_MM)[0] if len(pts) else np.full(len(head), np.inf)
    if np.isfinite(near).sum() < HEAD_MIN_POINTS:
        raise ValueError("над мыщелком нет кости ямки — проверьте сегментацию черепа")
    # нормали — из кости к суставу. По всей сетке черепа «наружу» не определить (внутренние поверхности свода,
    # пазух, слуховых проходов), поэтому — по поверхности у головки: она смотрит на головку
    d, i = cKDTree(head).query(pts, distance_upper_bound=NEAR_MM)
    near = np.isfinite(d)
    if near.any() and np.einsum("ij,ij->i", head[i[near]] - pts[near], normals[near]).sum() < 0:
        normals = -normals
    return Joint(head, pts, normals, -1.0 if side == "right" else 1.0)


def protrusion(j: Joint, disc_mode: str = "rest") -> dict:
    """Путь головки при протрузии: от покоя (прикус) вперёд, головка прижата вверх к «кость + диск»; ССП — по
    хорде CHORD_MM. path — N×2 (вперёд, вверх) от покоя, как ``guidance.condylar_path(...)["path"]``."""
    rest = j.superior_gap()
    out = {"rest_gap_mm": round(rest, 2), "disc": disc_mode, "notes": []}
    if j.allow.max() > REST_INSIDE_WARN_MM:
        out["notes"].append(f"головка в прикусе заходит в кость на {j.allow.max():.1f} мм — проверьте прикус и "
                            "сегментацию сустава")
    path, z = [(0.0, 0.0)], 0.0
    for y in np.arange(STEP_MM, PATH_MM + 1e-9, STEP_MM):
        zz = j.press_up(0.0, float(y), _disc(disc_mode, rest, float(y)), z)
        if zz is None:
            break
        z = zz
        path.append((float(y), z))
    path = np.array(path)
    if len(path) < int(CHORD_MM / STEP_MM) + 1:
        raise ValueError("головка не скользит по ямке — проверьте мыщелок и сегментацию сустава")
    d = np.hypot(path[:, 0], path[:, 1])
    k = int(np.argmax(d >= CHORD_MM)) if (d >= CHORD_MM).any() else len(d) - 1
    out["sagittal_deg"] = round(float(np.degrees(np.arctan2(-path[k, 1], path[k, 0]))), 1)
    out["path"] = path
    return out


def balancing(j: Joint, disc_mode: str = "rest") -> dict:
    """Балансирующий мыщелок при латеротрузии: вперёд по y; головка, прижатая вверх, съезжает внутрь, пока
    поверхность над ней не опускается внутрь круче WALL_DEG (вогнутость ската головку не держит, стенка —
    держит). Немедленный боковой сдвиг и угол Беннетта — по кости, справочно: на архиве КТ они скачут от долей мм
    положения головки (связки и капсула на КТ не видны), в монтаж не идут."""
    rest = j.superior_gap()
    out = {"notes": []}
    lat, z, m = [], 0.0, 0.0
    wall = False
    tan = np.tan(np.radians(WALL_DEG))
    for y in np.arange(0.0, LATERO_MM + 1e-9, 0.5):
        disc = _disc(disc_mode, rest, float(y))
        if y > 0:
            zz = j.press_up(j.medial * m, float(y), disc, z)
            if zz is None:
                break
            z = zz
        while m < MEDIAL_MAX_MM - 1e-9:
            mm = min(m + MEDIAL_STEP_MM, MEDIAL_MAX_MM)
            zz = j.press_up(j.medial * mm, float(y), disc, z)
            if zz is None or zz < z - (mm - m) * tan:
                wall = True
                break
            m, z = mm, zz
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


def analyze(j: Joint, disc_mode: str = "rest") -> dict:
    """Протрузия и латеротрузия (балансирующая сторона): углы и пути."""
    out = protrusion(j, disc_mode)
    lat = balancing(j, disc_mode)
    out["notes"] += lat.pop("notes")
    out.update(lat)
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


def case_joints(skull, mandible, frame: np.ndarray, condyles: dict, bite: np.ndarray | None = None) -> dict:
    """Суставы кейса: {"right" | "left": Joint или текст ошибки}. skull, mandible — сетки КТ (vertices, faces);
    frame — система монтажа (координаты КТ → монтаж); condyles — Co сторон в координатах КТ (как на снимке:
    ``Session.landmarks``); bite — нижняя челюсть с КТ в прикус (``Session.bite_motion``)."""
    from .landmarks import condylar_heads

    R = np.asarray(frame, float)[:3, :3]
    mv, mf = np.asarray(mandible.vertices, float), np.asarray(mandible.faces)
    heads = condylar_heads(mv, mf, up=R[2], left=-R[0], anterior=R[1])
    out = {}
    for side, key in (("right", "Co_R"), ("left", "Co_L")):
        if key not in heads or condyles.get(side) is None:
            out[side] = "головка мыщелка на нижней челюсти не найдена"
            continue
        try:
            out[side] = joint(skull.vertices, skull.faces, mv[heads[key]["vertices"]], frame, condyles[side], side,
                              bite=bite)
        except ValueError as e:
            out[side] = str(e)
    return out
