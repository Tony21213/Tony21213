"""Виртуальный артикулятор: движения нижней челюсти по суставным путям и по зубам.

Как полностью регулируемый артикулятор: мыщелки стоят на шарнирной оси и
скользят каждый по своему суставному пути. Путь задаётся сагиттальным углом
(ССП — наклон вниз от горизонтали выбранной плоскости), а на балансирующей
стороне ещё углом Беннетта (внутрь) и немедленным боковым сдвигом, который
набирается на первом миллиметре пути. Рабочий мыщелок при боковом движении
уходит наружу на столько же, на сколько балансирующий — внутрь.

Передняя часть идёт по зубам. Если заданы модели челюстей в прикусе
(`Occlusion`), на каждом шаге нижняя челюсть поворачивается вокруг оси
мыщелков до контакта: зубы скользят друг по другу, не проникая. Поэтому
резцовое и клыковое ведение берутся из реальной формы зубов пациента со
стёртостью, а не из средних значений резцового столика.

Монтаж и углы разделены. Модели «гипсуются» в системе монтажа
(motion.Anatomy.frame: x — вправо, y — вперёд, z — вверх), например по
эстетической плоскости из фото. Мыщелки — свои точки пациента: справа и слева
они могут стоять на разной высоте и глубине, ось вращения проходит через оба
реальных мыщелка, даже если она косая к системе монтажа. Углы суставных путей
задаются в своей системе (Settings.frame — обычно Франкфурт по КТ или система,
в которой анализировалась запись) и переносятся в систему монтажа без
искажения: смена монтажа не меняет движение челюсти пациента.

Движения возвращаются записями motion.Recording в координатах кейса — так же,
как записи аксиографа. Поэтому сгенерированные движения анализируются тем же
motion.analyze_case и сверяются с реальными записями P-ART.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .motion import Anatomy, Recording
from .register import apply, axis_angle, rigid, rotation_between

ISS_MM = 1.0  # немедленный боковой сдвиг набирается на первом миллиметре пути балансирующего мыщелка
FRAMES = 41
GUIDE_REACH_MM = 6.0  # нижние зубы дальше этого от верхних в прикусе не участвуют в ведении
GUIDE_POINTS = 8000  # столько точек нижних зубов проверяется на проникновение
CONTACT_TOL_MM = 0.02  # допуск проникновения сверх того, что было в исходном прикусе
STEP_DEG = 0.05  # шаг поиска контакта: на резцах ~0.08 мм — тоньше режущего края, насквозь не проскочить
CLOSE_LIMIT_DEG = -15.0
OPEN_LIMIT_DEG = 25.0

SIDES = ("right", "left")


@dataclass
class Settings:
    """Настройки суставных путей; sources — откуда каждое число (запись, КТ, формула, по умолчанию)."""

    sagittal_right_deg: float = 35.0
    sagittal_left_deg: float = 35.0
    bennett_right_deg: float = 10.0  # угол Беннетта правого мыщелка (он балансирующий при движении влево)
    bennett_left_deg: float = 10.0
    side_shift_right_mm: float = 0.0  # немедленный боковой сдвиг при движении вправо
    side_shift_left_mm: float = 0.0
    sources: dict = field(default_factory=dict)
    # координаты кейса → система, в которой заданы углы; None — система монтажа (Anatomy.frame)
    frame: np.ndarray | None = None

    def get(self, name: str, side: str) -> float:
        return float(getattr(self, f"{name}_{side}_{'mm' if name == 'side_shift' else 'deg'}"))

    def copy(self) -> "Settings":
        return Settings(**{k: v for k, v in vars(self).items() if k != "sources"}, sources=dict(self.sources))


def settings_from_analysis(analysis: dict, base: Settings | None = None, frame: np.ndarray | None = None) -> Settings:
    """Настройки из анализа записи (motion.analyze_case); чего в записи нет — из base.

    frame — система, в которой шёл анализ (Anatomy.frame): углы отсчитаны от неё.
    """
    s = (base or Settings()).copy()
    if frame is not None:
        s.frame = np.asarray(frame, float)
    for key, value in analysis["articulator"].items():
        if value is not None and hasattr(s, key) and not key.startswith("fischer"):
            setattr(s, key, float(value))
            s.sources[key] = "запись движения"
    return s


def _sagittal(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([0.0, np.cos(a), -np.sin(a)])


def condyles(anatomy: Anatomy) -> dict:
    """Мыщелки в анатомической системе."""
    return {s: apply(anatomy.frame, anatomy.points[f"condyle_{s}"][None])[0] for s in SIDES}


def jaw_pose(c0: dict, c1: dict, theta_deg: float, anchor: str = "mid") -> np.ndarray:
    """Положение нижней челюсти: мыщелки из c0 переходят в c1, рот открыт на theta вокруг оси мыщелков.

    Точно в заданное место попадает anchor (мыщелок или середина оси); второй
    мыщелок — на оси в ту же сторону, расстояние между мыщелками сохраняется.
    """
    u0, u1 = c0["right"] - c0["left"], c1["right"] - c1["left"]
    u1 = u1 / np.linalg.norm(u1)
    R = axis_angle(u1, -np.radians(theta_deg)) @ rotation_between(u0, u1)  # открывание — поворот вокруг −x
    a0 = (c0["right"] + c0["left"]) / 2 if anchor == "mid" else c0[anchor]
    a1 = (c1["right"] + c1["left"]) / 2 if anchor == "mid" else c1[anchor]
    return rigid(R, a1 - R @ a0)


def _to_mount(anatomy: Anatomy, s: Settings) -> np.ndarray:
    """Поворот направлений из системы углов в систему монтажа."""
    if s.frame is None:
        return np.eye(3)
    return anatomy.frame[:3, :3] @ np.asarray(s.frame, float)[:3, :3].T


def protrusion_targets(c0: dict, s: Settings, travel: float, V: np.ndarray = np.eye(3)) -> dict:
    """Мыщелки после протрузии на travel мм по суставным путям (V — из системы углов в систему монтажа)."""
    return {side: c0[side] + travel * (V @ _sagittal(s.get("sagittal", side))) for side in SIDES}


def laterotrusion_targets(c0: dict, s: Settings, working: str, forward: float,
                          V: np.ndarray = np.eye(3)) -> tuple[dict, str]:
    """Мыщелки при боковом движении в сторону working; forward — путь балансирующего мыщелка вперёд, мм."""
    balancing = "left" if working == "right" else "right"
    toward = V @ np.array([1.0 if working == "right" else -1.0, 0, 0])  # к рабочей стороне
    lateral = s.get("side_shift", working) * min(1.0, forward / ISS_MM) \
        + forward * np.tan(np.radians(s.get("bennett", balancing)))
    path = V @ np.array([0.0, forward, -forward * np.tan(np.radians(s.get("sagittal", balancing)))])
    return {balancing: c0[balancing] + lateral * toward + path, working: c0[working] + lateral * toward}, balancing


class Occlusion:
    """Верхние зубы как препятствие для нижних: ведение по реальной геометрии зубов.

    Только внутриротовые сканы (или CAD-модели) в прикусе: зубы из КТ для
    скольжения не годятся — разрешение и сглаживание искажают бугры и края.
    Модели — в координатах кейса. Проникновение считается по ближайшей
    вершине верхней модели и её нормали. Прикус сканов часто «провален» на
    десятые доли миллиметра, поэтому допуск свой у каждой точки: глубже, чем
    она была в исходном прикусе, ей уходить нельзя, а точкам, которые в прикусе
    не касались, — глубже CONTACT_TOL_MM. Шаги движения и поиска контакта малы,
    поэтому за шаг точка не проскакивает сквозь тонкий край.
    """

    def __init__(self, upper_vertices, upper_faces, lower_vertices, frame: np.ndarray, seed: int = 0):
        import trimesh

        upper = trimesh.Trimesh(apply(frame, upper_vertices), np.asarray(upper_faces), process=False)
        self.points = np.asarray(upper.vertices)
        self.normals = np.asarray(upper.vertex_normals)
        self.tree = cKDTree(self.points)
        lower = apply(frame, lower_vertices)
        d, _ = self.tree.query(lower, distance_upper_bound=GUIDE_REACH_MM)
        near = lower[np.isfinite(d)]
        if len(near) == 0:
            raise ValueError(f"модели не в прикусе: нижние зубы дальше {GUIDE_REACH_MM:g} мм от верхних")
        if len(near) > GUIDE_POINTS:
            near = near[np.random.default_rng(seed).choice(len(near), GUIDE_POINTS, replace=False)]
        self.lower = near
        _d, i = self.tree.query(near)
        if np.mean(np.sum((near - self.points[i]) * self.normals[i], axis=1)) < 0:
            self.normals = -self.normals  # нормали верхних зубов — наружу, к нижним
        start = self._signed(np.eye(4))
        self.floor = np.minimum(start, 0.0) - CONTACT_TOL_MM  # не глубже, чем в прикусе
        self.bite_penetration_mm = float(max(0.0, -start.min()))  # насколько «провален» прикус

    def _signed(self, M: np.ndarray) -> np.ndarray:
        """Расстояние каждой нижней точки до верхних зубов со знаком (минус — внутри); далёкие — inf."""
        p = apply(M, self.lower)
        d, i = self.tree.query(p, distance_upper_bound=1.5)
        out = np.full(len(p), np.inf)
        ok = np.isfinite(d)
        out[ok] = np.sum((p[ok] - self.points[i[ok]]) * self.normals[i[ok]], axis=1)
        return out

    def depth(self, M: np.ndarray) -> float:
        """Насколько при положении M (анатомическая система) нижние зубы уходят в верхние глубже допуска, мм."""
        return float(max(0.0, (self.floor - self._signed(M)).max()))

    def contact(self, c0: dict, c1: dict, anchor: str, start: float) -> float:
        """Угол открывания, при котором зубы в контакте без проникновения (ищется от start)."""
        def free(theta):
            return self.depth(jaw_pose(c0, c1, theta, anchor)) <= 0.0

        lo = hi = start
        if free(start):  # закрывать, пока зубы не встретятся
            while lo > CLOSE_LIMIT_DEG and free(lo - STEP_DEG):
                lo -= STEP_DEG
            hi, lo = lo, lo - STEP_DEG
            if lo <= CLOSE_LIMIT_DEG:
                return hi
        else:  # открывать, пока не разойдутся
            while hi < OPEN_LIMIT_DEG and not free(hi + STEP_DEG):
                hi += STEP_DEG
            lo, hi = hi, hi + STEP_DEG
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if free(mid) else (mid, hi)
        return hi


def contact_sectors(rec: Recording, anatomy: Anatomy, occlusion: Occlusion, every: int = 4, gap: float = 0.1,
                    anterior_mm: float = 12.0) -> dict:
    """Каких участков дуги касаются зубы и на какой части пути: {участок: [доля пути от, до]}.

    Участки — передние (не дальше anterior_mm назад от резцовой точки, с клыками)
    и жевательные справа и слева. Касание — нижняя точка ближе gap мм к верхним
    зубам. Контакты жевательных зубов балансирующей стороны при боковом
    движении — интерференции.
    """
    F = anatomy.frame
    back = np.linalg.inv(F)
    inc = apply(F, anatomy.points["incisal"][None])[0]
    n = len(rec.transforms)
    out = {}
    for k in range(0, n, every):
        p = apply(F @ rec.transforms[k] @ back, occlusion.lower)
        d, _ = occlusion.tree.query(p, distance_upper_bound=gap)
        hit = p[np.isfinite(d)] - inc
        if not len(hit):
            continue
        sector = np.where(hit[:, 1] > -anterior_mm, "передние",
                          np.where(hit[:, 0] > 0, "жевательные справа", "жевательные слева"))
        t = round(k / max(n - 1, 1), 2)
        for name in set(sector.tolist()):
            lo, hi = out.get(name, (t, t))
            out[name] = [min(lo, t), max(hi, t)]
    return dict(sorted(out.items()))


def _recording(name: str, anatomy: Anatomy, poses: list, duration: float) -> Recording:
    """Положения из анатомической системы — в координаты кейса."""
    F = anatomy.frame
    back = np.linalg.inv(F)
    T = np.array([back @ M @ F for M in poses])
    return Recording(name, np.linspace(0, duration, len(poses)), T, "виртуальный артикулятор")


def opening(anatomy: Anatomy, s: Settings, max_deg: float = 28.0, hinge_deg: float = 12.0,
            travel: float = 10.0, frames: int = FRAMES) -> Recording:
    """Открывание: сначала чистое вращение на шарнирной оси, потом мыщелки идут вперёд по путям."""
    c0, V = condyles(anatomy), _to_mount(anatomy, s)
    poses = []
    for theta in np.linspace(0, max_deg, frames):
        t = travel * max(0.0, theta - hinge_deg) / max(max_deg - hinge_deg, 1e-9)
        poses.append(jaw_pose(c0, protrusion_targets(c0, s, t, V), theta))
    return _recording("Открывание", anatomy, poses, 1.5)


def protrusion(anatomy: Anatomy, s: Settings, travel: float = 6.0, occlusion: Occlusion | None = None,
               frames: int = FRAMES) -> Recording:
    """Протрузия: мыщелки по суставным путям, передние зубы — по верхним (если заданы модели)."""
    c0, V = condyles(anatomy), _to_mount(anatomy, s)
    poses, theta = [], 0.0
    for t in np.linspace(0, travel, frames):
        c1 = protrusion_targets(c0, s, t, V)
        theta = occlusion.contact(c0, c1, "mid", theta) if occlusion else 0.0
        poses.append(jaw_pose(c0, c1, theta))
    return _recording("Протрузия", anatomy, poses, 1.2)


def laterotrusion(anatomy: Anatomy, s: Settings, working: str, travel: float = 6.0,
                  occlusion: Occlusion | None = None, frames: int = FRAMES) -> Recording:
    """Боковое движение в сторону working: балансирующий мыщелок вперёд, вниз и внутрь, рабочий — наружу."""
    c0, V = condyles(anatomy), _to_mount(anatomy, s)
    poses, theta = [], 0.0
    for f in np.linspace(0, travel, frames):
        c1, anchor = laterotrusion_targets(c0, s, working, f, V)
        theta = occlusion.contact(c0, c1, anchor, theta) if occlusion else 0.0
        poses.append(jaw_pose(c0, c1, theta, anchor))
    name = "Латеротрузия вправо" if working == "right" else "Латеротрузия влево"
    return _recording(name, anatomy, poses, 1.2)


def chewing(anatomy: Anatomy, s: Settings, working: str = "right", cycles: int = 3, max_deg: float = 12.0,
            travel: float = 3.0, frames_per_cycle: int = 30) -> Recording:
    """Жевание: циклы открывания со смещением в рабочую сторону и возвратом в прикус."""
    c0, V = condyles(anatomy), _to_mount(anatomy, s)
    poses = []
    for u in np.linspace(0, cycles, cycles * frames_per_cycle + 1):
        phase = np.sin(np.pi * (u % 1.0))
        c1, anchor = laterotrusion_targets(c0, s, working, travel * phase ** 2, V)
        poses.append(jaw_pose(c0, c1, max_deg * phase, anchor))
    return _recording("Жевание", anatomy, poses, 0.8 * cycles)


def standard_movements(anatomy: Anatomy, s: Settings, occlusion: Occlusion | None = None) -> list[Recording]:
    """Набор, как при записи аксиографом: открывание, протрузия, латеротрузии вправо и влево."""
    return [opening(anatomy, s), protrusion(anatomy, s, occlusion=occlusion),
            laterotrusion(anatomy, s, "right", occlusion=occlusion),
            laterotrusion(anatomy, s, "left", occlusion=occlusion)]
