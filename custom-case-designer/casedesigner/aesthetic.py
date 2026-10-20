"""Эстетическая система координат по фото анфас.

Фото — анфас, пациент в привычном физиологическом положении головы, камера
выставлена по уровню. Тогда:

* **горизонт** — истинная горизонталь кадра (после поворота по EXIF). Если в
  кадре есть уровень или отвес, горизонт задаётся по нему
  (`horizon_from_points`, `horizon_from_vertical`);
* **крен** эстетической системы — по горизонту фото; наклон вперёд-назад и
  разворот — от функциональной системы (Франкфурт/HIP по КТ или окклюзионная
  плоскость), по фото анфас их не видно;
* **средняя линия** — по нескольким срединным точкам лица на фото (глабелла,
  кончик носа, подносовая точка, фильтрум, подбородок) с ручной правкой; по
  ней — смещение средней линии зубов и боковое положение начала системы;
* зрачковая линия и другие линии лица из-за естественной асимметрии лица
  горизонт не задают — это подсказки с их углом к горизонту (`face_hints`).

Фото привязывается к зубам верхнего скана по 2+ парам точек «фото — скан»
(режущие края, бугры клыков), как в exocad Smile Creator: по ним находится
положение камеры (`fit_camera`), и направления кадра переносятся на модели.
Перспектива по точкам на зубах не определяется (зубы неглубокие, камера в
полуметре и дальше: фокус от 21 до 150 мм даёт одну и ту же ошибку), поэтому
расстояние до камеры задаётся — по умолчанию DEFAULT_DISTANCE_MM (exocad: 1072,55
мм; протокол съёмки exocad — с 1 м), фокус подбирается по масштабу. Взгляд — спереди
по функциональной системе; при 3+ парах он уточняется, держась у фронтального.

Автоматические подсказки по лицу — MediaPipe Face Landmarker (Apache-2.0,
модель face_landmarker.task); без неё всё работает по точкам, отмеченным вручную.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .register import rigid

FULL_FRAME_DIAGONAL_MM = 43.27  # диагональ кадра 36×24 мм — для фокусного расстояния в 35-мм эквиваленте
DEFAULT_FOCAL_35MM = 50.0  # если в EXIF нет фокусного расстояния: портретная съёмка
MIRROR_GAIN = 0.5  # зеркальное (фронтальная камера) — если с отражением ошибка меньше хотя бы вдвое
DEFAULT_DISTANCE_MM = 1000.0  # камера — до зубов (exocad Smile Creator: 1072,55 мм по умолчанию)
DISTANCE_RANGE_MM = (150.0, 5000.0)
MIN_PAIRS = 2
VIEW_PRIOR_PX_PER_DEG = 0.5  # 3+ пары: отход взгляда от фронтального на 1° стоит, как 0,5 px ошибки на паре
LIP_DEPTH_MM = 10.0  # средняя линия лица на высоте резцов лежит на губе — примерно на 10 мм впереди режущего края

# Точки MediaPipe Face Landmarker (478): зрачки, углы глаз, крылья носа, углы рта, срединные точки.
FACE_POINTS = {"pupil_right": 468, "pupil_left": 473, "canthus_right": 33, "canthus_left": 263,
               "endocanthion_right": 133, "endocanthion_left": 362,
               "ala_right": 64, "ala_left": 294, "commissure_right": 61, "commissure_left": 291}
MIDLINE_POINTS = {"glabella": 9, "nasion": 168, "pronasale": 1, "subnasale": 2, "labrale_superius": 0,
                  "labrale_inferius": 17, "menton": 152}
# Контуры губ MediaPipe (по кругу): внутренний — край открытого рта (вырез рта на фото с улыбкой), наружный.
LIPS_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
FACE_DETECT_PX = 1600  # MediaPipe — на уменьшенном снимке (быстрее, точность та же)
FACE_LINES = {"зрачковая": ("pupil_right", "pupil_left"), "углов глаз": ("canthus_right", "canthus_left"),
              "крыльев носа": ("ala_right", "ala_left"), "углов рта": ("commissure_right", "commissure_left")}


@dataclass
class Photo:
    image: np.ndarray  # RGB, уже повёрнут по EXIF
    focal_px: float | None  # фокусное расстояние в пикселях (по EXIF), если известно
    notes: list = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


def load_photo(path: str) -> Photo:
    """Фото с поворотом по EXIF (телефон хранит ориентацию флагом) и фокусным расстоянием из EXIF."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        exif = im.getexif()
        upright = ImageOps.exif_transpose(im).convert("RGB")
    image = np.asarray(upright)
    notes, focal_px = [], None
    sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
    f35 = sub.get(0xA405) or exif.get(0xA405)  # FocalLengthIn35mmFilm
    if f35:
        diag = float(np.hypot(*image.shape[:2]))
        focal_px = float(f35) * diag / FULL_FRAME_DIAGONAL_MM
    if exif.get(0x0112, 1) != 1:
        notes.append("фото повёрнуто по EXIF")
    return Photo(image, focal_px, notes)


@dataclass
class Camera:
    """Камера фото в координатах кейса: пиксель = K · (R · точка + t)."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    residual_px: float
    mirrored: bool = False
    size: tuple = (0, 0)

    def project(self, points: np.ndarray) -> np.ndarray:
        c = np.asarray(points, float) @ self.R.T + self.t
        uv = c[:, :2] / c[:, 2:3] * np.diag(self.K)[:2] + self.K[:2, 2]
        if self.mirrored:
            uv[:, 0] = self.size[0] - uv[:, 0]
        return uv

    def ray(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Луч через пиксель: начало (центр камеры) и направление, координаты кейса."""
        u, v = float(uv[0]), float(uv[1])
        if self.mirrored:
            u = self.size[0] - u
        d_cam = np.array([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], 1.0])
        d = self.R.T @ d_cam
        return -self.R.T @ self.t, d / np.linalg.norm(d)

    @property
    def centre(self) -> np.ndarray:
        """Центр камеры, координаты кейса."""
        return -self.R.T @ self.t

    def axis(self, image_direction) -> np.ndarray:
        """Направление в кадре (пиксели: x вправо, y вниз) — как направление в координатах кейса."""
        dx, dy = float(image_direction[0]), float(image_direction[1])
        if self.mirrored:
            dx = -dx
        d = self.R.T @ np.array([dx, dy, 0.0])
        return d / np.linalg.norm(d)


def _front_rotation(base_frame: np.ndarray) -> np.ndarray:
    """Камера спереди: правая сторона кадра — левая сторона пациента, вниз кадра — вниз, взгляд — назад."""
    x_f, y_f, z_f = base_frame[:3, :3]
    return np.array([-x_f, -z_f, -y_f])


def _fit(px, pts, size, focal, R0, fit_focal):
    w, h = size
    cx, cy = w / 2, h / 2
    c3, c2 = pts.mean(0), px.mean(0)
    spread3 = np.linalg.norm(pts - c3, axis=1).mean()
    spread2 = np.linalg.norm(px - c2, axis=1).mean()
    depth = focal * spread3 / max(spread2, 1e-6)
    cam_c = np.array([(c2[0] - cx) / focal * depth, (c2[1] - cy) / focal * depth, depth])
    x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), cam_c - R0 @ c3, np.log(focal)]

    def residual(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        f = np.exp(x[6]) if fit_focal else focal
        c = pts @ R.T + x[3:6]
        uv = c[:, :2] / c[:, 2:3] * f + [cx, cy]
        return (uv - px).ravel()

    sol = least_squares(residual if fit_focal else (lambda x: residual(np.r_[x, np.log(focal)])),
                        x0 if fit_focal else x0[:6], method="lm")
    x = sol.x if fit_focal else np.r_[sol.x, np.log(focal)]
    f = float(np.exp(x[6]))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    rms = float(np.sqrt(np.mean(residual(x) ** 2) * 2))
    return Camera(K, Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6], rms, size=size)


def view_rotation(base_frame: np.ndarray, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Камера спереди по функциональной системе, повёрнутая на pitch (плюс — смотрит сверху: верх модели к камере)
    и yaw (плюс — камера правее пациента). Строки — оси камеры в координатах кейса."""
    R0 = _front_rotation(base_frame)
    x_f, _y_f, z_f = np.asarray(base_frame, float)[:3, :3]
    turn = Rotation.from_rotvec(-np.radians(yaw_deg) * z_f) * Rotation.from_rotvec(np.radians(pitch_deg) * x_f)
    return R0 @ turn.as_matrix().T


def _fit_at_distance(px, pts, size, distance, R0, free_view, anchor=None):
    """Камера, у которой anchor (по умолчанию центр точек) на глубине distance: неизвестны поворот (при 2 парах —
    только вокруг взгляда), сдвиг anchor в кадре и фокус."""
    w, h = size
    cx, cy = w / 2, h / 2
    c3 = pts.mean(0) if anchor is None else np.asarray(anchor, float)
    rel = pts - c3
    spread3 = np.linalg.norm(rel, axis=1).mean()
    c2 = px.mean(0)
    spread2 = np.linalg.norm(px - c2, axis=1).mean()
    f0 = max(spread2, 1e-6) * distance / max(spread3, 1e-6)

    def camera(x):
        dR = Rotation.from_rotvec(x[:3] if free_view else np.r_[0.0, 0.0, x[0]]).as_matrix()
        R = dR @ R0
        k = 3 if free_view else 1
        f = float(np.exp(x[k + 2]))
        t = np.r_[x[k:k + 2], distance] - R @ c3
        return R, t, f

    def residual(x):
        R, t, f = camera(x)
        c = pts @ R.T + t
        uv = c[:, :2] / c[:, 2:3] * f + [cx, cy]
        r = (uv - px).ravel()
        if free_view:  # взгляд — у фронтального: наклон и разворот (повороты вокруг осей кадра x, y)
            r = np.r_[r, VIEW_PRIOR_PX_PER_DEG * np.degrees(x[:2])]
        return r

    off0 = [(c2[0] - cx) / f0 * distance, (c2[1] - cy) / f0 * distance]
    x0 = np.r_[np.zeros(3 if free_view else 1), off0, np.log(f0)]
    sol = least_squares(residual, x0, method="lm" if len(x0) <= len(residual(x0)) else "trf")
    R, t, f = camera(sol.x)
    data = residual(sol.x)[: 2 * len(px)]
    rms = float(np.sqrt(np.mean(data ** 2) * 2))
    return Camera(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), R, t, rms, size=size)


def fit_camera(points_px, points_3d, size, base_frame: np.ndarray, focal_px: float | None = None,
               distance_mm: float | None = None, view_deg=None, anchor=None) -> Camera:
    """Положение камеры по парам точек «фото — скан».

    points_px — пиксели на фото (x вправо, y вниз), points_3d — те же точки на
    верхнем скане (координаты кейса); base_frame — функциональная система (камера
    спереди). Как в exocad: anchor (резцовая точка; без неё — центр точек) — на
    глубине distance_mm от камеры (по умолчанию DEFAULT_DISTANCE_MM), фокус
    подбирается; хватит 2 пар. Взгляд —
    спереди, повёрнутый на view_deg = (наклон, разворот) градусов; при 3+ парах и
    view_deg=None — уточняется по точкам, держась у фронтального. focal_px без
    distance_mm — прежний расчёт: фокус задан (EXIF), расстояние подбирается
    (4+ пары). Зеркальное фото (фронтальная камера) распознаётся по ошибке.
    """
    px = np.asarray(points_px, float)
    pts = np.asarray(points_3d, float)
    w, h = size
    if focal_px is not None and distance_mm is None:
        if len(px) != len(pts) or len(px) < 4:
            raise ValueError("нужно минимум 4 пары точек «фото — скан»")
        R0 = _front_rotation(base_frame)
        direct = _fit(px, pts, size, float(focal_px), R0, False)
        flipped = _fit(np.c_[w - px[:, 0], px[:, 1]], pts, size, float(focal_px), R0, False)
    else:
        if len(px) != len(pts) or len(px) < MIN_PAIRS:
            raise ValueError(f"нужно минимум {MIN_PAIRS} пары точек «фото — скан»")
        d = float(DEFAULT_DISTANCE_MM if distance_mm is None else distance_mm)
        if not DISTANCE_RANGE_MM[0] <= d <= DISTANCE_RANGE_MM[1]:
            raise ValueError(f"расстояние до камеры — от {DISTANCE_RANGE_MM[0]:g} до {DISTANCE_RANGE_MM[1]:g} мм")
        pitch, yaw = (0.0, 0.0) if view_deg is None else (float(view_deg[0]), float(view_deg[1]))
        R0 = view_rotation(base_frame, pitch, yaw)
        free = view_deg is None and len(px) >= 3
        direct = _fit_at_distance(px, pts, size, d, R0, free, anchor)
        flipped = _fit_at_distance(np.c_[w - px[:, 0], px[:, 1]], pts, size, d, R0, free, anchor)
        if len(px) < 3:
            return direct  # по 2 парам зеркало не отличить: фото считается обычным
    if flipped.residual_px < MIRROR_GAIN * direct.residual_px:
        flipped.mirrored = True
        return flipped
    return direct


MANUAL_KEYS = ("pitch", "yaw", "roll", "dx", "dy", "scale")


def adjust_camera(camera: Camera, manual: dict | None, anchor) -> Camera:
    """Ручная поправка совмещения, как в exocad Smile Design: скан поворачивается вокруг anchor (резцовая точка) —
    pitch (вокруг оси кадра «вправо»; плюс — верх скана к камере), yaw (вокруг «вниз»), roll (вокруг взгляда),
    градусы; затем сдвигается на dx, dy (px снимка, как на экране) и масштабируется в scale раз вокруг anchor.
    Зеркальное фото — поправка как на экране."""
    from dataclasses import replace

    m = {k: float((manual or {}).get(k, 1.0 if k == "scale" else 0.0)) for k in MANUAL_KEYS}
    if m["scale"] <= 0:
        raise ValueError("масштаб поправки — больше нуля")
    sgn = -1.0 if camera.mirrored else 1.0
    Q = Rotation.from_euler("xyz", [m["pitch"], sgn * m["yaw"], sgn * m["roll"]], degrees=True).as_matrix()
    A = np.asarray(anchor, float)
    R = Q @ camera.R
    t = camera.R @ A + camera.t - Q @ camera.R @ A  # anchor — на прежнем месте перед камерой
    K = camera.K.copy()
    cam = replace(camera, R=R, t=t)
    uv_a = cam.project(A[None])[0]
    if camera.mirrored:
        uv_a[0] = camera.size[0] - uv_a[0]
    s_ = m["scale"]
    K[0, 0] *= s_
    K[1, 1] *= s_
    K[0, 2] = camera.K[0, 2] + (1 - s_) * (uv_a[0] - camera.K[0, 2]) + sgn * m["dx"]
    K[1, 2] = camera.K[1, 2] + (1 - s_) * (uv_a[1] - camera.K[1, 2]) + m["dy"]
    return replace(cam, K=K)


def horizon_from_points(p1, p2) -> float:
    """Угол истинной горизонтали в кадре по двум точкам на горизонтальной линии (уровень), градусы."""
    d = np.asarray(p2, float) - np.asarray(p1, float)
    if d[0] < 0:
        d = -d
    return float(np.degrees(np.arctan2(d[1], d[0])))


def horizon_from_vertical(p_top, p_bottom) -> float:
    """Угол истинной горизонтали по двум точкам на вертикали (отвес), градусы."""
    d = np.asarray(p_bottom, float) - np.asarray(p_top, float)
    return float(np.degrees(np.arctan2(d[1], d[0]))) - 90.0


@dataclass
class Aesthetic:
    frame: np.ndarray  # 4×4: координаты кейса → эстетическая система (x вправо, y вперёд, z вверх)
    roll_deg: float  # на сколько функциональная горизонталь наклонена к истинной; плюс — правая сторона выше
    midline_offset_mm: float | None  # средняя линия зубов относительно лица; плюс — смещена вправо пациента
    facial_midline_tilt_deg: float | None  # наклон средней линии лица к истинной вертикали; плюс — верх вправо
    hints: dict = field(default_factory=dict)  # углы линий лица к горизонту, градусы; плюс — правая сторона выше
    notes: list = field(default_factory=list)


def _line_through(points_px):
    p = np.asarray(points_px, float)
    c = p.mean(0)
    d = np.linalg.svd(p - c, full_matrices=False)[2][0]
    return c, d if d[1] > 0 else -d  # направление — вниз по кадру


def aesthetic_frame(base_frame: np.ndarray, camera: Camera, horizon_deg: float = 0.0, midline_px=None,
                    incisal_point=None, face=None, lip_depth_mm: float = LIP_DEPTH_MM) -> Aesthetic:
    """Эстетическая система: крен — по горизонту фото, наклон и разворот — от функциональной системы.

    base_frame — функциональная система (координаты кейса → x вправо, y вперёд,
    z вверх); camera — по fit_camera; horizon_deg — угол истинной горизонтали в
    кадре (0 — край кадра, плюс — линия идёт вниз вправо по кадру); midline_px —
    точки средней линии лица на фото; incisal_point — точка между центральными
    резцами (координаты кейса); face — точки лица (face_hints) для подсказок.
    Знаки углов: плюс — правая сторона пациента выше (для средней линии — верх
    смещён вправо пациента).
    """
    R_f = np.asarray(base_frame, float)[:3, :3]
    x_f, y_f, _z_f = R_f
    origin_f = -R_f.T @ np.asarray(base_frame, float)[:3, 3]
    h = np.radians(horizon_deg)
    d = camera.axis((np.cos(h), np.sin(h)))
    x_a = d - (d @ y_f) * y_f
    x_a /= np.linalg.norm(x_a)
    if x_a @ x_f < 0:
        x_a = -x_a
    y_a = y_f
    z_a = np.cross(x_a, y_a)
    roll = float(np.degrees(np.arctan2(x_f @ z_a, x_f @ x_a)))

    # для углов — вид «как смотрим на пациента»: на зеркальном фото отражаем по горизонтали
    w = camera.size[0]
    h_view = -horizon_deg if camera.mirrored else horizon_deg

    def view(p):
        p = np.asarray(p, float)
        return np.c_[w - p[..., 0], p[..., 1]].reshape(p.shape) if camera.mirrored else p

    notes, offset, tilt, origin = [], None, None, origin_f
    if midline_px is not None and len(midline_px) >= 2:
        _c, md = _line_through(view(midline_px))
        hv = np.radians(h_view)
        vertical = np.array([-np.sin(hv), np.cos(hv)])  # истинная вертикаль в кадре, вниз
        clockwise = float(np.degrees(np.arctan2(vertical[0] * md[1] - vertical[1] * md[0], vertical @ md)))
        tilt = -clockwise  # по часовой — верх уходит вправо по кадру, то есть влево пациента
        if incisal_point is not None:
            inc = np.asarray(incisal_point, float)
            c, md_img = _line_through(midline_px)
            u, v = camera.project(inc[None])[0]
            # средняя линия лица на высоте резцов — на губе, впереди резцов на lip_depth_mm; при камере,
            # развёрнутой к голове, точка на другой глубине сместилась бы вбок на Δглубины × tg разворота
            o, ray = camera.ray(c + (v - c[1]) / md_img[1] * md_img)
            m = o + ((inc + lip_depth_mm * y_a - o) @ y_a) / (ray @ y_a) * ray
            offset = float((inc - m) @ x_a)
            origin = origin_f + ((m - origin_f) @ x_a) * x_a  # сагиттальная плоскость — по средней линии лица
        else:
            notes.append("нет точки между резцами — смещение средней линии зубов не посчитано")
    hints = {}
    for name, (right, left) in FACE_LINES.items():
        if face and right in face and left in face:
            dv = view(face[left]) - view(face[right])  # правая сторона пациента на фото слева
            hints[name] = round(_wrap(float(np.degrees(np.arctan2(dv[1], dv[0]))) - h_view), 2)
    R_a = np.array([x_a, y_a, z_a])
    return Aesthetic(rigid(R_a, -R_a @ origin), round(roll, 3), None if offset is None else round(offset, 3),
                     None if tilt is None else round(tilt, 3), hints, notes)


SYMMETRY_KEEP = 0.7  # доля лучших зеркальных пар: остальное — асимметрия и края поля зрения КТ


SYMMETRY_MAX_DEG, SYMMETRY_MAX_MM = 15.0, 10.0  # поиск — рядом с начальной плоскостью (по зубной дуге)


def midsagittal_plane(points: np.ndarray, normal, origin, keep: float = SYMMETRY_KEEP,
                      max_deg: float = SYMMETRY_MAX_DEG, max_mm: float = SYMMETRY_MAX_MM,
                      sample: int = 20000) -> tuple[np.ndarray, np.ndarray, float]:
    """Срединная плоскость по зеркальной симметрии облака точек (кости лицевого скелета или лицо).

    Мера симметрии — среднее расстояние отражённых точек до исходных по лучшим парам (keep): худшие — это
    асимметрия и края поля зрения КТ. Плоскость ищется рядом с начальной (normal, origin — например, по
    зубной дуге): крен и разворот — до ±max_deg, сдвиг — до ±max_mm; поиск без ограничений на обрезанном поле
    зрения уходит к ложной симметрии. Сетка, затем уточнение. Возвращает нормаль, точку и СКО пар, мм.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    P = np.asarray(points, float)
    rng = np.random.default_rng(0)
    if len(P) > sample:
        P = P[rng.choice(len(P), sample, replace=False)]
    tree = cKDTree(P)
    coarse = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    n0 = np.asarray(normal, float) / np.linalg.norm(normal)
    o0 = np.asarray(origin, float)
    e1 = np.cross(n0, [0, 0, 1.0] if abs(n0[2]) < 0.9 else [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)

    def plane(q):
        n = Rotation.from_rotvec(np.radians(q[0]) * e1).apply(Rotation.from_rotvec(np.radians(q[1]) * e2).apply(n0))
        return n, o0 + q[2] * n0

    def cost(q, pts=P):
        q = np.clip(q, [-max_deg, -max_deg, -max_mm], [max_deg, max_deg, max_mm])
        n, o = plane(q)
        d = np.sort(tree.query(pts - 2 * ((pts - o) @ n)[:, None] * n)[0])
        return float(d[: int(len(d) * keep)].mean())

    grid = [(a, b, t) for a in np.arange(-max_deg, max_deg + 0.1, 3.0) for b in np.arange(-max_deg, max_deg + 0.1, 3.0)
            for t in np.arange(-max_mm, max_mm + 0.1, 2.0)]
    best = np.asarray(min(grid, key=lambda q: cost(q, coarse)), float)
    # Начальный симплекс — явно: у точки сетки (0, 0, 0) scipy берёт шаг 0.00025 и сразу останавливается.
    simplex = np.vstack([best, best + [1.5, 0, 0], best + [0, 1.5, 0], best + [0, 0, 1.0]])
    q = np.clip(minimize(cost, best, method="Nelder-Mead",
                         options={"xatol": 0.02, "fatol": 1e-5, "initial_simplex": simplex}).x,
                [-max_deg, -max_deg, -max_mm], [max_deg, max_deg, max_mm])
    n, o = plane(q)
    d = np.sort(tree.query(P - 2 * ((P - o) @ n)[:, None] * n)[0])
    rms = float(np.sqrt(np.mean(d[: int(len(d) * keep)] ** 2)))
    return (n if n @ n0 > 0 else -n), o, rms


def aesthetic_frame_ct(base_frame: np.ndarray, face_points: np.ndarray) -> Aesthetic:
    """Эстетическая система по КТ, пока нет фото: средняя линия, крен и разворот — по зеркальной симметрии
    лицевого скелета (face_points — кости черепа и верхней челюсти без нижней, или лицо); наклон вперёд-назад —
    от функциональной системы (base_frame), как и при эстетической системе по фото. Начало — как у
    base_frame, сдвинутое вбок на срединную плоскость лица."""
    F = np.asarray(base_frame, float)
    R_f = F[:3, :3]
    x_f, y_f, _z_f = R_f
    origin_f = -R_f.T @ F[:3, 3]
    x_a, mid, rms = midsagittal_plane(face_points, x_f, origin_f)
    y_a = y_f - (y_f @ x_a) * x_a
    y_a /= np.linalg.norm(y_a)
    z_a = np.cross(x_a, y_a)
    roll = float(np.degrees(np.arctan2(x_f @ z_a, x_f @ x_a)))
    yaw = float(np.degrees(np.arctan2(y_f @ x_a, y_f @ y_a)))
    origin = origin_f + ((mid - origin_f) @ x_a) * x_a
    R_a = np.array([x_a, y_a, z_a])
    notes = [f"эстетическая плоскость по КТ: средняя линия — по симметрии лицевого скелета (СКО {rms:.1f} мм); "
             f"к функциональной системе — крен {roll:+.1f}°, разворот {yaw:+.1f}°; наклон — от функциональной"]
    return Aesthetic(rigid(R_a, -R_a @ origin), round(roll, 3), None, None, {"yaw_deg": round(yaw, 3)}, notes)


# Размеры передних зубов по лицу — ориентиры (средние соотношения из литературы), не правило: подсказка к выбору
# библиотеки зубов в Smile Design, где масштаб фото подбирается вручную.
TOOTH_GUIDES = {
    "central_by_pupils": (1 / 6.6, "межзрачковое / 6,6 (Cesario, Latta, 1984)"),
    "centrals_by_canthi": (0.618, "две центральные — между внутренними углами глаз × 0,618 (Abdullah, 2002)"),
    "canines_by_alae": (1.0, "между буграми клыков по прямой — ширина крыльев носа (Hoffman и др., 1986)"),
}
CENTRAL_WIDTH_TO_LENGTH = 0.78  # ширина центрального резца к длине — 75–80 %
# Глубина точек лица за резцовой точкой по взгляду камеры (средняя пропорция профиля): зрачки и углы глаз — глубже,
# крылья носа — ближе. Ошибка масштаба ≈ ошибка глубины / расстояние до камеры (10 мм на 300 мм — 3 %).
FACE_DEPTH_MM = {"pupil": 18.0, "endocanthion": 16.0, "canthus": 20.0, "ala": 6.0, "commissure": 0.0}
FACE_WIDTHS = {"межзрачковое": "pupil", "между внутренними углами глаз": "endocanthion",
               "между наружными углами глаз": "canthus", "крыльев носа": "ala", "рта": "commissure"}


def face_widths_mm(camera: Camera, face: dict, incisal_point, depth_mm: dict | None = None) -> dict:
    """Расстояния по лицу в мм по фото с камерой совмещения (fit_camera): точки лица (face_hints или вручную,
    пиксели) — на лучах камеры, на глубине FACE_DEPTH_MM за резцовой точкой по взгляду камеры. Возвращает
    {название: мм} для пар, отмеченных на фото."""
    depth = {**FACE_DEPTH_MM, **(depth_mm or {})}
    view = camera.R.T @ np.array([0.0, 0.0, 1.0])
    inc = np.asarray(incisal_point, float)
    out = {}
    for name, key in FACE_WIDTHS.items():
        right, left = face.get(f"{key}_right"), face.get(f"{key}_left")
        if right is None or left is None:
            continue
        plane = inc + depth[key] * view
        pts = []
        for uv in (right, left):
            o, d = camera.ray(uv)
            pts.append(o + ((plane - o) @ view) / (d @ view) * d)
        out[name] = round(float(np.linalg.norm(pts[0] - pts[1])), 1)
    return out


def tooth_guide(widths: dict) -> dict:
    """Ориентиры размеров передних зубов по расстояниям лица (face_widths_mm), мм: ширина центрального резца — по
    межзрачковому и по внутренним углам глаз, длина — по ширине, между клыками — по крыльям носа; к каждому —
    откуда (TOOTH_GUIDES)."""
    out = {}
    if "межзрачковое" in widths:
        k, why = TOOTH_GUIDES["central_by_pupils"]
        out["ширина центрального резца (по зрачкам)"] = (round(widths["межзрачковое"] * k, 1), why)
    if "между внутренними углами глаз" in widths:
        k, why = TOOTH_GUIDES["centrals_by_canthi"]
        out["ширина центрального резца (по углам глаз)"] = (round(widths["между внутренними углами глаз"] * k / 2, 1),
                                                            why)
    centrals = [v for key, (v, _w) in out.items() if key.startswith("ширина центрального")]
    if centrals:
        w = float(np.mean(centrals))
        out["длина центрального резца"] = (round(w / CENTRAL_WIDTH_TO_LENGTH, 1),
                                           f"ширина {w:.1f} мм / {CENTRAL_WIDTH_TO_LENGTH:g}".replace(".", ","))
    if "крыльев носа" in widths:
        k, why = TOOTH_GUIDES["canines_by_alae"]
        out["между клыками (по прямой)"] = (round(widths["крыльев носа"] * k, 1), why)
    return out


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def face_hints(image: np.ndarray, model_path: str) -> dict:
    """Точки лица по MediaPipe Face Landmarker: зрачки, углы глаз, крылья носа, углы рта, срединные точки,
    контуры губ (lips_inner — край открытого рта, lips_outer), пиксели снимка; горизонт по линии глаз — eye_line_deg
    (как exocad Smile Creator: линия глаз — горизонталь кадра)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image

    h, w = image.shape[:2]
    k = min(1.0, FACE_DETECT_PX / max(w, h))
    small = image if k == 1.0 else np.asarray(Image.fromarray(np.ascontiguousarray(image[..., :3])).resize(
        (max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR))
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    try:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small[..., :3])))
    finally:
        landmarker.close()
    if not result.face_landmarks:
        raise ValueError("лицо на фото не найдено")
    lm = result.face_landmarks[0]

    def px(i):
        return (lm[i].x * w, lm[i].y * h)

    points = {name: px(i) for name, i in FACE_POINTS.items()}
    points["midline"] = [px(i) for i in MIDLINE_POINTS.values()]
    points["lips_inner"] = [px(i) for i in LIPS_INNER]
    points["lips_outer"] = [px(i) for i in LIPS_OUTER]
    points["eye_line_deg"] = horizon_from_points(points["pupil_right"], points["pupil_left"])
    return points
