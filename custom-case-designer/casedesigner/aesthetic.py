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

Фото привязывается к зубам верхнего скана по 4+ парам точек «фото — скан»
(режущие края, бугры клыков): по ним находится положение камеры
(`fit_camera`), и направления кадра переносятся на модели.

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
LIP_DEPTH_MM = 10.0  # средняя линия лица на высоте резцов лежит на губе — примерно на 10 мм впереди режущего края

# Точки MediaPipe Face Landmarker (478): зрачки, углы глаз, крылья носа, углы рта, срединные точки.
FACE_POINTS = {"pupil_right": 468, "pupil_left": 473, "canthus_right": 33, "canthus_left": 263,
               "ala_right": 64, "ala_left": 294, "commissure_right": 61, "commissure_left": 291}
MIDLINE_POINTS = {"glabella": 9, "nasion": 168, "pronasale": 1, "subnasale": 2, "labrale_superius": 0,
                  "labrale_inferius": 17, "menton": 152}
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
    else:
        notes.append("в EXIF нет фокусного расстояния — камера будет подобрана с фокусом (нужно 6+ пар точек)")
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


def fit_camera(points_px, points_3d, size, base_frame: np.ndarray, focal_px: float | None = None) -> Camera:
    """Положение камеры по парам точек «фото — скан».

    points_px — пиксели на фото (x вправо, y вниз), points_3d — те же точки на
    верхнем скане (координаты кейса); base_frame — функциональная система (для
    начального приближения: камера спереди). Фокус — из EXIF; если неизвестен,
    подбирается (нужно 6+ пар). Зеркальное фото (фронтальная камера)
    распознаётся по ошибке.
    """
    px = np.asarray(points_px, float)
    pts = np.asarray(points_3d, float)
    if len(px) != len(pts) or len(px) < 4:
        raise ValueError("нужно минимум 4 пары точек «фото — скан»")
    fit_focal = focal_px is None
    if fit_focal and len(px) < 6:
        raise ValueError("фокусное расстояние неизвестно — нужно минимум 6 пар точек или фото с EXIF")
    w, h = size
    focal = focal_px or DEFAULT_FOCAL_35MM * float(np.hypot(w, h)) / FULL_FRAME_DIAGONAL_MM
    R0 = _front_rotation(base_frame)
    direct = _fit(px, pts, size, focal, R0, fit_focal)
    flipped = _fit(np.c_[w - px[:, 0], px[:, 1]], pts, size, focal, R0, fit_focal)
    if flipped.residual_px < MIRROR_GAIN * direct.residual_px:
        flipped.mirrored = True
        return flipped
    return direct


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


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def face_hints(image: np.ndarray, model_path: str) -> dict:
    """Точки лица по MediaPipe Face Landmarker: зрачки, углы глаз, крылья носа, углы рта и срединные точки (пиксели)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    try:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)))
    finally:
        landmarker.close()
    if not result.face_landmarks:
        raise ValueError("лицо на фото не найдено")
    lm = result.face_landmarks[0]
    h, w = image.shape[:2]

    def px(i):
        return (lm[i].x * w, lm[i].y * h)

    points = {name: px(i) for name, i in FACE_POINTS.items()}
    points["midline"] = [px(i) for i in MIDLINE_POINTS.values()]
    return points
