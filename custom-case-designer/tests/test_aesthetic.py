"""Эстетическая система по фото анфас — на синтетическом снимке с известной камерой."""

import os

import numpy as np
import pytest

from casedesigner import aesthetic as ae
from casedesigner.register import apply, axis_angle

# Верхние зубы в функциональной системе (x — вправо пациента, y — вперёд, z — вверх), мм.
TEETH = {11: (4.3, 0, 0), 21: (-4.3, 0, 0), 12: (11, -2, 0.6), 22: (-11, -2, 0.6), 13: (17, -6, 0.3),
         23: (-17, -6, 0.3), 14: (21, -12, 1.0), 24: (-21, -12, 1.0), 16: (24, -25, 1.5), 26: (-24, -25, 1.5)}
POINTS = np.array(list(TEETH.values()), float)
INCISAL = np.zeros(3)
SIZE = (4000, 3000)
F_PX = 3000.0


def scene(roll_deg=3.0, yaw_deg=2.0, pitch_deg=-5.0, face_offset=1.5, face_tilt_deg=0.0, noise=0.2, seed=0):
    """Голова наклонена к истинной горизонтали на roll_deg (правая сторона выше), камера стоит по уровню."""
    rng = np.random.default_rng(seed)
    Q = axis_angle(np.array([0, 1.0, 0]), np.radians(roll_deg))  # истинные оси в системе головы
    x_t, z_t = Q @ [1.0, 0, 0], Q @ [0, 0, 1.0]
    y_t = np.array([0, 1.0, 0])
    yaw, pitch = axis_angle(z_t, np.radians(yaw_deg)), None
    cam_x, cam_y, cam_z = yaw @ -x_t, yaw @ -z_t, yaw @ -y_t  # спереди: вправо по кадру — влево пациента
    pitch = axis_angle(cam_x, np.radians(pitch_deg))
    R = np.array([cam_x, pitch @ cam_y, pitch @ cam_z])
    centre = INCISAL - 300.0 * R[2] + [0, 0, 0]
    cam = ae.Camera(np.array([[F_PX, 0, SIZE[0] / 2], [0, F_PX, SIZE[1] / 2], [0, 0, 1]]), R, -R @ centre, 0.0,
                    size=SIZE)
    teeth_px = cam.project(POINTS) + rng.normal(0, noise, (len(POINTS), 2))
    heights = np.array([70.0, 45, 22, 12, 6, -18, -42])  # глабелла … подбородок, мм от резцов
    depth = np.array([5.0, 8, 20, 15, 12, 10, 8])
    face_mid = INCISAL + face_offset * x_t + heights[:, None] * z_t + depth[:, None] * y_t \
        + (heights * np.tan(np.radians(face_tilt_deg)))[:, None] * x_t
    face = {"pupil_right": INCISAL + 31 * x_t + 75 * z_t + 1.0 * z_t, "pupil_left": INCISAL - 31 * x_t + 75 * z_t,
            "commissure_right": INCISAL + 24 * x_t - 2 * z_t, "commissure_left": INCISAL - 24 * x_t - 2 * z_t}
    face_px = {k: cam.project(np.asarray(v)[None])[0] for k, v in face.items()}
    return cam, teeth_px, cam.project(face_mid), face_px


def test_camera_from_teeth_points():
    cam, px, _mid, _face = scene()
    fit = ae.fit_camera(px, POINTS, SIZE, np.eye(4), focal_px=F_PX)
    angle = np.degrees(np.arccos(np.clip((np.trace(fit.R.T @ cam.R) - 1) / 2, -1, 1)))
    assert angle < 0.3 and fit.residual_px < 1.0 and not fit.mirrored


def test_aesthetic_frame_roll_midline_and_hints():
    cam, px, mid, face = scene(roll_deg=3.0, face_offset=1.5)
    fit = ae.fit_camera(px, POINTS, SIZE, np.eye(4), focal_px=F_PX)
    a = ae.aesthetic_frame(np.eye(4), fit, midline_px=mid, incisal_point=INCISAL, face=face)
    assert a.roll_deg == pytest.approx(3.0, abs=0.15)  # функциональная горизонталь: правая сторона выше на 3°
    assert a.midline_offset_mm == pytest.approx(-1.5, abs=0.1)  # зубы на 1.5 мм левее средней линии лица
    assert a.facial_midline_tilt_deg == pytest.approx(0.0, abs=0.3)
    # подсказки — углы на снимке: при голове, развёрнутой к камере на 2°, перспектива меняет их на доли градуса
    assert 0.3 < a.hints["зрачковая"] < 1.3  # правый зрачок выше на 1 мм на 62 мм — около 0.9°
    assert a.hints["углов рта"] == pytest.approx(0.0, abs=0.3)
    cam0, px0, _m, face0 = scene(yaw_deg=0.0, pitch_deg=0.0)
    straight = ae.aesthetic_frame(np.eye(4), ae.fit_camera(px0, POINTS, SIZE, np.eye(4), focal_px=F_PX), face=face0)
    assert straight.hints["зрачковая"] == pytest.approx(np.degrees(np.arctan2(1.0, 62)), abs=0.1)
    # эстетическая система: x — по истинному горизонту, y — как в функциональной, начало — на средней линии лица
    R = a.frame[:3, :3]
    assert R[1] @ [0, 1, 0] == pytest.approx(1.0)
    assert apply(a.frame, (INCISAL + 1.5 * (R[0]))[None])[0][0] == pytest.approx(0.0, abs=0.15)


def test_tilted_facial_midline_and_mirrored_photo():
    cam, px, mid, face = scene(face_tilt_deg=2.0)
    a = ae.aesthetic_frame(np.eye(4), ae.fit_camera(px, POINTS, SIZE, np.eye(4), focal_px=F_PX),
                           midline_px=mid, incisal_point=INCISAL, face=face)
    assert a.facial_midline_tilt_deg == pytest.approx(2.0, abs=0.3)  # верх средней линии — вправо пациента

    def mirror(p):
        p = np.asarray(p, float)
        return np.c_[SIZE[0] - p[..., 0], p[..., 1]].reshape(p.shape)

    fit = ae.fit_camera(mirror(px), POINTS, SIZE, np.eye(4), focal_px=F_PX)
    assert fit.mirrored
    b = ae.aesthetic_frame(np.eye(4), fit, midline_px=mirror(mid), incisal_point=INCISAL,
                           face={k: mirror(v) for k, v in face.items()})
    assert b.roll_deg == pytest.approx(a.roll_deg, abs=0.15)
    assert b.midline_offset_mm == pytest.approx(a.midline_offset_mm, abs=0.1)
    assert b.facial_midline_tilt_deg == pytest.approx(a.facial_midline_tilt_deg, abs=0.3)
    assert b.hints["зрачковая"] == pytest.approx(a.hints["зрачковая"], abs=0.2)


def test_level_line_in_the_photo():
    """Камера завалена на 4°, но в кадре виден уровень: горизонт по нему, крен — тот же."""
    cam, px, mid, face = scene(roll_deg=3.0)
    tilt = axis_angle(np.array([0, 0, 1.0]), np.radians(4.0))  # поворот кадра вокруг оси взгляда
    cam4 = ae.Camera(cam.K, tilt @ cam.R, tilt @ cam.t, 0.0, size=SIZE)
    level = cam4.project(np.array([[-40.0, 0, 30], [40.0, 0, 30]]) @ np.eye(3).T)  # линия по истинной горизонтали
    Q = axis_angle(np.array([0, 1.0, 0]), np.radians(3.0))
    level = cam4.project(np.array([30 * (Q @ [1.0, 0, 0]), -30 * (Q @ [1.0, 0, 0])]) + [0, 0, 30])
    fit = ae.fit_camera(cam4.project(POINTS), POINTS, SIZE, np.eye(4), focal_px=F_PX)
    a = ae.aesthetic_frame(np.eye(4), fit, horizon_deg=ae.horizon_from_points(*level))
    assert a.roll_deg == pytest.approx(3.0, abs=0.15)
    assert ae.horizon_from_vertical((0, 0), (0, 100)) == pytest.approx(0.0)


def test_photo_exif_orientation_and_focal(tmp_path):
    from PIL import Image

    im = Image.new("RGB", (400, 300), (200, 180, 170))
    exif = Image.Exif()
    exif[0x0112] = 6  # повёрнуто: телефон держали вертикально
    exif.get_ifd(0x8769)[0xA405] = 26  # 26 мм в 35-мм эквиваленте
    path = tmp_path / "face.jpg"
    im.save(path, exif=exif)
    photo = ae.load_photo(str(path))
    assert photo.size == (300, 400)
    assert photo.focal_px == pytest.approx(26 * 500 / ae.FULL_FRAME_DIAGONAL_MM, rel=1e-6)


def test_too_few_points():
    cam, px, _mid, _face = scene()
    with pytest.raises(ValueError, match="6 пар"):
        ae.fit_camera(px[:5], POINTS[:5], SIZE, np.eye(4))


MODEL = os.environ.get("CCD_FACE_MODEL", os.path.join(os.path.dirname(__file__), "..", "models",
                                                      "face_landmarker.task"))


@pytest.mark.skipif(not os.path.isfile(MODEL), reason="нет модели MediaPipe face_landmarker.task")
def test_face_hints_on_a_public_domain_portrait():
    pytest.importorskip("mediapipe")
    from skimage import data

    image = data.astronaut()  # фото NASA, общественное достояние
    face = ae.face_hints(image, MODEL)
    assert face["pupil_right"][0] < face["pupil_left"][0]  # анфас: правый глаз пациента — слева на снимке
    mid = np.asarray(face["midline"])
    assert np.ptp(mid[:, 0]) < 0.25 * np.ptp(mid[:, 1])  # срединные точки — почти вертикаль
