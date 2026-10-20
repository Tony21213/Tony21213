"""Камера фото по самому снимку (casedesigner/photo_camera.py): подбор крена, наклона и фокуса по полям GeoCalib,
фокус и сдвиг по карте точек MoGe-2 — на синтетике; сети ONNX — если есть в models/."""

import os

import numpy as np
import pytest

from casedesigner import photo_camera as pc

MODELS = os.environ.get("CCD_MODELS", os.path.join(os.path.dirname(__file__), "..", "models"))


def test_fields_give_back_roll_pitch_focal():
    """Поля идеальной камеры (крен 4°, наклон −6°, фокус 700 px) → тот же крен, наклон и фокус."""
    w, h = 320, 480
    roll, pitch, f = np.radians(4.0), np.radians(-6.0), 700.0
    up, lat = pc.perspective_fields(roll, pitch, f, w, h)
    conf = np.ones((h, w))
    r, p, f2, sigma = pc.calibrate_fields(up.transpose(2, 0, 1), conf, lat[None], conf)
    assert np.degrees(r) == pytest.approx(4.0, abs=0.05) and np.degrees(p) == pytest.approx(-6.0, abs=0.05)
    assert f2 == pytest.approx(700.0, rel=0.01) and sigma is not None


def test_up_field_points_up_and_horizon_sign():
    """Без крена и наклона «вверх» — к верху кадра; крен + — горизонт в кадре под −крен (horizon_deg)."""
    up, lat = pc.perspective_fields(0.0, 0.0, 500.0, 64, 48)
    assert np.allclose(up[24, 32], [0, -1], atol=1e-6) and abs(lat[24, 32]) < 1e-6
    g = pc.gravity_from_rp(np.radians(10), 0.0)  # «вверх» в кадре наклонён влево
    assert g[0] < 0 and g[1] < 0


def test_moge_focal_shift_from_point_map():
    """Карта точек с точностью до сдвига глубины: фокус и сдвиг восстанавливаются (как recover_focal_shift авторов)."""
    h, w = 60, 80
    uv = pc.view_plane_uv(w, h)
    focal, depth = 1.7, 2.0 + 0.3 * np.random.default_rng(0).random((h, w))
    xy = uv * depth[..., None] / focal
    points = np.concatenate([xy, (depth - 0.8)[..., None]], -1)  # сдвиг 0,8 — сеть его не знает
    f, s = pc.recover_focal_shift(points, np.ones((h, w), bool))
    assert f == pytest.approx(focal, rel=1e-3) and s == pytest.approx(0.8, abs=1e-3)


def test_geocalib_input_is_multiple_of_32():
    img = np.zeros((6240, 4160, 3), np.uint8)
    x, (sx, sy) = pc.geocalib_input(img)
    assert x.shape[:2] == (1, 3) and x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0 and min(x.shape[2:]) == 320
    assert sx == pytest.approx(320 / 4160, rel=1e-3)


def test_estimate_without_models_says_why(tmp_path):
    lens = pc.estimate(np.zeros((100, 80, 3), np.uint8), str(tmp_path))
    assert lens.focal_px is None and lens.distance_mm is None and len(lens.notes) == 2


@pytest.mark.skipif(not (pc.geocalib_available(MODELS) and pc.moge_available(MODELS)), reason="нет GeoCalib и MoGe-2 в models/")
def test_networks_on_a_synthetic_room():
    """Сети ONNX работают: «комната» с полом и стенами — вертикаль без крена, фокус и глубина конечны."""
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[yy > h * 0.6] = (120, 90, 60)  # пол
    img[yy <= h * 0.6] = (200, 200, 190)  # стена
    img[(xx // 40) % 2 == 0] //= 2  # вертикальные полосы — дверные косяки
    lens = pc.estimate(img, MODELS)
    assert lens.focal_px and lens.focal_px > 0 and abs(lens.roll_deg) < 10 and lens.distance_mm and lens.distance_mm > 0
