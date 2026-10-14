"""Фото пациента в наборе Smile Design по камере совмещения (exocad_views.photo_view, write_photo_images; «Эстетика»,
часть 1): в ортогональном виде exocad по ViewDirection фото ложится на модели на глубине резцовой точки."""

import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from casedesigner import aesthetic as ae
from casedesigner import exocad_views as ev
from casedesigner.register import apply, rigid

W, H = 1200, 800


def camera(mirrored=False, turn_deg=8.0):
    """Камера спереди, в 300 мм, слегка развёрнута и наклонена: x кадра — влево пациента, y — вниз, взгляд — назад."""
    R0 = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
    R = R0 @ Rotation.from_euler("zx", [turn_deg, -5], degrees=True).as_matrix()
    centre = R.T @ np.array([0.0, 0, -300.0])
    K = np.array([[2400.0, 0, W / 2], [0, 2400.0, H / 2], [0, 0, 1]])
    return ae.Camera(K, R, -R @ centre, 0.3, mirrored=mirrored, size=(W, H))


def on_photo(geometry, uv):
    c = geometry["corners"]
    u, v = uv[0] / W, uv[1] / H
    return c["TopLeft"] + u * (c["TopRight"] - c["TopLeft"]) + v * (c["BottomLeft"] - c["TopLeft"])


def test_photo_overlays_models_at_incisal_depth():
    cam, anchor = camera(), np.array([2.0, 5.0, -3.0])
    g = ev.photo_view(cam, anchor)
    view = g["view"]
    assert np.allclose(np.cross(view, g["axis_y"]), g["axis_x"]) and abs(view @ g["axis_y"]) < 1e-9
    # точки на плоскости резцов поперёк взгляда: их пиксель на фото — ровно за ними по взгляду
    a, b = np.cross(view, [0, 0, 1.0]), None
    a /= np.linalg.norm(a)
    b = np.cross(view, a)
    for s, t in ((0, 0), (20, 0), (-15, 10), (8, -12)):
        p = anchor + s * a + t * b
        q = on_photo(g, cam.project(p[None])[0])
        d = q - p
        assert np.linalg.norm(d - (d @ view) * view) < 1e-6 and d @ view > 0  # фото — за моделями
    # углы — прямоугольник: верх и низ параллельны оси экрана «вправо»
    c = g["corners"]
    top = c["TopRight"] - c["TopLeft"]
    assert np.allclose(top / np.linalg.norm(top), g["axis_x"], atol=1e-9)


def test_scene_transform_and_mirrored_photo(tmp_path):
    cam = camera(mirrored=True)
    anchor = np.array([0.0, 0, 0])
    to_scene = rigid(Rotation.from_euler("z", 30, degrees=True).as_matrix(), [30, -80, 60])
    plain = ev.photo_view(cam, anchor)
    g = ev.photo_view(cam, anchor, to_scene)
    for key in plain["corners"]:
        assert np.allclose(g["corners"][key], apply(to_scene, plain["corners"][key][None])[0])
    image = np.zeros((H, W, 3), np.uint8)
    image[:, : W // 2] = (250, 20, 20)  # левая половина — красная
    names = ev.write_photo_images(str(tmp_path), "case", [
        {"kind": "smile", "image": image, "camera": cam, "anchor": anchor},
        {"kind": "retracted", "image": image, "camera": camera(), "anchor": anchor}], to_scene)
    assert names == ["case.treatmentImages", "case-smile.jpg", "case-retracted.jpg"]
    smile = np.asarray(Image.open(tmp_path / "case-smile.jpg"))
    assert smile[H // 2, W - 10, 0] > 200 and smile[H // 2, 10, 0] < 60  # зеркальное — отражено обратно
    root = ET.parse(tmp_path / "case.treatmentImages").getroot()
    images = root.findall("Images/Image")
    assert [i.findtext("ImageType") for i in images] == ["SmileImage", "SmileImageRetracted"]
    assert [i.findtext("Filename") for i in images] == ["case-smile.jpg", "case-retracted.jpg"]
    assert images[0].findtext("ImagePixelsX") == str(W)
    view = np.array([float(images[0].findtext(f"ViewDirection/{k}")) for k in "xyz"])
    assert np.allclose(view, g["view"])
