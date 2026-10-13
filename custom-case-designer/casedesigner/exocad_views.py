"""Эстетический вид для exocad: набор изображений Smile Design (``<проект>.treatmentImages``).

exocad пишет этот XML («Набор изображений XML (*.treatmentImages)») при закрытии Smile Design и сам читает его при
загрузке проекта. Если в наборе есть фото улыбки (``SmileImage``), под «Сохранить вид» появляется кнопка
«Вид Smile Design»: камера встаёт точно по направлению взгляда снимка (``ViewDirection``), ось экрана «вправо» —
``ViewAxisX`` (= взгляд × «вверх» экрана), «вверх» — ``ViewAxisY``; углы плоскости снимка ``TopLeft…BottomRight`` —
в мм сцены (проверено в exocad 3.3 на кейсе 3, docs/exocad.md, «Эстетический вид»).

Здесь снимок — светлая сетка с красной средней линией лица и синей эстетической горизонталью; он стоит за головой
(фон за моделями, перекрестье — на резцовой точке). Пустые числовые поля exocad читает только как
``xsi:nil="true"``; пустой ``imageHash`` принимает.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

SIZE_MM = (240.0, 180.0)  # ширина, высота снимка
PIXELS = (2400, 1800)
DEPTH_MM = 100.0  # за резцовой точкой по эстетическому «вперёд»: фон за головой
GRID_PX = 125  # 12,5 мм


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def front_view(axes, anchor, depth_mm: float = DEPTH_MM, size_mm=SIZE_MM) -> dict:
    """Геометрия фронтального снимка: axes — строки «вправо (пациента), вперёд, вверх» эстетической системы в мм
    сцены exocad, anchor — резцовая точка. Камера спереди смотрит назад: взгляд = −вперёд, ось экрана «вправо»
    = взгляд × вверх = −вправо пациента (правая сторона пациента — слева на экране, как на фото анфас)."""
    right, forward, up = (_unit(v) for v in np.asarray(axes, float))
    view = -forward
    axis_x = _unit(np.cross(view, up))
    c = np.asarray(anchor, float) - depth_mm * forward
    hw, hh = size_mm[0] / 2 * axis_x, size_mm[1] / 2 * up
    return {"view": view, "axis_x": axis_x, "axis_y": up, "center": c,
            "corners": {"TopLeft": c - hw + hh, "TopRight": c + hw + hh, "BottomLeft": c - hw - hh,
                        "BottomRight": c + hw - hh}}


def picture(path: str, pixels=PIXELS, label: str = "KStom: эстетика — средняя линия и горизонталь"):
    """Снимок-фон: сетка 12,5 мм, средняя линия (красная), горизонталь (синяя), круг — правая сторона пациента."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = pixels
    img = Image.new("RGB", (w, h), (236, 236, 231))
    d = ImageDraw.Draw(img)
    for x in range(0, w, GRID_PX):
        d.line([(x, 0), (x, h)], fill=(206, 206, 201), width=2)
    for y in range(0, h, GRID_PX):
        d.line([(0, y), (w, y)], fill=(206, 206, 201), width=2)
    d.line([(w // 2, 0), (w // 2, h)], fill=(220, 30, 30), width=8)
    d.line([(0, h // 2), (w, h // 2)], fill=(30, 90, 220), width=8)
    d.ellipse([(80, 80), (190, 190)], fill=(220, 30, 30))
    try:
        font = ImageFont.truetype("arial.ttf", 56)
    except OSError:
        font = ImageFont.load_default()
    d.text((230, 105), f"{label} (круг — правая сторона пациента)", fill=(60, 60, 60), font=font)
    img.save(path, quality=92)


def _vec(parent, tag, v):
    e = ET.SubElement(parent, tag)
    for k, x in zip("xyz", np.asarray(v, float)):
        ET.SubElement(e, k).text = repr(float(x))


def treatment_images_xml(image_file: str, geometry: dict, pixels=PIXELS, project: str | None = None,
                         display: str = "Эстетика KStom") -> bytes:
    """XML набора с одним фото улыбки (как его пишет exocad 3.3; лишние поля — пустые, ``xsi:nil``)."""
    root = ET.Element("TreatmentImages", {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                          "xmlns:xsd": "http://www.w3.org/2001/XMLSchema"})
    im = ET.SubElement(ET.SubElement(root, "Images"), "Image")
    ET.SubElement(im, "Filename").text = image_file
    ET.SubElement(im, "DisplayName").text = display
    for key, p in geometry["corners"].items():
        _vec(im, key, p)
    for tag, val in (("ImageType", "SmileImage"), ("AutodetectDepth", "false"),
                     ("ApplyImage2DToArticulatorCoordinateSystem", "false"), ("IsRegisteredToJawScanData", "true")):
        ET.SubElement(im, tag).text = val
    ET.SubElement(im, "BoundaryCurves")
    ET.SubElement(im, "EyeLeftLocation", {"xsi:nil": "true"})
    ET.SubElement(im, "EyeRightLocation", {"xsi:nil": "true"})
    ET.SubElement(im, "imageHash")
    _vec(im, "ViewDirection", geometry["view"])
    _vec(im, "ViewAxisX", geometry["axis_x"])
    _vec(im, "ViewAxisY", geometry["axis_y"])
    ET.SubElement(im, "FovFocalPoint", {"xsi:nil": "true"})
    ET.SubElement(im, "DistanceToCamera", {"xsi:nil": "true"})
    ET.SubElement(im, "ImagePixelsX").text = str(pixels[0])
    ET.SubElement(im, "ImagePixelsY").text = str(pixels[1])
    ET.SubElement(root, "Helpers")
    if project:
        ET.SubElement(root, "GeneratedForProjectName").text = project
    m = ET.SubElement(root, "MatrixToScene")
    for r in range(4):
        for c in range(4):
            ET.SubElement(m, f"_{r}{c}").text = "1" if r == c else "0"
    ET.indent(root, "  ")
    return b'<?xml version="1.0" encoding="utf-8"?>\r\n' + ET.tostring(root, encoding="utf-8").replace(b"\n", b"\r\n")


def write_treatment_images(folder: str, stem: str, axes, anchor) -> list[str]:
    """Записать ``<stem>.treatmentImages`` и снимок ``<stem>-smile.jpg`` в папку проекта; вернуть имена файлов."""
    geometry = front_view(axes, anchor)
    image = f"{stem}-smile.jpg"
    picture(os.path.join(folder, image))
    name = f"{stem}.treatmentImages"
    with open(os.path.join(folder, name), "wb") as f:
        f.write(treatment_images_xml(image, geometry, project=stem))
    return [name, image]
