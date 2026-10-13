"""Эстетический вид для exocad — набор изображений Smile Design (casedesigner/exocad_views.py, docs/exocad.md).

Проверено в exocad 3.3 (кейс 3): набор читается при загрузке проекта, кнопка «Вид Smile Design» ставит камеру по
ViewDirection фото улыбки; пустые числовые поля — только xsi:nil, иначе «ошибка в XML документе».
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from casedesigner import exocad_views as ev
from casedesigner.register import axis_angle

NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"


def vec(e):
    return np.array([float(e.findtext(k)) for k in "xyz"])


def tilted_axes(deg=8.0):
    """Эстетика передом вниз на deg (как к осям артикулятора у кейса 3 при оси-орбитальной гипсовке)."""
    R = axis_angle(np.array([1.0, 0, 0]), np.radians(-deg))
    return (R @ np.eye(3)).T  # строки: вправо, вперёд, вверх


def test_front_view_looks_back_along_aesthetic_forward():
    axes = tilted_axes()
    right, forward, up = axes
    anchor = np.array([1.0, 70.0, -50.0])
    g = ev.front_view(axes, anchor)
    assert g["view"] == pytest.approx(-forward)
    assert g["axis_x"] == pytest.approx(-right)  # правая сторона пациента — слева на экране
    assert g["axis_y"] == pytest.approx(up)
    assert g["center"] == pytest.approx(anchor - ev.DEPTH_MM * forward)  # фон — за головой
    c = g["corners"]
    assert c["TopLeft"] - c["TopRight"] == pytest.approx(ev.SIZE_MM[0] * right)
    assert c["TopLeft"] - c["BottomLeft"] == pytest.approx(ev.SIZE_MM[1] * up)
    for p in c.values():  # плоскость снимка — поперёк взгляда
        assert (p - g["center"]) @ g["view"] == pytest.approx(0, abs=1e-9)


def test_written_set_as_exocad_reads_it(tmp_path):
    names = ev.write_treatment_images(str(tmp_path), "case-KStom", np.eye(3), [0, 70, -50])
    assert names == ["case-KStom.treatmentImages", "case-KStom-smile.jpg"]
    raw = (tmp_path / names[0]).read_bytes()
    assert raw.startswith(b'<?xml version="1.0" encoding="utf-8"?>\r\n') and b"\r\n" in raw
    root = ET.fromstring(raw)
    assert root.tag == "TreatmentImages" and root.findtext("GeneratedForProjectName") == "case-KStom"
    images = root.findall("Images/Image")
    assert len(images) == 1
    im = images[0]
    assert im.findtext("Filename") == names[1] and im.findtext("ImageType") == "SmileImage"
    assert im.findtext("IsRegisteredToJawScanData") == "true"
    for tag in ("EyeLeftLocation", "EyeRightLocation", "FovFocalPoint", "DistanceToCamera"):
        assert im.find(tag).get(NIL) == "true" and not list(im.find(tag)) and not (im.find(tag).text or "").strip()
    assert vec(im.find("ViewDirection")) == pytest.approx([0, -1, 0])
    assert vec(im.find("ViewAxisX")) == pytest.approx([-1, 0, 0])
    assert vec(im.find("ViewAxisY")) == pytest.approx([0, 0, 1])
    assert vec(im.find("TopLeft")) == pytest.approx([120, 70 - ev.DEPTH_MM, -50 + 90])
    m = root.find("MatrixToScene")
    assert [[float(m.findtext(f"_{r}{c}")) for c in range(4)] for r in range(4)] == np.eye(4).tolist()
    with Image.open(tmp_path / names[1]) as img:
        assert img.size == (int(im.findtext("ImagePixelsX")), int(im.findtext("ImagePixelsY"))) == ev.PIXELS
