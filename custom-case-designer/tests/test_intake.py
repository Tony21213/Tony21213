"""Входной контроль кейса (casedesigner/intake.py): воксель и поле КТ, дыры сканов, прикус, фото, отчёт."""

import numpy as np

from casedesigner import intake, reports
from casedesigner.volume import Volume


def volume(spacing=0.3):
    return Volume(np.zeros((100, 120, 140), np.int16), np.full(3, spacing), np.zeros(3), np.eye(3))


def grid(n=30, hole=None):
    """Открытая сетка n×n с шагом 1 мм; hole — (i0, i1): квадрат граней, вырезанный в середине."""
    ij = np.array([(i, j) for j in range(n) for i in range(n)], float)
    v = np.c_[ij, np.zeros(len(ij))]
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            if hole and hole[0] <= i < hole[1] and hole[0] <= j < hole[1]:
                continue
            a = j * n + i
            faces += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    return v, np.array(faces)


def test_ct_voxel_and_field_of_view():
    vol = volume()
    landmarks = {"Co_R": np.array([20.0, 18, 15]), "Co_L": np.array([1.0, 18, 15]),
                 "Or_R": np.array([20.0, 18, 10]), "Or_L": np.array([22.0, 18, 10])}
    found = {f.title: f for f in intake.ct_findings(vol, landmarks)}
    assert found["Размер вокселя"].level == "ok"
    assert found["Мыщелки: правый"].level == "ok"
    left = found["Мыщелки: левый"]
    assert left.level == "error" and "у края" in left.text and left.image is not None  # срез через точку
    assert found["Уши (порионы): правый"].level == "warn" and "не найдена" in found["Уши (порионы): правый"].text
    assert intake.ct_findings(volume(0.35), {})[0].level == "warn"
    assert intake.ct_findings(volume(0.5), {})[0].level == "error"


def test_scan_holes_and_bite():
    v, f = grid(hole=(12, 16))
    holes = intake.holes(v, f)
    assert len(holes) == 1  # внешний край скана — не дыра
    lengths = np.linalg.norm(v[holes[0][:, 0]] - v[holes[0][:, 1]], axis=1).sum()
    assert lengths == np.float64(16.0)
    clean_v, clean_f = grid()
    assert intake.holes(clean_v, clean_f) == []
    found = intake.scan_findings({"upper": (v, f), "lower": (clean_v, clean_f)}, has_bite=False)
    by = {x.title: x for x in found}
    assert by["Скан верхней челюсти"].level == "warn" and by["Скан верхней челюсти"].image is not None
    assert by["Скан нижней челюсти"].level == "ok" and by["Прикус"].level == "warn"
    assert intake.scan_findings({}, has_bite=True)[0].level == "error"  # нет скана


def test_report_lists_missing_first_and_renders():
    findings = (intake.ct_findings(volume(), {"Co_R": np.array([1.0, 18, 15])})
                + intake.scan_findings({"upper": grid(hole=(12, 16))}, has_bite=True)
                + intake.photo_findings(smile=True, retractor=False))
    rep = intake.report(findings, "Кейс 7 · 14.10.2026")
    assert [s.title for s in rep.sections] == ["Чего не хватает", "КТ", "Сканы", "Фото"]
    assert rep.sections[0].level == "error" and any("Фото с ретрактором" in n for n in rep.sections[0].notes)
    assert len(rep.figures) == sum(f.image is not None for f in findings) == 2
    pages = reports.render(rep)
    assert pages and pages[0].size == reports.PAGE
