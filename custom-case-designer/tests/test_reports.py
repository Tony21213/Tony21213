"""Генератор отчётов (casedesigner/reports.py): страницы A4, PNG и PDF, паспорт кейса без имени пациента."""

import pytest
from PIL import Image

from casedesigner import reports

EXPORT = {  # сводка exocad_articulator_export.export: имена файлов и папка — с именем пациента
    "files": {"проект": "1488_Иванов_И.И-KStom.dentalProject"}, "folder": r"Y:\CAD\1488_Иванов_И.И-KStom",
    "articulator": {"TiltCondylarGuideRight": 37.7, "TiltCondylarGuideLeft": 44.0, "BennettAngleRight": 16.7,
                    "BennettAngleLeft": 17.5, "ImmediateSideshiftRight": 0.0, "ImmediateSideshiftLeft": 0.2},
    "icd_mm": 101.3, "project_parameters": {"TiltCondylarGuideRight": 37.7, "TiltCondylarGuideLeft": 44.0},
    "inserts": {"right": {"insert": "Изгиб 1 мм", "tilt_deg": 41.2, "rms_mm": 0.05},
                "left": {"insert": "Прямая", "tilt_deg": 44.0, "rms_mm": 0.08}},
    "notes": ["мыщелки несимметричны: ось наклонена на 1,6°", r"скопируйте Y:\CAD\1488_Иванов_И.И-KStom\x один раз"],
}


def _text(report):
    parts = [report.title, report.case_label]
    for s in report.sections:
        parts += [s.title, *s.notes, *[f"{a} {b}" for a, b in s.rows]]
    return " ".join(parts)


def test_passport_from_export_without_patient_data():
    rep = reports.passport(EXPORT, "Кейс 4 · 14.10.2026", mount={"plane": "Эстетическая через мыщелки",
                                                                  "aesthetic_deg": 1.6},
                           condyles={"right": "подтверждён врачом", "left": "автоматически"},
                           bite={"shift_mm": 0.31, "rotation_deg": 0.4})
    text = _text(rep)
    assert "Иванов" not in text and "1488" not in text and "Y:" not in text  # ни имён файлов, ни путей
    assert "справа 37,7° · слева 44,0°" in text and "101,3 мм" in text
    assert "«Изгиб 1 мм», 41,2°" in text and "Эстетическая через мыщелки" in text and "0,31 мм" in text
    assert [s.title for s in rep.sections] == ["Артикулятор", "Монтаж", "Мыщелки", "Исправление прикуса",
                                               "Предупреждения"]
    assert rep.sections[-1].notes == ["мыщелки несимметричны: ось наклонена на 1,6°"]  # заметка с путём отброшена


def test_render_pages_and_save_png_pdf(tmp_path):
    rep = reports.passport(EXPORT, "Кейс 4 · 14.10.2026")
    rep.sections.append(reports.Section("Длинный раздел", rows=[(f"Параметр {i}", "значение " * 12) for i in range(60)]))
    rep.figures = [reports.Figure("Модели спереди", Image.new("RGB", (800, 500), (200, 180, 160))),
                   reports.Figure("Артикулятор с осью", Image.new("RGB", (640, 640), (120, 140, 200)))]
    pages = reports.render(rep)
    assert len(pages) >= 2 and all(p.size == reports.PAGE for p in pages)
    title = pages[0].crop((reports.MARGIN, reports.MARGIN, reports.MARGIN + 400, reports.MARGIN + 50)).convert("L")
    assert min(title.getdata()) < 100  # заголовок нарисован
    names = reports.save(rep, str(tmp_path), "passport")
    assert names == ["passport.png", "passport.pdf"]
    assert Image.open(tmp_path / "passport.png").size == reports.PAGE
    pdf = (tmp_path / "passport.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages") == len(pages)


def test_no_cyrillic_font_is_an_error(monkeypatch):
    monkeypatch.setattr(reports, "_font_paths", lambda bold: [])
    with pytest.raises(RuntimeError, match="шрифта"):
        reports.render(reports.Report("Паспорт кейса", "Кейс 1"))
