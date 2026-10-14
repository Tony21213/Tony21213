"""Отчёты со скриншотами — общий генератор (порядок работ владельца, п. 9): паспорт кейса при выгрузке, затем
входной контроль и отчёт цефалометрии на том же механизме.

Контракт:

* ``Report`` — заголовок, подпись кейса (номер и дата, **без имени пациента**), разделы «параметр — значение» с
  заметками и картинки с подписями. ``render`` — страницы A4 (150 точек на дюйм, PIL), ``save`` — PNG первой
  страницы (картинка паспорта) и PDF всех страниц.
* Имён пациентов в отчёте нет: подпись кейса задаёт вызывающий (номер кейса или дата), сводки выгрузок берутся по
  ролям и числам — имена файлов и пути (в них бывает имя пациента) в отчёт не попадают; заметка с путём
  отбрасывается (``_clean``).
* Шрифт с кириллицей: DejaVu Sans из matplotlib, иначе Segoe UI / Arial (Windows) или DejaVu (Linux); нет ни одного —
  ``RuntimeError``.
* ``passport`` — паспорт кейса из сводки ``exocad_articulator_export.export`` и сведений сессии (монтаж, мыщелки,
  прикус); картинки (модели с фото спереди, артикулятор с осью) передаёт вызывающий.
"""

import os
import re
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

PAGE = (1240, 1754)  # A4, 150 точек на дюйм
DPI = 150
MARGIN = 70
COLORS = {"text": (30, 34, 40), "muted": (96, 104, 114), "line": (210, 214, 220), "info": (58, 143, 214),
          "warn": (214, 150, 40), "error": (200, 60, 60)}
PATH_RE = re.compile(r"[A-Za-z]:[\\/]|\\\\")  # путь Windows или UNC


@dataclass
class Section:
    title: str
    rows: list = field(default_factory=list)  # (параметр, значение)
    notes: list = field(default_factory=list)  # предупреждения и пояснения раздела
    level: str = "info"  # info / warn / error — цвет полосы раздела


@dataclass
class Figure:
    caption: str
    image: Image.Image


@dataclass
class Report:
    title: str
    case_label: str  # «Кейс 4 · 14.10.2026» — без имени пациента
    sections: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    footer: str = "KStom Case Designer"


def _font_paths(bold: bool) -> list[str]:
    paths = []
    try:
        import matplotlib

        paths.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf",
                                  "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    except ImportError:
        pass
    windows = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    paths += [os.path.join(windows, "segoeuib.ttf" if bold else "segoeui.ttf"),
              os.path.join(windows, "arialbd.ttf" if bold else "arial.ttf"),
              "/usr/share/fonts/truetype/dejavu/" + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    return paths


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in _font_paths(bold):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    raise RuntimeError("нет шрифта с кириллицей для отчёта (DejaVu Sans, Segoe UI или Arial)")


def _wrap(text: str, f, width: float) -> list[str]:
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}".strip()
            if f.getlength(trial) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


class _Pages:
    def __init__(self, report: Report):
        self.report, self.pages = report, []
        self.f = {"title": font(40, True), "label": font(24), "h": font(28, True), "row": font(22),
                  "rowb": font(22, True), "small": font(18)}
        self.new_page()

    def new_page(self):
        self.page = Image.new("RGB", PAGE, "white")
        self.draw = ImageDraw.Draw(self.page)
        self.pages.append(self.page)
        y = MARGIN
        if len(self.pages) == 1:
            self.draw.text((MARGIN, y), self.report.title, font=self.f["title"], fill=COLORS["text"])
            y += 56
            self.draw.text((MARGIN, y), self.report.case_label, font=self.f["label"], fill=COLORS["muted"])
            y += 44
            self.draw.line((MARGIN, y, PAGE[0] - MARGIN, y), fill=COLORS["line"], width=2)
            y += 24
        self.y = y
        self.footer()

    def footer(self):
        text = f"{self.report.footer} · {self.report.case_label} · стр. {len(self.pages)}"
        self.draw.text((MARGIN, PAGE[1] - MARGIN + 20), text, font=self.f["small"], fill=COLORS["muted"])

    def room(self, height: int):
        if self.y + height > PAGE[1] - MARGIN:
            self.new_page()

    def section(self, s: Section):
        width = PAGE[0] - 2 * MARGIN
        self.room(90)
        color = COLORS.get(s.level, COLORS["info"])
        self.draw.rectangle((MARGIN, self.y + 4, MARGIN + 8, self.y + 34), fill=color)
        self.draw.text((MARGIN + 22, self.y), s.title, font=self.f["h"], fill=COLORS["text"])
        self.y += 48
        label_w = int(width * 0.45)
        for label, value in s.rows:
            left = _wrap(label, self.f["row"], label_w - 20)
            right = _wrap(value, self.f["rowb"], width - label_w)
            h = 30 * max(len(left), len(right))
            self.room(h)
            for i, line in enumerate(left):
                self.draw.text((MARGIN + 22, self.y + 30 * i), line, font=self.f["row"], fill=COLORS["muted"])
            for i, line in enumerate(right):
                self.draw.text((MARGIN + label_w, self.y + 30 * i), line, font=self.f["rowb"], fill=COLORS["text"])
            self.y += h + 4
        for note in s.notes:
            lines = _wrap(note, self.f["row"], width - 60)
            self.room(30 * len(lines))
            self.draw.text((MARGIN + 22, self.y), "!", font=self.f["rowb"], fill=COLORS["warn"])
            for i, line in enumerate(lines):
                self.draw.text((MARGIN + 50, self.y + 30 * i), line, font=self.f["row"], fill=COLORS["text"])
            self.y += 30 * len(lines) + 4
        self.y += 20

    def figures(self, figures: list):
        width = PAGE[0] - 2 * MARGIN
        cell = (width - 30) // 2
        for i in range(0, len(figures), 2):
            row = figures[i:i + 2]
            scaled = []
            for fig in row:
                im = fig.image.convert("RGB")
                k = cell / im.width
                scaled.append((fig, im.resize((cell, max(1, round(im.height * k))))))
            h = max(im.height for _f, im in scaled) + 40
            self.room(h)
            for j, (fig, im) in enumerate(scaled):
                x = MARGIN + j * (cell + 30)
                self.page.paste(im, (x, self.y))
                self.draw.rectangle((x, self.y, x + im.width - 1, self.y + im.height - 1), outline=COLORS["line"])
                self.draw.text((x, self.y + im.height + 8), fig.caption, font=self.f["small"], fill=COLORS["muted"])
            self.y += h + 10


def render(report: Report) -> list[Image.Image]:
    """Страницы отчёта (A4, PIL): заголовок и подпись кейса, разделы, картинки по две в ряд, нижний колонтитул."""
    pages = _Pages(report)
    for s in report.sections:
        pages.section(s)
    pages.figures(report.figures)
    return pages.pages


def save(report: Report, folder: str, stem: str = "passport") -> list[str]:
    """PNG первой страницы и PDF всех страниц в folder; возвращает имена файлов. stem задаёт программа (не имя
    пациента)."""
    pages = render(report)
    os.makedirs(folder, exist_ok=True)
    png, pdf = f"{stem}.png", f"{stem}.pdf"
    pages[0].save(os.path.join(folder, png))
    pages[0].save(os.path.join(folder, pdf), save_all=True, append_images=pages[1:], resolution=DPI)
    return [png, pdf]


def _clean(notes) -> list[str]:
    """Заметки без путей: в пути бывает имя пациента."""
    return [str(n) for n in notes or [] if not PATH_RE.search(str(n))]


def _num(v, unit: str = "", digits: int = 1) -> str:
    return "—" if v is None else f"{float(v):.{digits}f}".replace(".", ",") + unit


def _sides(values: dict, key: str, unit: str) -> str:
    return f"справа {_num(values.get(key + 'Right'), unit)} · слева {_num(values.get(key + 'Left'), unit)}"


def passport(export: dict, case_label: str, mount: dict | None = None, condyles: dict | None = None,
             bite: dict | None = None, figures: list | None = None) -> Report:
    """Паспорт кейса: что сделано на выходе — для техника (параметры прямо из проекта exocad), клиники
    (согласование плана) и архива (с какими настройками делали).

    export — сводка exocad_articulator_export.export (берутся только числа и заметки: articulator, icd_mm, inserts,
    project_parameters, notes); mount — {"plane": название, "aesthetic_deg": расхождение с эстетикой, "aesthetic_mm"};
    condyles — {"right"/"left": источник точки — «подтверждён врачом», «автоматически»…}; bite — {"shift_mm",
    "rotation_deg"} исправления прикуса; figures — Figure (модели с фото спереди, артикулятор с осью)."""
    values = export.get("articulator") or {}
    sections = []
    rows = [("Угол суставного пути (ССП)", _sides(values, "TiltCondylarGuide", "°")),
            ("Угол Беннетта", _sides(values, "BennettAngle", "°")),
            ("Мгновенный боковой сдвиг", _sides(values, "ImmediateSideshift", " мм")),
            ("Межмыщелковое расстояние", _num(export.get("icd_mm"), " мм"))]
    inserts = export.get("inserts")
    if inserts:
        rows.append(("Вставки ССП (по пути мыщелка)", " · ".join(
            f"{'справа' if side == 'right' else 'слева'} «{inserts[side]['insert']}», {_num(inserts[side]['tilt_deg'], '°')}"
            for side in ("right", "left") if side in inserts)))
    else:
        rows.append(("Вставки ССП", "стандартные"))
    in_project = sorted(export.get("project_parameters") or {})
    rows.append(("В проекте exocad", "углы и сдвиг у каждого зуба" + (", межмыщелковое"
                                                                    if "IntercondylarDistance" in in_project else "")))
    sections.append(Section("Артикулятор", rows))
    if mount:
        rows = [("Плоскость монтажа", str(mount.get("plane", "—")))]
        if mount.get("aesthetic_deg") is not None:
            rows.append(("Расхождение с эстетикой", _num(mount["aesthetic_deg"], "°")
                         + ("" if mount.get("aesthetic_mm") is None else f", {_num(mount['aesthetic_mm'], ' мм')}")))
        sections.append(Section("Монтаж", rows))
    if condyles:
        sections.append(Section("Мыщелки", [("Правый", str(condyles.get("right", "—"))),
                                            ("Левый", str(condyles.get("left", "—")))]))
    if bite:
        sections.append(Section("Исправление прикуса", [("Сдвиг", _num(bite.get("shift_mm"), " мм", 2)),
                                                        ("Поворот", _num(bite.get("rotation_deg"), "°", 2))]))
    notes = _clean(export.get("notes"))
    if notes:
        sections.append(Section("Предупреждения", notes=notes, level="warn"))
    return Report("Паспорт кейса", case_label, sections, list(figures or []))
