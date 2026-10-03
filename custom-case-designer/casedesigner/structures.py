"""Какие структуры сегментируются из КТ и как называются их файлы.

Два прохода, у каждого своя модель:

* основной — девять анатомических структур;
* зубы — каждый зуб отдельно с номером FDI, плюс импланты, коронки на
  имплантах и мосты. Модель зубов смотрит только на область зубных рядов,
  найденную основным проходом.

Номер класса в модели — позиция в списке, начиная с 1 (0 — фон).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Structure:
    key: str  # имя файла без расширения
    name: str  # название для интерфейса
    color: str  # цвет в интерфейсе, sRGB
    # С какой челюстью двигается структура в режиме «прикус сканов»: "upper" — с верхней
    # челюстью и черепом, "lower" — с нижней челюстью, None — бывает в обеих (делится по частям).
    jaw: str | None = "upper"


ANATOMY = (
    Structure("mandible", "Нижняя челюсть", "#E6D5B8", "lower"),
    Structure("upper_skull", "Верхняя челюсть и череп", "#D4C3A3"),
    Structure("upper_teeth", "Зубы верхней челюсти", "#FBF8EE"),
    Structure("lower_teeth", "Зубы нижней челюсти", "#F1ECDD", "lower"),
    Structure("mandibular_canal", "Нижнечелюстной канал", "#E8412E", "lower"),
    Structure("maxillary_sinus", "Гайморовы пазухи", "#3A8FD6"),
    Structure("nasal_cavity", "Полость носа", "#5CC6E4"),
    Structure("pharynx", "Глотка", "#2C64A8"),
    Structure("soft_palate", "Мягкое нёбо", "#EFA18E"),
)

# Порядок зубов в модели: квадранты 1–4, в каждом от центрального резца к восьмёрке.
FDI = tuple(10 * q + n for q in (1, 2, 3, 4) for n in range(1, 9))

TEETH = tuple(
    Structure(f"tooth_{fdi}", f"Зуб {fdi}", "#FBF8EE" if fdi < 30 else "#F1ECDD", "upper" if fdi < 30 else "lower")
    for fdi in FDI
) + (
    Structure("implant", "Импланты", "#8E9399", None),
    Structure("implant_crown", "Коронки на имплантах", "#EDF2F7", None),
    Structure("bridge", "Мосты", "#E6EDF5", None),
)

# Наборы классов по имени модели: так модель описывается в своём model.json.
SCHEMES = {"anatomy": ANATOMY, "teeth": TEETH}

# Область для модели зубов: зубные ряды из основного прохода плюс запас, мм.
TEETH_MARGIN_MM = 5.0
TEETH_SOURCE = ("upper_teeth", "lower_teeth")


def jaw_of(key: str) -> str | None:
    """Челюсть структуры по ключу её файла ("mandible", "teeth/tooth_36", ...); None — смешанная или неизвестная."""
    name = key.rsplit("/", 1)[-1]
    for scheme in SCHEMES.values():
        for structure in scheme:
            if structure.key == name:
                return structure.jaw
    return None
