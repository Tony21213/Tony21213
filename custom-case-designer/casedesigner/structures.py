"""Каталог структур из КТ: как называются, каким цветом показываются и с какой челюстью двигаются.

Какие структуры выдаёт модель сегментации, решает её model.json (поле
outputs: ключ структуры → метки сети). Ключ — путь файла без расширения:
"mandible", "teeth/tooth_36", "pulp/pulp_36". Здесь — всё, что о структуре
нужно знать интерфейсу и экспорту.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Structure:
    name: str  # название для интерфейса
    color: str  # цвет в интерфейсе, sRGB
    # С какой челюстью двигается структура в режиме «прикус сканов»: "upper" — с верхней
    # челюстью и черепом, "lower" — с нижней, None — бывает в обеих (делится по частям).
    jaw: str | None = "upper"


FDI = tuple(10 * q + n for q in (1, 2, 3, 4) for n in range(1, 9))


def _catalogue() -> dict[str, Structure]:
    c = {
        "mandible": Structure("Нижняя челюсть", "#E6D5B8", "lower"),
        "maxilla": Structure("Верхняя челюсть", "#D9C8A9"),
        "skull": Structure("Череп", "#D4C3A3"),
        "upper_teeth": Structure("Зубы верхней челюсти", "#FBF8EE"),
        "lower_teeth": Structure("Зубы нижней челюсти", "#F1ECDD", "lower"),
        "mandibular_canal": Structure("Нижнечелюстной канал", "#E8412E", "lower"),
        "incisive_canal": Structure("Резцовый канал", "#F07A5A", "lower"),
        "lingual_canal": Structure("Язычный канал", "#F09A7A", "lower"),
        "maxillary_sinus": Structure("Гайморовы пазухи", "#3A8FD6"),
        "frontal_sinus": Structure("Лобные пазухи", "#4FA3E0"),
        "nasal_cavity": Structure("Полость носа", "#5CC6E4"),
        "pharynx": Structure("Глотка", "#2C64A8"),
        "nasopharynx": Structure("Носоглотка", "#2F6FB5"),
        "oropharynx": Structure("Ротоглотка", "#2A5E9E"),
        "hypopharynx": Structure("Гортаноглотка", "#244F85"),
        "soft_palate": Structure("Мягкое нёбо", "#EFA18E"),
        "auditory_canal_right": Structure("Слуховой проход правый", "#9C7BE0"),
        "auditory_canal_left": Structure("Слуховой проход левый", "#9C7BE0"),
        "hard_palate": Structure("Твёрдое нёбо", "#E3B49A"),
        "teeth/implant": Structure("Импланты", "#8E9399", None),
        "teeth/crown": Structure("Коронки (ортопедические)", "#EDF2F7", None),
        "teeth/bridge": Structure("Мосты", "#E6EDF5", None),
    }
    for fdi in FDI:
        jaw = "upper" if fdi < 30 else "lower"
        c[f"teeth/tooth_{fdi}"] = Structure(f"Зуб {fdi}", "#FBF8EE" if jaw == "upper" else "#F1ECDD", jaw)
        c[f"pulp/pulp_{fdi}"] = Structure(f"Пульпа {fdi}", "#E57373", jaw)
    return c


CATALOGUE = _catalogue()


def structure(key: str) -> Structure:
    """Описание структуры; для неизвестного ключа — серая смешанная структура."""
    return CATALOGUE.get(key, Structure(key, "#B0B0B0", None))


def jaw_of(key: str) -> str | None:
    """Челюсть структуры по ключу её файла; None — смешанная или неизвестная (делится по частям)."""
    known = CATALOGUE.get(key)
    return known.jaw if known else None
