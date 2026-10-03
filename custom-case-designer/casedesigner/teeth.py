"""Коронки зубов в КТ — опорная геометрия для совмещения со сканами.

Совмещение идёт только по зубам: кость и десна на скане и в КТ выглядят
по-разному (десны в КТ почти не видно), а эмаль и дентин видны в обоих.
Коронка в КТ — это часть поверхности зуба, которая граничит не с костью,
а с мягкими тканями или воздухом.
"""

from dataclasses import dataclass

import numpy as np

from .surface import Surface, extract_surface, refine_edges, tissue_levels

# Насколько снаружи от поверхности зуба смотреть, что за ней: кость или мягкие ткани.
PROBE_MM = 0.8


@dataclass
class Levels:
    hard: float  # мягкие ткани / кость
    dense: float  # кость / зубы

    @classmethod
    def of(cls, vol) -> "Levels":
        return cls(*tissue_levels(vol))


def crown_surface(vol, levels: Levels, roi=None, step: int = 1, refine: bool = False) -> Surface:
    """Поверхность коронок: зубы, граничащие с мягкими тканями или воздухом.

    refine=True — точки сдвигаются на реальную границу (максимум перепада
    яркости), а точки без чёткой границы отбрасываются.
    """
    teeth = extract_surface(vol, levels.dense, roi=roi, step=step)
    if not len(teeth.points):
        return teeth
    outside = vol.sample(teeth.points + PROBE_MM * teeth.normals)
    keep = outside < levels.hard
    crowns = Surface(teeth.points[keep], teeth.normals[keep])
    if refine and len(crowns.points):
        contrast = 0.5 * (levels.dense - levels.hard)
        crowns, ok = refine_edges(vol, crowns, min_contrast=contrast)
        crowns = Surface(crowns.points[ok], crowns.normals[ok])
    return crowns


def split_jaws(crowns: Surface):
    """Коронки верхней и нижней челюсти.

    В координатах DICOM ось z направлена к голове пациента. Жевательные
    поверхности нижних зубов смотрят вверх (+z), верхних — вниз; граница
    между челюстями — посередине между ними.
    """
    nz = crowns.normals[:, 2]
    up, down = crowns.points[nz > 0.5, 2], crowns.points[nz < -0.5, 2]
    if not len(up) or not len(down):
        raise ValueError("в КТ не найдены жевательные поверхности обеих челюстей")
    z_split = 0.5 * (np.median(up) + np.median(down))
    lower = crowns.points[:, 2] < z_split
    return (Surface(crowns.points[~lower], crowns.normals[~lower]),
            Surface(crowns.points[lower], crowns.normals[lower]))


# Направление жевательной поверхности каждой челюсти в координатах DICOM.
OCCLUSAL = {"upper": np.array([0.0, 0.0, -1.0]), "lower": np.array([0.0, 0.0, 1.0])}
