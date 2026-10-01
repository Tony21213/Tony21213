"""Генерация сот в модели, экспортированной из exocad.

Модель делается полой: снаружи остаётся стенка заданной толщины, а полость
разбивается на шестигранные ячейки, открытые со стороны основания. Это
экономит материал при печати и уменьшает деформации.

Полость строится послойно: каждый горизонтальный срез модели сужается на
толщину стенки (с учётом соседних срезов, чтобы стенка выдерживалась и на
наклонных участках), пересекается с сеткой шестигранников и вычитается из
модели. Наружная поверхность модели при этом не меняется.

Использование:
    python honeycomb.py model.stl model_honeycomb.stl --wall 1.2 --cell 5 --inner-wall 1.2
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import trimesh
from manifold3d import CrossSection, Error, Manifold, Mesh, OpType

# Срезы берутся чуть в стороне от границ слоёв, чтобы не попадать точно в
# горизонтальные грани модели (например, в плоское дно).
SLICE_EPS = 1e-4
# Насколько нижний слой сот выходит за дно, чтобы ячейки гарантированно открылись.
BOTTOM_OVERCUT = 1.0


@dataclass
class HoneycombParams:
    wall: float = 1.2  # толщина наружной стенки, мм
    cell: float = 5.0  # размер ячейки между параллельными гранями, мм
    inner_wall: float = 1.2  # толщина стенок между ячейками, мм
    max_height: float | None = None  # высота сот над основанием; None — на всю модель
    perf_diameter: float = 0.0  # диаметр перфорации внутренних стенок; 0 — без неё
    perf_height: float | None = None  # высота центра перфорации над основанием
    open_side: str = "bottom"  # с какой стороны открыты соты: bottom или top
    layer: float = 0.2  # шаг послойного построения, мм

    def validate(self):
        for name in ("wall", "cell", "inner_wall", "layer"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть больше нуля")
        if self.max_height is not None and self.max_height <= 0:
            raise ValueError("max_height должен быть больше нуля")
        if self.perf_diameter < 0:
            raise ValueError("perf_diameter не может быть отрицательным")
        if self.perf_diameter >= self.cell + self.inner_wall:
            raise ValueError("perf_diameter должен быть меньше шага ячеек")
        if self.open_side not in ("bottom", "top"):
            raise ValueError("open_side должен быть bottom или top")


def load_model(path: str) -> Manifold:
    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    model = Manifold(
        Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
    )
    if model.status() != Error.NoError or model.is_empty():
        raise ValueError(
            f"{path}: модель не замкнута или содержит дефекты сетки ({model.status().name}). "
            "Закройте отверстия в модели перед генерацией сот."
        )
    return model


def save_model(model: Manifold, path: str):
    mesh = model.to_mesh()
    trimesh.Trimesh(
        vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts, process=False
    ).export(path)


def hex_cells(bounds, center, p: HoneycombParams) -> CrossSection:
    """Сетка шестигранных ячеек, покрывающая bounds = (xmin, ymin, xmax, ymax)."""
    pitch = p.cell + p.inner_wall  # расстояние между центрами соседних ячеек
    dx = pitch * math.sqrt(3) / 2  # шаг между столбцами
    hexagon = CrossSection.circle(p.cell / math.sqrt(3), 6)
    cx, cy = center
    xmin, ymin, xmax, ymax = bounds

    cells = []
    for i in range(math.floor((xmin - cx) / dx) - 1, math.ceil((xmax - cx) / dx) + 2):
        y_shift = pitch / 2 if i % 2 else 0.0
        j_from = math.floor((ymin - cy - y_shift) / pitch) - 1
        j_to = math.ceil((ymax - cy - y_shift) / pitch) + 2
        for j in range(j_from, j_to):
            cells.append(hexagon.translate((cx + i * dx, cy + j * pitch + y_shift)))
    return CrossSection.batch_boolean(cells, OpType.Add)


def perforation_strips(bounds, center, p: HoneycombParams, half_width: float) -> CrossSection:
    """Полосы через центры ячеек по трём направлениям сетки: прорезают все внутренние стенки."""
    pitch = p.cell + p.inner_wall
    dx = pitch * math.sqrt(3) / 2  # расстояние между соседними линиями центров
    cx, cy = center
    xmin, ymin, xmax, ymax = bounds
    reach = math.hypot(xmax - xmin, ymax - ymin) + 2 * pitch
    count = math.ceil(reach / dx)

    strips = []
    for angle in (30.0, 90.0, 150.0):
        a = math.radians(angle)
        nx, ny = -math.sin(a), math.cos(a)
        strip = CrossSection.square((2 * reach, 2 * half_width), center=True).rotate(angle)
        for k in range(-count, count + 1):
            strips.append(strip.translate((cx + k * dx * nx, cy + k * dx * ny)))
    return CrossSection.batch_boolean(strips, OpType.Add)


def _same(a: CrossSection, b: CrossSection) -> bool:
    return (a - b).area() + (b - a).area() < 1e-6


def build_cavity(model: Manifold, p: HoneycombParams) -> Manifold:
    """Полость с сотами для модели, у которой открытая сторона — нижняя (минимум Z)."""
    xmin, ymin, z0, xmax, ymax, ztop = model.bounding_box()
    z_end = ztop if p.max_height is None else min(ztop, z0 + p.max_height)
    h = p.layer
    n_layers = math.ceil((z_end - z0) / h)
    reach = math.ceil(p.wall / h + 0.5)  # сколько соседних срезов влияет на слой

    bounds = (xmin, ymin, xmax, ymax)
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    cells = hex_cells(bounds, center, p)

    slices = {}

    def boundary_slice(i):
        # Ниже основания модель считается продолжением дна вниз, поэтому
        # стенка у открытой стороны не образуется и ячейки остаются открытыми.
        if i not in slices:
            slices[i] = model.slice(max(z0 + i * h, z0) + SLICE_EPS)
        return slices[i]

    offsets = {}

    def shrunk(i, radius):
        key = (max(i, 0), radius)
        if key not in offsets:
            offsets[key] = boundary_slice(i).offset(-radius)
        return offsets[key]

    perf_r = p.perf_diameter / 2
    perf_z = None
    if perf_r > 0:
        perf_z = z0 + (p.perf_height if p.perf_height is not None else perf_r + p.wall)

    layers = []  # [нижняя граница, верхняя граница, сечение полости]
    for k in range(n_layers):
        lo, hi = z0 + k * h, min(z0 + (k + 1) * h, z_end)

        # Точки слоя [lo, hi] должны быть не ближе p.wall к поверхности модели:
        # срез на расстоянии d от слоя сужается на sqrt(wall² - d²) (сечение шара).
        # d уменьшено на полшага: так учитывается поверхность между срезами.
        parts = []
        for i in range(k - reach, k + 2 + reach):
            zi = z0 + i * h
            d = max(0.0, lo - zi - h / 2, zi - hi - h / 2)
            if d >= p.wall:
                continue
            parts.append(shrunk(i, math.sqrt(p.wall**2 - d**2)))
        inner = CrossSection.batch_boolean(parts, OpType.Intersect)
        if inner.is_empty():
            continue

        pattern = cells
        if perf_z is not None:
            # Перфорация — горизонтальные цилиндры, в срезе — полосы переменной ширины.
            d = max(0.0, lo - perf_z, perf_z - hi)
            if d < perf_r:
                strips = perforation_strips(bounds, center, p, math.sqrt(perf_r**2 - d**2))
                pattern = cells + strips
        section = inner ^ pattern
        if section.is_empty():
            continue

        if k == 0:
            lo -= BOTTOM_OVERCUT
        if layers and math.isclose(layers[-1][1], lo) and _same(layers[-1][2], section):
            layers[-1][1] = hi
        else:
            layers.append([lo, hi, section])

    pieces = [Manifold.extrude(s, top - bottom).translate((0, 0, bottom)) for bottom, top, s in layers]
    if not pieces:
        return Manifold()
    return Manifold.batch_boolean(pieces, OpType.Add)


def make_honeycomb(model: Manifold, p: HoneycombParams) -> Manifold:
    p.validate()
    flip = p.open_side == "top"
    if flip:
        model = model.mirror((0, 0, 1))
    cavity = build_cavity(model, p)
    if cavity.is_empty():
        raise ValueError("Модель слишком тонкая для сот с такими параметрами")
    result = model - cavity
    if flip:
        result = result.mirror((0, 0, 1))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Генерация сот в модели (STL из exocad)")
    parser.add_argument("input", help="исходная модель (STL, PLY, OBJ)")
    parser.add_argument("output", help="куда сохранить результат (STL)")
    parser.add_argument("--wall", type=float, default=1.2, help="толщина наружной стенки, мм (1.2)")
    parser.add_argument("--cell", type=float, default=5.0, help="размер ячейки, мм (5)")
    parser.add_argument("--inner-wall", type=float, default=1.2, help="толщина внутренних стенок, мм (1.2)")
    parser.add_argument("--max-height", type=float, help="высота сот над основанием, мм (по умолчанию — вся модель)")
    parser.add_argument("--perf-diameter", type=float, default=0.0, help="диаметр перфорации внутренних стенок, мм (0 — без неё)")
    parser.add_argument("--perf-height", type=float, help="высота центра перфорации над основанием, мм")
    parser.add_argument("--open-side", choices=("bottom", "top"), default="bottom", help="сторона открытых сот по оси Z (bottom)")
    parser.add_argument("--layer", type=float, default=0.2, help="шаг построения, мм (0.2)")
    args = parser.parse_args(argv)

    params = HoneycombParams(
        wall=args.wall,
        cell=args.cell,
        inner_wall=args.inner_wall,
        max_height=args.max_height,
        perf_diameter=args.perf_diameter,
        perf_height=args.perf_height,
        open_side=args.open_side,
        layer=args.layer,
    )
    started = time.perf_counter()
    try:
        model = load_model(args.input)
        result = make_honeycomb(model, params)
    except ValueError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    save_model(result, args.output)

    before, after = model.volume(), result.volume()
    print(f"Объём: {before:.0f} → {after:.0f} мм³ (экономия {100 * (1 - after / before):.0f}%)")
    print(f"Готово за {time.perf_counter() - started:.1f} с: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
