"""Генерация сот в модели, экспортированной из exocad.

Пользователь выбирает плоские наружные грани модели (как в Materialise
Magics), и каждая выбранная грань превращается в открытые соты: шестигранные
ячейки идут от грани вглубь модели, а от остальной поверхности их отделяет
стенка заданной толщины. Это экономит материал при печати и уменьшает
деформации.

Для каждой грани модель поворачивается так, чтобы грань оказалась внизу, и
полость строится послойно: каждый срез сужается на толщину стенки (с учётом
соседних срезов, чтобы стенка выдерживалась и на наклонных участках),
пересекается с сеткой шестигранников и вычитается из модели. Наружная
поверхность модели при этом не меняется.

Использование:
    python honeycomb.py model.stl --list-planes
    python honeycomb.py model.stl out.stl --planes 1,3 --wall 1.2 --cell 5 --inner-wall 1.2
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
    depth: float | None = None  # глубина сот от грани; None — насколько позволяет модель
    perf_diameter: float = 0.0  # диаметр перфорации внутренних стенок; 0 — без неё
    perf_height: float | None = None  # расстояние от грани до центра перфорации
    layer: float = 0.2  # шаг послойного построения, мм

    def validate(self):
        for name in ("wall", "cell", "inner_wall", "layer"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть больше нуля")
        if self.depth is not None and self.depth <= 0:
            raise ValueError("depth должен быть больше нуля")
        if self.perf_diameter < 0:
            raise ValueError("perf_diameter не может быть отрицательным")
        if self.perf_diameter >= self.cell + self.inner_wall:
            raise ValueError("perf_diameter должен быть меньше шага ячеек")


@dataclass
class Plane:
    """Плоская наружная грань: точки x с dot(x, normal) == offset, normal смотрит наружу."""

    normal: np.ndarray
    offset: float
    area: float = 0.0
    center: np.ndarray | None = None

    def to_base(self) -> np.ndarray:
        """Матрица 4×4, которая кладёт грань в плоскость z = 0, а модель — выше неё."""
        m = trimesh.geometry.align_vectors(self.normal, (0.0, 0.0, -1.0))
        m[2, 3] += self.offset
        return m


def find_planes(model: Manifold, min_area: float = 10.0) -> list[Plane]:
    """Плоские наружные грани модели, на которых можно открыть соты, по убыванию площади."""
    mesh = model.to_mesh()
    verts = mesh.vert_properties[:, :3].astype(np.float64)
    tris = verts[mesh.tri_verts]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > 1e-12
    normals = cross[keep] / double_area[keep, None]
    areas = double_area[keep] / 2
    centers = tris[keep].mean(axis=1)
    offsets = np.einsum("ij,ij->i", normals, centers)

    # Треугольники одной плоскости: одинаковые нормаль и смещение (с допуском).
    key = np.c_[np.round(normals, 3), np.round(offsets, 2)]
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.reshape(-1)
    group_area = np.bincount(group, areas)

    planes = []
    for g in np.argsort(-group_area):
        if group_area[g] < 1.0:
            break
        sel = group == g
        w = areas[sel]
        normal = (normals[sel] * w[:, None]).sum(axis=0)
        normal /= np.linalg.norm(normal)
        offset = float((offsets[sel] * w).sum() / w.sum())
        center = (centers[sel] * w[:, None]).sum(axis=0) / w.sum()
        # Округление могло разбить одну плоскость на несколько групп — объединяем.
        for pl in planes:
            if pl.normal @ normal > 0.9999 and abs(pl.offset - offset) < 0.02:
                total = pl.area + w.sum()
                pl.center = (pl.center * pl.area + center * w.sum()) / total
                pl.area = total
                break
        else:
            planes.append(Plane(normal, offset, float(w.sum()), center))

    # Наружная грань — та, за плоскость которой модель нигде не выходит.
    planes = [pl for pl in planes if pl.area >= min_area and (verts @ pl.normal).max() - pl.offset < 0.02]
    return sorted(planes, key=lambda pl: -pl.area)


def bottom_plane(model: Manifold) -> Plane:
    return Plane(np.array([0.0, 0.0, -1.0]), -model.bounding_box()[2])


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
    """Полость с сотами для модели, у которой открытая грань лежит внизу (минимум Z)."""
    xmin, ymin, z0, xmax, ymax, ztop = model.bounding_box()
    z_end = ztop if p.depth is None else min(ztop, z0 + p.depth)
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


def make_honeycomb(model: Manifold, p: HoneycombParams, planes: list[Plane] | None = None) -> Manifold:
    """Открывает соты на каждой из граней planes (по умолчанию — на дне модели)."""
    p.validate()
    if not planes:
        planes = [bottom_plane(model)]

    cavities = []
    for n, plane in enumerate(planes, 1):
        to_base = plane.to_base()
        cavity = build_cavity(model.transform(to_base[:3]), p)
        if cavity.is_empty():
            raise ValueError(f"Грань {n}: модель слишком тонкая для сот с такими параметрами")
        cavities.append(cavity.transform(np.linalg.inv(to_base)[:3]))
    # Полость возвращается в исходные координаты, поэтому поверхность модели остаётся нетронутой.
    return model - Manifold.batch_boolean(cavities, OpType.Add)


def print_planes(planes: list[Plane]):
    if not planes:
        print("Плоских наружных граней не найдено")
        return
    print(" №   площадь, мм²   нормаль               центр, мм")
    for n, pl in enumerate(planes, 1):
        nx, ny, nz = pl.normal
        cx, cy, cz = pl.center
        print(f"{n:2}   {pl.area:11.1f}   ({nx:5.2f} {ny:5.2f} {nz:5.2f})   ({cx:6.1f} {cy:6.1f} {cz:6.1f})")


def select_planes(model: Manifold, spec: str) -> list[Plane]:
    spec = spec.strip().lower()
    if spec == "bottom":
        return [bottom_plane(model)]
    planes = find_planes(model)
    if not planes:
        raise ValueError("у модели нет плоских наружных граней")
    if spec == "auto":
        return planes[:1]
    try:
        numbers = [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"не понял --planes {spec!r}: нужны номера граней через запятую, auto или bottom")
    bad = [n for n in numbers if not 1 <= n <= len(planes)]
    if bad or not numbers:
        raise ValueError(f"нет граней с номерами {bad or spec}; всего граней: {len(planes)} (см. --list-planes)")
    return [planes[n - 1] for n in dict.fromkeys(numbers)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Генерация сот в модели (STL из exocad)")
    parser.add_argument("input", help="исходная модель (STL, PLY, OBJ)")
    parser.add_argument("output", nargs="?", help="куда сохранить результат (STL)")
    parser.add_argument("--list-planes", action="store_true", help="показать плоские грани модели и выйти")
    parser.add_argument(
        "--planes",
        default="bottom",
        help="грани под соты: номера из --list-planes через запятую, auto (самая большая) или bottom (дно, по умолчанию)",
    )
    parser.add_argument("--wall", type=float, default=1.2, help="толщина наружной стенки, мм (1.2)")
    parser.add_argument("--cell", type=float, default=5.0, help="размер ячейки, мм (5)")
    parser.add_argument("--inner-wall", type=float, default=1.2, help="толщина внутренних стенок, мм (1.2)")
    parser.add_argument("--depth", type=float, help="глубина сот от грани, мм (по умолчанию — насколько позволяет модель)")
    parser.add_argument("--perf-diameter", type=float, default=0.0, help="диаметр перфорации внутренних стенок, мм (0 — без неё)")
    parser.add_argument("--perf-height", type=float, help="расстояние от грани до центра перфорации, мм")
    parser.add_argument("--layer", type=float, default=0.2, help="шаг построения, мм (0.2)")
    args = parser.parse_args(argv)

    params = HoneycombParams(
        wall=args.wall,
        cell=args.cell,
        inner_wall=args.inner_wall,
        depth=args.depth,
        perf_diameter=args.perf_diameter,
        perf_height=args.perf_height,
        layer=args.layer,
    )
    if not args.list_planes and not args.output:
        parser.error("укажите файл результата")
    started = time.perf_counter()
    try:
        model = load_model(args.input)
        if args.list_planes:
            print_planes(find_planes(model))
            return 0
        result = make_honeycomb(model, params, select_planes(model, args.planes))
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
