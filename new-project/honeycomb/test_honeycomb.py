import pytest
from manifold3d import Error, Manifold, OpType

from honeycomb import HoneycombParams, load_model, make_honeycomb, save_model

WALL = 1.2


def base_with_teeth():
    """Основание 40×30×12 мм со «скошенными» стенками и несколькими «зубами» сверху."""
    base = Manifold.cube((40, 30, 12), center=True).translate((0, 0, 6))
    base = base.warp(lambda v: (v[0] * (1 - 0.01 * v[2]), v[1] * (1 - 0.01 * v[2]), v[2]))
    teeth = [Manifold.sphere(4, 48).translate((x, 0, 13)) for x in (-12, 0, 12)]
    return Manifold.batch_boolean([base, *teeth], OpType.Add)


def test_box_shell_and_open_cells():
    box = Manifold.cube((40, 30, 15)).translate((-20, -15, 0))
    result = make_honeycomb(box, HoneycombParams(wall=WALL))

    assert result.status() == Error.NoError
    assert result.bounding_box() == pytest.approx(box.bounding_box())
    assert result.volume() < 0.6 * box.volume()

    # На дне ячейки открыты: в срезе есть отверстия.
    assert result.slice(0.05).num_contour() > 1
    # Под верхней гранью остаётся сплошная стенка.
    top = result.slice(15 - WALL + 0.05)
    assert top.num_contour() == 1 and top.area() == pytest.approx(40 * 30)
    # Боковые стенки не тоньше заданной: полость в пределах сужённого контура.
    cavity = box - result
    xmin, ymin, _, xmax, ymax, zmax = cavity.bounding_box()
    assert xmin >= -20 + WALL - 1e-3 and xmax <= 20 - WALL + 1e-3
    assert ymin >= -15 + WALL - 1e-3 and ymax <= 15 - WALL + 1e-3
    assert zmax <= 15 - WALL + 1e-3


def wall_gap(model, result):
    """Минимальное расстояние от полости до наружной поверхности (открытое дно не считается)."""
    cavity = (model - result).trim_by_plane((0, 0, 1), 0.01)
    outside = Manifold.cube((200, 200, 200), center=True) - model
    outside = outside.trim_by_plane((0, 0, 1), 0.01)
    return cavity.min_gap(outside, 2 * WALL)


def test_min_wall_on_sloped_model():
    model = base_with_teeth()
    result = make_honeycomb(model, HoneycombParams(wall=WALL))

    assert result.status() == Error.NoError
    assert result.volume() < 0.8 * model.volume()
    assert wall_gap(model, result) >= WALL * 0.995


def test_max_height_and_perforation():
    box = Manifold.cube((40, 30, 15)).translate((-20, -15, 0))
    limited = make_honeycomb(box, HoneycombParams(wall=WALL, max_height=6))
    assert (box - limited).bounding_box()[5] == pytest.approx(6, abs=1e-3)

    plain = make_honeycomb(box, HoneycombParams(wall=WALL))
    perforated = make_honeycomb(box, HoneycombParams(wall=WALL, perf_diameter=1.0, perf_height=3))
    assert perforated.volume() < plain.volume()
    # Перфорация не прорезает наружную стенку.
    assert wall_gap(box, perforated) >= WALL * 0.995


def test_open_top():
    box = Manifold.cube((40, 30, 15)).translate((-20, -15, 0))
    result = make_honeycomb(box, HoneycombParams(wall=WALL, open_side="top"))
    assert result.slice(15 - 0.05).num_contour() > 1
    assert result.slice(WALL - 0.05).num_contour() == 1


def test_stl_roundtrip(tmp_path):
    src, dst = tmp_path / "in.stl", tmp_path / "out.stl"
    save_model(base_with_teeth(), str(src))
    result = make_honeycomb(load_model(str(src)), HoneycombParams())
    save_model(result, str(dst))
    assert load_model(str(dst)).volume() == pytest.approx(result.volume(), rel=1e-3)


def test_rejects_bad_params():
    box = Manifold.cube((10, 10, 10))
    with pytest.raises(ValueError):
        make_honeycomb(box, HoneycombParams(wall=0))
    with pytest.raises(ValueError):
        make_honeycomb(box, HoneycombParams(wall=6))  # стенка толще половины модели
