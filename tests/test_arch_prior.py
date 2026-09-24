from crownai.arch_prior import adjacent_pair_distance_mm, axis_tilt_for_fdi


def test_adjacent_pair_distance_is_sane():
    d = adjacent_pair_distance_mm(16, 15)
    assert d is not None
    assert 5.0 < d["median_mm"] < 15.0
    assert d["p10_mm"] <= d["median_mm"] <= d["p90_mm"]
    assert d["n"] > 100


def test_adjacent_pair_distance_order_independent():
    assert adjacent_pair_distance_mm(16, 15) == adjacent_pair_distance_mm(15, 16)


def test_non_adjacent_pair_is_none():
    assert adjacent_pair_distance_mm(16, 11) is None


def test_axis_tilt_covers_every_fdi_tooth():
    for quadrant in (1, 2, 3, 4):
        for pos in range(1, 9):
            fdi = quadrant * 10 + pos
            tilt = axis_tilt_for_fdi(fdi)
            assert tilt is not None, fdi
            assert 0.0 <= tilt.median_tilt_from_arch_up_deg < 45.0
            assert tilt.n > 100
