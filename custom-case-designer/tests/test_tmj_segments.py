"""Сустав отдельно от остальной кости: головки мыщелков, отрезанные от нижней челюсти (landmarks.condylar_heads), и
скаты — суставная ямка с задним скатом бугорка, отделённые от черепа (guidance.articular_surface).

Кейс 4 (13.10): «самая верхняя точка ветви» слева — вершина венечного отростка, ALI — на стенке ямки; под
скатом в сечении попадалась посторонняя кость, ССП выходил −70°.
"""

import numpy as np
import pytest
from skimage import measure

from casedesigner import condyles as cond
from casedesigner import guidance as gd
from casedesigner import landmarks as lmk

from test_motion import plate, temporal_bone

STEP = 0.5  # мм на воксель


def mandible_volume():
    """Нижняя челюсть в LPS (x — влево, y — назад, z — вверх): тело и две ветви; на ветви — мыщелковый
    отросток сзади (верх z = 15) и венечный спереди (верх z = 22, выше головки), вырезка — z = 0."""
    lo = np.array([-60.0, -40.0, -45.0])
    shape = (241, 131, 141)
    x, y, z = np.meshgrid(*(lo[i] + STEP * np.arange(shape[i]) for i in range(3)), indexing="ij")
    vol = np.zeros(shape, bool)
    for cx in (-45.0, 45.0):
        ramus = (np.abs(x - cx) <= 4) & (y >= -10) & (y <= 20) & (z >= -40) & (z <= 0)
        condyle = (np.abs(x - cx) <= 4) & (y >= 10) & (y <= 20) & (z > 0) & (z <= 15)
        coronoid = (np.abs(x - cx) <= 3) & (y >= -10) & (y <= -3) & (z > 0) & (z <= 22)
        vol |= ramus | condyle | coronoid
    vol |= (np.abs(x) <= 49) & (y >= -35) & (y <= 10) & (z >= -40) & (z <= -32)  # тело
    v, f, _n, _ = measure.marching_cubes(np.pad(vol, 1).astype(float), 0.5)
    return (v - 1) * STEP + lo, f


def test_heads_cut_from_mandible_ignore_higher_coronoid():
    v, f = mandible_volume()
    heads = lmk.condylar_heads(v, f)
    for key, cx in (("Co_R", -45.0), ("Co_L", 45.0)):
        top = heads[key]["top"]
        assert top[0] == pytest.approx(cx, abs=1.0) and top[1] == pytest.approx(15.0, abs=2.0)
        assert top[2] == pytest.approx(15.0, abs=1.0)  # верх головки, а не венечного (22)
        assert heads[key]["notch_mm"] == pytest.approx(22.0, abs=1.5)  # вырезка — на 22 мм ниже верха ветви
        assert (v[heads[key]["vertices"], 1] >= 9).all()  # кусок — только мыщелковый отросток
    old = lmk.suggest_condyles(v)  # «самая верхняя точка ветви» — вершина венечного
    assert old["Co_L"][2] == pytest.approx(22.0, abs=1.0)
    new = lmk.suggest_condyles(v, faces=f)
    assert new["Co_L"] == pytest.approx(heads["Co_L"]["top"])


def test_ali_point_replaced_by_head_when_far():
    head = np.array([45.0, 15.0, 15.0])
    near = head + [0, 3.0, 0]
    point, source, note = cond.prefer_head(near, head)
    assert source == cond.SOURCE_ALI and note is None and point == pytest.approx(near)
    far = head + [0, 6.0, 0]  # ALI на задней стенке ямки
    point, source, note = cond.prefer_head(far, head)
    assert source == cond.SOURCE_MANDIBLE and point == pytest.approx(head) and "6.0 мм" in note
    assert cond.prefer_head(far, None)[1] == cond.SOURCE_ALI  # головка не выделилась — точка ALI


@pytest.mark.parametrize("alpha", [30, 45])
def test_slope_excludes_foreign_bone_under_eminence(alpha):
    """Посторонняя кость под скатом (не связана с ямкой): в профиль и ССП не попадает."""
    v, f = temporal_bone(alpha)
    dv, df = plate(np.array([-5.0, 3.0, -7.0]), np.array([10.0, 0, 0]), np.array([0, 9.0, 0]), nu=12, nv=20)
    vertices, faces = np.vstack([v, dv]), np.vstack([f, df + len(v)])
    patch = gd.articular_surface(vertices, faces, np.eye(4), np.zeros(3))
    assert patch is not None and (patch < len(f)).all() and len(patch) > 100  # только височная кость
    info = gd.condylar_path(*gd.eminence_profile(vertices, faces, np.eye(4), np.zeros(3)))
    assert info["sagittal_deg"] == pytest.approx(alpha, abs=0.5)
    # без выделения ската (как раньше) — нижняя кость в столбце, профиль ломается
    y, z = gd.eminence_profile(dv, df, np.eye(4), np.zeros(3))
    assert np.nanmin(z) == pytest.approx(-7.0, abs=1e-6)


def test_implausible_sagittal_not_applied():
    """Профиль снят не с сустава (ССП вне 5–75°): угол не идёт в настройки — стандартный, с заметкой."""
    from types import SimpleNamespace

    from casedesigner import exocad_webview as ew
    from test_mount_planes import CO, scans

    bones = [temporal_bone(-20, x=(CO[s][0] - 8, CO[s][0] + 8)) for s in ("right", "left")]
    skull = SimpleNamespace(vertices=np.vstack([b[0] for b in bones]),
                            faces=np.vstack([bones[0][1], bones[1][1] + len(bones[0][0])]))
    upper, lower = scans()
    _a, settings, eminence, notes = ew.mount(upper[0], lower[0], None, skull, condyles=CO)
    assert not eminence and "sagittal_right_deg" not in settings.sources
    assert sum("вне 5–75°" in n for n in notes) == 2
    assert gd.plausible_sagittal(35) and not gd.plausible_sagittal(-70) and not gd.plausible_sagittal(80)


def test_no_bone_above_condyle_gives_no_slope():
    dv, df = plate(np.array([-5.0, 3.0, -7.0]), np.array([10.0, 0, 0]), np.array([0, 9.0, 0]))
    assert gd.articular_surface(dv, df, np.eye(4), np.zeros(3)) is None
