"""Полная реабилитация (casedesigner/rehab.py): подъём прикуса вокруг оси мыщелков и шаблон окклюзионной плоскости
(сфера Монсона) — на синтетическом монтаже: x — вправо, y — вперёд, z — вверх."""

import numpy as np
import pytest

from casedesigner import motion as mo
from casedesigner import rehab as rh
from casedesigner.register import apply

CR, CL = np.array([50.0, -90.0, 0.0]), np.array([-50.0, -90.0, 0.0])
INC = np.array([0.0, 0.0, -30.0])
ANATOMY = mo.Anatomy(np.eye(4), {"condyle_right": CR, "condyle_left": CL, "incisal": INC}, "тест")


def lower_arch():
    """Нижняя дуга: точки бугров по параболе от резцов назад, поверхность чуть ниже."""
    out = []
    for x in np.linspace(-27, 27, 271):
        y = -0.05 * x ** 2
        for z in (-30.0, -31.0, -32.0):
            out.append([x, y, z])
    return np.array(out)


def test_raise_about_condyles():
    T, deg = rh.raise_transform(ANATOMY, 4.0)
    assert apply(T, INC[None])[0][2] == pytest.approx(INC[2] - 4.0, abs=1e-6)
    for c in (CR, CL):  # мыщелки на оси — на месте
        assert np.allclose(apply(T, c[None])[0], c, atol=1e-9)
    assert 2.0 < deg < 3.0
    rep = rh.raise_report(ANATOMY, T, lower_arch())
    assert rep["incisal_mm"] == pytest.approx(4.0, abs=1e-3)
    assert set(rep["molars"]) == {"right", "left"}
    for v in rep["molars"].values():  # ближе к оси — меньше
        assert 2.0 < v < 3.6
    assert rh.raise_transform(ANATOMY, 0) == (pytest.approx(np.eye(4)), 0.0)
    with pytest.raises(ValueError, match="от 0 до"):
        rh.raise_transform(ANATOMY, 30)


def test_monson_sphere_and_template():
    c = rh.monson_center(ANATOMY, INC)
    for p in (CR, CL, INC):
        assert np.linalg.norm(p - c) == pytest.approx(rh.MONSON_MM, abs=1e-6)
    assert c[2] > INC[2]  # центр — над окклюзионной плоскостью
    with pytest.raises(ValueError, match="радиус"):
        rh.monson_center(ANATOMY, INC, 40.0)
    landmarks = {"Po_R": (60.0, -95.0, 25.0), "Po_L": (-60.0, -95.0, 25.0), "Or_R": (35.0, -15.0, 25.0),
                 "Or_L": (-35.0, -15.0, 25.0), "S": (0.0, -70.0, 40.0), "N": (0.0, 0.0, 45.0)}
    t = rh.template(ANATOMY, INC, landmarks=landmarks)
    v = np.asarray(t["surface"].vertices)
    assert np.allclose(np.linalg.norm(v - t["center"], axis=1), rh.MONSON_MM, atol=1e-6)  # лоскут — на сфере
    assert len(t["surface"].faces) % 2 == 0 and len(t["curves"].faces) > 0
    assert 0.5 < t["spee_depth_mm"] < 5.0  # кривая Шпее — вогнутая, неглубокая
    assert set(t["tilt"]) == {"fh", "sn"} and 0 <= t["tilt"]["fh"]["deg"] < 40
    assert t["tilt"]["fh"]["norm"] == rh.NORM_FH
    # больше радиус — площе кривая Шпее
    assert rh.template(ANATOMY, INC, 140.0)["spee_depth_mm"] < t["spee_depth_mm"]


def test_session_rehab_and_case_from_session(tmp_path):
    """Сессия: подъём и шаблон по монтажу; выгрузка с артикулятором — нижняя челюсть в поднятом положении,
    шаблон — с верхней; состояние — в .ccdcase."""
    from types import SimpleNamespace

    import trimesh

    from casedesigner import exocad_articulator_export as xa
    from casedesigner import kinematics as kin
    from casedesigner.app.session import Session

    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    with pytest.raises(ValueError, match="смонтируйте"):
        s.articulation_rehab(raise_mm=2)
    s.jaw.set_scans(SimpleNamespace(vertices=lower_arch() + [0, 0, 40.0], faces=np.zeros((0, 3), int), name="u"),
                    SimpleNamespace(vertices=lower_arch(), faces=np.zeros((0, 3), int), name="l"))
    s.jaw.anatomy = ANATOMY  # монтаж (set_scans его сбрасывает)
    info = s.articulation_rehab(raise_mm=4, template=True)
    assert info["raise_mm"] == 4 and 2 < info["angle_deg"] < 3 and info["report"]["incisal_mm"] == pytest.approx(4, abs=1e-3)
    assert info["template_info"]["radius"] == rh.MONSON_MM and info["display_bite"] != np.eye(4).tolist()
    assert len(s.rehab_mesh("surface")) > 8 and len(s.rehab_mesh("curves")) > 8
    with pytest.raises(ValueError, match="радиус"):
        s.articulation_rehab(radius=30)
    assert s.rehab["radius"] == rh.MONSON_MM  # неудачная правка не записана
    # выгрузка: нижний скан, его кость — в поднятом положении; шаблон — отдельно
    R = s.rehab_transform()
    lo = lower_arch()
    up_item = {"role": "jaw", "path": None, "reg": SimpleNamespace(jaw="upper", transform=np.eye(4)),
               "scan": SimpleNamespace(vertices=lo + [0, 0, 40.0], faces=np.zeros((0, 3), int))}
    lo_item = {"role": "jaw", "path": None, "reg": SimpleNamespace(jaw="lower", transform=np.eye(4)),
               "scan": SimpleNamespace(vertices=lo, faces=np.zeros((0, 3), int))}
    mandible = trimesh.creation.box((80, 60, 10))
    fake = SimpleNamespace(jaw=SimpleNamespace(anatomy=ANATOMY, lower_fix=np.eye(4), settings=kin.Settings(), movements={},
                                               mounting="ct"),
                           bite_motion=None, scans={"u": up_item, "l": lo_item}, structures={"mandible": mandible},
                           landmarks={}, suggested=set(), landmark_sources={}, _jaw_items=lambda: (up_item, lo_item),
                           rehab_transform=lambda: R, rehab_info=s.rehab_info, rehab_template=s.rehab_template)
    case = xa.case_from_session(fake)
    assert np.allclose(case.lower.vertices, apply(R, lo)) and np.allclose(case.upper.vertices, lo + [0, 0, 40.0])
    assert np.allclose(case.bones["lower"].vertices, apply(R, mandible.vertices))
    assert set(case.occlusal_template) == {"surface", "curves"} and case.rehab_info["raise_mm"] == 4
