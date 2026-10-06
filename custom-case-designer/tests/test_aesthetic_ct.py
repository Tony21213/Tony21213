"""Эстетическая система по КТ (пока нет фото): средняя линия по симметрии лицевого скелета; мыщелок — задний
отросток ветви, даже если венечный выше; рассчитанные движения — в файле лицевой дуги для exocad."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from casedesigner import aesthetic
from casedesigner import exocad_facebow as ef
from casedesigner.landmarks import suggest_condyles
from casedesigner.register import apply, rigid


def face_cloud(seed=0):
    """Симметричный «лицевой скелет» (x — вправо) с асимметричным обрывом поля зрения справа."""
    rng = np.random.default_rng(seed)
    pts = []
    for cx, cy, cz, r in ((35, 10, 20, 12), (20, 40, 10, 8), (10, 60, 0, 6), (45, -10, 30, 9), (25, 20, 35, 10)):
        for sx in (1, -1):
            p = rng.normal(size=(3000, 3))
            p = p / np.linalg.norm(p, axis=1, keepdims=True) * r + [sx * cx, cy, cz]
            pts.append(p)
    p = np.vstack(pts)
    return p[p[:, 0] < 52]  # поле зрения обрезано справа


def test_midsagittal_plane_found_near_the_dental_one():
    true = rigid(Rotation.from_euler("xyz", [0, 4, -3], degrees=True).as_matrix(), np.array([3.0, 0, 0]))
    pts = apply(true, face_cloud())  # лицо повёрнуто (крен 4°, разворот −3°) и сдвинуто на 3 мм
    n, o, rms = aesthetic.midsagittal_plane(pts, [1.0, 0, 0], [0.0, 0, 0])  # начальная — по зубам, без поворота
    expect = true[:3, 0]
    assert np.degrees(np.arccos(abs(n @ expect))) < 0.5 and abs((o - true[:3, 3]) @ expect) < 0.3 and rms < 1.0


def test_aesthetic_frame_ct():
    roll = rigid(Rotation.from_euler("y", 3, degrees=True).as_matrix(), np.zeros(3))
    base = np.eye(4)  # функциональная система совпадает с координатами кейса
    aes = aesthetic.aesthetic_frame_ct(base, apply(roll, face_cloud()))
    assert aes.roll_deg == pytest.approx(-3.0, abs=0.5) or aes.roll_deg == pytest.approx(3.0, abs=0.5)
    assert abs(aes.hints["yaw_deg"]) < 0.5 and aes.notes
    z = np.linalg.inv(aes.frame)[:3, 1]  # «вперёд» — от функциональной системы
    assert np.degrees(np.arccos(z @ [0, 1, 0])) < 0.5


def test_condyle_is_the_posterior_process_even_below_the_coronoid():
    """Ветвь: мыщелок сзади (верх z = 0), венечный отросток спереди выше (z = +5), вырезка неглубокая."""
    rng = np.random.default_rng(1)
    parts = []
    for side in (1, -1):
        x = side * 47
        ramus = np.c_[np.full(4000, x) + rng.normal(0, 2, 4000), rng.uniform(-25, 10, 4000), rng.uniform(-40, -14, 4000)]
        condyle = np.c_[np.full(800, x) + rng.normal(0, 3, 800), rng.uniform(-25, -15, 800), rng.uniform(-14, 0, 800)]
        coronoid = np.c_[np.full(800, x) + rng.normal(0, 1, 800), rng.uniform(0, 8, 800), rng.uniform(-14, 5, 800)]
        parts += [ramus, condyle, coronoid]
    co = suggest_condyles(np.vstack(parts), up=(0, 0, 1), left=(-1, 0, 0), anterior=(0, 1, 0))
    for key in ("Co_R", "Co_L"):
        assert co[key][1] < -14 and co[key][2] > -1.6, key  # задний отросток, его верх


def test_movements_in_facebow_file(tmp_path):
    marks = np.array([[0, 0, 0], [25, 0, 30.5], [-25, 0, 30.5]], float)
    reg = ef.Register(trimesh.creation.box((60, 10, 60)), marks)
    frame = np.eye(4)
    fb = ef.facebow(frame, [50, 0, 0], [-50, 0, 0], [0, 95, -35], reg)
    a = np.radians(np.linspace(0, 10, 11))
    opening = np.array([rigid(Rotation.from_rotvec([t, 0, 0]).as_matrix(), np.zeros(3)) for t in a])  # вокруг оси x кейса
    shift = np.array([rigid(np.eye(3), [k, 0, 0]) for k in np.linspace(0, 5, 6)])  # вправо пациента (x монтажа)
    path = tmp_path / "m.jawmotion"
    path.write_bytes(ef.jawmotion_xml(fb, movements=[("Открывание", opening, 60), ("Латеротрузия вправо", shift, 30)]))
    got = ef.read_jawmotion(str(path))
    assert set(got["movements"]) == {"opening", "lateral_rt"}
    tr = got["movements"]["lateral_rt"][0]
    # в системе регистратора x — влево: сдвиг вправо пациента — x уменьшается на 5 мм
    assert np.abs(tr["mark_1"][-1] - tr["mark_1"][0] - [-5, 0, 0]).max() < 1e-3
    first = np.array([got["movements"]["opening"][0][f"mark_{i}"][0] for i in (1, 2, 3)])
    assert np.abs(first - got["marks"]).max() < 1e-3  # первый кадр — положение скана
