"""Суставной путь в 3D (casedesigner/joint3d.py) на искусственном суставе известной формы: головка — эллипсоид,
крыша ямки — плоская, скат бугорка — под ALPHA, внутренняя стенка — в W0 от головки и уходит внутрь под BETA."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from casedesigner import exocad_articulator_export as xa
from casedesigner import joint3d as j3
from casedesigner import kinematics as kin
from casedesigner.app.session import Session
from casedesigner.register import rigid

ALPHA, BETA, GAP, W0 = 40.0, 12.0, 1.0, 1.0
HEAD = (9.0, 4.0, 5.0)  # полуоси головки: вбок, вперёд, вверх


def head_points():
    s = trimesh.creation.icosphere(subdivisions=4)
    v = np.asarray(s.vertices) * HEAD
    return v + [0, 0, -HEAD[2]]  # верхушка (Co) — в начале координат


def fossa(slope=ALPHA, beta=BETA, wall=True):
    """Поверхность кости (сетка): крыша и скат — высотой над головкой; стенка — внутри (−x, правый сустав)."""
    xs = np.arange(-14.0, 14.01, 0.5)
    ys = np.arange(-10.0, 24.01, 0.5)

    # скат — касательная к головке на зазоре GAP: точка эллипсоида с нормалью «вперёд-вверх» под углом ската
    a, b, c = HEAD
    n = np.array([0.0, np.sin(np.radians(slope)), np.cos(np.radians(slope))])
    p = np.array([0.0, 0.0, -c]) + np.array([a * a * n[0], b * b * n[1], c * c * n[2]]) / np.sqrt(
        (a * n[0]) ** 2 + (b * n[1]) ** 2 + (c * n[2]) ** 2)
    q = p + GAP * n
    tan = np.tan(np.radians(slope))

    def zf(y):
        slope_z = q[2] - (y - q[1]) * tan
        crest = q[2] - 12 * tan  # вершина бугорка — на 12 мм вперёд
        return min(GAP, max(slope_z, crest))

    grid = np.array([[(x, y, zf(y)) for x in xs] for y in ys])
    ny, nx = grid.shape[:2]
    idx = np.arange(nx * ny).reshape(ny, nx)
    q = np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
    faces = np.r_[q[:, [0, 2, 1]], q[:, [0, 3, 2]]]  # нормали — вниз, к суставу
    parts = [trimesh.Trimesh(grid.reshape(-1, 3), faces, process=False)]
    if wall:
        zs = np.arange(-15.0, GAP + 0.01, 0.5)
        wg = np.array([[(-(HEAD[0] + W0) - max(y, 0.0) * np.tan(np.radians(beta)), y, z) for z in zs] for y in ys])
        ny, nz = wg.shape[:2]
        idx = np.arange(ny * nz).reshape(ny, nz)
        q = np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
        wf = np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
        w = trimesh.Trimesh(wg.reshape(-1, 3), wf, process=False)
        if w.face_normals[:, 0].mean() < 0:  # нормаль стенки — к головке (+x)
            w = trimesh.Trimesh(wg.reshape(-1, 3), wf[:, ::-1], process=False)
        parts.append(w)
    m = trimesh.util.concatenate(parts)
    return np.asarray(m.vertices), np.asarray(m.faces)


def joint(bite=None, **kw):
    v, f = fossa(**kw)
    return j3.joint(v, f, head_points(), np.eye(4), np.zeros(3), "right", bite=bite)


def test_known_slope_and_medial_wall():
    r = j3.analyze(joint())
    assert r["rest_gap_mm"] == pytest.approx(GAP, abs=0.15)
    assert r["sagittal_deg"] == pytest.approx(ALPHA, abs=4.0)
    assert r["side_shift_mm"] == pytest.approx(W0, abs=0.3)
    assert r["bennett_deg"] == pytest.approx(BETA, abs=3.0)
    path = r["path"]
    assert path[0, 0] == 0 and path[-1, 1] < 0  # вперёд и вниз
    flat = j3.analyze(joint(slope=20.0))
    assert flat["sagittal_deg"] < r["sagittal_deg"] - 10


def test_no_medial_wall_and_anatomic_disc():
    r = j3.analyze_both(joint(wall=False))
    assert r["bennett_deg"] is None and any("стенки" in n for n in r["notes"])
    assert r["anatomic"]["sagittal_deg"] is not None
    with pytest.raises(ValueError, match="кости ямки"):
        j3.joint(*fossa(), head_points() + [0, 0, -60.0], np.eye(4), np.array([0, 0, -60.0]) + [0, 0, 80.0], "right")


def test_rest_is_the_bite_not_the_ct():
    """Покой — головка в прикусе: ниже на КТ-снимке или чуть «в кости» — путь всё равно от неё и по тому же скату."""
    lower = j3.protrusion(joint(bite=rigid(np.eye(3), [0.0, 0.0, -0.5])))  # в прикусе ниже — щель больше
    assert lower["rest_gap_mm"] == pytest.approx(GAP + 0.5, abs=0.15)
    assert tuple(lower["path"][0]) == (0.0, 0.0) and lower["sagittal_deg"] == pytest.approx(ALPHA, abs=5.0)
    inside = j3.protrusion(joint(bite=rigid(np.eye(3), [0.0, 0.0, GAP + 0.3])))  # прикус по сканам неточен
    assert inside["rest_gap_mm"] == 0.0 and inside["sagittal_deg"] == pytest.approx(ALPHA, abs=5.0)
    assert not inside["notes"]  # 0,3 мм — в пределах точности
    deep = j3.protrusion(joint(bite=rigid(np.eye(3), [0.0, 0.0, GAP + 0.8])))
    assert any("заходит в кость" in n for n in deep["notes"])


def _session(tmp_path, monkeypatch, left="головка мыщелка на нижней челюсти не найдена"):
    v, f = fossa()
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.case = SimpleNamespace()
    s.structures = {"skull": trimesh.Trimesh(v, f, process=False), "mandible": trimesh.creation.box((1, 1, 1))}
    s.landmarks = {"Co_R": np.zeros(3), "Co_L": np.array([-100.0, 0.0, 0.0])}
    s.jaw.anatomy = SimpleNamespace(frame=np.eye(4))
    s.bite_motion = rigid(np.eye(3), [0.0, -0.2, 0.0])  # в прикусе мыщелок на 0,2 мм назад
    calls = []

    def case_joints(skull, mandible, frame, condyles, bite):
        calls.append(bite)
        return {"right": j3.joint(skull.vertices, skull.faces, head_points(), frame, condyles["right"], "right", bite),
                "left": left}
    monkeypatch.setattr(j3, "case_joints", case_joints)
    return s, calls


def test_session_applies_3d_sagittal_and_keeps_manual(tmp_path, monkeypatch):
    s, calls = _session(tmp_path, monkeypatch)
    st = s.jaw.settings
    st.sources["sagittal_right_deg"] = "КТ: суставной бугорок"
    st.sources["sagittal_left_deg"] = "КТ: суставной бугорок"
    left_before = st.sagittal_left_deg
    assert s._apply_joint3d()
    assert st.sagittal_right_deg == pytest.approx(ALPHA, abs=5.0) and st.sources["sagittal_right_deg"] == s.JOINT3D_SOURCE
    assert st.bennett_right_deg == pytest.approx(st.sagittal_right_deg / 8 + 12, abs=0.1)
    assert st.sagittal_left_deg == left_before and st.sources["sagittal_left_deg"] == "КТ: суставной бугорок"
    assert np.allclose(calls[0], s.bite_motion)  # от прикуса
    assert s.joint3d_paths() is None  # левой стороны в 3D нет — вставки по плоскому профилю
    info = s.articulation_joint3d()
    r = info["sides"]["right"]
    assert info["available"] and not info["applied"] and r["applied"] and "error" in info["sides"]["left"]
    assert np.allclose(r["path"][0], [0.0, -0.2, 0.0])  # путь в окне — от мыщелка в прикусе
    assert r["flat_deg"] is not None
    st.sources["sagittal_right_deg"] = "вручную"
    st.sagittal_right_deg = 22.0
    s._apply_joint3d()
    assert st.sagittal_right_deg == 22.0  # поставленное врачом не трогается
    assert len(calls) == 1  # суставы — из кэша


def test_export_takes_3d_paths(tmp_path, monkeypatch):
    s, _calls = _session(tmp_path, monkeypatch, left=None)
    monkeypatch.setattr(j3, "case_joints", lambda skull, mandible, frame, condyles, bite: {
        side: j3.joint(skull.vertices, skull.faces, head_points(), frame, np.zeros(3), "right", bite)
        for side in ("right", "left")})
    s._apply_joint3d()
    paths = s.joint3d_paths()
    assert set(paths) == {"right", "left"} and paths["right"].shape[1] == 2
    info = s.articulation_joint3d(details=True)
    assert info["applied"] and info["sides"]["right"]["bennett_deg"] == pytest.approx(BETA, abs=3.0)
    assert info["sides"]["right"]["anatomic_deg"] is not None
    # выгрузка: пути из 3D, плоский профиль не считается
    box = trimesh.creation.box((10, 10, 10))
    item = {"role": "jaw", "path": None, "reg": SimpleNamespace(jaw="upper", transform=np.eye(4)),
            "scan": SimpleNamespace(vertices=box.vertices, faces=box.faces)}
    lo = dict(item, reg=SimpleNamespace(jaw="lower", transform=np.eye(4)))
    session = SimpleNamespace(jaw=SimpleNamespace(anatomy=SimpleNamespace(frame=np.eye(4)), lower_fix=np.eye(4),
                                                  settings=kin.Settings(), movements={}, mounting="ct"),
                              bite_motion=None, scans={"u": item, "l": lo}, structures={}, landmarks={},
                              suggested=set(), landmark_sources={}, _jaw_items=lambda: (item, lo),
                              joint3d_paths=lambda: paths)
    monkeypatch.setattr(xa, "ArticulatorCase", lambda *a, **k: SimpleNamespace(args=a, kwargs=k))
    got = xa.case_from_session(session)
    assert got.args[6] is paths
