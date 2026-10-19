"""Толщина десны над костью и корнями по КТ (casedesigner/gingiva.py) — на синтетике: «кость» — плита, «корень» —
цилиндр, «скан» — точки на известной высоте; сессия, окно (HTTP) и выгрузки."""

import json
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from casedesigner import gingiva as gg
from casedesigner.register import apply, rigid


def slab():
    """Кость: плита 140×60×10 мм, верх — z = 0."""
    return trimesh.creation.box(extents=(140, 60, 10), transform=rigid(np.eye(3), [0, 0, -5.0]))


def grid(z, n=12, half=8.0):
    xs = np.linspace(-half, half, n)
    return np.array([[x, y, z] for x in xs for y in xs])


def test_thickness_over_bone_bins_and_colors():
    heights = [0.5, 1.5, 2.5, 4.0, 8.0]
    pts = np.vstack([grid(h) + [20.0 * (i - 2), 0, 0] for i, h in enumerate(heights)])
    g = gg.thickness(pts, None, [slab()], {})
    for i, h in enumerate(heights):
        t = g.thickness[i * 144:(i + 1) * 144]
        if h < gg.FAR_MM:
            assert np.allclose(t, h, atol=0.12), (h, np.nanmin(t), np.nanmax(t))
            assert (g.under[i * 144:(i + 1) * 144] == gg.BONE).all()
        else:
            assert np.isnan(t).all()  # дальше FAR_MM — не десна над костью
    colors = g.colors()
    expected = [gg.BINS[0][1], gg.BINS[1][1], gg.BINS[2][1], gg.BINS[3][1], gg.NONE]
    for i, color in enumerate(expected):
        assert tuple(colors[i * 144]) == color
    s = g.summary()
    assert s["vertices"] == 4 * 144 and s["shares"]["< 1 мм — тонкая"] == pytest.approx(0.25)


def test_inside_bone_and_inverted_normals():
    pts = np.vstack([grid(1.0), grid(-0.8)])  # вторая половина — глубже кости
    for bone in (slab(), trimesh.Trimesh(slab().vertices, slab().faces[:, ::-1], process=False)):
        g = gg.thickness(pts, None, [bone], {})
        assert np.allclose(g.thickness[:144], 1.0, atol=0.12)
        assert g.inside[144:].all() and (g.thickness[144:] == 0).all()
        assert tuple(g.colors()[200]) == gg.INSIDE
        assert any("глубже кости" in n for n in g.notes)


def test_root_biotype_and_crown_excluded():
    root = trimesh.creation.cylinder(radius=3.0, height=20.0, sections=48).subdivide_to_size(0.8)  # зуб 11, ось z
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    gum = np.vstack([np.c_[3.8 * np.cos(ang), 3.8 * np.sin(ang), np.full_like(ang, z)] for z in (-4.0, -2.0, 0.0)])
    crown = np.c_[3.1 * np.cos(ang), 3.1 * np.sin(ang), np.full_like(ang, 6.0)]  # на зубе — коронка скана
    pts = np.vstack([gum, crown])
    g = gg.thickness(pts, None, [], {11: root})
    assert np.allclose(g.thickness[:180], 0.8, atol=0.12) and (g.tooth[:180] == 11).all()
    assert np.isnan(g.thickness[180:]).all()  # у самой поверхности зуба снаружи — коронка, не десна
    teeth = g.by_tooth()
    assert teeth == [{"fdi": 11, "median_mm": pytest.approx(0.8, abs=0.12), "thin": True, "thin_share": 1.0,
                      "inside_share": 0.0}]
    deep = np.vstack([gum, np.c_[2.0 * np.cos(ang), 2.0 * np.sin(ang), np.full_like(ang, -6.0)],
                      np.c_[2.2 * np.cos(ang), 2.2 * np.sin(ang), np.full_like(ang, -7.0)],
                      np.c_[2.4 * np.cos(ang), 2.4 * np.sin(ang), np.full_like(ang, -8.0)],
                      np.c_[2.6 * np.cos(ang), 2.6 * np.sin(ang), np.full_like(ang, -9.0)]])  # больше половины — внутри корня
    row = gg.thickness(deep, None, [], {11: root}).by_tooth()[0]
    assert row["inside_share"] > 0.5 and row["thin"] is None and row["median_mm"] == pytest.approx(0.8, abs=0.12)
    mask = np.zeros(len(pts), bool)
    mask[:60] = True  # совмещение отметило их коронкой — не десна
    assert np.isnan(gg.thickness(pts, mask, [], {11: root}).thickness[:60]).all()


def _session(tmp_path):
    from casedesigner.app.common import Registration, Scan
    from casedesigner.app.session import Session

    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.case = SimpleNamespace()  # КТ «открыт»: карте нужны только структуры и совмещение
    s.structures = {"maxilla": slab(), "mandible": trimesh.Trimesh(slab().vertices * [1, 1, -1], slab().faces[:, ::-1])}
    up = grid(1.5, n=20)
    lo = grid(-1.5, n=20)
    for jaw, pts in (("upper", up), ("lower", lo)):
        faces = np.zeros((0, 3), int)
        scan = Scan(jaw, pts, faces)
        dev = np.full(len(pts), np.nan)
        dev[:10] = 0.05  # 10 вершин легли на коронки КТ
        reg = Registration(scan, jaw, np.eye(4), dev)
        s.scans[jaw] = {"scan": scan, "path": None, "color": "#fff", "jaw": jaw, "reg": reg, "auto": reg, "role": "jaw",
                        "lower_bite": None, "transform": np.eye(4), "accepted": True, "guided": True}
    return s


def test_session_map_cache_http(tmp_path):
    from casedesigner.app import server as app_server

    s = _session(tmp_path)
    g = s.gum_map("upper")
    assert g is s.gum_map("upper")  # кэш
    assert np.isnan(g.thickness[:10]).all() and np.allclose(g.thickness[10:], 1.5, atol=0.12)
    assert len(s.scan_gingiva("lower")) == 3 * 400
    httpd = app_server.serve(s)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api"
    try:
        with urllib.request.urlopen(f"{base}/gingiva") as r:
            info = json.loads(r.read())
        assert set(info["jaws"]) == {"upper", "lower"} and info["jaws"]["upper"]["median_mm"] == pytest.approx(1.5, abs=0.12)
        assert [l["label"] for l in info["legend"]][0] == "< 1 мм — тонкая"
        with urllib.request.urlopen(f"{base}/scans/upper/gingiva") as r:
            assert len(r.read()) == 3 * 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_exports_write_gingiva_ply(tmp_path):
    """Выгрузка в папку — <скан>_gingiva.ply рядом с картой отклонений; с артикулятором — скан в положении выгрузки."""
    from casedesigner import exocad_articulator_export as xa
    from casedesigner import kinematics as kin
    from casedesigner import motion as mo
    from casedesigner.fusion import export_case

    s = _session(tmp_path)
    regs = [s.scans["upper"]["reg"], s.scans["lower"]["reg"]]
    export_case(str(tmp_path / "out"), regs, {}, frame="dicom", gingiva=s._gum_colors(regs))
    ply = trimesh.load(tmp_path / "out" / "upper_gingiva.ply", process=False)
    assert len(ply.vertices) == 400 and tuple(np.asarray(ply.visual.vertex_colors)[50][:3]) == gg.BINS[1][1][:3]
    # с артикулятором: нижний скан карты — в прикусе выгрузки
    bite = rigid(np.eye(3), [0.0, 0.0, -1.0])
    s.jaw = SimpleNamespace(anatomy=mo.Anatomy(np.eye(4), {"incisal": np.zeros(3), "condyle_right": np.array([50.0, -80, 30]),
                                                           "condyle_left": np.array([-50.0, -80, 30])}, "тест"),
                            lower_fix=np.eye(4), settings=kin.Settings(), movements={}, mounting="ct")
    s.bite_motion = bite
    case = xa.case_from_session(s)
    assert set(case.gingiva) == {"upper", "lower"}
    assert np.allclose(case.gingiva["lower"].vertices, apply(bite, s.scans["lower"]["scan"].vertices))
    assert case.gingiva["upper"].colors.shape == (400, 4)
