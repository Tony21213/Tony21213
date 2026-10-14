"""Цветные сканы (PLY с цветом вершин; запись владельца «цветные сканы», 13.10): цвет читается в Scan, хранится в
файле кейса и выгружается PLY с цветом; заголовок исходного PLY (там бывают имя и данные пациента) не переносится."""

from types import SimpleNamespace

import numpy as np
import trimesh

from casedesigner import exocad_articulator_export as xa
from casedesigner.fusion import Registration, Scan, export_case
from test_exocad_articulator_export import case  # noqa: F401


def colored_ply(path, comment="PatientName TESTNAME"):
    box = trimesh.creation.box((10, 8, 6))
    n = len(box.vertices)
    colors = np.c_[np.arange(n) * 30 % 255, np.full(n, 120), np.arange(n) * 7, np.full(n, 255)].astype(np.uint8)
    box.visual.vertex_colors = colors
    data = box.export(file_type="ply", encoding="ascii")
    path.write_bytes(data.replace(b"format ascii 1.0\n", b"format ascii 1.0\ncomment " + comment.encode() + b"\n", 1))
    return box, colors


def test_scan_reads_vertex_colors(tmp_path):
    box, colors = colored_ply(tmp_path / "upper.ply")
    scan = Scan.load(str(tmp_path / "upper.ply"))
    assert np.array_equal(scan.vertices, box.vertices) and np.array_equal(scan.colors[:, :3], colors[:, :3])
    box.export(str(tmp_path / "gray.stl"))
    assert Scan.load(str(tmp_path / "gray.stl")).colors is None
    flat = box.copy()
    flat.visual.vertex_colors = [200, 200, 200, 255]  # одна заливка — не цвет сканера
    flat.export(str(tmp_path / "flat.ply"))
    assert Scan.load(str(tmp_path / "flat.ply")).colors is None


def test_export_writes_colored_scan_as_ply_without_source_header(tmp_path):
    colored_ply(tmp_path / "upper.ply")
    upper = Scan.load(str(tmp_path / "upper.ply"))
    lower = Scan("lower", upper.vertices - [0, 0, 7], upper.faces)
    regs = [Registration(s, jaw, np.eye(4), np.full(len(s.vertices), np.nan))
            for s, jaw in ((upper, "upper"), (lower, "lower"))]
    out = tmp_path / "out"
    report = export_case(str(out), regs, without_ct=True)
    assert "upper.ply" in report["files"] and "lower.stl" in report["files"]  # бесцветный — как раньше
    raw = (out / "upper.ply").read_bytes()
    assert b"TESTNAME" not in raw
    mesh = trimesh.load_mesh(str(out / "upper.ply"), process=False)
    assert np.array_equal(np.asarray(mesh.visual.vertex_colors)[:, :3], upper.colors[:, :3])


def test_case_file_keeps_scan_colors(tmp_path):
    from casedesigner.app.session import Session

    colored_ply(tmp_path / "upper.ply")
    scan = Scan.load(str(tmp_path / "upper.ply"))
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.scans["a"] = {"scan": scan, "path": str(tmp_path / "upper.ply"), "color": "#fff", "jaw": "upper", "reg": None,
                    "transform": None, "auto": None, "accepted": False, "role": "jaw"}
    saved = s.save_case(str(tmp_path / "color.ccdcase"))
    reopened = Session(memory_path=str(tmp_path / "memory2.jsonl"))
    reopened.open_case(saved["path"])
    got = next(iter(reopened.scans.values()))["scan"]
    assert np.array_equal(got.colors, scan.colors)


def test_articulator_export_takes_colors_from_scan(case):
    up = Scan("upper", case.upper.vertices, case.upper.faces, np.asarray(case.upper.colors, np.uint8))
    lo = Scan("lower", case.lower.vertices, case.lower.faces)
    items = {name: {"role": "jaw", "path": None, "reg": SimpleNamespace(transform=np.eye(4)), "scan": s}
             for name, s in (("upper", up), ("lower", lo))}
    session = SimpleNamespace(
        jaw=SimpleNamespace(anatomy=case.anatomy, settings=case.settings, lower_fix=np.eye(4), movements={},
                            mounting="ct"),
        bite_motion=None, scans=items, structures={}, landmarks={}, suggested=set(), landmark_sources={},
        _jaw_items=lambda: (items["upper"], items["lower"]))
    got = xa.case_from_session(session)
    assert np.array_equal(got.upper.colors, up.colors) and got.lower.colors is None
