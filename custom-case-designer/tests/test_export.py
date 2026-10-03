import json

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.cli import main
from casedesigner.fusion import Scan, export_case
from casedesigner.register import apply


@pytest.fixture(scope="module")
def two_scans(jaw_ct):
    regs, truth = [], {}
    for jaw, seed in (("upper", 3), ("lower", 2)):
        verts, faces = phantom.make_scan(jaw)
        regs.append(jaw_ct.register(Scan(jaw, apply(phantom.scan_pose(seed), verts), faces)))
        truth[jaw] = verts
    return regs, truth


def load(path):
    """Вершины STL по треугольникам — в том порядке, в каком их записал экспорт."""
    return np.asarray(trimesh.load_mesh(str(path), process=False).vertices)


def per_triangle(points, faces):
    return np.asarray(points)[faces].reshape(-1, 3)


def test_export_in_ct_frame(tmp_path, jaw_ct, two_scans):
    regs, truth = two_scans
    surfaces = jaw_ct.surfaces(step=2)
    report = export_case(str(tmp_path), regs, surfaces, frame="ct")
    for jaw in ("upper", "lower"):
        faces = next(r.scan.faces for r in regs if r.jaw == jaw)
        assert np.abs(load(tmp_path / f"{jaw}.stl") - per_triangle(truth[jaw], faces)).max() < 0.15
        assert (tmp_path / f"{jaw}_deviation.ply").exists()
    teeth = surfaces["ct_teeth"]
    assert np.abs(load(tmp_path / "ct_teeth.stl") - per_triangle(teeth.vertices, teeth.faces)).max() < 1e-4
    saved = json.loads((tmp_path / "case.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(report))
    assert saved["scans"]["upper"]["jaw"] == "upper"


def test_exocad_frame_separate_scans(tmp_path, jaw_ct, two_scans):
    """Сканы выгружены каждый в своих координатах: первый остаётся как есть, второй ставится по КТ."""
    regs, truth = two_scans
    surfaces = jaw_ct.surfaces(step=2)
    report = export_case(str(tmp_path), regs, surfaces)
    upper, lower = regs
    assert np.abs(load(tmp_path / "upper.stl") - per_triangle(upper.scan.vertices, upper.scan.faces)).max() < 1e-4
    to_scan = np.linalg.inv(upper.transform)
    teeth = surfaces["ct_teeth"]
    assert np.abs(load(tmp_path / "ct_teeth.stl") - per_triangle(apply(to_scan, teeth.vertices), teeth.faces)).max() < 1e-3
    assert np.abs(load(tmp_path / "lower.stl") - per_triangle(apply(to_scan, truth["lower"]), lower.scan.faces)).max() < 0.2
    assert report["scans"]["lower"]["placement"] == "ct"


def shared_session(jaw_ct, lower_offset=(0.0, 0.0, 0.0)):
    """Оба скана из одной сессии сканера: общие координаты, прикус со сканера."""
    pose = phantom.scan_pose(4)
    regs = []
    for jaw, offset in (("upper", (0.0, 0.0, 0.0)), ("lower", lower_offset)):
        verts, faces = phantom.make_scan(jaw)
        regs.append(jaw_ct.register(Scan(jaw, apply(pose, verts + offset), faces), jaw=jaw))
    return regs


def test_exocad_frame_shared_session(tmp_path, jaw_ct):
    regs = shared_session(jaw_ct)
    report = export_case(str(tmp_path), regs)
    for reg in regs:  # оба скана — ровно как пришли со сканера
        assert np.abs(load(tmp_path / f"{reg.jaw}.stl") - per_triangle(reg.scan.vertices, reg.scan.faces)).max() < 1e-4
    lower = report["scans"]["lower"]
    assert lower["placement"] == "scanner" and lower["scanner_vs_ct_mm"] < 0.05 and not lower["warnings"]


def test_exocad_frame_bite_mismatch_warns(tmp_path, jaw_ct):
    regs = shared_session(jaw_ct, lower_offset=(0.0, 0.0, -0.3))  # прикус на сканере ошибся на 0.3 мм
    lower = export_case(str(tmp_path), regs)["scans"]["lower"]
    assert lower["placement"] == "scanner"
    assert lower["scanner_vs_ct_mm"] == pytest.approx(0.3, abs=0.05)
    assert any("Прикус" in w for w in lower["warnings"])


def test_cli(tmp_path):
    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(tmp_path / "ct.nii.gz"))
    verts, faces = phantom.make_scan("lower")
    trimesh.Trimesh(apply(phantom.scan_pose(2), verts), faces).export(tmp_path / "lower_scan.stl")

    out = tmp_path / "out"
    assert main(["register", str(tmp_path / "ct.nii.gz"), "--scan", str(tmp_path / "lower_scan.stl"),
                 "--jaw", "lower_scan=lower", "--frame", "ct", "-o", str(out)]) == 0
    assert np.abs(load(out / "lower_scan.stl") - per_triangle(verts, faces)).max() < 0.15
    assert main(["register", str(tmp_path / "ct.nii.gz"), "--scan", "missing.stl", "-o", str(out)]) == 1
