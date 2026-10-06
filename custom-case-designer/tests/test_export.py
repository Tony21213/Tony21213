import json

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.cli import main
from casedesigner.fusion import CaseCT, Scan, export_case, split_by_jaw
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


def test_export_in_dicom_frame(tmp_path, jaw_ct, two_scans):
    regs, truth = two_scans
    surfaces = jaw_ct.surfaces(step=2)
    report = export_case(str(tmp_path), regs, surfaces, bite="ct", frame="dicom", ct=jaw_ct)
    for jaw in ("upper", "lower"):
        faces = next(r.scan.faces for r in regs if r.jaw == jaw)
        assert np.abs(load(tmp_path / f"{jaw}.stl") - per_triangle(truth[jaw], faces)).max() < 0.02
        assert (tmp_path / f"{jaw}_deviation.ply").exists()
    teeth = surfaces["ct_teeth"]
    assert np.abs(load(tmp_path / "ct_teeth.stl") - per_triangle(teeth.vertices, teeth.faces)).max() < 1e-4
    saved = json.loads((tmp_path / "case.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(report))
    assert saved["scans"]["upper"]["jaw"] == "upper"


OPEN_DEG = 4.0  # на КТ рот приоткрыт, на сканах — прикус


@pytest.fixture(scope="module")
def open_case():
    """КТ с приоткрытым ртом и оба скана из одной сессии сканера (общие координаты, прикус)."""
    ct = CaseCT(phantom.make_volume(open_deg=OPEN_DEG))
    pose = phantom.scan_pose(4)
    regs = [ct.register(Scan(jaw, apply(pose, phantom.make_scan(jaw)[0]), phantom.make_scan(jaw)[1]), jaw=jaw)
            for jaw in ("upper", "lower")]
    parts = split_by_jaw(ct.surfaces(step=2)["ct_teeth"], ct)
    meshes = {"upper_teeth": parts["upper"], "lower_teeth": parts["lower"], "ct_teeth": ct.surfaces(step=2)["ct_teeth"]}
    return ct, regs, meshes, pose


def test_bite_of_scans(tmp_path, open_case):
    ct, regs, meshes, pose = open_case
    report = export_case(str(tmp_path), regs, meshes, ct=ct)
    upper, lower = regs
    # Сканы — как пришли со сканера.
    for reg in regs:
        assert np.abs(load(tmp_path / f"{reg.jaw}.stl") - per_triangle(reg.scan.vertices, reg.scan.faces)).max() < 1e-4
        assert report["scans"][reg.jaw]["placement"] == "scanner"
    # Верхние структуры — по верхнему скану, нижние переехали к нижнему скану (рот на КТ закрылся).
    closed = pose @ np.linalg.inv(phantom.jaw_opening(OPEN_DEG))
    up, lo = meshes["upper_teeth"], meshes["lower_teeth"]
    assert np.abs(load(tmp_path / "upper_teeth.stl") - per_triangle(apply(pose, up.vertices), up.faces)).max() < 0.05
    assert np.abs(load(tmp_path / "lower_teeth.stl") - per_triangle(apply(closed, lo.vertices), lo.faces)).max() < 0.05
    # Смешанная поверхность разделена: каждая часть ушла со своей челюстью.
    mixed = load(tmp_path / "ct_teeth.stl")
    expected = np.vstack([per_triangle(apply(pose, up.vertices), up.faces),
                          per_triangle(apply(closed, lo.vertices), lo.faces)])
    assert np.abs(mixed - expected).max() < 0.05
    bite = report["ct_bite_vs_scans"]
    assert bite["rotation_deg"] == pytest.approx(OPEN_DEG, abs=0.1) and bite["lower_jaw_on_ct_vs_scans_mean_mm"] > 1


def test_static_ct_bite(tmp_path, open_case):
    ct, regs, meshes, pose = open_case
    export_case(str(tmp_path), regs, meshes, bite="ct", ct=ct)
    upper, lower = regs
    lo = meshes["lower_teeth"]
    # Нижние структуры остались как на КТ (рот открыт), нижний скан переехал на них.
    assert np.abs(load(tmp_path / "lower_teeth.stl") - per_triangle(apply(pose, lo.vertices), lo.faces)).max() < 0.05
    opened = pose @ phantom.jaw_opening(OPEN_DEG) @ np.linalg.inv(pose)
    expected = per_triangle(apply(opened, lower.scan.vertices), lower.scan.faces)
    assert np.abs(load(tmp_path / "lower.stl") - expected).max() < 0.05


def test_bite_of_scans_in_dicom(tmp_path, open_case):
    """Прикус сканов в координатах КТ: верхняя челюсть — как на КТ, нижняя со структурами — сомкнута по сканам."""
    ct, regs, meshes, pose = open_case
    report = export_case(str(tmp_path), regs, meshes, bite="scan", frame="dicom", ct=ct)
    upper, lower = regs
    up, lo = meshes["upper_teeth"], meshes["lower_teeth"]
    closed = np.linalg.inv(phantom.jaw_opening(OPEN_DEG))  # нижняя челюсть КТ → в прикус сканов, в координатах КТ
    assert np.abs(load(tmp_path / "upper_teeth.stl") - per_triangle(up.vertices, up.faces)).max() < 0.05
    assert np.abs(load(tmp_path / "lower_teeth.stl") - per_triangle(apply(closed, lo.vertices), lo.faces)).max() < 0.05
    to_ct = np.linalg.inv(pose)  # сканы — где они на КТ в прикусе сканера
    for reg in regs:
        assert np.abs(load(tmp_path / f"{reg.jaw}.stl")
                      - per_triangle(apply(to_ct, reg.scan.vertices), reg.scan.faces)).max() < 0.05
    assert report["bite"] == "scans" and report["frame"].startswith("DICOM")
    assert report["ct_bite_vs_scans"]["rotation_deg"] == pytest.approx(OPEN_DEG, abs=0.1)


def test_ct_bite_in_scanner_frame(tmp_path, open_case):
    """Прикус КТ в координатах сканера: верхний скан — как со сканера, нижний и все структуры — как на КТ."""
    ct, regs, meshes, pose = open_case
    export_case(str(tmp_path), regs, meshes, bite="ct", frame="exocad", ct=ct)
    upper, lower = regs
    assert np.abs(load(tmp_path / "upper.stl") - per_triangle(upper.scan.vertices, upper.scan.faces)).max() < 1e-4
    lo = meshes["lower_teeth"]
    assert np.abs(load(tmp_path / "lower_teeth.stl") - per_triangle(apply(pose, lo.vertices), lo.faces)).max() < 0.05


def test_separately_exported_scans(tmp_path, jaw_ct, two_scans):
    """Сканы не в прикусе (каждый в своих координатах): второй ставится по КТ."""
    regs, truth = two_scans
    report = export_case(str(tmp_path), regs, ct=jaw_ct)
    upper, lower = regs
    assert report["scans"]["lower"]["placement"] == "ct" and report["notes"]
    to_scan = np.linalg.inv(upper.transform)
    assert np.abs(load(tmp_path / "lower.stl") - per_triangle(apply(to_scan, truth["lower"]), lower.scan.faces)).max() < 0.05


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
                 "--jaw", "lower_scan=lower", "--bite", "ct", "--frame", "dicom", "-o", str(out)]) == 0
    assert np.abs(load(out / "lower_scan.stl") - per_triangle(verts, faces)).max() < 0.15
    assert main(["register", str(tmp_path / "ct.nii.gz"), "--scan", "missing.stl", "-o", str(out)]) == 1
