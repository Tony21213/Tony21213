"""Проект exocad: матрицы сцены (.scanInfo главнее .matrix4, запись «точка-строка») и поиск своего скана."""

import numpy as np
import pytest
import trimesh

import phantom
from casedesigner import exocad_project as ep
from casedesigner.register import apply


def matrix_xml(tag: str, M: np.ndarray) -> str:
    """Матрица в записи exocad: «точка-строка» — ячейки транспонированной матрицы-столбца."""
    rows = np.asarray(M).T
    return f"<{tag}>" + "".join(f"<_{r}{c}>{float(rows[r, c])!r}</_{r}{c}>" for r in range(4) for c in range(4)) + f"</{tag}>"


SCANNER = phantom.scan_pose(6)  # «матрица сканера»: файл скана → сцена exocad
OTHER = phantom.scan_pose(7)


@pytest.fixture
def project(tmp_path):
    verts, faces = phantom.make_scan("lower")
    trimesh.Trimesh(verts, faces).export(tmp_path / "p-lowerjaw.stl")
    trimesh.Trimesh(verts + 5, faces).export(tmp_path / "p-lowerjaw-situ.stl")
    trimesh.Trimesh(verts, faces).export(tmp_path / "p-36-crown_cad.stl")  # моделировка — не скан
    (tmp_path / "p.dentalProject").write_text("<Treatment/>", encoding="utf-8")
    return tmp_path, verts


def test_matrix_is_point_row():
    """Сдвиг в последней строке (_30.._32) — это сдвиг в нашей записи-столбце."""
    import xml.etree.ElementTree as ET

    M = ep.read_matrix(ET.fromstring(matrix_xml("Matrix4", SCANNER)))
    assert M == pytest.approx(SCANNER)
    row = ET.fromstring("<M>" + "".join(f"<_{r}{c}>{(r == c) * 1.0 + (r == 3) * (c < 3) * [3, 4, 5, 0][c]}</_{r}{c}>"
                                         for r in range(4) for c in range(4)) + "</M>")
    assert ep.read_matrix(row)[:3, 3] == pytest.approx([3, 4, 5])


def test_matrix4_and_matching_scan(project):
    folder, verts = project
    assert ep.find(str(folder / "нет")) is None
    (folder / "p.matrix4").write_text(matrix_xml("Matrix4", SCANNER), encoding="utf-8")
    proj = ep.find(str(folder))
    assert proj.source == ".matrix4" and proj.default == pytest.approx(SCANNER)
    assert sorted(p.split("\\")[-1].split("/")[-1] for p in proj.scans) == ["p-lowerjaw-situ.stl", "p-lowerjaw.stl"]
    loaded = trimesh.load_mesh(folder / "p-lowerjaw.stl", process=True).vertices
    assert proj.match(loaded).endswith("p-lowerjaw.stl")
    assert proj.match(apply(OTHER, loaded)) is None  # тот же скан, но в других координатах — не он


def test_scaninfo_wins_over_matrix4(project):
    """exocad пишет в .scanInfo обратные матрицы (сцена → файлы); они главнее .matrix4."""
    folder, _ = project
    (folder / "p.matrix4").write_text(matrix_xml("Matrix4", np.eye(4)), encoding="utf-8")  # перезаписанный
    (folder / "p.scanInfo").write_text(
        "<ScanInfo>" + matrix_xml("MatrixToScanDataFiles", np.linalg.inv(SCANNER))
        + "<ScanFiles><ScanFile><FileName>C:\\cad\\p\\p-lowerjaw.stl</FileName>"
        + matrix_xml("TransformationMatrix", np.linalg.inv(OTHER)) + "</ScanFile></ScanFiles></ScanInfo>",
        encoding="utf-8")
    proj = ep.find(str(folder))
    assert proj.source == ".scanInfo"
    assert proj.default == pytest.approx(SCANNER)
    assert proj.to_scene(str(folder / "p-lowerjaw.stl")) == pytest.approx(OTHER)  # своя матрица файла
    assert proj.to_scene(str(folder / "p-lowerjaw-situ.stl")) == pytest.approx(SCANNER)


def test_no_matrices_is_identity(project):
    folder, _ = project
    proj = ep.find(str(folder))
    assert proj.default == pytest.approx(np.eye(4)) and proj.source.startswith("нет")
