"""Цифровая лицевая дуга для exocad: шарнир — на мыщелках пациента, горизонталь артикулятора — горизонталь
монтажа. Проверяется вся цепочка exocad: скан маркера → метки вилки → система регистратора → артикулятор."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from casedesigner import exocad_facebow as ef
from casedesigner.register import apply, rigid

MARKS = np.array([[0, 0, 0], [25, 0, 30.5], [-25, 0, 30.5]], float)  # метки вилки Zebris SD (метаданные exocad)
EXOCAD = ef.find_exocad()


@pytest.fixture
def register():
    fork = trimesh.creation.box((62, 12, 60), trimesh.transformations.translation_matrix((0, 2.7, 20)))
    return ef.Register(fork, MARKS.copy())


@pytest.fixture
def case():
    """Система монтажа, повёрнутая и сдвинутая относительно координат сканов; мыщелки — несимметрично."""
    frame = rigid(Rotation.from_euler("xyz", [8, -5, 30], degrees=True).as_matrix(), np.array([12.0, -40.0, 7.0]))
    inv = np.linalg.inv(frame)
    mounted = {"right": np.array([52.0, -1.5, 0.8]), "left": np.array([-53.0, 1.5, -0.8]),
               "incisal": np.array([0.5, 98.0, -38.0])}
    return frame, {k: apply(inv, v) for k, v in mounted.items()}


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return rigid(R, cb - R @ ca)


def to_articulator(points):
    """Как exocad переводит систему регистратора в артикулятор: запись «точка-строка»."""
    M = ef.REGISTER_TO_ARTICULATOR
    return np.asarray(points) @ M[:3, :3] + M[3, :3]


def test_models_go_into_articulator_by_condyles_and_horizontal(case, register):
    frame, p = case
    fb = ef.facebow(frame, p["right"], p["left"], p["incisal"], register)
    # exocad: вилка на скане маркера (fork_pose) и её метки в файле дают «сканы → регистратор».
    fitted = kabsch(apply(fb.fork_pose, register.marks), fb.marks)
    assert np.abs(fitted - fb.case_to_register).max() < 1e-9
    # Середина между мыщелками — середина шарнирной оси артикулятора (30, −80, 60).
    mid = (p["right"] + p["left"]) / 2
    assert np.abs(to_articulator(apply(fb.case_to_register, mid)) - [30, -80, 60]).max() < 1e-9
    # Горизонталь монтажа — горизонталь артикулятора: оси монтажа = оси артикулятора (x вправо, y вперёд, z вверх).
    axes = np.linalg.inv(frame)[:3, :3]  # столбцы — оси монтажа в координатах сканов
    for k, expected in ((0, [40, -80, 60]), (1, [30, -70, 60]), (2, [30, -80, 70])):  # вправо, вперёд, вверх
        got = to_articulator(apply(fb.case_to_register, mid + axes[:, k] * 10))
        assert np.abs(got - expected).max() < 1e-9, k
    # Мыщелки несимметричны к горизонтали: ось — через середину, отклонение каждого — в отчёте.
    assert fb.off_axis_mm["right"] == pytest.approx(np.hypot(1.5, 0.8), abs=0.01) and fb.notes
    # Вилка — перед резцами, челюсти не пересекает.
    first = apply(frame @ fb.fork_pose, register.marks[:1])[0]
    assert np.abs(first - (np.array([0.5, 98.0, -38.0]) + ef.FORK_OFFSET_MM)).max() < 1e-9


def test_jawmotion_file(case, register, tmp_path):
    frame, p = case
    fb = ef.facebow(frame, p["right"], p["left"], p["incisal"], register)
    path = tmp_path / "facebow.jawmotion"
    path.write_bytes(ef.jawmotion_xml(fb))
    got = ef.read_jawmotion(str(path))
    assert got["coordinate_system"] == "axis_orbital" and got["upper_type"] == "bite_fork"
    assert np.abs(got["marks"] - fb.marks).max() < 1e-3
    assert set(got["positions"]) == {"scan_position", "habitual_occlusion", "intercuspitation_max"}
    assert got["points"]["orbital"][1] == 0  # точка горизонтали
    tracks = got["movements"]["opening"][0]
    assert np.abs(tracks["mark_1"][0] - fb.marks[0]).max() < 1e-3
    # Открывание — вращение вокруг шарнирной оси (ось x регистратора): расстояние до оси не меняется.
    for k, mark in enumerate(fb.marks, 1):
        r = np.hypot(tracks[f"mark_{k}"][:, 1], tracks[f"mark_{k}"][:, 2])
        assert np.abs(r - np.hypot(mark[1], mark[2])).max() < 2e-3
    assert tracks["mark_1"][-1][1] < tracks["mark_1"][0][1]  # резцы — вниз
    text = path.read_text(encoding="utf-8")
    assert "<first_name />" in text or "<first_name></first_name>" in text  # без данных пациента


def test_own_articulator_folder(tmp_path):
    files = ef.write_articulator(str(tmp_path / "art"), icd=104.0, settings={"TiltCondylarGuideLeft": 41.5})
    root = ET.parse(tmp_path / "art" / "articulatorparameters.xml").getroot()
    assert root.tag == "ArticulatorSettings" and root.findtext("IntercondylarDistance") == "104"
    M = root.find("MovementregisterToArticulatorTransformation")
    got = np.array([[float(M.findtext(f"_{i}{j}")) for j in range(4)] for i in range(4)])
    assert np.array_equal(got, ef.REGISTER_TO_ARTICULATOR)
    p = root.find("TiltCondylarGuideLeftParam")
    assert p.findtext("Value") == "41.5" and float(p.findtext("MinValue")) < 41.5 < float(p.findtext("MaxValue"))
    for part in root.find("ArticulatorMainParts"):
        assert part.findtext("Filename") in files
    for name in files:
        if name.endswith(".off"):
            mesh = trimesh.load_mesh(str(tmp_path / "art" / name), process=False)
            assert len(mesh.faces) > 0, name
    head = trimesh.load_mesh(str(tmp_path / "art" / "condylar_head.off"))
    assert np.abs(np.linalg.norm(head.vertices, axis=1) - 4.0).max() < 0.01  # головка — сфера в начале координат


def test_export(case, register, tmp_path):
    frame, p = case
    res = ef.export(str(tmp_path), frame, p["right"], p["left"], p["incisal"], register)
    for name in res["files"]:
        assert (tmp_path / name).is_file(), name
    marker = trimesh.load_mesh(str(tmp_path / "movementmarker.stl"))
    assert len(marker.faces) == len(register.fork.faces)  # только вилка: копия скана мешала бы в exocad
    assert res["icd_mm"] == pytest.approx(np.linalg.norm(p["right"] - p["left"]), abs=0.05)


@pytest.mark.skipif(EXOCAD is None, reason="exocad не установлен")
def test_real_exocad_register_and_samples():
    """На ПК с exocad: вилка Zebris SD из его библиотеки и образцы файлов движений читаются."""
    reg = ef.load_register(EXOCAD)
    assert np.array_equal(reg.marks, MARKS)
    from scipy.spatial import cKDTree
    d = cKDTree(reg.fork.vertices).query(reg.marks)[0]
    assert np.ptp(d) < 0.05  # метки — центры маркеров вилки, на равном расстоянии от поверхности
    samples = os.path.join(os.path.dirname(EXOCAD), "CAD-Data")
    for name, system in (("2025-03-25_99999-012/Facebow_articulator_settings.jawmotion", "axis_orbital"),
                         ("2025-03-25_99999-049/2025-03-25_99999-049.jawmotion", "upper_arch")):
        path = os.path.join(samples, *name.split("/"))
        if os.path.isfile(path):
            got = ef.read_jawmotion(path)
            assert got["coordinate_system"] == system and got["marks"].shape == (3, 3) and got["movements"]
