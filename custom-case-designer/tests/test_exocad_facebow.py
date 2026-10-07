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
    # Вилка — перед резцами, как у пациента: оси вилки = оси регистратора (горизонтально, ручкой вперёд,
    # стороной +y вверх). Так вид exocad по умолчанию показывает её той же стороной, что и библиотечную;
    # перевёрнутая вилка давала в exocad перевёрнутые движения.
    first = apply(frame @ fb.fork_pose, register.marks[:1])[0]
    assert np.abs(first - (np.array([0.5, 98.0, -38.0]) + ef.FORK_OFFSET_MM)).max() < 1e-9
    assert np.abs((fb.case_to_register @ fb.fork_pose)[:3, :3] - np.eye(3)).max() < 1e-9


def test_fork_in_mouth_like_real_one(case, register):
    """Вилка во рту, как настоящая в образце exocad 012: метка 1 — на высоте резцовой точки в 34 мм позади неё,
    метки 2–3 — у клыков (в 4 мм позади резцов, по 25 мм в стороны). Метки и треки exocad — на зубном ряду."""
    frame, p = case
    fb = ef.facebow(frame, p["right"], p["left"], p["incisal"], register)
    marks = apply(frame @ fb.fork_pose, register.marks) - apply(frame, p["incisal"][None])[0]  # вправо, вперёд, вверх
    assert np.abs(marks[0] - [0, -34.4, 0]).max() < 1e-6
    for m in marks[1:]:
        assert abs(abs(m[0]) - 25) < 1e-6 and -5 < m[1] < -3 and abs(m[2]) < 1e-6


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


def _roof(mesh, y):
    """Нижняя грань вставки (крыша, по которой скользит головка) над точкой y дорожки, по средней линии."""
    v = np.asarray(mesh.vertices)
    near = v[np.abs(v[:, 1] - y) < 1e-6]
    return near[:, 2].min()


def test_condylar_insert_straight_and_curved():
    """Вставка ССП в системе дорожки, как у exocad: прямой путь под углом дорожки — плоская вставка (нижняя
    грань на высоте радиуса головки, как «Planar» у Harman OSH); изогнутый путь — изгиб, повторяющий отклонение
    пути от прямой; головка, катящаяся по пути, касается крыши и не входит в неё."""
    tilt = 35.0
    t = np.linspace(0, 12, 49)
    straight = np.c_[t * np.cos(np.radians(tilt)), -t * np.sin(np.radians(tilt))]
    flat = ef.condylar_insert(straight, tilt)
    assert flat.is_watertight
    for y in (-4.0, 0.0, 6.0, 15.0):
        assert _roof(flat, y) == pytest.approx(ef.CONDYLAR_HEAD_MM, abs=1e-6)
    # путь круче вначале и положе дальше (выпуклый, как у бугорка): вставка — ниже прямой, потом выше
    ang = np.radians(np.linspace(50, 20, 49))
    step = np.c_[np.cos(ang), -np.sin(ang)] * 0.25
    curved = np.vstack([[0, 0], np.cumsum(step, axis=0)])
    ins = ef.condylar_insert(curved, tilt)
    a = np.radians(tilt)
    for y_fwd, z_up in curved[::8]:
        yl, zl = y_fwd * np.cos(a) - z_up * np.sin(a), y_fwd * np.sin(a) + z_up * np.cos(a)
        y = np.round(yl / 0.25) * 0.25
        if y <= 16:
            assert _roof(ins, y) == pytest.approx(ef.CONDYLAR_HEAD_MM + zl, abs=0.15)


def test_articulator_with_patient_inserts_and_mounting_plane(tmp_path):
    t = np.linspace(0, 12, 49)
    path = np.c_[t * np.cos(np.radians(40)), -t * np.sin(np.radians(40))]
    files = ef.write_articulator(str(tmp_path / "art"), icd=96.0, settings={"TiltCondylarGuideRight": 40,
                                 "TiltCondylarGuideLeft": 40}, inserts={"right": path, "left": path}, plane_height=26.5)
    root = ET.parse(tmp_path / "art" / "articulatorparameters.xml").getroot()
    assert root.findtext("CurrentCondylarInsertColorRight") == ef.INSERT_ID
    meshes = root.findall("ArticulatorMainParts/CondylarInserts/ArticulatorMesh")
    assert {(m.findtext("Side"), m.findtext("Filename")) for m in meshes} == {
        ("Right", "condylar_insert_right.off"), ("Left", "condylar_insert_left.off")}
    for m in meshes:
        assert m.findtext("Id") == ef.INSERT_ID and m.findtext("Filename") in files
        mesh = trimesh.load_mesh(str(tmp_path / "art" / m.findtext("Filename")))
        assert _roof(mesh, 0.0) == pytest.approx(ef.CONDYLAR_HEAD_MM, abs=1e-3)
    # плоскость гипсовки: горизонталь на высоте 26.5 (точки — от ArticulatorPosition)
    pos = float(root.findtext("ArticulatorPosition/z"))
    heights = [float(root.findtext(f"{tag}/z")) + pos for tag in
               ("ArticulationPlaneLegRight", "ArticulationPlaneLegLeft", "ArticulationPlaneIncisalNeedle")]
    assert np.allclose(heights, 26.5)


def test_export(case, register, tmp_path):
    frame, p = case
    upper = trimesh.creation.icosphere(subdivisions=3, radius=20).apply_translation(p["incisal"] + [0, 0, 5])
    res = ef.export(str(tmp_path), frame, p["right"], p["left"], p["incisal"], register, upper=upper,
                    meshes={"upperjaw.stl": upper})
    root = ET.parse(os.path.join(res["articulator"], "articulatorparameters.xml")).getroot()
    plane = float(root.findtext("ArticulationPlaneIncisalNeedle/z")) + float(root.findtext("ArticulatorPosition/z"))
    incisal_art = apply(np.array(res["case_to_articulator"]), p["incisal"][None])[0]
    assert plane == pytest.approx(incisal_art[2], abs=0.01)  # плоскость гипсовки — на высоте резцов
    for name in res["files"]:
        assert (tmp_path / name).is_file(), name
    assert res["icd_mm"] == pytest.approx(np.linalg.norm(p["right"] - p["left"]), abs=0.05)
    from scipy.spatial import cKDTree
    # Всё для проекта — в координатах артикулятора (как сканы образца exocad 012): мыщелки — на шарнирной оси
    # (30, −80, 60), горизонталь монтажа — горизонталь артикулятора.
    fb = ef.facebow(frame, p["right"], p["left"], p["incisal"], register)
    to_art = np.array(res["case_to_articulator"])
    assert np.abs(to_art - fb.case_to_articulator).max() < 1e-5
    mid = apply(to_art, (p["right"] + p["left"])[None] / 2)[0]
    assert np.abs(mid - [30, -80, 60]).max() < 1e-4
    axes = to_art[:3, :3] @ np.linalg.inv(frame)[:3, :3]  # оси монтажа (вправо, вперёд, вверх) в артикуляторе
    assert np.abs(axes - np.eye(3)).max() < 1e-5
    condyles = trimesh.load_mesh(str(tmp_path / "condyles.stl"))
    assert np.abs(condyles.vertices.mean(0) - mid).max() < 0.01
    up_art = trimesh.load_mesh(str(tmp_path / "upperjaw.stl"))
    assert cKDTree(up_art.vertices).query(apply(to_art, upper.vertices))[0].max() < 1e-3
    # Скан маркера «верхняя челюсть на вилке», как UpperJawOnStand у SDI Matrix: копия верхнего скана — на нём
    # самом (exocad совмещает по ней маркер с верхним сканом), вилка — во рту.
    marker = trimesh.load_mesh(str(tmp_path / "movementmarker.stl"))
    assert len(marker.faces) == len(upper.faces) + len(register.fork.faces)
    assert cKDTree(marker.vertices).query(up_art.vertices)[0].max() < 1e-3
    extra = marker.vertices[cKDTree(up_art.vertices).query(marker.vertices)[0] > 0.01]
    fork = apply(to_art @ fb.fork_pose, register.fork.vertices)
    assert cKDTree(extra).query(fork)[0].max() < 1e-3  # остальное — вилка на своём месте
    # exocad: вилка библиотеки на маркере и метки из файла движений → «файл → регистратор»; с переводом
    # регистратор → артикулятор это тождество: модели остаются, где лежат в файлах.
    marks = ef.read_jawmotion(str(tmp_path / "facebow.jawmotion"))["marks"]
    file_to_register = kabsch(apply(to_art @ fb.fork_pose, register.marks), marks)
    assert np.abs(ef.register_to_articulator() @ file_to_register - np.eye(4)).max() < 1e-3


@pytest.mark.skipif(EXOCAD is None, reason="exocad не установлен")
def test_sample_012_models_in_articulator_like_exocad():
    """Образец exocad с настоящей лицевой дугой Zebris (012): вилка библиотеки на его скане маркера и его
    .jawmotion дают «сканы → регистратор»; наша цепочка для тех же сканов ставит модели в артикулятор туда же
    (≤ 0.5 мм, ≤ 0.5°), а система анатомически верна: верхняя челюсть над нижней, резцы впереди оси."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "verify_facebow_sample.py")
    spec = importlib.util.spec_from_file_location("verify_facebow_sample", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.main(EXOCAD)
    assert res["share"] > 0.5 and res["max_mm"] <= 0.5 and res["deg"] <= 0.5
    assert res["upper_y"] > res["lower_y"] and 70 < res["incisal_z"] < 120
    assert res["fork_vs_head_deg"] < 10 and res["ours_vs_real_fork_deg"] < 10  # вилка — как у пациента


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
