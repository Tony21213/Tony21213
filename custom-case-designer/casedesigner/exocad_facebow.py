"""Цифровая лицевая дуга для exocad: шарнир артикулятора — на мыщелках пациента, гипсовка — по горизонтали
монтажа (эстетической плоскости лица или функциональной). Движения exocad считает сам — по зубам или по
резцовому столику; траектории не нужны.

Только официальные форматы exocad (docs/exocad.md, «Артикуляторы»):

* файл движений Zebris (`dental_measurement`, .jawmotion) в системе «ось — плоскость» (axis_orbital).
  У нас это система монтажа: начало — середина между мыщелками, x — влево, y — вверх (нормаль
  горизонтали), z — вперёд. Верхняя челюсть привязана вилкой (bite_fork): в файле — метки вилки;
* скан маркера — вилка Zebris SD (STL из библиотеки exocad, library\\movementregister\\zebris_type_sd)
  перед резцами, в координатах сканов. Ни с одной челюстью она не пересекается и не совпадает: копия
  верхнего скана в маркере ложится на сам скан поверхность в поверхность, вилка между челюстями проходит
  сквозь нижние зубы — и в exocad щелчок попадает в маркер, ничего не выбрать. exocad находит вилку на
  скане маркера по геометрии, по меткам переводит модели в систему регистратора и ставит их в артикулятор;
* свой артикулятор (папка для library\\articulator): перевод из системы регистратора без наклона, как у
  SAM 2P, — горизонталь артикулятора и есть горизонталь монтажа, середина шарнирной оси — середина между
  мыщелками пациента.

Ось артикулятора прямая, а мыщелки пациента могут стоять на разной высоте и глубине: ось идёт через
середину между ними вдоль горизонтали монтажа, насколько каждый мыщелок от неё — в отчёте.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import trimesh

from .register import apply, rigid

ZEBRIS_SD = os.path.join("library", "movementregister", "zebris_type_sd")
ZEBRIS_SD_STL = "movementregister-zebris_type_sd.stl"
ZEBRIS_SD_NAME, ZEBRIS_SD_ID = "Bite fork type SD", "REF1960320"
# Система монтажа (x — вправо, y — вперёд, z — вверх) → регистратор Zebris «ось — плоскость» (x — влево,
# y — вверх, z — вперёд). В STL вилки оси те же, что у регистратора: метки (0,0,0) и (±25, 0, 30.5).
TO_REGISTER = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
# Регистратор → артикулятор exocad (запись «точка-строка»): у всех артикуляторов exocad середина шарнирной
# оси в (30, −80, 60); без наклона (как у SAM 2P) горизонталь артикулятора — горизонталь регистратора.
REGISTER_TO_ARTICULATOR = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [30, -80, 60, 1]], float)
# Метка 1 вилки от резцовой точки (вправо, вперёд, вверх), мм: вилка — перед резцами на их высоте (её задний
# край в 4.5 мм перед режущим краем), не пересекает челюсти. Настоящая вилка лежит на окклюзионной плоскости
# (пример exocad 012: метка 1 — в 34 мм позади режущего края), но в прикусе сканов она проходила бы сквозь
# нижние зубы; exocad находит вилку на скане маркера по геометрии, где бы она ни стояла.
FORK_OFFSET_MM = np.array([0.0, 15.0, 0.0])
ORBITAL = (-30.0, 0.0, 70.0)  # точка горизонтали справа (в файле Zebris — орбитальная)
OPENING_DEG, OPENING_FRAMES, FREQUENCY = 6.0, 61, 60  # короткое шарнирное открывание: в файле должно быть движение
ARTICULATOR_NAME = "Custom Case Designer"


@dataclass
class Register:
    """Вилка регистратора: сетка и три метки в её системе (из библиотеки exocad)."""

    fork: trimesh.Trimesh
    marks: np.ndarray  # 3×3


@dataclass
class Facebow:
    case_to_register: np.ndarray  # 4×4: координаты кейса (сканов) → система регистратора
    fork_pose: np.ndarray  # 4×4: вилка → координаты кейса
    marks: np.ndarray  # метки вилки в системе регистратора
    condyles: dict  # right/left: мыщелки в координатах кейса
    off_axis_mm: dict  # right/left: насколько мыщелок от оси артикулятора
    notes: list = field(default_factory=list)


def find_exocad(roots=(r"C:\Exo", r"C:\exocad", r"C:\Program Files\exocad", r"D:\exocad")) -> str | None:
    """Папка DentalCADApp, в которой есть вилка Zebris SD (самая новая версия)."""
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _files in os.walk(root):
            if os.path.basename(dirpath) == "DentalCADApp":
                if os.path.isfile(os.path.join(dirpath, ZEBRIS_SD, ZEBRIS_SD_STL)):
                    found.append(dirpath)
                dirnames[:] = []
            elif dirpath.count(os.sep) - root.count(os.sep) >= 3:
                dirnames[:] = []
    return max(found, key=os.path.getmtime) if found else None


def load_register(path: str) -> Register:
    """Вилка Zebris SD: path — DentalCADApp exocad или сама папка вилки."""
    folder = path if os.path.isfile(os.path.join(path, ZEBRIS_SD_STL)) else os.path.join(path, ZEBRIS_SD)
    stl = os.path.join(folder, ZEBRIS_SD_STL)
    if not os.path.isfile(stl):
        raise FileNotFoundError(f"нет вилки Zebris SD: {folder}")
    meta = ET.parse(os.path.join(folder, ZEBRIS_SD_STL.replace(".stl", ".metadata"))).getroot()
    marks = np.array([[float(p.findtext(k)) for k in "xyz"] for p in meta.iter("Point")])
    if marks.shape != (3, 3):
        raise ValueError(f"у вилки должно быть три метки: {folder}")
    return Register(trimesh.load_mesh(stl, process=False), marks)


def facebow(frame: np.ndarray, condyle_right, condyle_left, incisal, register: Register) -> Facebow:
    """Лицевая дуга по системе монтажа (frame: координаты кейса → x вправо, y вперёд, z вверх) и мыщелкам."""
    F = np.asarray(frame, float)
    cases = {"right": np.asarray(condyle_right, float), "left": np.asarray(condyle_left, float)}
    mounted = {side: apply(F, p) for side, p in cases.items()}
    mid = (mounted["right"] + mounted["left"]) / 2
    to_register = TO_REGISTER @ rigid(np.eye(3), -mid) @ F
    off_axis = {side: round(float(np.linalg.norm((p - mid)[1:])), 2) for side, p in mounted.items()}
    # Вилка: оси — как у регистратора, на окклюзионной плоскости верхних зубов (как настоящая).
    R = TO_REGISTER[:3, :3].T
    start = apply(F, incisal) + FORK_OFFSET_MM
    fork_pose = np.linalg.inv(F) @ rigid(R, start - R @ register.marks[0])
    marks = apply(to_register @ fork_pose, register.marks)
    notes = []
    worst = max(off_axis.values())
    if worst > 1.0:
        notes.append(f"Мыщелки пациента стоят несимметрично к горизонтали монтажа: ось артикулятора проходит через "
                     f"середину между ними, мыщелок от неё — до {worst:.1f} мм.")
    return Facebow(to_register, fork_pose, marks, cases, off_axis, notes)


def _sub(parent, tag, text=None):
    e = ET.SubElement(parent, tag)
    if text is not None:
        e.text = text
    return e


def _xyz(parent, tag, p):
    e = _sub(parent, tag)
    for k, v in zip("xyz", p):
        _sub(e, k, f"{v:.3f}")
    return e


def _hinge_rotation(deg: float) -> np.ndarray:
    """Поворот нижней челюсти вокруг шарнирной оси (ось x регистратора): резцы — вниз."""
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


MOVEMENT_TYPES = {"Открывание": "opening", "Протрузия": "protrusion", "Жевание": "chewing"}  # типы Zebris


def movement_type(name: str) -> str:
    """Тип движения Zebris по названию записи (латеротрузия вправо — lateral_rt)."""
    low = name.lower()
    if "латеро" in low or "lateral" in low:
        return "lateral_rt" if ("прав" in low or "right" in low) else "lateral_lt"
    return next((v for k, v in MOVEMENT_TYPES.items() if k.lower() in low), "custom")


def tracks(fb: Facebow, transforms: np.ndarray) -> np.ndarray:
    """Траектории меток (кадры × 3 × xyz) в системе регистратора: метки движутся с нижней челюстью;
    transforms — положения нижней челюсти относительно верхней по кадрам, координаты кейса."""
    A = fb.case_to_register
    marks_case = apply(np.linalg.inv(A), fb.marks)
    return np.array([apply(A @ M, marks_case) for M in np.asarray(transforms, float)])


def jawmotion_xml(fb: Facebow, description: str = "", movements=None) -> bytes:
    """Файл лицевой дуги в формате Zebris (`dental_measurement`) — без данных пациента.

    movements — [(название, положения нижней челюсти (N×4×4, координаты кейса), частота кадров)]; без них —
    короткое шарнирное открывание (в файле должно быть движение)."""
    from . import __version__

    root = ET.Element("dental_measurement", {"xmlns": "http://www.zebris.de/JMA"})
    _sub(root, "program", "Custom Case Designer")
    _sub(root, "program_version", __version__)
    _sub(root, "format_version", "1.0")
    _sub(root, "measuring_system", "")
    _sub(root, "bite_fork_name", ZEBRIS_SD_NAME)
    _sub(root, "bite_fork_id", ZEBRIS_SD_ID)
    patient = _sub(root, "patient")
    for tag in ("first_name", "last_name", "born", "sex", "code"):
        _sub(patient, tag, "")
    _sub(root, "measured", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _sub(root, "description", description)
    _sub(root, "coordinate_system", "axis_orbital")

    def marks(parent):
        for i, p in enumerate(fb.marks, 1):
            _xyz(parent, f"mark_{i}", p)

    upper = _sub(root, "upper_position")
    _sub(upper, "type", "bite_fork")
    marks(_sub(upper, "points"))
    positions = _sub(root, "positions")
    for kind in ("scan_position", "habitual_occlusion", "intercuspitation_max"):
        pos = _sub(positions, "position")
        _sub(pos, "type", kind)
        _sub(pos, "id", kind)
        marks(pos)
    point = _sub(_sub(root, "points"), "point")
    _sub(point, "type", "orbital")
    _sub(point, "id", "orbital")
    for k, v in zip("xyz", ORBITAL):
        _sub(point, k, f"{v:.3f}")
    if not movements:
        hinge = np.array([[_hinge_rotation(a) @ p for p in fb.marks] for a in np.linspace(0.0, OPENING_DEG, OPENING_FRAMES)])
        moves = [("opening", hinge, FREQUENCY)]
    else:
        moves = [(movement_type(name), tracks(fb, T), freq) for name, T, freq in movements]
    parent = _sub(root, "movements")
    used = {}
    for kind, frames, freq in moves:
        movement = _sub(parent, "movement")
        used[kind] = used.get(kind, 0) + 1
        _sub(movement, "type", kind)
        _sub(movement, "id", kind if used[kind] == 1 else f"{kind}_{used[kind]}")
        node = _sub(movement, "tracks")
        for i in range(3):
            track = _sub(node, "track")
            _sub(track, "type")
            _sub(track, "id", f"mark_{i + 1}")
            _sub(track, "size", str(len(frames)))
            _sub(track, "frequency", f"{freq:g}")
            quants = _sub(track, "quants")
            for p in frames[:, i]:
                _xyz(quants, "quant", p)
    ET.indent(root, "\t")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")


def read_jawmotion(path: str) -> dict:
    """Чтение файла Zebris/ModJaw (`dental_measurement`): система, метки, точки, движения, настройки."""
    root = ET.parse(path).getroot()
    for e in root.iter():
        e.tag = e.tag.split("}")[-1]

    def pts(e):
        return np.array([[float(m.findtext(k)) for k in "xyz"] for m in e if m.tag.startswith("mark_")])

    upper = root.find("upper_position")
    movements = {}
    for mv in root.findall("movements/movement"):
        tracks = {t.findtext("id"): np.array([[float(q.findtext(k)) for k in "xyz"] for q in t.find("quants")])
                  for t in mv.find("tracks")}
        movements.setdefault(mv.findtext("type"), []).append(tracks)
    return {"coordinate_system": root.findtext("coordinate_system"),
            "upper_type": None if upper is None else upper.findtext("type"),
            "marks": None if upper is None else pts(upper.find("points")),
            "positions": {p.findtext("type"): pts(p) for p in root.findall("positions/position")},
            "points": {p.findtext("type"): np.array([float(p.findtext(k)) for k in "xyz"])
                       for p in root.findall("points/point")},
            "movements": movements,
            "articulators": [a.tag for a in root.findall("articulator_settings/*")]}


def marker_mesh(register: Register, fb: Facebow) -> trimesh.Trimesh:
    """Скан маркера: только вилка на своём месте, без копии верхнего скана (она легла бы на сам скан)."""
    fork = register.fork.copy()
    fork.apply_transform(fb.fork_pose)
    return fork


# --- свой артикулятор ---------------------------------------------------------------------------------

def _param(parent, kind: str, value, low, high):
    p = _sub(parent, f"{kind}Param")
    for tag, v in (("Type", kind), ("Value", value), ("MinValue", low), ("MaxValue", high)):
        _sub(p, tag, f"{v:g}" if not isinstance(v, str) else v)


def _matrix(parent, tag, M):
    m = _sub(parent, tag)
    for i in range(4):
        for j in range(4):
            _sub(m, f"_{i}{j}", f"{M[i, j]:g}")


def write_off(path: str, mesh: trimesh.Trimesh):
    """OFF текстом (exocad читает и текстовый, и свой двоичный)."""
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    with open(path, "w", encoding="ascii", newline="\n") as out:
        out.write(f"OFF\n{len(v)} {len(f)} 0\n")
        out.writelines(f"{a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in v)
        out.writelines(f"3 {a} {b} {c}\n" for a, b, c in f)


def _guide(side: int) -> trimesh.Trimesh:
    """Дорожка мыщелка в его системе: паз над головкой, вперёд (+y) на 15 мм, наружу — стенка Беннетта."""
    roof = trimesh.creation.box((10, 22, 2), trimesh.transformations.translation_matrix((0, 4, 5)))
    wall = trimesh.creation.box((2, 22, 10), trimesh.transformations.translation_matrix((-side * 5, 4, 0)))
    return trimesh.util.concatenate([roof, wall])


def write_articulator(folder: str, icd: float = 110.0, settings: dict | None = None) -> list[str]:
    """Папка своего артикулятора для library\\articulator exocad: перевод из регистратора без наклона,
    геометрия — как у SAM 2P (та же система). settings — значения по умолчанию (TiltCondylarGuideLeft=…)."""
    os.makedirs(folder, exist_ok=True)
    values = {"TiltCondylarGuideLeft": 35, "TiltCondylarGuideRight": 35, "BennettAngleLeft": 10,
              "BennettAngleRight": 10, "ImmediateSideshiftLeft": 0, "ImmediateSideshiftRight": 0,
              "TiltFrontplate": 40, **(settings or {})}
    root = ET.Element("ArticulatorSettings", {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                              "xmlns:xsd": "http://www.w3.org/2001/XMLSchema"})
    for tag, value in (("AnteriorPosteriorDistance", "129.54"), ("IntercondylarDistance", f"{icd:g}"),
                       ("HeightUpperArticulatorPart", "91"), ("HeightFrontPlate", "0.001"),
                       ("HeightIncisalNeedle", "0")):
        _sub(root, tag, value)
    for tag, p in (("ArticulatorPosition", (30, -80, -31)), ("ArticulationPlaneLegRight", (-60, 0, 50.26)),
                   ("ArticulationPlaneLegLeft", (60, 0, 50.26)), ("ArticulationPlaneIncisalNeedle", (0, 0, 24.34))):
        e = _sub(root, tag)
        for k, v in zip("xyz", p):
            _sub(e, k, f"{v:g}")
    vis = _sub(root, "ArticulationPlaneVisualization")
    material = _sub(vis, "Material")
    _sub(material, "Color", "#3A8FD6")
    _sub(material, "Opacity", "0.35")
    _sub(vis, "StippleTransparency", "false")
    _matrix(root, "MovementregisterToArticulatorTransformation", REGISTER_TO_ARTICULATOR)
    for tag, value in (("MoveLowerJawByDefault", "true"), ("ArticulatorOnlyShowDecorationMeshes", "false"),
                       ("ShowArticulatorMeshes", "true"), ("CollapseBennettAngle", "false"),
                       ("CollapseTiltCondylarGuide", "false"), ("CollapseImmediateSideshift", "false"),
                       ("CollapseHeightIncisalNeedleOffset", "false"), ("CollapseRotationAxis", "true"),
                       ("CollapseRotationFrontplateY", "false"), ("CollapseRotationFrontplateZ", "true"),
                       ("CollapseTiltFrontplate", "false"), ("FrontplateAdjustable", "true")):
        _sub(root, tag, value)
    parts = _sub(root, "ArticulatorMainParts")
    for tag, name in (("Incisalneedle", "incisal_needle.off"), ("FrontplateLeft", "incisal_plate_left.off"),
                      ("FrontplateRight", "incisal_plate_right.off")):
        _sub(_sub(parts, tag), "Filename", name)
    for kind, value, low, high in (("Protrusion", 5, 0, 12), ("Retrusion", 0, 0, 2), ("LaterotrusionRight", 5, 0, 12),
                                   ("LaterotrusionLeft", 5, 0, 12), ("BennettAngleLeft", None, -5, 45),
                                   ("BennettAngleRight", None, -5, 45), ("TiltCondylarGuideLeft", None, -20, 75),
                                   ("TiltCondylarGuideRight", None, -20, 75), ("ImmediateSideshiftLeft", None, 0, 3),
                                   ("ImmediateSideshiftRight", None, 0, 3), ("HeightIncisalNeedleOffset", 0, -10, 10),
                                   ("RotationFrontplateYLeft", 0, 0, 60), ("RotationFrontplateYRight", 0, 0, 60),
                                   ("TiltFrontplate", None, 0, 89)):
        _param(root, kind, values[kind] if value is None else value, low, high)
    ET.indent(root, "\t")
    files = {"articulatorparameters.xml": b'<?xml version="1.0"?>\n' + ET.tostring(root, encoding="utf-8")}
    meshes = {
        "condylar_head.off": trimesh.creation.icosphere(subdivisions=3, radius=4.0),
        "condylar_guide_left.off": _guide(-1), "condylar_guide_right.off": _guide(1),
        "incisal_needle.off": trimesh.creation.cylinder(radius=1.5, height=40,
                                                        transform=trimesh.transformations.translation_matrix((0, 0, 20))),
        "incisal_plate_left.off": trimesh.creation.box((12, 16, 2), trimesh.transformations.translation_matrix((6, 0, -1))),
        "incisal_plate_right.off": trimesh.creation.box((12, 16, 2), trimesh.transformations.translation_matrix((-6, 0, -1))),
    }
    for name, data in files.items():
        with open(os.path.join(folder, name), "wb") as f:
            f.write(data)
    for name, mesh in meshes.items():
        write_off(os.path.join(folder, name), mesh)
    return sorted([*files, *meshes])


def export(out_dir: str, frame, condyle_right, condyle_left, incisal, register: Register,
           icd: float | None = None, settings: dict | None = None,
           movements=None, jawmotion_name: str = "facebow.jawmotion", marker_name: str = "movementmarker.stl",
           articulator_dir: str | None = None) -> dict:
    """Всё для exocad: файл лицевой дуги (с движениями, если даны), скан маркера, мыщелки (сферы для проверки)
    и свой артикулятор (в articulator_dir или рядом)."""
    fb = facebow(frame, condyle_right, condyle_left, incisal, register)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, jawmotion_name), "wb") as f:
        f.write(jawmotion_xml(fb, "Custom Case Designer: шарнир на мыщелках пациента, гипсовка по горизонтали монтажа",
                              movements))
    marker_mesh(register, fb).export(os.path.join(out_dir, marker_name))
    spheres = [trimesh.creation.icosphere(subdivisions=2, radius=2.5).apply_translation(p) for p in fb.condyles.values()]
    trimesh.util.concatenate(spheres).export(os.path.join(out_dir, "condyles.stl"))
    if icd is None:
        icd = float(np.linalg.norm(fb.condyles["right"] - fb.condyles["left"]))
    art = os.path.join(articulator_dir or out_dir, ARTICULATOR_NAME)
    files = write_articulator(art, round(icd, 1), settings)
    return {"files": [jawmotion_name, marker_name, "condyles.stl"],
            "articulator": art, "articulator_files": files,
            "off_axis_mm": fb.off_axis_mm, "icd_mm": round(icd, 1), "notes": fb.notes,
            "case_to_register": fb.case_to_register.round(6).tolist()}
