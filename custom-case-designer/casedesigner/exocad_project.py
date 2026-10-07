"""Проект exocad: сканы в его папке и координаты сцены exocad (подробно — docs/exocad.md).

Читаются только официальные файлы проекта (.dentalProject, .scanInfo, .matrix4) и только
матрицы и имена файлов сканов — данные пациента не читаются. Матрицы exocad записаны
«точка-строка» (сдвиг в последней строке); здесь они переводятся в привычную запись
(столбец): p' = M · [x y z 1]ᵀ.

Сцена exocad = файл скана × матрица выгрузки сканера (.matrix4). Когда exocad открыл
проект, он пишет в .scanInfo обратную ей матрицу — из сцены к файлам сканов
(MatrixToScanDataFiles и TransformationMatrix каждого скана); она главнее .matrix4,
который бывает перезаписан.
"""

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

MESH_EXT = (".stl", ".ply", ".obj")


def read_matrix(element) -> np.ndarray:
    """Матрица exocad (ячейки _00…_33, «точка-строка») в записи «столбец»."""
    rows = np.array([[float(element.findtext(f"_{r}{c}")) for c in range(4)] for r in range(4)])
    return rows.T


@dataclass
class ExocadProject:
    folder: str
    scans: list[str]  # файлы сеток в папке проекта (без моделировок *_cad)
    default: np.ndarray  # файл скана → сцена exocad, если у файла нет своей матрицы
    source: str  # откуда матрица: ".scanInfo", ".matrix4" или "нет (единичная)"
    per_file: dict[str, np.ndarray] = field(default_factory=dict)  # имя файла скана → его матрица в сцену
    stem: str = ""  # имя проекта (файл .dentalProject без расширения)

    def to_scene(self, path: str) -> np.ndarray:
        return self.per_file.get(os.path.basename(path).lower(), self.default)

    def match(self, vertices: np.ndarray, tol: float = 1e-3) -> str | None:
        """Файл скана проекта с той же геометрией (тот же скан, в тех же координатах файла)."""
        import trimesh

        v = np.asarray(vertices, float)
        lo, hi = v.min(axis=0), v.max(axis=0)
        sample = v[np.random.default_rng(0).choice(len(v), min(len(v), 500), replace=False)]
        for path in self.scans:
            try:
                mesh = trimesh.load_mesh(path, process=True)
            except (OSError, ValueError):
                continue
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            w = np.asarray(mesh.vertices, float)
            if len(w) != len(v) or not (np.allclose(w.min(axis=0), lo, atol=tol) and np.allclose(w.max(axis=0), hi, atol=tol)):
                continue
            from scipy.spatial import cKDTree

            if cKDTree(w).query(sample)[0].max() <= tol:
                return path
        return None


def find(folder: str) -> ExocadProject | None:
    """Проект exocad в этой папке (рядом лежит .dentalProject) или None."""
    projects = sorted(glob.glob(os.path.join(folder, "*.dentalProject")))
    if not projects:
        return None
    stem = os.path.splitext(projects[0])[0]
    scans = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                   if p.lower().endswith(MESH_EXT) and "_cad" not in os.path.basename(p).lower())
    scan_info = next(iter(glob.glob(stem + ".scanInfo") or sorted(glob.glob(os.path.join(folder, "*.scanInfo")))), None)
    matrix4 = next(iter(glob.glob(stem + ".matrix4") or sorted(glob.glob(os.path.join(folder, "*.matrix4")))), None)
    if scan_info:
        root = ET.parse(scan_info).getroot()
        to_files = root.find("MatrixToScanDataFiles")
        default = np.linalg.inv(read_matrix(to_files)) if to_files is not None else np.eye(4)
        per_file = {}
        for item in root.iter("ScanFile"):
            name, matrix = item.findtext("FileName"), item.find("TransformationMatrix")
            if name and matrix is not None:
                per_file[os.path.basename(name.replace("\\", "/")).lower()] = np.linalg.inv(read_matrix(matrix))
        return ExocadProject(folder, scans, default, ".scanInfo", per_file, os.path.basename(stem))
    if matrix4:
        return ExocadProject(folder, scans, read_matrix(ET.parse(matrix4).getroot()), ".matrix4",
                             stem=os.path.basename(stem))
    return ExocadProject(folder, scans, np.eye(4), "нет (единичная)", stem=os.path.basename(stem))


FRAMEWORK_STL = "{stem}-{jaw}jaw-partialframework_cad.stl"
FRAMEWORK_INFO = "{stem}-{jaw}jaw.partialInfo"
BONE_MATERIAL = "Кость из КТ (KStom Case Designer)"


def _partial_info(file_name: str) -> bytes:
    """Список каркасов в формате exocad (.partialInfo): сетка и её положение (единичное — сетка уже на месте)."""
    from datetime import datetime

    root = ET.Element("PartialInfo")
    item = ET.SubElement(ET.SubElement(root, "PartialFileList"), "PartialFile")
    ET.SubElement(item, "FileName").text = file_name
    m = ET.SubElement(item, "TransformationMatrix")
    for r in range(4):
        for c in range(4):
            ET.SubElement(m, f"_{r}{c}").text = f"{1.0 if r == c else 0.0:.16f}"
    ET.SubElement(item, "MaterialName").text = BONE_MATERIAL
    ET.SubElement(item, "Material").text = "NP_L"
    ET.SubElement(item, "Optimization").text = "Mill"
    axis = ET.SubElement(item, "Axis")
    for k, v in zip("xyz", (0.0, 0.0, 1.0)):
        ET.SubElement(axis, k).text = f"{v:.16f}"
    ET.SubElement(item, "MillingDiameter").text = "0.1000000000000000"
    ET.SubElement(root, "UsedReconstructionFileList")
    ET.SubElement(root, "ProductName").text = "KStom Case Designer"
    ET.SubElement(root, "SaveTime").text = datetime.now().strftime("%Y-%m-%d-%H-%M")
    ET.indent(root, "    ")
    return ET.tostring(root, encoding="utf-8")


def write_bone_frameworks(project: ExocadProject, bones: dict) -> tuple[list[str], list[str]]:
    """Кости из КТ — бюгельными каркасами проекта: exocad подгружает каркасы сам и двигает каждый со своей
    челюстью в артикуляторе (череп и верхняя челюсть — с ВЧ, нижняя челюсть — с НЧ).

    bones — {"upper" | "lower": trimesh} в координатах файла скана своей челюсти: так exocad хранит результаты
    моделировки (пример exocad с ModJaw: модели зубов лежат на скане в координатах файла, а не сцены). Пишутся
    сетка …-upperjaw-partialframework_cad.stl и список каркасов …-upperjaw.partialInfo; состояние моделировки
    (.partialCAD, закрытый формат) не пишется. Настоящий каркас в проекте не перезаписывается.
    Возвращает (записанные файлы, заметки)."""
    files, notes = [], []
    for jaw, mesh in bones.items():
        stl = FRAMEWORK_STL.format(stem=project.stem, jaw=jaw)
        info = FRAMEWORK_INFO.format(stem=project.stem, jaw=jaw)
        stl_path, info_path = os.path.join(project.folder, stl), os.path.join(project.folder, info)
        if os.path.exists(info_path) and b"KStom Case Designer" not in open(info_path, "rb").read():
            notes.append(f"В проекте уже есть бюгельный каркас {'верхней' if jaw == 'upper' else 'нижней'} "
                         "челюсти — кость не записана, каркас не тронут.")
            continue
        mesh.export(stl_path)
        with open(info_path, "wb") as f:
            f.write(_partial_info(stl))
        files += [stl, info]
    return files, notes
