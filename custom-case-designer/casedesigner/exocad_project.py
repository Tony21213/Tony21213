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
        return ExocadProject(folder, scans, default, ".scanInfo", per_file)
    if matrix4:
        return ExocadProject(folder, scans, read_matrix(ET.parse(matrix4).getroot()), ".matrix4")
    return ExocadProject(folder, scans, np.eye(4), "нет (единичная)")
