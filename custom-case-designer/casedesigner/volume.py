"""Чтение КТ: папки DICOM, отдельные файлы, архивы, NIfTI/MHA/NRRD.

Все координаты — миллиметры в системе пациента, как в DICOM (LPS: x — вправо
от пациента к его левой стороне, y — спереди назад, z — снизу вверх).
"""

import os
import tempfile
import zipfile
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

VOLUME_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")


@dataclass
class Volume:
    data: np.ndarray  # яркости, оси (z, y, x) — как отдаёт SimpleITK
    spacing: np.ndarray  # размер вокселя по осям индекса x, y, z, мм
    origin: np.ndarray  # центр вокселя (0, 0, 0) в мм пациента
    direction: np.ndarray  # 3×3, столбцы — направления осей индекса в мм пациента

    @property
    def index_to_world(self) -> np.ndarray:
        """Матрица 4×4: индекс (x, y, z), можно дробный → мм пациента."""
        m = np.eye(4)
        m[:3, :3] = self.direction * self.spacing
        m[:3, 3] = self.origin
        return m

    def to_index(self, points: np.ndarray) -> np.ndarray:
        """Мм пациента → дробный индекс (x, y, z)."""
        m = self.index_to_world
        return np.linalg.solve(m[:3, :3], (np.asarray(points, float) - m[:3, 3]).T).T

    def to_world(self, index: np.ndarray) -> np.ndarray:
        m = self.index_to_world
        return np.asarray(index, float) @ m[:3, :3].T + m[:3, 3]

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Яркость в точках (мм пациента), трилинейная интерполяция; вне объёма — минимум."""
        idx = self.to_index(points)
        return ndimage.map_coordinates(
            self.data, idx[:, ::-1].T, order=1, mode="constant", cval=float(self.data.min())
        )

    def gradient(self, points: np.ndarray, h: float = 0.5) -> np.ndarray:
        """Градиент яркости в мм пациента (центральные разности, шаг h вокселя)."""
        idx = self.to_index(points)
        g = np.empty_like(idx)
        for axis in range(3):
            d = np.zeros(3)
            d[axis] = h
            hi = ndimage.map_coordinates(self.data, (idx + d)[:, ::-1].T, order=1, mode="nearest")
            lo = ndimage.map_coordinates(self.data, (idx - d)[:, ::-1].T, order=1, mode="nearest")
            g[:, axis] = (hi - lo) / (2 * h)
        # Градиент по индексу переводится в мм: g_world = M⁻ᵀ g_index.
        return np.linalg.solve(self.index_to_world[:3, :3].T, g.T).T


def _from_sitk(image: sitk.Image) -> Volume:
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        size[3] = 0
        image = sitk.Extract(image, size, [0, 0, 0, 0])
    image = sitk.Cast(image, sitk.sitkFloat32)
    spacing = np.array(image.GetSpacing(), float)
    if not np.all((spacing >= 0.04) & (spacing <= 1.2)):
        raise ValueError(f"размер вокселя {spacing.round(3).tolist()} мм похож на испорченные метаданные")
    data = sitk.GetArrayFromImage(image)
    if min(data.shape) < 2:
        raise ValueError("в файле один срез, а не объём: укажите папку со всей серией")
    return Volume(
        data=data,
        spacing=spacing,
        origin=np.array(image.GetOrigin(), float),
        direction=np.array(image.GetDirection(), float).reshape(3, 3),
    )


def _largest_series(folder: str):
    """Самая длинная DICOM-серия в папке и вложенных папках (рядом часто лежат скауты)."""
    best = []
    for root, _dirs, _files in os.walk(folder):
        for series in sitk.ImageSeriesReader.GetGDCMSeriesIDs(root) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series)
            if len(files) > len(best):
                best = files
    return best


def _read_dicom_folder(folder: str) -> sitk.Image:
    files = _largest_series(folder)
    if not files:
        raise ValueError(f"{folder}: DICOM-серия не найдена")
    if len(files) == 1:
        return sitk.ReadImage(files[0])  # многокадровый .dcm
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute()


def load_volume(path) -> Volume:
    """КТ из папки DICOM, одного .dcm, .zip или NIfTI/MHA/NRRD."""
    path = os.fspath(path)
    if os.path.isdir(path):
        return _from_sitk(_read_dicom_folder(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory(prefix="casedesigner_") as tmp:
            with zipfile.ZipFile(path) as z:
                for member in z.infolist():
                    target = os.path.realpath(os.path.join(tmp, member.filename))
                    if target.startswith(os.path.realpath(tmp) + os.sep) and not member.is_dir():
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with z.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
            return _from_sitk(_read_dicom_folder(tmp))
    if path.lower().endswith(VOLUME_SUFFIXES):
        return _from_sitk(sitk.ReadImage(path))
    # Один файл DICOM: многокадровый — читаем как есть, срез серии — берём всю папку.
    image = sitk.ReadImage(path)
    if image.GetDimension() >= 3 and min(image.GetSize()[:3]) > 1:
        return _from_sitk(image)
    return _from_sitk(_read_dicom_folder(os.path.dirname(os.path.abspath(path))))
