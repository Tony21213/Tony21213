"""Чтение КТ: папки DICOM, отдельные файлы, архивы, NIfTI/MHA/NRRD.

Все координаты — миллиметры в системе пациента, как в DICOM (LPS: x — вправо
от пациента к его левой стороне, y — спереди назад, z — снизу вверх).
"""

import os
import re
import atexit
import shutil
import subprocess
import tarfile
import tempfile
import threading
import zipfile
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

VOLUME_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")
# Архивы с КТ: zip и tar читаются сами; 7z и rar — через tar.exe Windows 10/11 (libarchive).
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".7z", ".rar")


@dataclass
class Volume:
    data: np.ndarray  # яркости, оси (z, y, x) — как отдаёт SimpleITK
    spacing: np.ndarray  # размер вокселя по осям индекса x, y, z, мм
    origin: np.ndarray  # центр вокселя (0, 0, 0) в мм пациента
    direction: np.ndarray  # 3×3, столбцы — направления осей индекса в мм пациента
    device: str = ""  # производитель и модель аппарата из DICOM, если есть

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


DEVICE_TAGS = ("0008|0070", "0008|1090")  # Manufacturer, Manufacturer's Model Name


def _device(get) -> str:
    """«Производитель Модель» из тегов DICOM; get(tag) возвращает значение или None."""
    return " ".join(v.strip() for v in (get(t) for t in DEVICE_TAGS) if v and v.strip())


def _from_sitk(image: sitk.Image, device: str | None = None) -> Volume:
    if device is None:
        device = _device(lambda t: image.GetMetaData(t) if image.HasMetaDataKey(t) else None)
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
        device=device,
    )


def _read_image(path: str) -> sitk.Image:
    """sitk.ReadImage, в том числе по пути не из латиницы.

    На Windows читатели NIfTI, NRRD и MetaImage открывают файл по пути в
    кодировке ANSI, и путь с кириллицей («D:\\Пациенты\\…») не находится
    (DICOM читается). Тогда файл читается из копии во временной папке.
    """
    try:
        return sitk.ReadImage(path)
    except RuntimeError:
        if path.isascii():
            raise
    tmp_root = tempfile.gettempdir()
    if not tmp_root.isascii():
        raise ValueError("не удалось открыть файл КТ по пути с нелатинскими буквами: переименуйте папку "
                         "латиницей или откройте папку DICOM")
    name = os.path.basename(path)
    suffix = next((s for s in VOLUME_SUFFIXES if name.lower().endswith(s)), os.path.splitext(name)[1])
    with tempfile.TemporaryDirectory(prefix="casedesigner_") as tmp:
        copy = os.path.join(tmp, "volume" + suffix)
        shutil.copyfile(path, copy)
        if suffix == ".mhd":  # данные лежат в отдельном файле рядом с заголовком
            with open(path, encoding="latin-1") as f:
                header = f.read()
            data = re.search(r"^ElementDataFile\s*=\s*(.+?)\s*$", header, re.M)
            if data and data.group(1) != "LOCAL":
                shutil.copyfile(os.path.join(os.path.dirname(path), data.group(1)), os.path.join(tmp, "volume.raw"))
                with open(copy, "w", encoding="latin-1") as f:
                    f.write(header[:data.start(1)] + "volume.raw" + header[data.end(1):])
        return sitk.ReadImage(copy)


def _largest_series(folder: str):
    """Самая длинная DICOM-серия в папке и вложенных папках (рядом часто лежат скауты)."""
    best = []
    for root, _dirs, _files in os.walk(folder):
        for series in sitk.ImageSeriesReader.GetGDCMSeriesIDs(root) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series)
            if len(files) > len(best):
                best = files
    return best


def dicom_series(folder: str) -> list[dict]:
    """DICOM-серии папки (и вложенных): описание, модальность, размер — от самой длинной к самой короткой.

    В папке со снимком часто лежат скауты, реконструкции, серии с другим полем;
    по ним пользователь выбирает нужную. Имена пациентов здесь не читаются.
    """
    out = []
    for root, _dirs, _files in os.walk(folder):
        for series in sitk.ImageSeriesReader.GetGDCMSeriesIDs(root) or ():
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series)
            info = {"id": series, "files": len(files), "description": "", "modality": "", "size": None}
            try:
                r = sitk.ImageFileReader()
                r.SetFileName(files[0])
                r.ReadImageInformation()
                get = lambda t: r.GetMetaData(t).strip() if r.HasMetaDataKey(t) else ""  # noqa: E731
                size = list(r.GetSize())
                info.update(description=get("0008|103e"), modality=get("0008|0060"),
                            size=size[:2] + [len(files) if len(files) > 1 else (size[2] if len(size) > 2 else 1)])
            except RuntimeError:
                pass
            out.append(info)
    return sorted(out, key=lambda x: -x["files"])


def _series_files(folder: str, series: str):
    for root, _dirs, _files in os.walk(folder):
        if series in (sitk.ImageSeriesReader.GetGDCMSeriesIDs(root) or ()):
            return sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, series)
    raise ValueError(f"{folder}: нет серии {series}")


def _read_dicom_folder(folder: str, series: str | None = None) -> Volume:
    files = _series_files(folder, series) if series else _largest_series(folder)
    if not files:
        raise ValueError(f"{folder}: DICOM-серия не найдена")
    if len(files) == 1:
        return _from_sitk(sitk.ReadImage(files[0]))  # многокадровый .dcm
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    return _from_sitk(image, _device(lambda t: reader.GetMetaData(0, t) if reader.HasMetaDataKey(0, t) else None))


def is_archive(path: str) -> bool:
    path = os.fspath(path)
    if not os.path.isfile(path):
        return False
    low = path.lower()
    if low.endswith(VOLUME_SUFFIXES):  # .nii.gz — не архив, а сжатый файл
        return False
    return low.endswith(ARCHIVE_SUFFIXES) or zipfile.is_zipfile(path)


def _inside(root: str, name: str) -> str | None:
    """Путь внутри root для имени из архива или None, если имя ведёт наружу."""
    target = os.path.realpath(os.path.join(root, name))
    return target if target.startswith(os.path.realpath(root) + os.sep) else None


def _extract(path: str, folder: str):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                target = _inside(folder, member.filename)
                if target and not member.is_dir():
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
        return
    if not path.lower().endswith((".7z", ".rar")) and tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            for member in t:
                target = _inside(folder, member.name)
                if target and member.isfile():  # ссылки и устройства не нужны
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with t.extractfile(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
        return
    # 7z, rar: tar.exe Windows (bsdtar) распаковывает их и не пишет за пределы папки.
    tar = shutil.which("tar")
    if tar:
        try:
            done = subprocess.run([tar, "-xf", path, "-C", folder], capture_output=True,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            done = None
        if done is not None and done.returncode == 0:
            return
    raise ValueError(f"{os.path.basename(path)}: не удалось распаковать архив — распакуйте его и откройте папку")


class _Unpacked:
    """Последний распакованный архив: выбор серии и загрузка не распаковывают его дважды."""

    def __init__(self):
        self.lock = threading.Lock()
        self.key, self.tmp = None, None
        atexit.register(self.clear)

    def folder(self, path: str) -> str:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
        with self.lock:
            if key != self.key:
                self.clear()
                tmp = tempfile.TemporaryDirectory(prefix="casedesigner_")
                try:
                    _extract(path, tmp.name)
                except BaseException:
                    tmp.cleanup()
                    raise
                self.key, self.tmp = key, tmp
            return self.tmp.name

    def clear(self):
        if self.tmp is not None:
            self.tmp.cleanup()
        self.key, self.tmp = None, None


_unpacked = _Unpacked()


def _volume_files(folder: str) -> list[str]:
    return sorted(os.path.join(root, f) for root, _d, files in os.walk(folder) for f in files
                  if f.lower().endswith(VOLUME_SUFFIXES) and not f.lower().endswith(".raw"))


def series_of(path) -> list[dict]:
    """Серии DICOM папки или архива — чтобы выбрать, какую открыть (у одиночного файла КТ — пусто)."""
    path = os.fspath(path)
    if os.path.isdir(path):
        return dicom_series(path)
    if is_archive(path):
        return dicom_series(_unpacked.folder(path))
    return []


def load_volume(path, series: str | None = None) -> Volume:
    """КТ из папки DICOM (series — какая серия; по умолчанию самая длинная), одного .dcm, архива
    (zip, tar, tar.gz, 7z, rar — с DICOM или файлом NIfTI/MHA/NRRD внутри) или NIfTI/MHA/NRRD."""
    path = os.fspath(path)
    if os.path.isdir(path):
        return _read_dicom_folder(path, series)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if is_archive(path):
        folder = _unpacked.folder(path)
        if series or dicom_series(folder):
            return _read_dicom_folder(folder, series)
        files = _volume_files(folder)
        if not files:
            raise ValueError(f"{os.path.basename(path)}: в архиве нет КТ (DICOM, NIfTI, MHA, NRRD)")
        return _from_sitk(_read_image(max(files, key=os.path.getsize)))
    if path.lower().endswith(VOLUME_SUFFIXES):
        return _from_sitk(_read_image(path))
    # Один файл DICOM: многокадровый — читаем как есть, срез серии — берём всю папку.
    image = sitk.ReadImage(path)
    if image.GetDimension() >= 3 and min(image.GetSize()[:3]) > 1:
        return _from_sitk(image)
    return _read_dicom_folder(os.path.dirname(os.path.abspath(path)))
