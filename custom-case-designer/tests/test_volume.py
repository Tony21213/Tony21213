import os
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk

from casedesigner.volume import load_volume


def rotated_image(spacing=(0.25, 0.25, 0.3)):
    arr = np.random.default_rng(0).integers(-1000, 3000, (20, 24, 28)).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    a = np.radians(12)
    direction = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    img.SetDirection(direction.ravel().tolist())
    img.SetOrigin((-12.5, 30.0, -80.0))
    img.SetSpacing(spacing)
    return img, arr


def check_geometry(vol, img, arr):
    assert vol.data.shape == arr.shape
    assert np.allclose(vol.data, arr)
    for idx in [(0, 0, 0), (27, 23, 19), (3.5, 10.25, 7.75)]:
        expected = img.TransformContinuousIndexToPhysicalPoint(idx)
        assert vol.to_world(np.array([idx]))[0] == pytest.approx(expected, abs=1e-4)
    p = vol.to_world(np.array([[5.0, 6.0, 7.0]]))
    assert vol.to_index(p)[0] == pytest.approx((5, 6, 7))
    assert vol.sample(p)[0] == pytest.approx(arr[7, 6, 5])


def test_nifti(tmp_path):
    img, arr = rotated_image()
    path = str(tmp_path / "ct.nii.gz")
    sitk.WriteImage(img, path)
    check_geometry(load_volume(path), img, arr)


def write_dicom_series(img, folder):
    os.makedirs(folder, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetImageIO("GDCMImageIO")  # у файлов серии нет расширения, как у многих аппаратов
    direction = img.GetDirection()
    series = "1.2.826.0.1.3680043.2.1125.1.42"
    for k in range(img.GetDepth()):
        s = img[:, :, k]
        s.SetMetaData("0020|000e", series)
        s.SetMetaData("0020|0037", "\\".join(f"{v:.9f}" for v in (direction[0], direction[3], direction[6],
                                                                       direction[1], direction[4], direction[7])))
        s.SetMetaData("0020|0032", "\\".join(f"{v:.6f}" for v in img.TransformIndexToPhysicalPoint((0, 0, k))))
        s.SetMetaData("0020|0013", str(k + 1))
        s.SetMetaData("0028|0030", f"{img.GetSpacing()[1]:.6f}\\{img.GetSpacing()[0]:.6f}")
        s.SetMetaData("0018|0050", f"{img.GetSpacing()[2]:.6f}")
        s.SetMetaData("0008|0060", "CT")
        writer.SetFileName(os.path.join(folder, f"IM{k:04d}"))
        writer.Execute(s)


def test_dicom_folder_and_zip(tmp_path):
    img, arr = rotated_image()
    # GDCM на Windows не пишет по пути с кириллицей — пишем латиницей и переименовываем: проверяется чтение.
    write_dicom_series(img, str(tmp_path / "patient" / "series"))
    os.replace(tmp_path / "patient", tmp_path / "Пациент КТ")
    folder = tmp_path / "Пациент КТ" / "series"
    check_geometry(load_volume(str(tmp_path / "Пациент КТ")), img, arr)
    # Один срез серии — читается вся папка.
    check_geometry(load_volume(str(folder / "IM0003")), img, arr)

    archive = tmp_path / "ct.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in os.listdir(folder):
            z.write(folder / name, f"export/{name}")
    check_geometry(load_volume(str(archive)), img, arr)


@pytest.mark.parametrize("suffix", [".nii.gz", ".mha", ".nrrd", ".mhd"])
def test_volume_file_in_cyrillic_folder(tmp_path, monkeypatch, suffix):
    """NIfTI, MHA, NRRD, MHD из папки пациента с кириллицей (на Windows SimpleITK такой путь не открывает)."""
    img, arr = rotated_image()
    (tmp_path / "patient").mkdir()
    sitk.WriteImage(img, str(tmp_path / "patient" / f"ct{suffix}"))
    os.replace(tmp_path / "patient", tmp_path / "Пациент КТ")
    read = sitk.ReadImage

    def windows_like(path, *args, **kwargs):
        if not str(path).isascii():
            raise RuntimeError(f"Exception thrown in SimpleITK ImageFileReader_Execute: unable to open {path}")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(sitk, "ReadImage", windows_like)
    check_geometry(load_volume(str(tmp_path / "Пациент КТ" / f"ct{suffix}")), img, arr)


def test_rejects_broken_spacing(tmp_path):
    img, _ = rotated_image(spacing=(1.0, 1.0, 5.0))
    path = str(tmp_path / "bad.nii.gz")
    sitk.WriteImage(img, path)
    with pytest.raises(ValueError):
        load_volume(path)
