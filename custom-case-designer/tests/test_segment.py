import json

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.cli import main
from casedesigner.segment import Model, Segmenter, predict, taubin, to_grid
from casedesigner.structures import ANATOMY, FDI, TEETH
from casedesigner.volume import Volume
from toy_onnx import linear_model, make_models


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return make_models(str(tmp_path_factory.mktemp("models")))


@pytest.fixture(scope="module")
def segmented(models):
    return Segmenter.from_folder(models, device="cpu").run(phantom.make_volume())


def test_scheme():
    assert len(ANATOMY) == 9 and len(FDI) == 32 and len(TEETH) == 35
    assert FDI[:3] == (11, 12, 13) and FDI[-1] == 48
    assert [s.key for s in TEETH[-3:]] == ["implant", "implant_crown", "bridge"]


def test_model_validation(tmp_path, models):
    with pytest.raises(FileNotFoundError):
        Model.load(str(tmp_path))
    (tmp_path / "model.json").write_text(json.dumps({"scheme": "bones", "spacing": 0.3, "patch": [8, 8, 8],
                                                     "normalization": {"clip": [0, 1], "mean": 0, "std": 1}}))
    with pytest.raises(ValueError):
        Model.load(str(tmp_path))
    with pytest.raises(ValueError):  # модель зубов на месте анатомии
        Segmenter(f"{models}/teeth")


def test_structures_found(segmented):
    assert set(segmented.meshes) == {"mandible", "upper_teeth", "teeth/tooth_11"}
    for name, mesh in segmented.meshes.items():
        tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
        assert tm.is_watertight, name
        assert tm.volume > 0, name  # грани смотрят наружу


def test_surfaces_in_patient_coordinates(segmented):
    # Поверхность зубов — там, где у фантома граница зубов (заглушка режет по яркости 1900,
    # чуть внутри настоящей границы, поэтому допуск — полвокселя).
    v = segmented.meshes["upper_teeth"].vertices
    sdf = np.minimum(phantom.teeth_sdf(v), phantom.teeth_sdf(phantom._mirror(v), phantom.UPPER_VARIANT))
    assert np.abs(sdf).mean() < 0.15
    # Второй проход считался только в области зубов, но дал ту же поверхность.
    a = trimesh.Trimesh(segmented.meshes["upper_teeth"].vertices, segmented.meshes["upper_teeth"].faces)
    b = trimesh.Trimesh(segmented.meshes["teeth/tooth_11"].vertices, segmented.meshes["teeth/tooth_11"].faces)
    assert b.volume == pytest.approx(a.volume, rel=0.01)
    assert b.bounds == pytest.approx(a.bounds, abs=0.05)


def test_sliding_window_matches_single_pass(tmp_path):
    linear_model(str(tmp_path / "small"), "anatomy", [0, 1] + [0] * 8, [0, -0.7] + [-100] * 8, (16, 24, 20))
    linear_model(str(tmp_path / "big"), "anatomy", [0, 1] + [0] * 8, [0, -0.7] + [-100] * 8, (128, 128, 128))
    image = phantom.make_volume().data[40:90, 30:100, 50:120]
    small, big = Model.load(str(tmp_path / "small")), Model.load(str(tmp_path / "big"))
    a = predict(small.session("cpu"), small, image)
    b = predict(big.session("cpu"), big, image)  # снимок меньше окна — дополняется
    assert a.shape == b.shape == (10, *image.shape)
    assert np.allclose(a, b, atol=1e-4)


def test_grid_keeps_patient_coordinates():
    rng = np.random.default_rng(0)
    a = np.radians(20)
    direction = np.array([[np.cos(a), 0, np.sin(a)], [0, -1, 0], [np.sin(a), 0, -np.cos(a)]])  # с отражениями
    vol = Volume(rng.normal(size=(30, 40, 50)).astype(np.float32), np.array([0.2, 0.25, 0.4]),
                 np.array([10.0, -5.0, 3.0]), direction)
    from scipy import ndimage

    vol.data = ndimage.gaussian_filter(vol.data, 3).astype(np.float32)
    grid = to_grid(vol, 0.3)
    assert np.allclose(grid.spacing, 0.3)
    centre = vol.to_world(np.array([[25.0, 20.0, 15.0]]))
    pts = centre + rng.uniform(-2, 2, (50, 3))
    assert np.allclose(grid.sample(pts), vol.sample(pts), atol=0.02 * np.ptp(vol.data))
    # Область расчёта вырезается по мм пациента.
    roi = (centre[0] - 2, centre[0] + 2)
    crop = to_grid(vol, 0.3, roi)
    assert np.all(crop.data.shape[::-1] <= np.ceil(4 * 1.8 / 0.3) + 2)
    assert np.allclose(crop.sample(centre), vol.sample(centre), atol=0.02 * np.ptp(vol.data))


def test_taubin_smooths_without_shrinking():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    noisy = sphere.vertices + np.random.default_rng(0).normal(0, 0.15, sphere.vertices.shape)
    smooth = taubin(noisy, sphere.faces, iterations=10)
    r_noisy, r_smooth = np.linalg.norm(noisy, axis=1), np.linalg.norm(smooth, axis=1)
    assert r_smooth.std() < 0.5 * r_noisy.std()
    assert r_smooth.mean() == pytest.approx(10.0, rel=0.005)


def test_cli_segment(tmp_path, models):
    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(tmp_path / "ct.nii.gz"))
    out = tmp_path / "out"
    assert main(["segment", str(tmp_path / "ct.nii.gz"), "--models", models, "--device", "cpu", "-o", str(out)]) == 0
    assert (out / "mandible.stl").exists() and (out / "teeth" / "tooth_11.stl").exists()
    assert main(["segment", str(tmp_path / "ct.nii.gz"), "--models", str(tmp_path), "-o", str(out)]) == 1
