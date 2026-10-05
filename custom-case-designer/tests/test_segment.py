import json

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.cli import main
from casedesigner.segment import Model, Segmenter, predict, taubin, to_grid
from casedesigner.structures import CATALOGUE, FDI, jaw_of, structure
from casedesigner.volume import Volume
from toy_onnx import NORMALIZATION, linear_model, make_models


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return make_models(str(tmp_path_factory.mktemp("models")))


@pytest.fixture(scope="module")
def segmented(models):
    return Segmenter(models, device="cpu").run(phantom.make_volume())


def test_catalogue():
    assert len(FDI) == 32 and FDI[:3] == (11, 12, 13) and FDI[-1] == 48
    assert jaw_of("mandible") == "lower" and jaw_of("teeth/tooth_36") == "lower" and jaw_of("pulp/pulp_21") == "upper"
    assert jaw_of("teeth/implant") is None and jaw_of("ct_bone") is None  # делятся по частям
    assert structure("teeth/tooth_11").name == "Зуб 11" and structure("что-то").jaw is None
    assert all(s.color.startswith("#") for s in CATALOGUE.values())


def test_model_validation(tmp_path, models):
    with pytest.raises(FileNotFoundError):
        Model.load(str(tmp_path))
    bad = {"name": "x", "spacing": 0.3, "patch": [8, 8, 8], "normalization": NORMALIZATION,
           "labels": ["background", "bone"], "outputs": {"mandible": ["кость"]}}
    (tmp_path / "model.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):  # метка из outputs не описана в labels
        Model.load(str(tmp_path))
    with pytest.raises(ValueError):
        Segmenter(models, only=["нет такой"])
    assert [m.name for m in Segmenter(models).models] == ["bones", "teeth"]  # по priority


def test_structures_found(segmented):
    assert set(segmented.meshes) == {"mandible", "upper_teeth", "hard_tissue", "teeth/tooth_11"}
    assert set(segmented.labels) == {"bones", "teeth"}
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
    # Модель зубов считалась только в области зубов, на другой сетке (RAS, другой шаг),
    # но дала ту же поверхность; «upper_teeth» взят из модели с меньшим priority.
    a = trimesh.Trimesh(segmented.meshes["upper_teeth"].vertices, segmented.meshes["upper_teeth"].faces)
    b = trimesh.Trimesh(segmented.meshes["teeth/tooth_11"].vertices, segmented.meshes["teeth/tooth_11"].faces)
    assert b.volume == pytest.approx(a.volume, rel=0.02)
    assert b.bounds == pytest.approx(a.bounds, abs=0.1)
    # Структура из нескольких меток — их объединение.
    hard = trimesh.Trimesh(segmented.meshes["hard_tissue"].vertices, segmented.meshes["hard_tissue"].faces)
    mand = trimesh.Trimesh(segmented.meshes["mandible"].vertices, segmented.meshes["mandible"].faces)
    assert hard.volume == pytest.approx(mand.volume + a.volume, rel=0.02)


def test_only_wanted_structures(models):
    """Нужна только нижняя челюсть — модель зубов не считается, лишние поверхности не строятся."""
    seg = Segmenter(models, device="cpu")
    assert [m.name for m in seg.plan(lambda k: k == "mandible")] == ["bones"]
    assert [m.name for m in seg.plan(lambda k: k == "teeth/tooth_11")] == ["teeth"]  # у bones нет меток зубов
    result = seg.run(phantom.make_volume(), want=lambda k: k == "mandible")
    assert set(result.meshes) == {"mandible"} and set(result.labels) == {"bones"}


def test_teeth_region_from_whole_head_model(tmp_path, monkeypatch):
    """Область модели зубов — по меткам зубов модели всего снимка (craniofacial), а не по плотности:
    на КЛКТ с большим полем по плотности область раздувается на всю голову."""
    from casedesigner.segment import labels_box

    grid = Volume(np.zeros((40, 50, 60), np.float32), np.full(3, 0.5), np.array([-10.0, -20.0, -5.0]), np.eye(3))
    labels = np.zeros(grid.data.shape, np.int64)
    labels[10:20, 5:25, 30:50] = 2  # «зубы»: z 10–19, y 5–24, x 30–49 вокселей
    lo, hi = labels_box(grid, labels, [2], (1.0, 2.0, 3.0))
    assert lo == pytest.approx(grid.to_world(np.array([[30, 5, 10]]))[0] - [1, 2, 3], abs=0.6)
    assert hi == pytest.approx(grid.to_world(np.array([[49, 24, 19]]))[0] + [1, 2, 3], abs=0.6)
    assert labels_box(grid, np.zeros_like(labels), [2], (1, 1, 1)) is None

    # Сегментатор: модель всего снимка с меткой teeth_upper задаёт область модели зубов.
    linear_model(str(tmp_path / "head"), "head", ["background", "bone", "teeth_upper"], [0.0, 1.0, 2.0],
                 [0.0, -0.7, -2.6], {"mandible": ["bone"]}, (64, 96, 96), priority=1)
    linear_model(str(tmp_path / "teeth"), "teeth", ["background", "tooth_11"], [0.0, 2.0], [0.0, -3.8],
                 {"teeth/tooth_11": ["tooth_11"]}, (64, 96, 96), region={"around": "teeth", "margin_mm": [1, 1, 1]},
                 priority=2)
    from casedesigner import segment

    monkeypatch.setattr(segment, "teeth_box", lambda *a: pytest.fail("область по плотности, а не по модели головы"))
    seg = Segmenter(str(tmp_path), device="cpu")
    assert [m.name for m in seg.plan(lambda k: k == "teeth/tooth_11")] == ["head", "teeth"]  # head — ради области
    result = seg.run(phantom.make_volume(), want=lambda k: k == "teeth/tooth_11")
    assert set(result.meshes) == {"teeth/tooth_11"}
    head_grid, head_labels = result.labels["head"]
    teeth_grid, _ = result.labels["teeth"]
    lo, hi = labels_box(head_grid, head_labels, [2], (1, 1, 1))
    n = np.array(teeth_grid.data.shape[::-1]) - 1
    corners = teeth_grid.to_world(np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])]))
    assert np.all(corners.min(axis=0) <= lo + 0.5) and np.all(corners.max(axis=0) >= hi - 0.5)


def test_canal_kept_only_inside_mandible():
    """Куски канала вне нижней челюсти отбрасываются, внутри — остаются целиком."""
    from casedesigner.segment import Mesh, keep_inside

    grid = Volume(np.zeros((40, 40, 80), np.float32), np.full(3, 0.5), np.zeros(3), np.eye(3))
    mask = np.zeros(grid.data.shape, bool)
    mask[5:35, 5:35, 5:40] = True  # «челюсть»: x 2.5–20 мм, y и z 2.5–17.5 мм
    inside = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    outside = inside.copy()
    inside.apply_translation([10, 10, 10])
    outside.apply_translation([32, 10, 10])  # за челюстью
    both = trimesh.util.concatenate([inside, outside])
    kept = keep_inside(Mesh(both.vertices, both.faces), grid, mask)
    assert len(kept.faces) == len(inside.faces)
    assert np.allclose(kept.vertices.mean(axis=0), [10, 10, 10], atol=0.1)
    assert keep_inside(Mesh(outside.vertices, outside.faces), grid, mask) is None


def test_sliding_window_matches_single_pass(tmp_path):
    for name, patch in (("small", (16, 24, 20)), ("big", (128, 128, 128))):
        linear_model(str(tmp_path / name), name, ["background", "bone"], [0, 1], [0, -0.7], {"mandible": ["bone"]}, patch)
    image = phantom.make_volume().data[40:90, 30:100, 50:120]
    small, big = Model.load(str(tmp_path / "small")), Model.load(str(tmp_path / "big"))
    a = predict(small.session("cpu"), small, image)
    b = predict(big.session("cpu"), big, image)  # снимок меньше окна — дополняется
    assert a.shape == b.shape == (2, *image.shape)
    assert np.allclose(a, b, atol=1e-4)


def test_grid_keeps_patient_coordinates():
    rng = np.random.default_rng(0)
    a = np.radians(20)
    direction = np.array([[np.cos(a), 0, np.sin(a)], [0, -1, 0], [np.sin(a), 0, -np.cos(a)]])  # с отражениями
    vol = Volume(rng.normal(size=(30, 40, 50)).astype(np.float32), np.array([0.2, 0.25, 0.4]),
                 np.array([10.0, -5.0, 3.0]), direction)
    from scipy import ndimage

    vol.data = ndimage.gaussian_filter(vol.data, 3).astype(np.float32)
    grid = to_grid(vol, (0.4, 0.3, 0.3), "RAS")
    assert np.allclose(grid.spacing, (0.3, 0.3, 0.4))  # шаг (z, y, x) → по осям индекса (x, y, z)
    assert np.allclose(grid.direction, np.diag([-1, -1, 1]), atol=0.4)  # оси к R, A, S
    centre = vol.to_world(np.array([[25.0, 20.0, 15.0]]))
    pts = centre + rng.uniform(-2, 2, (50, 3))
    assert np.allclose(grid.sample(pts), vol.sample(pts), atol=0.02 * np.ptp(vol.data))
    # Область расчёта вырезается по мм пациента.
    roi = (centre[0] - 2, centre[0] + 2)
    crop = to_grid(vol, (0.3, 0.3, 0.3), "LPS", roi)
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
    assert main(["segment", str(tmp_path / "ct.nii.gz"), "--models", models, "--only", "bones", "--device", "cpu",
                 "-o", str(tmp_path / "only")]) == 0
    assert not (tmp_path / "only" / "teeth").exists()
    assert main(["segment", str(tmp_path / "ct.nii.gz"), "--models", str(tmp_path), "-o", str(out)]) == 1
