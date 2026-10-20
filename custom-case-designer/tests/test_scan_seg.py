"""Зубы на внутриротовом скане без КТ (casedesigner/scan_seg.py): виды, отрисовка по соглашениям PyTorch3D (на них
сеть обучали — зеркальный вид перепутал бы стороны), голосование и чистка; сеть ONNX — если есть в models/."""

import os

import numpy as np
import pytest
import trimesh

from casedesigner import scan_seg as ss

MODELS = os.environ.get("CCD_MODELS", os.path.join(os.path.dirname(__file__), "..", "models"))


def test_42_views_on_sphere():
    c = ss.view_positions()
    assert c.shape == (42, 3) and np.allclose(np.linalg.norm(c, axis=1), ss.RADIUS)
    assert len(np.unique(c.round(6), axis=0)) == 42
    for p in c:  # у полюса +Y вид не вырождается
        R = ss._look_at(p)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_view_axes_like_pytorch3d():
    """Камера на +Z смотрит в начало: ось x вида — к −X мира (x вида — влево), y — вверх."""
    R = ss._look_at(np.array([0.0, 0.0, 5.0]))
    assert np.allclose(R, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], atol=1e-9)


def test_render_sides_depth_background():
    """Вид не зеркальный: с +Z (верх — +Y) точка мира на +X — в правой половине снимка, как в PyTorch3D (ось x вида
    смотрит к −X мира — влево от зрителя); глубина — по оси вида."""
    box = trimesh.creation.box((0.2, 0.2, 0.2))
    box.apply_translation([0.4, 0.0, 0.0])
    v, f = np.asarray(box.vertices), np.asarray(box.faces)
    colors = np.full((len(v), 3), 0.5)
    img, pf = ss.render(v, f, colors, np.array([0.0, 0.0, 1.35]), 64)
    cols = np.nonzero((pf >= 0).any(0))[0]
    assert cols.min() > 32  # справа
    assert img[0][pf < 0].min() == 1.0 and (img[3][pf < 0] == -1).all()  # фон белый, глубина −1
    assert img[3][pf >= 0].min() == pytest.approx(1.35 - 0.1, abs=0.01)  # ближняя грань коробки
    assert np.allclose(img[:3][:, pf >= 0], 0.5)


def test_unit_sphere_and_islands():
    v = np.array([[0.0, 0, 0], [10, 4, 2]])
    u = ss.unit(v)
    assert np.isclose(np.linalg.norm(u[1]), 1.0) and np.allclose(u.mean(0), 0)
    # полоса из треугольников: 1000 вершин номера 5 и островок из 3 вершин номера 7 внутри — получает 5
    n = 1000
    f = np.array([[i, i + 1, i + 2] for i in range(n - 2)])
    lab = np.full(n, 5)
    lab[500:503] = 7
    out = ss.clean_islands(f, lab, n, scale=1.0)
    assert (out == 5).all()


def test_aesthetics_takes_scan_numbers_without_ct(monkeypatch, tmp_path):
    """«Эстетика»: сегментации зубов на КТ нет — номера для совмещения фото по самому скану (сеть подменена)."""
    import phantom
    from casedesigner.app.session import Session

    for jaw in ("upper", "lower"):
        vv, ff = phantom.make_scan(jaw, resolution=0.6)
        trimesh.Trimesh(vv, ff).export(tmp_path / f"{jaw}.stl")
    s = Session(memory_path=str(tmp_path / "memory.jsonl"), models_dir=str(tmp_path))
    for jaw in ("upper", "lower"):
        s.register(s.add_scan(str(tmp_path / f"{jaw}.stl"))["id"], jaw)
    seen = []
    monkeypatch.setattr(ss, "available", lambda folder: True)
    monkeypatch.setattr(ss.DentalModelSeg, "labels", lambda self, v, f, jaw=None, progress=None: seen.append(jaw) or
                        np.where(np.asarray(v)[:, 2] > np.median(np.asarray(v)[:, 2]), 11, 0))
    labels, normals = s._aes_teeth()
    assert seen == ["upper"] and labels is not None and (labels == 11).sum() > 200 and normals is not None
    assert s._upper_ct()[2]["teeth_source"] == "DentalModelSeg"
    s._aes_teeth()
    assert len(seen) == 1  # кэш


@pytest.mark.skipif(not ss.available(MODELS), reason="нет DentalModelSeg в models/dentalmodelseg")
def test_network_runs_on_phantom():
    import phantom

    v, f = phantom.make_scan("upper", resolution=0.8)
    lab = ss.DentalModelSeg(MODELS).labels(v, f, "upper")
    assert lab.shape == (len(v),) and set(np.unique(lab).tolist()) <= {0} | set(range(11, 29))
