"""Зубы на фото и уточнение камеры по контурам (casedesigner/photo_teeth.py) — на искусственных зубах: коробочки по
дуге, камера спереди на 1 м; маски «как у нейросети» — проекция настоящей камерой."""

import os

import numpy as np
import pytest
import trimesh

from casedesigner import aesthetic as ae
from casedesigner import photo_teeth as pt

SIZE = (3000, 2000)
ARCH = {11: (4.5, 0.0), 21: (-4.5, 0.0), 12: (11.5, -2.5), 22: (-11.5, -2.5), 13: (17.0, -7.0), 23: (-17.0, -7.0),
        14: (20.5, -13.0), 24: (-20.5, -13.0)}


def teeth():
    """Точки поверхности верхних зубов (коронки 7×6×9 мм, режущий край — z = 0) с нормалями и номерами."""
    pts, nrm, lab = [], [], []
    for n, (x, y) in ARCH.items():
        box = trimesh.creation.box((7.0 if n % 10 < 3 else 6.0, 6.0, 9.0))
        box.apply_translation([x, y, 4.5])
        p, f = trimesh.sample.sample_surface_even(box, 20000, seed=n)
        pts.append(p)
        nrm.append(box.face_normals[f])
        lab.append(np.full(len(p), n))
    return np.vstack(pts), np.vstack(nrm), np.concatenate(lab)


def camera(distance=1000.0, yaw=0.0, pitch=0.0, focal=12000.0):
    R = ae.view_rotation(np.eye(4), pitch, yaw)
    c = np.array([0.0, distance, 4.0]) @ np.eye(3)
    K = np.array([[focal, 0, SIZE[0] / 2], [0, focal, SIZE[1] / 2], [0, 0, 1.0]])
    return ae.Camera(K, R, -R @ c, 0.0, size=SIZE)


def test_crown_labels_skip_gingiva():
    t = {11: np.array([[0.0, 0, 0], [1.0, 0, 0]]), 21: np.array([[10.0, 0, 0]])}
    v = np.array([[0.2, 0, 0], [10.3, 0, 0], [0.0, 0, 2.0], [5.0, 0, 0]])
    assert pt.crown_labels(v, t).tolist() == [11, 21, 0, 0]
    assert pt.crown_labels(v, t, crown=[True, False, True, True]).tolist() == [11, 0, 0, 0]


def test_projection_occluders_and_refine():
    P, N, L = teeth()
    true = camera()
    window = pt.teeth_window(true, P, SIZE)
    grid = pt.grid_for(true, P, L, window, N)
    view = pt.project(true, P, L, window, grid, N)
    assert set(view.labels[view.labels > 0].tolist()) == set(ARCH)
    masks = {n: (view.labels == n, 0.99) for n in ARCH}
    assert all(v == 1.0 for v in pt.iou(view, masks).values())
    # нижние зубы спереди и ниже режущего края загораживают низ коронок
    occl = np.c_[np.random.default_rng(1).uniform(-25, 25, 20000), np.full(20000, 2.0), np.random.default_rng(2).uniform(-6, 1.5, 20000)]
    hidden = pt.project(true, P, L, window, grid, N, occl)
    assert (hidden.labels == -1).sum() > 0 and (hidden.labels > 0).sum() < (view.labels > 0).sum()
    # грубая камера (как по 2 парам точек): развёрнута вокруг зубов на 1,5°, сдвинута на 1,5 мм, фокус — на 4 %;
    # уточнение по краям возвращает её
    from dataclasses import replace

    from scipy.spatial.transform import Rotation

    Q = Rotation.from_rotvec(np.radians([0.8, 1.0, 0.6])).as_matrix()
    c_bad = Q @ true.centre + [1.5, 0.0, -1.0]
    R_bad = true.R @ Q.T
    K = true.K.copy()
    K[0, 0] = K[1, 1] = true.K[0, 0] * 1.04
    bad = replace(true, K=K, R=R_bad, t=-R_bad @ c_bad)
    err = lambda c: float(np.median(np.linalg.norm(c.project(P) - true.project(P), axis=1)))  # noqa: E731
    assert err(bad) > 15
    r = pt.refine(bad, P, L, masks, window, grid, anchor=np.zeros(3), distance_mm=1000.0, normals=N)
    assert err(r.camera) < 2.0 and r.edge_px < 2.5
    assert np.mean(list(r.iou_after.values())) > np.mean(list(r.iou_before.values())) + 0.1


MODELS = os.environ.get("CCD_MODELS", os.path.join(os.path.dirname(__file__), "..", "models"))


@pytest.mark.skipif(not pt.Sam2.available(MODELS), reason="нет SAM 2.1 в models/sam2.1-tiny")
def test_sam_masks_a_disk():
    yy, xx = np.mgrid[0:600, 0:800]
    img = np.full((600, 800, 3), 40, np.uint8)
    disk = (xx - 300) ** 2 + (yy - 320) ** 2 < 110 ** 2
    img[disk] = (230, 225, 210)
    sam = pt.Sam2(MODELS)
    state = sam.encode(img, (100, 100, 700, 560))
    m, score = sam.mask(state, (300, 320))
    ref = disk[100:560, 100:700]
    assert score > 0.8 and (m & ref).sum() / (m | ref).sum() > 0.9
