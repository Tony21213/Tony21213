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


def _ellipse(cam, z_top, z_bottom, half_mm, n=40):
    """Контур губ на фото (px): эллипс по ширине half_mm от середины и по высоте — от z_top до z_bottom (мм)."""
    (ul, vt), (ur, vb) = cam.project(np.array([[half_mm, 0.0, z_top], [-half_mm, 0.0, z_bottom]]))
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[(ul + ur) / 2 + abs(ur - ul) / 2 * np.cos(a), (vt + vb) / 2 + (vb - vt) / 2 * np.sin(a)]


def _smile(cam, P, N, L):
    """Фото с улыбкой: кожа, губы, тёмный рот, зубы — проекция камерой cam; верхняя губа закрывает шейки зубов."""
    inner = _ellipse(cam, 6.0, -7.0, 24.0)
    outer = _ellipse(cam, 10.0, -12.0, 30.0)
    window = pt.teeth_window(cam, P, SIZE)
    grid = pt.grid_for(cam, P, L, window, N)
    view = pt.project(cam, P, L, window, grid, N, mouth=inner)
    from skimage.draw import polygon2mask

    img = np.empty((SIZE[1], SIZE[0], 3), np.uint8)
    img[:] = (150, 100, 85)
    img[polygon2mask(SIZE[::-1], outer[:, ::-1])] = (165, 55, 65)
    img[polygon2mask(SIZE[::-1], inner[:, ::-1])] = (50, 15, 20)
    teeth = np.kron(view.labels > 0, np.ones((grid, grid), bool))
    x0, y0 = window[:2]
    img[y0:y0 + teeth.shape[0], x0:x0 + teeth.shape[1]][teeth] = (235, 230, 220)
    return img, inner, outer, view, window, grid


def test_lips_hide_teeth_and_coarse_camera_without_pairs():
    """Вырез губ: зубы вне него закрыты (−2), край по губе в уточнении не участвует; без пар точек — грубое
    положение по силуэту зубов, затем уточнение по маскам."""
    P, N, L = teeth()
    true = camera()
    img, inner, outer, view, window, grid = _smile(true, P, N, L)
    seen = view.index[view.labels > 0]
    assert (view.labels == -2).sum() > 0 and P[seen[seen >= 0], 2].max() < 6.5  # шейки — под губой
    err = lambda c: float(np.median(np.linalg.norm(c.project(P) - true.project(P), axis=1)))  # noqa: E731
    cam, score = pt.coarse_camera(img, P, L, np.eye(4), np.zeros(3), 1000.0, N, lips=outer, mouth=inner)
    assert score > 0.6 and err(cam) < 12.0
    masks = {n: (view.labels == n, 0.99) for n in ARCH if (view.labels == n).sum() >= pt.MIN_CELLS}
    r = pt.refine(cam, P, L, masks, window, grid, anchor=np.zeros(3), distance_mm=1000.0, normals=N, mouth=inner)
    assert err(r.camera) < 3.0 and np.mean(list(r.iou_after.values())) > 0.85  # премоляры — почти под губой


def test_teeth_color_mask_and_mouth_box():
    img = np.full((200, 300, 3), (150, 100, 85), np.uint8)
    img[80:120, 100:200] = (235, 230, 220)
    img[120:130, 100:200] = (165, 55, 65)
    m = pt.teeth_color_mask(img, (50, 50, 250, 150))
    assert m[35:65, 55:145].all() and not m[75:, :].any()
    assert pt.mouth_box((300, 200), [[100, 90], [200, 90], [150, 130]]) == (85, 76, 215, 144)
    assert pt.mouth_box((300, 200)) == (60, 90, 240, 170)


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
