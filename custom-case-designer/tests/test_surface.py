import numpy as np
from scipy import ndimage

from casedesigner.surface import extract_surface, refine_edges
from casedesigner.volume import Volume


def blurred_sphere(radius=8.0, spacing=0.3, inside=2500.0, outside=150.0):
    n = int(2 * (radius + 4) / spacing)
    vol = Volume(np.zeros((n, n, n), np.float32), np.full(3, spacing), np.full(3, -n * spacing / 2), np.eye(3))
    zz, yy, xx = np.indices(vol.data.shape)
    r = np.linalg.norm(vol.to_world(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]), axis=1)
    occ = np.clip(0.5 - (r - radius) / spacing, 0, 1).reshape(vol.data.shape)
    vol.data = ndimage.gaussian_filter(outside + (inside - outside) * occ, 1.0).astype(np.float32)
    return vol


def test_refine_moves_threshold_surface_to_true_edge():
    vol = blurred_sphere()
    # Порог ближе к яркости шара — изоповерхность заметно внутри настоящей границы.
    iso = extract_surface(vol, 2200.0)
    iso_err = np.abs(np.linalg.norm(iso.points, axis=1) - 8.0)
    refined, ok = refine_edges(vol, iso)
    ref_err = np.abs(np.linalg.norm(refined.points[ok], axis=1) - 8.0)
    assert iso_err.mean() > 0.15
    assert ok.mean() > 0.99
    assert ref_err.mean() < 0.02 and ref_err.max() < 0.04
    # Нормали смотрят наружу.
    assert (np.einsum("ij,ij->i", iso.normals, iso.points) > 0).all()


def test_roi_limits_surface():
    vol = blurred_sphere()
    part = extract_surface(vol, 1300.0, roi=(np.array([0.0, -20, -20]), np.array([20.0, 20, 20])))
    assert len(part.points) and part.points[:, 0].min() > -0.5
