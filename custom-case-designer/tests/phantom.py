"""Синтетическая челюсть: КЛКТ-объём и «внутриротовой скан» той же челюсти.

Форма задаётся функцией со знаком (отрицательна внутри), поэтому и объём, и
скан строятся от одной и той же границы — истинное совмещение известно точно.
"""

import functools

import numpy as np
from skimage import measure

from casedesigner.register import rigid
from casedesigner.volume import Volume

AIR, SOFT, BONE, TOOTH = -1000.0, 150.0, 1300.0, 2500.0
N_TEETH = 12
OCCLUSAL_Z = 10.5  # верхняя челюсть — зеркало нижней относительно этой плоскости
UPPER_VARIANT = 5  # у верхних зубов другой рисунок бугров


def _capsule(p, a, b, r):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1) - r


def _sphere(p, c, r):
    return np.linalg.norm(p - c, axis=1) - r


def _smin(a, b, k=0.8):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def tooth_centres():
    angles = np.linspace(np.radians(12), np.radians(168), N_TEETH)
    return np.c_[22 * np.cos(angles), 20 * np.sin(angles), np.zeros(N_TEETH)], angles


def _mirror(p):
    return np.c_[p[:, 0], p[:, 1], 2 * OCCLUSAL_Z - p[:, 2]]


def teeth_sdf(p, variant=0):
    centres, angles = tooth_centres()
    f = np.full(len(p), 10.0)  # далеко от зубов точное значение не нужно
    for i, (c, a) in enumerate(zip(centres, angles)):
        near = np.flatnonzero((np.linalg.norm(p[:, :2] - c[:2], axis=1) < 8) & (p[:, 2] > -14) & (p[:, 2] < 14))
        f[near] = np.minimum(f[near], _tooth(p[near], c, a, i + variant))
    return f


def _tooth(p, c, a, i):
    radial = np.array([np.cos(a), np.sin(a), 0.0])
    size = 3.6 if abs(np.cos(a)) > 0.6 else 3.0  # «моляры» шире «резцов»
    crown = _capsule(p, c + [0, 0, 4.5], c + [0, 0, 6.5], size)
    root = _capsule(p, c + [0, 0, -10], c + [0, 0, 4], 2.2)
    tooth = _smin(crown, root, 1.0)
    # Бугры разные у каждого зуба — чтобы совмещение не могло «съехать» вдоль дуги.
    for j, ang in enumerate((0.3, 2.2, 4.1)):
        tang = np.cross([0, 0, 1], radial)
        off = 0.45 * size * (np.cos(ang + i) * radial + np.sin(ang + i) * tang)
        tooth = _smin(tooth, _sphere(p, c + off + [0, 0, 8.6 + 0.4 * np.sin(i + j)], 1.4), 0.6)
    return tooth


_ARCH_LO, _ARCH_RES, _ARCH_MAP = np.array([-45.0, -30.0]), 0.05, None


def _arch_distance(p):
    """Расстояние в плоскости XY от дуги зубов (эллипс 22×20 мм, передняя половина)."""
    global _ARCH_MAP
    from scipy import ndimage

    if _ARCH_MAP is None:
        shape = (1800, 1700)  # x, y
        theta = np.linspace(0, np.pi, 20000)
        pix = np.round((np.c_[22 * np.cos(theta), 20 * np.sin(theta)] - _ARCH_LO) / _ARCH_RES).astype(int)
        mask = np.ones(shape, bool)
        mask[pix[:, 0], pix[:, 1]] = False
        _ARCH_MAP = ndimage.distance_transform_edt(mask) * _ARCH_RES
    ij = (p[:, :2] - _ARCH_LO) / _ARCH_RES
    return ndimage.map_coordinates(_ARCH_MAP, ij.T, order=1, mode="nearest")


def bone_sdf(p):
    return np.maximum(_arch_distance(p) - 6.5, np.maximum(p[:, 2] - 2.0, -15 - p[:, 2]))


def gum_sdf(p):
    return np.maximum(_arch_distance(p) - 7.5, np.maximum(p[:, 2] - 3.5, -16 - p[:, 2]))


@functools.lru_cache(maxsize=None)
def make_volume(spacing=0.3, rotation_deg=10.0, noise=25.0, blur=0.6, seed=0) -> Volume:
    """КЛКТ фантома: повёрнутая сетка вокселей с ненулевым началом координат."""
    from scipy import ndimage

    a = np.radians(rotation_deg)
    direction = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    lo, hi = np.array([-34.0, -14.0, -18.0]), np.array([34.0, 34.0, 39.0])
    shape = np.ceil((hi - lo) / spacing).astype(int) + 8  # x, y, z
    origin = np.array([-30.0, -20.0, -18.0])
    vol = Volume(np.zeros(shape[::-1], np.float32), np.full(3, spacing), origin, direction)

    zz, yy, xx = np.indices(vol.data.shape)
    world = vol.to_world(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    h = spacing  # ширина размытия границы при «частичном объёме»
    occ = lambda f: np.clip(0.5 - f / h, 0, 1)
    up = _mirror(world)
    img = AIR + (SOFT - AIR) * occ(np.minimum(gum_sdf(world), gum_sdf(up)))
    b = occ(np.minimum(bone_sdf(world), bone_sdf(up)))
    img = img * (1 - b) + BONE * b
    t = occ(np.minimum(teeth_sdf(world), teeth_sdf(up, UPPER_VARIANT)))
    img = img * (1 - t) + TOOTH * t
    img = ndimage.gaussian_filter(img.reshape(vol.data.shape), blur)
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    vol.data = img.astype(np.float32)
    return vol


def make_scan(jaw="lower", resolution=0.15, seed=1):
    """Скан челюсти: коронки и десна, только видимая со стороны прикуса часть, в мм фантома."""
    lo, hi = np.array([-30.0, -6.0, -4.0]), np.array([30.0, 30.0, 11.0])
    shape = np.ceil((hi - lo) / resolution).astype(int) + 1
    grid = np.stack(np.meshgrid(*[lo[i] + resolution * np.arange(shape[i]) for i in range(3)], indexing="ij"), -1)
    p = grid.reshape(-1, 3)
    variant = UPPER_VARIANT if jaw == "upper" else 0
    f = np.minimum(teeth_sdf(p, variant), gum_sdf(p)).reshape(shape)
    verts, faces, normals, _ = measure.marching_cubes(f, 0.0, spacing=(resolution,) * 3)
    verts += lo
    tri_n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    # Сканер видит то, что обращено к нему (со стороны прикуса), и десну до переходной складки.
    visible = (tri_n[:, 2] > -0.2 * np.linalg.norm(tri_n, axis=1)) & (verts[faces].mean(axis=1)[:, 2] > -1.0)
    faces = faces[visible]
    used = np.unique(faces)
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    verts = verts[used] + np.random.default_rng(seed).normal(0, 0.01, (len(used), 3))
    faces = remap[faces]
    if jaw == "upper":
        verts, faces = _mirror(verts), faces[:, ::-1]
    return verts, faces


def scan_pose(seed=2):
    """Случайное положение скана: его координаты не совпадают с КТ."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(20, 60))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return rigid(R, rng.uniform(-40, 40, 3))
