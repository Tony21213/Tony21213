import numpy as np

from crownai.mesh import Mesh, load_stl, nearest_distance, raycast, save_stl


def cube():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    m = Mesh(v, f)
    return m if m.volume() > 0 else m.flipped()


def test_cube_volume_and_watertight():
    m = cube()
    assert abs(m.volume() - 1.0) < 1e-9
    assert m.is_watertight()
    assert not Mesh(m.vertices, m.faces[:-1]).is_watertight()


def test_stl_roundtrip(tmp_path):
    m = cube()
    save_stl(m, tmp_path / "c.stl")
    back = load_stl(tmp_path / "c.stl")
    assert len(back.vertices) == 8 and len(back.faces) == 12
    assert abs(back.volume() - 1.0) < 1e-6


def test_ascii_stl(tmp_path):
    (tmp_path / "t.stl").write_text(
        "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid t\n")
    m = load_stl(tmp_path / "t.stl")
    assert m.faces.shape == (1, 3)


def test_raycast_near_far_and_parallel():
    m = cube()
    origin = np.array([[0.5, 0.5, 0.5]])
    d = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    assert np.allclose(raycast(origin, d, m), [0.5, 0.5])
    below = np.array([[0.2, 0.3, -2.0], [0.7, 0.6, -2.0], [5.0, 5.0, -2.0]])
    up = np.tile([0.0, 0.0, 1.0], (3, 1))
    assert np.allclose(raycast(below, up, m)[:2], [2.0, 2.0])
    assert np.isinf(raycast(below, up, m)[2])
    assert np.allclose(raycast(below, up, m, farthest=True)[:2], [3.0, 3.0])


def test_nearest_distance():
    d, i = nearest_distance(np.array([[0.0, 0, 0], [2.0, 0, 0]]), np.array([[0.0, 0, 1], [3.0, 0, 0]]))
    assert np.allclose(d, [1.0, 1.0]) and list(i) == [0, 1]
