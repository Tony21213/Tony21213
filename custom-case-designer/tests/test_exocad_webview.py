"""HTML-экспорт exocad: синтетическая сцена в том же двоичном формате (без данных пациентов)."""

import base64
import json
import lzma
import struct

import numpy as np
import pytest
import trimesh

from casedesigner import exocad_webview as ew
from casedesigner.register import apply, axis_angle, rigid

from test_motion import plate, temporal_bone

_FILTER = {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 16, "lc": 3, "lp": 0, "pb": 2}


def _i(v):
    return struct.pack("<i", v)


def _s(text, pad=True):
    b = text.encode()
    return _i(len(b)) + b + (b"\0" * (-len(b) % 4) if pad else b"")


def _packed(values):
    """Упакованные целые OpenCTM: байтовые плоскости, сжатие LZMA1."""
    v = np.asarray(values, dtype=np.uint32).reshape(len(values), -1)
    planes = np.stack([(v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255]).astype(np.uint8)
    comp = lzma.compress(planes.transpose(0, 2, 1).tobytes(), format=lzma.FORMAT_RAW, filters=[_FILTER])
    return struct.pack("<I", len(comp)) + bytes([(2 * 5 + 0) * 9 + 3]) + struct.pack("<I", 1 << 16) + comp


def ctm(verts, tris, method="RAW"):
    head = b"OCTM" + _i(5) + method.encode().ljust(4, b"\0") + _i(len(verts)) + _i(len(tris))
    head += _i(0) + _i(0) + _i(0) + _s("", pad=False)
    if method == "RAW":
        return head + b"INDX" + np.asarray(tris, "<u4").tobytes() + b"VERT" + np.asarray(verts, "<f4").tobytes()
    t = np.array([np.roll(tr, -int(np.argmin(tr))) for tr in tris])
    t = t[np.lexsort((t[:, 1], t[:, 0]))]
    e = t.copy()
    e[0, 1:] -= t[0, 0]
    for k in range(1, len(t)):
        e[k, 0] = t[k, 0] - t[k - 1, 0]
        e[k, 1] = t[k, 1] - (t[k - 1, 1] if t[k, 0] == t[k - 1, 0] else t[k, 0])
        e[k, 2] = t[k, 2] - t[k, 0]
    bits = np.asarray(verts, "<f4").reshape(-1).view("<u4")
    return head + b"INDX" + _packed(e) + b"VERT" + _packed(bits)


def scene_html(path, objects, password=""):
    b = _i(6) + _s("russian") + _i(0) + _s("a") + _s("b") + _s(password)
    b += _s("build") + _s("1.0")
    b += struct.pack("<3f", 1, 0, 0) + struct.pack("<3f", 0, 0, 1) + _i(1) + struct.pack("<3f", 0, 0, 0) + _i(0)
    b += b"\0\0\0\0" * 4 + struct.pack("<3f", 1, 1, 1)
    b += _i(0) + _i(0) + _i(0) + _i(0)
    b += _i(len(objects))
    for tree, data, world in objects:
        b += _i(0) * 3 + b"\x80\x80\x80\0" * 4 + struct.pack("<3f", 1, 1, 0) + b"\0\0\0\0" + struct.pack("<f", 0)
        b += np.asarray(world, "<f4").T.tobytes()  # матрица по столбцам
        b += _i(len(data)) + data + b"\0" * (-len(data) % 4)
        b += _i(0) + struct.pack("<f", 1)
        b += _i(len(tree)) + b"".join(_s(p) + b"\x80\x80\x80\0" for p in tree)
        b += _i(1) + _i(1) + _i(0)
    path.write_text('<script>DentalWebGL.m_Data = {"data": "' + base64.b64encode(b).decode() + '"};</script>',
                    encoding="utf-8")
    return path


def arch_ribbon(z, width=8.0, n=60):
    """Полоса по зубной дуге (x — вправо, y — вперёд): резцы при y = 90, моляры при y = 50."""
    u = np.linspace(-1, 1, n)
    centre = np.c_[25 * u, 90 - 40 * u ** 2]
    tangent = np.c_[np.full(n, 25.0), -80 * u]
    normal = np.c_[tangent[:, 1], -tangent[:, 0]] / np.linalg.norm(tangent, axis=1)[:, None]
    rows = [np.c_[centre + r * normal, np.full(n, z)] for r in np.linspace(-width / 2, width / 2, 5)]
    v = np.vstack(rows)
    k = (np.arange(4)[:, None] * n + np.arange(n - 1)[None]).ravel()
    f = np.vstack([np.c_[k, k + n, k + 1], np.c_[k + 1, k + n, k + n + 1]])
    return v, f


# Анатомия → сцена exocad: оси повёрнуты на 180° вокруг вертикали и сдвинуты (как в реальных экспортах).
TO_SCENE = rigid(axis_angle(np.array([0, 0, 1.0]), np.pi), np.array([30.0, 40, 15]))


def build_scene(tmp_path):
    def obj(tree, mesh, method="RAW"):
        v, f = mesh
        return tree, ctm(apply(TO_SCENE, v), f, method), np.eye(4)

    upper = arch_ribbon(-29.9)
    lower_top, lower_base = arch_ribbon(-30.0), arch_ribbon(-38.0)
    lower = (np.vstack([lower_top[0], lower_base[0]]), np.vstack([lower_top[1], lower_base[1] + len(lower_top[0])]))
    caps = [trimesh.creation.icosphere(2, 4.0) for _ in range(2)]
    for cap, x in zip(caps, (50.0, -50.0)):
        cap.apply_translation([x, 0, -4.0])  # верхушка головки (Co) — в (±50, 0, 0)
    coronoids = [trimesh.creation.icosphere(2, 3.0) for _ in range(2)]
    for cor, x in zip(coronoids, (45.0, -45.0)):
        cor.apply_translation([x, 22.0, -1.0])  # венечные отростки впереди и на 2 мм выше головок
    body = plate(np.array([-45.0, 10, -50]), np.array([90.0, 0, 0]), np.array([0, 80.0, 0]))
    mand = trimesh.util.concatenate([trimesh.Trimesh(*body, process=False), *caps, *coronoids])
    bones = []
    for x, alpha in ((50.0, 40.0), (-50.0, 30.0)):
        bones.append(temporal_bone(alpha, x=(x - 8, x + 8)))
    skull = (np.vstack([b[0] for b in bones]), np.vstack([bones[0][1], bones[1][1] + len(bones[0][0])]))
    pre = plate(np.array([20.0, 60, -29.9]), np.array([6.0, 0, 0]), np.array([0, 6.0, 0]))
    return scene_html(tmp_path / "case.html", [
        obj(["Объекты визуализации", "Иванов Иван Иванович-30.09.2026"], upper),
        obj(["Сканы предпрепов", "верх выровнен.stl (15)"], pre),
        obj(["Полная анатомия", "Полная анатомия (Верх. челюсть)", "15: Понтик полная анатомия"], pre),
        obj(["Бюгельный каркас", "Upper Jaw Skull.stl (17-...-27)"], skull, "MG1"),
        obj(["Бюгельный каркас", "Lower Jaw.stl (37-...-47)"], (mand.vertices, mand.faces)),
        obj(["Сканы челюстей", "Скан челюсти (17-...-27)"], upper, "MG1"),
        obj(["Сканы челюстей", "Скан челюсти (37-...-47)"], lower),
    ])


def test_scene_parts_and_anonymity(tmp_path):
    scene = ew.load(str(build_scene(tmp_path)))
    p = scene.parts()
    assert len(p["upper_scan"].vertices) == 300 and p["upper_scan"].jaw == "upper"  # полный скан, не предпреп
    assert p["lower_scan"].jaw == "lower" and set(p["designs"]) == {15}
    assert p["skull"].kind == "ct" and p["mandible"].kind == "ct" and p["mandible"].jaw == "lower"
    assert "Иванов" not in json.dumps(scene.summary(), ensure_ascii=False)
    v0 = apply(TO_SCENE, arch_ribbon(-29.9)[0])
    assert np.abs(np.sort(p["upper_scan"].vertices, 0) - np.sort(v0, 0)).max() < 1e-4  # координаты сцены


def test_dynamics_from_scene(tmp_path):
    report, case, anatomy = ew.analyze(ew.load(str(build_scene(tmp_path))))
    assert "Иванов" not in json.dumps(report, ensure_ascii=False)
    back = np.linalg.inv(TO_SCENE)
    right = apply(back, anatomy.points["condyle_right"][None])[0]
    assert right == pytest.approx([50, 0, 0], abs=1.0)  # правый — справа, хотя оси сцены развёрнуты
    assert report["intercondylar_mm"] == pytest.approx(100, abs=1.0)
    assert report["eminence"]["right"]["sagittal_deg"] == pytest.approx(40, abs=1.0)
    assert report["eminence"]["left"]["sagittal_deg"] == pytest.approx(30, abs=1.0)
    assert report["settings"]["bennett_right_deg"] == pytest.approx(40 / 8 + 12, abs=0.2)
    assert set(report["contacts"]) == {r.name for r in case.recordings}
    assert all(v is not None for v in report["guidance_deg"].values())


def test_password_protected_export_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="паролем"):
        ew.load(str(scene_html(tmp_path / "p.html", [], password="salt")))


def test_ctm_methods_agree():
    v, f = arch_ribbon(0.0)
    for method in ("RAW", "MG1"):
        verts, tris = ew.decode_ctm(ctm(v, f, method))
        assert np.allclose(verts, v, atol=1e-5)
        assert {tuple(sorted(t)) for t in tris.tolist()} == {tuple(sorted(t)) for t in f.tolist()}
