"""exocad webview import, on a synthetic scene written in the same binary layout."""

import base64
import lzma
import struct

import numpy as np
import pytest

from crownai.webview import case_for_tooth, classify, decode_ctm, load_webview

_FILTER = {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 16, "lc": 3, "lp": 0, "pb": 2}


def _i(v):
    return struct.pack("<i", v)


def _s(text, pad=True):
    b = text.encode()
    return _i(len(b)) + b + (b"\0" * (-len(b) % 4) if pad else b"")


def _packed(values: np.ndarray) -> bytes:
    """OpenCTM packed-integer stream: byte planes, LZMA1 compressed."""
    v = np.asarray(values, dtype=np.uint32).reshape(len(values), -1)
    planes = np.stack([(v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255]).astype(np.uint8)
    raw = planes.transpose(0, 2, 1).tobytes()
    comp = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=[_FILTER])
    props = bytes([(2 * 5 + 0) * 9 + 3]) + struct.pack("<I", 1 << 16)
    return struct.pack("<I", len(comp)) + props + comp


def _ctm(verts, tris, method="RAW"):
    head = b"OCTM" + _i(5) + method.encode().ljust(4, b"\0") + _i(len(verts)) + _i(len(tris))
    head += _i(0) + _i(0) + _i(0) + _s("", pad=False)
    if method == "RAW":
        return (head + b"INDX" + np.asarray(tris, "<u4").tobytes()
                + b"VERT" + np.asarray(verts, "<f4").tobytes())
    # MG1: triangles start with their smallest index, sorted, delta coded
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


def _cube(offset):
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) + offset
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v, np.array(f)


def _scene(objects):
    b = _i(6) + _s("english") + _i(0) + _s("a") + _s("b") + _s("")  # no password
    b += _s("build") + _s("1.0")
    b += struct.pack("<3f", 1, 0, 0) + struct.pack("<3f", 0, 0, 1) + _i(1) + struct.pack("<3f", 0, 0, 0) + _i(0)
    b += b"\0\0\0\0" * 4 + struct.pack("<3f", 1, 1, 1)
    b += _i(0) + _i(0) + _i(0) + _i(0)  # image, 2 view lists, annotations
    b += _i(len(objects))
    for path, ctm, shift in objects:
        b += _i(0) * 3 + b"\x80\x80\x80\0" * 4 + struct.pack("<3f", 1, 1, 0) + b"\0\0\0\0" + struct.pack("<f", 0)
        m = np.eye(4, dtype="<f4")
        m[:3, 3] = shift
        b += m.T.tobytes()  # column-major
        b += _i(len(ctm)) + ctm + b"\0" * (-len(ctm) % 4)
        b += _i(0) + struct.pack("<f", 1)
        b += _i(len(path)) + b"".join(_s(p) + b"\x80\x80\x80\0" for p in path)
        b += _i(1) + _i(1) + _i(0)
    return b


def _html(tmp_path, objects):
    data = base64.b64encode(_scene(objects)).decode()
    f = tmp_path / "case.html"
    f.write_text('<script>DentalWebGL.m_Data = {"data": "' + data + '"};</script>')
    return f


def test_decode_raw_and_mg1():
    v, f = _cube(0.0)
    for method in ("RAW", "MG1"):
        verts, tris = decode_ctm(_ctm(v, f, method))
        assert np.allclose(verts, v)
        assert {tuple(sorted(t)) for t in tris.tolist()} == {tuple(sorted(t)) for t in f.tolist()}


def test_load_and_classify_scene(tmp_path):
    v, f = _cube(0.0)
    path = _html(tmp_path, [
        (["Сканы челюстей", "Сканы челюстей (Ниж. челюсть)", "33: Препарирование"], _ctm(v, f, "MG1"), (0, 0, 0)),
        (["Сканы челюстей", "Сканы челюстей (Ниж. челюсть)", "Скан челюсти (37-...-47)"], _ctm(v, f), (5, 0, 0)),
        (["Ваксап", "Ваксап (Ниж. челюсть)", "33: Ваксап"], _ctm(v, f), (0, 0, 1)),
        (["Антагонисты"], _ctm(v, f), (0, 0, 9)),
        (["2D-изображения", "IMG_1.JPG"], _ctm(v, f), (0, 0, 0)),
    ])
    objs = load_webview(path)
    assert len(objs) == 5
    assert np.allclose(objs[1].mesh.vertices.min(0), [5, 0, 0])  # world matrix applied
    kinds = [classify(o)["kind"] for o in objs]
    assert kinds == ["prep", "jaw", "waxup", "antagonist", "image"]
    assert classify(objs[0])["teeth"] == [33] and classify(objs[0])["jaw"] == "lower"
    case = case_for_tooth(objs, 33)
    assert case.axis == (0.0, 0.0, 1.0)
    assert case.used == {"prep": 1, "jaw_scans": 1, "antagonist_objects": 1, "reference_waxup": True}
    with pytest.raises(ValueError, match="preparations: \\[33\\]"):
        case_for_tooth(objs, 46)


def test_password_protected_is_rejected(tmp_path):
    b = _i(6) + _s("english") + _i(0) + _s("a") + _s("b") + _s("salt")
    f = tmp_path / "p.html"
    f.write_text('DentalWebGL.m_Data = {"data": "' + base64.b64encode(b).decode() + '"}')
    with pytest.raises(ValueError, match="password"):
        load_webview(f)
