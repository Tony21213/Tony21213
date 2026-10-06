"""Сканы прикуса: ставятся на сканы челюстей (не на КТ) и задают прикус, снятый врачом.

КТ фантома — с приоткрытым ртом (прикус на КТ не тот), сканы — в прикусе: так видно,
по какому прикусу встал нижний скан.
"""

import numpy as np
import pytest

import phantom
from casedesigner.fusion import BITE, CaseCT, Scan, export_case, is_bite_name, place_bites
from casedesigner.register import apply

OPEN_DEG = 4.0
POSE = phantom.scan_pose(4)  # координаты сканера: общие для сканов одной сессии


def bite_patch(side: int = 1):
    """Щёчный скан прикуса одной стороны: коронки обеих челюстей у окклюзионной плоскости, видимые снаружи."""
    parts, offset, faces = [], 0, []
    for jaw in ("upper", "lower"):
        v, f = phantom.make_scan(jaw)
        tri = v[f]
        c = tri.mean(axis=1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        out = np.c_[c[:, 0], c[:, 1], np.zeros(len(c))]
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        buccal = np.einsum("ij,ij->i", n, out) > 0.2 * np.linalg.norm(n, axis=1)
        keep = (side * c[:, 0] > 12) & (np.abs(c[:, 2] - phantom.OCCLUSAL_Z) < 5) & buccal
        used = np.unique(f[keep])
        remap = np.full(len(v), -1)
        remap[used] = np.arange(len(used))
        parts.append(v[used])
        faces.append(remap[f[keep]] + offset)
        offset += len(used)
    return np.vstack(parts), np.vstack(faces)


@pytest.fixture(scope="module")
def case():
    ct = CaseCT(phantom.make_volume(open_deg=OPEN_DEG))
    scans = {jaw: phantom.make_scan(jaw) for jaw in ("upper", "lower")}
    upper = ct.register(Scan("upper", apply(POSE, scans["upper"][0]), scans["upper"][1]), jaw="upper")
    bites = [bite_patch(1), bite_patch(-1)]
    return ct, scans, upper, bites


def lower_error(T, moved_lower, truth):
    return float(np.linalg.norm(apply(T, moved_lower) - truth, axis=1).max())


def test_bite_name():
    for name in ("p-TotalJaw0", "bite_right", "BuccalBite", "Occlusion 2", "прикус слева"):
        assert is_bite_name(name), name
    for name in ("p-UpperJaw", "p-lowerjaw", "maxillary", "mandibular"):
        assert not is_bite_name(name), name


def test_bite_in_scanner_coordinates(case):
    """Выгрузка одной сессией сканера: скан прикуса стоит вместе со сканами, прикус — со сканера."""
    ct, scans, upper, bites = case
    lower = ct.register(Scan("lower", apply(POSE, scans["lower"][0]), scans["lower"][1]), jaw="lower")
    res = place_bites([Scan(f"bite{k}", apply(POSE, v), f) for k, (v, f) in enumerate(bites)], upper, lower, ct)
    assert res.lower is None and not res.failed  # прикус уже со сканера
    for reg in res.regs:
        assert reg.jaw == BITE and np.allclose(reg.transform, upper.transform)
        assert reg.stats["matched_fraction"] > 0.9 and "вместе с ними" in reg.warnings[0]


@pytest.mark.parametrize("together", [False, True])
def test_doctors_bite_when_lower_exported_separately(case, together):
    """Нижний скан выгружен отдельно (не в прикусе): встаёт в прикус по сканам прикуса, а не по КТ
    (рот на КТ приоткрыт). together — нижний и сканы прикуса в общих координатах, верхний — отдельно."""
    ct, scans, upper, bites = case
    R = phantom.scan_pose(9)  # координаты отдельной выгрузки
    lower_v = apply(R, scans["lower"][0])
    lower = ct.register(Scan("lower", lower_v, scans["lower"][1]), jaw="lower")
    truth = apply(upper.transform, apply(POSE, scans["lower"][0]))  # нижний в прикусе сканов, в КТ
    assert lower_error(lower.transform, lower_v, truth) > 1.0  # прикус на КТ — другой
    pose = R if together else POSE
    res = place_bites([Scan(f"bite{k}", apply(pose, v), f) for k, (v, f) in enumerate(bites)], upper, lower, ct)
    assert res.lower is not None and not res.failed
    assert lower_error(res.lower, lower_v, truth) < 0.1

    # Экспорт в координатах сканера: нижний скан — в прикусе со сканов прикуса.
    lower.bite = res.lower
    import tempfile

    with tempfile.TemporaryDirectory() as out:
        report = export_case(out, [upper, lower, *res.regs], ct=ct)
        assert report["scans"]["lower"]["placement"] == "bite scan"
        assert all(report["scans"][f"bite{k}"]["placement"] == "jaws" for k in range(2))
        import trimesh

        got = np.asarray(trimesh.load_mesh(f"{out}/lower.stl", process=False).vertices)
        want = apply(POSE, scans["lower"][0])[scans["lower"][1]].reshape(-1, 3)
        assert np.abs(got - want).max() < 0.1


def test_bite_view_moves_lower_jaw_structures(tmp_path):
    """«Прикус» в окне: нижний скан и все структуры нижней челюсти из КТ — в прикус сканов (рот на КТ
    приоткрыт), верхние — на месте; смешанные делятся по челюстям. Выключено — как на КТ."""
    import SimpleITK as sitk
    import trimesh

    from casedesigner.app.session import Session

    vol = phantom.make_volume(open_deg=OPEN_DEG)
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(tmp_path / "ct.nii.gz"))
    for jaw in ("upper", "lower"):
        v, f = phantom.make_scan(jaw)
        trimesh.Trimesh(apply(POSE, v), f).export(tmp_path / f"{jaw}.stl")
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(tmp_path / "ct.nii.gz"))
    ids = {jaw: s.add_scan(str(tmp_path / f"{jaw}.stl"))["id"] for jaw in ("upper", "lower")}
    for jaw in ("upper", "lower"):
        s.register(ids[jaw], jaw=jaw)
    opened = phantom.jaw_opening(OPEN_DEG)
    up, lo = phantom.teeth_mesh("upper"), phantom.teeth_mesh("lower")
    s.structures = {"upper_teeth": trimesh.Trimesh(up.vertices, up.faces, process=False),
                    "lower_teeth": trimesh.Trimesh(apply(opened, lo.vertices), lo.faces, process=False),
                    "teeth/implant": trimesh.util.concatenate([  # смешанная: части в обеих челюстях
                        trimesh.Trimesh(up.vertices, up.faces, process=False),
                        trimesh.Trimesh(apply(opened, lo.vertices), lo.faces, process=False)])}
    info = s.bite_info()
    assert info["available"] and not info["on"] and info["shift_mm"] > 1  # прикус сканера, на КТ рот открыт

    def verts(key):
        raw = s.structure_mesh(key)
        nv = int(np.frombuffer(raw[:4], np.uint32)[0])
        return np.frombuffer(raw[8:8 + nv * 12], np.float32).reshape(-1, 3)

    s.set_bite_view(True)
    assert np.abs(verts("lower_teeth") - lo.vertices).max() < 0.15  # нижние зубы КТ сомкнулись вслед за сканом
    assert np.abs(verts("upper_teeth") - up.vertices).max() < 1e-4  # верхние — на месте
    mixed = verts("teeth/implant")
    from scipy.spatial import cKDTree
    closed = np.vstack([up.vertices, lo.vertices])
    assert cKDTree(closed).query(mixed)[0].max() < 0.15  # нижняя часть смешанной — тоже за челюстью
    lower_scan = trimesh.load_mesh(str(tmp_path / "lower.stl"), process=True).vertices
    shown = np.array(s.scan_info(ids["lower"])["transform"])
    assert np.abs(apply(shown, lower_scan) - apply(np.linalg.inv(POSE), lower_scan)).max() < 0.15
    with pytest.raises(ValueError, match="Прикус"):
        s.evaluate(ids["lower"], shown)  # коррекция нижнего — только как на КТ

    # Выгрузка в координатах КТ в прикусе сканов — как показано в 3D с «Прикусом».
    s.export(str(tmp_path / "out"), "scan", "dicom")
    got = trimesh.load_mesh(str(tmp_path / "out" / "lower_teeth.stl"), process=False).vertices
    assert np.abs(got - lo.vertices[lo.faces].reshape(-1, 3)).max() < 0.15
    got = trimesh.load_mesh(str(tmp_path / "out" / "upper_teeth.stl"), process=False).vertices
    assert np.abs(got - up.vertices[up.faces].reshape(-1, 3)).max() < 1e-3

    s.set_bite_view(False)
    assert np.abs(verts("lower_teeth") - apply(opened, lo.vertices)).max() < 1e-4


def test_misplaced_bite_is_not_used(case):
    """Скан прикуса, который не удаётся поставить, не ставится молча не туда: либо точно, либо с причиной."""
    ct, scans, upper, bites = case
    lower = ct.register(Scan("lower", apply(phantom.scan_pose(9), scans["lower"][0]), scans["lower"][1]), jaw="lower")
    v, f = bites[0]
    res = place_bites([Scan("bite", apply(phantom.scan_pose(13), v), f)], upper, lower, ct)
    if res.regs[0] is None:
        assert "одной сессии сканера" in res.failed[0]
    else:
        assert np.abs(apply(res.regs[0].transform, apply(phantom.scan_pose(13), v))
                      - apply(upper.transform, apply(POSE, v))).max() < 0.2
