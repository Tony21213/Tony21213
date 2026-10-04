import io
import zipfile

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from casedesigner import guidance as gd
from casedesigner import kinematics as kin
from casedesigner import motion as mo
from casedesigner.cli import main
from casedesigner.register import apply, axis_angle, rigid

# Анатомическая система «истины»: x — вправо, y — вперёд, z — вверх, начало — середина шарнирной оси.
POINTS = {"incisal": np.array([0, 90, -30.0]), "condyle_right": np.array([50.0, 0, 0]),
          "condyle_left": np.array([-50.0, 0, 0])}
# Координаты кейса (выгрузки) — произвольно повёрнуты и сдвинуты относительно анатомии.
CASE_TO_ANAT = rigid(axis_angle(np.array([0.3, -0.5, 1.0]), 0.7), np.array([12.0, -40, 85]))
ANAT_TO_CASE = np.linalg.inv(CASE_TO_ANAT)
SETTINGS = kin.Settings(33, 38, 8, 13, 0.6, 0.3)


def truth() -> mo.Anatomy:
    return mo.Anatomy(CASE_TO_ANAT, {k: apply(ANAT_TO_CASE, v[None])[0] for k, v in POINTS.items()}, "истина")


def recordings():
    a = truth()
    return kin.standard_movements(a, SETTINGS) + [kin.chewing(a, SETTINGS)]


def lower_arch() -> np.ndarray:
    """Модель нижних зубов: дуга от резцов (y = 90) до моляров, высотой 15 мм, толщиной 8 мм."""
    u = np.linspace(-1, 1, 60)
    arch = np.c_[25 * u, 90 - 40 * u ** 2, np.full_like(u, -30.0)]
    layers = [arch + [0, r, -dz] for dz in np.linspace(0, 15, 8) for r in (-4, 0, 4)]
    return apply(ANAT_TO_CASE, np.vstack(layers))


def plate(origin, u, v, nu=30, nv=30):
    a, b = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    vertices = (origin + a[..., None] * u + b[..., None] * v).reshape(-1, 3)
    k = (np.arange(nu - 1)[:, None] * nv + np.arange(nv - 1)[None]).ravel()
    faces = np.vstack([np.c_[k, k + nv, k + 1], np.c_[k + 1, k + nv, k + nv + 1]])
    return vertices, faces


# --- форматы файлов ---

def xml_matrix_text(recs):
    """Кадры с 16 числами по строкам и временем в атрибуте."""
    out = ['<?xml version="1.0"?><JawMotion units="mm"><Patient><Name>Иванов Иван</Name></Patient>']
    for r in recs:
        out.append(f'<Movement name="{r.name}">')
        for t, M in zip(r.times, r.transforms):
            out.append(f'<Frame time="{t:.4f}">{" ".join(f"{v:.9f}" for v in M.ravel())}</Frame>')
        out.append("</Movement>")
    out.append('<Angles><Bennett side="right">8.1</Bennett></Angles></JawMotion>')
    return "".join(out).encode("utf-8")


def xml_matrix4_cells(rec):
    """Как .matrix4 из выгрузки для exocad: ячейки _00…_33, сдвиг в последней строке, время в мс."""
    out = ["<Recording><Samples>"]
    for t, M in zip(rec.times, rec.transforms):
        cells = "".join(f"<_{i}{j}>{v:.9f}</_{i}{j}>" for i, row in enumerate(M.T) for j, v in enumerate(row))
        out.append(f"<Sample><TimeMs>{t * 1000:.1f}</TimeMs><Matrix4>{cells}</Matrix4></Sample>")
    return ("".join(out) + "</Samples></Recording>").encode()


def xml_two_bodies(rec, upper_world):
    """В кадре два тела в системе трекера: верхняя и нижняя челюсть."""
    out = ['<Track fps="100">']
    for M in rec.transforms:
        U = upper_world
        L = U @ M
        body = "".join(f'<Body name="{n}">{" ".join(f"{v:.9f}" for v in B.ravel())}</Body>'
                       for n, B in (("UpperJaw", U), ("LowerJaw", L)))
        out.append(f"<Frame>{body}</Frame>")
    return ("".join(out) + "</Track>").encode()


def xml_quaternion(rec):
    from scipy.spatial.transform import Rotation

    out = ["<Data>"]
    for t, M in zip(rec.times, rec.transforms):
        x, y, z, w = Rotation.from_matrix(M[:3, :3]).as_quat()
        tx, ty, tz = M[:3, 3]
        out.append(f'<Pose t="{t}"><Position x="{tx}" y="{ty}" z="{tz}"/>'
                   f'<Orientation w="{w}" x="{x}" y="{y}" z="{z}"/></Pose>')
    return ("".join(out) + "</Data>").encode()


def csv_comma_decimal(rec):
    lines = ["time;m00;…"]
    for t, M in zip(rec.times, rec.transforms):
        lines.append(";".join(f"{v:.6f}".replace(".", ",") for v in [t, *M.ravel()]))
    return "\n".join(lines).encode()


def read_one(name, data):
    case = mo.MotionCase("x")
    if name.endswith(".csv"):
        mo._read_table(data.decode(), name, case)
    else:
        mo._read_xml(data, name, case)
    return case


@pytest.mark.parametrize("layout", ["text", "cells", "bodies", "quaternion", "csv"])
def test_motion_formats(layout):
    recs = recordings()
    rec = recs[1]
    if layout == "text":
        case = read_one("m.xml", xml_matrix_text(recs))
        assert [r.name for r in case.recordings] == [r.name for r in recs]
        got = case.recordings[1]
        assert case.values["m.xml:JawMotion/Angles/Bennett"] == 8.1
        assert not any("Name" in k for k in case.values)
    elif layout == "cells":
        got = read_one("m.xml", xml_matrix4_cells(rec)).recordings[0]
    elif layout == "bodies":
        U = rigid(axis_angle(np.array([1.0, 2, 0.5]), 1.1), np.array([300.0, -20, 900]))
        got = read_one("m.xml", xml_two_bodies(rec, U)).recordings[0]
        assert "LowerJaw относительно UpperJaw" in got.source
    elif layout == "quaternion":
        got = read_one("m.xml", xml_quaternion(rec)).recordings[0]
    else:
        got = read_one("m.csv", csv_comma_decimal(rec)).recordings[0]
    tol = 1e-4 if layout == "csv" else 1e-6
    assert len(got.transforms) == len(rec.transforms)
    assert np.abs(got.transforms - rec.transforms).max() < tol
    if layout != "bodies":
        assert got.times == pytest.approx(rec.times, abs=1e-3) and got.timed
    else:
        assert got.duration == pytest.approx((len(rec.times) - 1) / 100)


def test_describe_xml_hides_values():
    lines = "\n".join(mo.describe_xml(xml_matrix_text(recordings())))
    assert "Иванов" not in lines and "Frame ×" in lines and "чисел в тексте: 16" in lines
    assert "units: 'mm'" in lines


def case_zip(tmp_path, meters=False):
    """Выгрузка как из P-ART: модели с именами exocad, XML движений, описание проекта."""
    recs = recordings()
    if meters:
        for r in recs:
            r.transforms[:, :3, 3] /= 1000
    path = tmp_path / "case.zip"
    lower = lower_arch()
    hull = trimesh.convex.convex_hull(lower)
    with zipfile.ZipFile(path, "w") as z:
        buf = io.BytesIO()
        trimesh.Trimesh(hull.vertices, hull.faces).export(buf, file_type="stl")
        z.writestr("Case 01/Case 01-LowerJaw.stl", buf.getvalue())
        z.writestr("Case 01/Case 01-UpperJaw.stl", buf.getvalue())
        z.writestr("Case 01/motion.xml", xml_matrix_text(recs).replace(b' units="mm"', b""))
        z.writestr("Case 01/Case 01.dentalProject",
                   b"<Treatment><Patient><PatientName>X</PatientName></Patient>"
                   b"<MovementMarkerScan>true</MovementMarkerScan></Treatment>")
        z.writestr("Case 01/readme.txt", "просто текст")
    return path


def test_read_case_and_cli(tmp_path, capsys):
    case = mo.read_case(str(case_zip(tmp_path)))
    assert case.roles == {"lower": "Case 01/Case 01-LowerJaw.stl", "upper": "Case 01/Case 01-UpperJaw.stl"}
    assert case.project == {"MovementMarkerScan": "true"} and len(case.recordings) == 5
    assert mo.reference_pose(case) is not None

    out = tmp_path / "report"
    assert main(["motion", str(tmp_path / "case.zip"), "-o", str(out)]) == 0
    assert (out / "motion.json").is_file() and (out / "paths.csv").is_file() and (out / "motion.png").is_file()
    assert "латеротрузия вправо" in capsys.readouterr().out

    assert main(["motion", str(tmp_path / "case.zip"), "--inspect"]) == 0
    printed = capsys.readouterr().out
    assert "Movement ×5" in printed and "PatientName" in printed and ">X<" not in printed


def test_meters_are_converted(tmp_path):
    case = mo.read_case(str(case_zip(tmp_path, meters=True)))
    assert any("метры" in n for n in case.notes)
    assert np.abs(case.recordings[1].transforms - recordings()[1].transforms).max() < 1e-5


# --- анализ ---

def test_analysis_recovers_articulator_settings():
    case = mo.MotionCase("synthetic", recordings())
    report = mo.analyze_case(case, truth())
    assert [r["kind"] for r in report["recordings"]] == \
        ["opening", "protrusion", "laterotrusion_right", "laterotrusion_left", "chewing"]
    s = report["articulator"]
    assert s["sagittal_right_deg"] == pytest.approx(33, abs=0.1)
    assert s["sagittal_left_deg"] == pytest.approx(38, abs=0.1)
    assert s["bennett_right_deg"] == pytest.approx(8, abs=0.1)
    assert s["bennett_left_deg"] == pytest.approx(13, abs=0.1)
    assert s["side_shift_right_mm"] == pytest.approx(0.6, abs=0.02)
    assert s["side_shift_left_mm"] == pytest.approx(0.3, abs=0.02)
    assert report["max_opening_mm"] > 40 and report["recordings"][4]["cycles"] == 3
    back = kin.settings_from_analysis(report)
    assert back.sagittal_left_deg == pytest.approx(38, abs=0.1) and back.sources["bennett_left_deg"]


def test_anatomy_estimated_from_motion_and_lower_model():
    case = mo.MotionCase("synthetic", recordings())
    est = mo.estimate_anatomy(case, lower=lower_arch())
    assert est.hinge_rms_mm < 1e-6
    inc = apply(CASE_TO_ANAT, est.points["incisal"][None])[0]
    assert inc == pytest.approx([0, 92, -30], abs=2.5)
    for side, x in (("right", 50), ("left", -50)):  # право и лево — из направления открывания
        c = apply(CASE_TO_ANAT, est.points[f"condyle_{side}"][None])[0]
        assert c == pytest.approx([x + inc[0], 0, 0], abs=1e-6)
    # горизонталь — плоскость «ось — резцовая точка»: ССП меньше на её наклон
    tilt = np.degrees(np.arctan2(-inc[2], inc[1]))
    s = mo.analyze_case(case, est)["articulator"]
    assert s["sagittal_right_deg"] == pytest.approx(33 - tilt, abs=0.3)
    assert [r["kind"] for r in mo.analyze_case(case, est)["recordings"]][1:4] == \
        ["protrusion", "laterotrusion_right", "laterotrusion_left"]


def test_anatomy_from_ct_uses_plane_and_condyles():
    pts = {"Po_R": [-60, 10, 0], "Po_L": [60, 10, 0], "Or_R": [-33, -70, 0], "Or_L": [33, -70, 0],
           "Co_R": [-50, 5, -12], "Co_L": [50, 5, -12], "N": [0, -85, 15], "ANS": [0, -78, -30],
           "PNS": [0, -30, -32], "Ba": [0, 15, -20], "IP": [0, -62, -38]}
    pts = {k: np.array(v, float) + [3, -4, 100] for k, v in pts.items()}  # мм КТ (LPS)
    case_to_ct = rigid(axis_angle(np.array([0, 0, 1.0]), 0.4), np.array([1.0, 2, 3]))
    a = mo.anatomy_from_ct(pts, "frankfurt", case_to_ct, incisal=np.zeros(3))
    co = apply(a.frame, a.points["condyle_right"][None])[0]
    assert co == pytest.approx([50, 0, 0], abs=1e-6)  # правый мыщелок — справа (+x), на оси
    assert "Франкфурт" in a.source


def test_tooth_guidance_follows_incisor_surface():
    """Нижний резец скользит по нёбной поверхности верхнего (50°), хотя суставной путь — 35°."""
    angle = np.radians(50)
    along = np.array([0, np.cos(angle), -np.sin(angle)])
    normal = np.array([0, -np.sin(angle), -np.cos(angle)])
    incisal = POINTS["incisal"]
    up_v, up_f = plate(incisal - 4 * along + [-6, 0, 0], np.array([12.0, 0, 0]), 12 * along)
    edge = np.c_[np.linspace(-4, 4, 41), np.full(41, 90.0), np.full(41, -30.0)] + 0.05 * normal
    low_v = np.vstack([edge, edge + [0, -1.5, -1.0], edge + [0, -3, -6]])
    a = mo.Anatomy(np.eye(4), dict(POINTS), "истина")
    occ = kin.Occlusion(up_v, up_f, low_v, np.eye(4))
    s = kin.Settings(35, 35)
    guided = mo.analyze_case(mo.MotionCase("x", [kin.protrusion(a, s, occlusion=occ)]), a)
    free = mo.analyze_case(mo.MotionCase("x", [kin.protrusion(a, s)]), a)
    assert guided["guidance_deg"]["protrusion"] == pytest.approx(50, abs=0.5)
    assert free["guidance_deg"]["protrusion"] == pytest.approx(35, abs=0.5)
    assert guided["articulator"]["sagittal_right_deg"] == pytest.approx(35, abs=0.1)  # мыщелки — по своим путям
    with pytest.raises(ValueError, match="не в прикусе"):
        kin.Occlusion(up_v, up_f, low_v + [0, 0, -20], np.eye(4))


# --- ССП по КТ ---

def temporal_bone(alpha, frame_inv=np.eye(4), x=(-8, 8), roof=2.5, drop=8.5):
    """Нижняя поверхность височной кости: плоская крыша ямки, скат бугорка под alpha, вершина."""
    y = np.arange(-8, 24, 0.2)
    z = np.maximum(np.where(y < 1, roof, roof - np.tan(np.radians(alpha)) * (y - 1)), roof - drop)
    v, f = plate(np.array([x[0], 0, 0.0]), np.array([x[1] - x[0], 0, 0.0]), np.zeros(3), nu=20, nv=len(y))
    v = v.reshape(20, len(y), 3)
    v[..., 1], v[..., 2] = y, z
    return apply(frame_inv, v.reshape(-1, 3)), f


@pytest.mark.parametrize("alpha", [30, 45])
def test_condylar_path_from_eminence(alpha):
    v, f = temporal_bone(alpha)
    info = gd.condylar_path(*gd.eminence_profile(v, f, np.eye(4), np.zeros(3)))
    assert info["sagittal_deg"] == pytest.approx(alpha, abs=0.5)
    assert info["eminence_deg"] == pytest.approx(alpha, abs=1.0)
    assert info["fossa_depth_mm"] == pytest.approx(8.5, abs=0.3)


def test_ct_settings_relative_to_plane():
    from casedesigner.landmarks import reference_frame

    from test_landmarks import skull_points

    pts, _ = skull_points(tilt_deg=12)  # голова на снимке наклонена — углы всё равно от Франкфурта
    frame = reference_frame(pts, "frankfurt")
    meshes = []
    for side, alpha in (("Co_R", 32), ("Co_L", 41)):
        co = apply(frame, pts[side][None])[0]
        v, f = temporal_bone(alpha, x=(co[0] - 8, co[0] + 8))
        v = v + [0, co[1], co[2]]
        meshes.append((apply(np.linalg.inv(frame), v), f))
    v = np.vstack([meshes[0][0], meshes[1][0]])
    f = np.vstack([meshes[0][1], meshes[1][1] + len(meshes[0][0])])
    s, details = gd.ct_settings(v, f, pts, "frankfurt")
    assert s.sagittal_right_deg == pytest.approx(32, abs=0.5) and s.sagittal_left_deg == pytest.approx(41, abs=0.5)
    assert s.bennett_right_deg == pytest.approx(32 / 8 + 12, abs=0.1)
    assert "КТ" in s.sources["sagittal_right_deg"] and details["left"]["rolling_radius_mm"] > 0


def test_aesthetic_mounting_with_asymmetric_condyles():
    """Модели «загипсованы» по эстетической плоскости (крен и поворот к Франкфурту), мыщелки
    справа и слева на разной высоте и глубине; углы заданы от Франкфурта — движение то же."""
    frankfurt = CASE_TO_ANAT
    roll_yaw = rigid(axis_angle(np.array([0, 1.0, 0]), np.radians(4)) @ axis_angle(np.array([0, 0, 1.0]),
                                                                                    np.radians(3)), np.zeros(3))
    aesthetic = roll_yaw @ frankfurt  # система монтажа
    condyles = {"condyle_right": np.array([49.0, -2.5, 1.8]), "condyle_left": np.array([-51.5, 1.5, -2.2])}
    points = {k: apply(ANAT_TO_CASE, v[None])[0] for k, v in {**POINTS, **condyles}.items()}
    mount = mo.Anatomy(aesthetic, points, "эстетическая плоскость")
    s = kin.Settings(33, 38, 8, 13, 0.6, 0.3, frame=frankfurt)
    recs = kin.standard_movements(mount, s)
    # мыщелки в системе монтажа действительно разные по y и z
    c = kin.condyles(mount)
    assert abs(c["right"][1] - c["left"][1]) > 1 and abs(c["right"][2] - c["left"][2]) > 1
    # анализ от Франкфурта по тем же точкам восстанавливает настройки
    report = mo.analyze_case(mo.MotionCase("x", recs), mo.Anatomy(frankfurt, points, "Франкфурт"))
    got = report["articulator"]
    for key, value in (("sagittal_right_deg", 33), ("sagittal_left_deg", 38), ("bennett_right_deg", 8),
                       ("bennett_left_deg", 13)):
        assert got[key] == pytest.approx(value, abs=0.15), key
    assert got["side_shift_right_mm"] == pytest.approx(0.6, abs=0.03)
    # ось открывания — реальная межмыщелковая, а не ось X монтажа: мыщелки на ней не сдвигаются
    opening = recs[0]
    early = opening.transforms[: len(opening.transforms) // 3]
    for p in (points["condyle_right"], points["condyle_left"]):
        assert np.abs(mo.track(early, p) - p).max() < 1e-6


def test_average_articulator_from_scans_only():
    """Только сканы: окклюзионная плоскость и «вперёд» — по дуге, мыщелки — по Бонвиллю и Балквиллу."""
    lower = lower_arch()
    upper = apply(ANAT_TO_CASE, apply(CASE_TO_ANAT, lower) + [0, 1.5, 9])
    a = mo.anatomy_average(lower, upper)
    R = a.frame[:3, :3] @ ANAT_TO_CASE[:3, :3]  # оси оценки в анатомической системе
    assert R[1] @ [0, 1, 0] > 0.999 and R[2] @ [0, 0, 1] > 0.999  # вперёд и вверх найдены
    inc = apply(CASE_TO_ANAT, a.points["incisal"][None])[0]
    for side, sign in (("right", 1), ("left", -1)):
        c = apply(CASE_TO_ANAT, a.points[f"condyle_{side}"][None])[0]
        assert np.linalg.norm(c - inc) == pytest.approx(mo.BONWILL_MM, abs=1e-6)
        assert np.sign(c[0] - inc[0]) == sign and c[2] > inc[2]  # справа — справа, мыщелки выше резцов
    with pytest.raises(ValueError, match="где верх"):
        mo.anatomy_average(lower)


def test_arch_direction_with_curve_of_spee():
    """Моляры выше резцов (кривая Шпее): самые высокие точки — одни моляры, «вперёд» всё равно к резцам."""
    u = np.linspace(-1, 1, 80)
    spee = 2.5 * u ** 2  # к молярам дуга поднимается
    arch = np.c_[25 * u, 90 - 40 * u ** 2, -30 + spee]
    lower = np.vstack([arch + [0, r, -dz] for dz in np.linspace(0, 8, 6) for r in (-4, 0, 4)])
    up, anterior = mo.arch_axes(lower, [0, 0, 1.0])
    assert anterior @ [0, 1, 0] > 0.99 and up @ [0, 0, 1] > 0.98


def ramp_case(thickness=None, molars_over_closed=None):
    """Верхний резец — нёбная поверхность под 50° (и губная, если задана толщина края); нижний режущий край у неё.

    molars_over_closed — «провал» прикуса: нижние жевательные бугры на столько мм внутри верхних.
    """
    angle = np.radians(50)
    along = np.array([0, np.cos(angle), -np.sin(angle)])
    toward_lower = np.array([0, -np.sin(angle), -np.cos(angle)])
    incisal = POINTS["incisal"]
    meshes = [plate(incisal - 4 * along + [-6, 0, 0], np.array([12.0, 0, 0]), 12 * along)]
    if thickness:  # губная поверхность: та же плоскость, сдвинутая внутрь зуба, нормаль наружу
        meshes.append(plate(incisal - 4 * along + [-6, 0, 0] - thickness * toward_lower, 12 * along,
                            np.array([12.0, 0, 0])))
    edge = np.c_[np.linspace(-4, 4, 41), np.full(41, 90.0), np.full(41, -30.0)] + 0.05 * toward_lower
    lower = [edge, edge + [0, -1.5, -1.0], edge + [0, -3, -6]]
    if molars_over_closed is not None:  # верхние бугры — плоскость z = −32, нижние — чуть внутри неё
        meshes.append(plate(np.array([-25.0, 40, -32]), np.array([50.0, 0, 0]), np.array([0, 20.0, 0])))
        grid = np.stack(np.meshgrid(np.linspace(-20, 20, 21), np.linspace(42, 58, 9)), -1).reshape(-1, 2)
        lower.append(np.c_[grid, np.full(len(grid), -32 + molars_over_closed)])
    verts, faces, off = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return np.vstack(verts), np.vstack(faces), np.vstack(lower)


def guided_protrusion(up_v, up_f, low_v, sagittal=35.0):
    a = mo.Anatomy(np.eye(4), dict(POINTS), "истина")
    occ = kin.Occlusion(up_v, up_f, low_v, np.eye(4))
    rec = kin.protrusion(a, kin.Settings(sagittal, sagittal), occlusion=occ)
    return mo.analyze_case(mo.MotionCase("x", [rec]), a)["guidance_deg"]["protrusion"], occ


def test_over_closed_bite_does_not_let_incisors_sink():
    """Прикус «провален» на 0.4 мм в жевательных зубах — резцы всё равно не уходят друг в друга."""
    exact, _ = guided_protrusion(*ramp_case(molars_over_closed=0.0))
    sunk, occ = guided_protrusion(*ramp_case(molars_over_closed=0.4))
    assert occ.bite_penetration_mm == pytest.approx(0.4, abs=0.05)
    assert sunk == pytest.approx(exact, abs=0.2) and exact == pytest.approx(50, abs=1.0)


def test_thin_incisal_edge_is_not_skipped():
    """Край резца 0.2 мм, суставной путь круче ведения: челюсть закрывается до контакта и не проскакивает край."""
    angle, _ = guided_protrusion(*ramp_case(thickness=0.2), sagittal=60.0)
    assert angle == pytest.approx(50, abs=0.6)


def test_fgp_follows_the_guiding_surface():
    """Огибающая нижнего резца, скользящего по нёбной поверхности, лежит на этой поверхности, не выше её."""
    from casedesigner import fgp

    up_v, up_f, low_v = ramp_case()
    a = mo.Anatomy(np.eye(4), dict(POINTS), "истина")
    s = kin.Settings(35, 35)
    occ = kin.Occlusion(up_v, up_f, low_v, np.eye(4))
    rec = kin.protrusion(a, s, occlusion=occ, frames=81)
    verts, faces, _ = fgp.fgp(up_v, up_f, low_v, a, s, recordings=[rec], cell=0.1)
    assert len(faces) > 0
    angle = np.radians(50)
    along = np.array([0, np.cos(angle), -np.sin(angle)])
    incisal = POINTS["incisal"]
    # точки огибающей впереди резца: высота над нёбной плоскостью (по вертикали) около −0.05…0 мм, не выше
    rel = verts - incisal
    front = verts[(np.abs(rel[:, 0]) < 3) & (rel[:, 1] > 0.5) & (rel[:, 1] < 4)]
    on_ramp_z = incisal[2] + (front[:, 1] - incisal[1]) * along[2] / along[1]
    gap = front[:, 2] - on_ramp_z
    # клетка берёт максимум: на склоне 50° запас до половины клетки × tg 50° (+ допуск контакта)
    assert gap.max() < 0.05 * np.tan(angle) + 0.05 and np.median(gap) > -0.3
    # огибающая не ниже исходного положения нижних зубов (у края резца, где они касаются верхних)
    tree = cKDTree(verts[:, :2])
    edge = low_v[:41]
    for p in edge:
        around = tree.query_ball_point(p[:2], 0.12)
        assert around and verts[around, 2].max() >= p[2] - 1e-6


@pytest.mark.parametrize("over_closed", [0.4, -0.6])
def test_seat_bite_on_hinge_axis(over_closed):
    """Провален на 0.4 мм — открыть до касания; не сомкнут на 0.6 мм — закрыть до первого контакта."""
    up_v, up_f, low_v = ramp_case(molars_over_closed=over_closed)
    a = mo.Anatomy(np.eye(4), dict(POINTS), "истина")
    T, info = kin.seat_bite(up_v, up_f, low_v, a)
    assert info["penetration_after_mm"] <= kin.CONTACT_TOL_MM
    seated = kin.Occlusion(up_v, up_f, apply(T, low_v), np.eye(4), max_points=kin.SEAT_POINTS)
    assert seated._signed(np.eye(4)).min() == pytest.approx(0.0, abs=kin.CONTACT_TOL_MM)  # касание
    if over_closed > 0:
        assert info["theta_deg"] > 0 and info["penetration_before_mm"] == pytest.approx(0.4, abs=0.05)
    else:
        assert info["theta_deg"] < 0
    # точки на шарнирной оси не сдвигаются — только поворот по дуге закрывания
    for c in ("condyle_right", "condyle_left"):
        assert apply(T, POINTS[c][None])[0] == pytest.approx(POINTS[c], abs=1e-9)
