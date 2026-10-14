"""Цефалометрия (casedesigner/cephalometry.py): анализ — данные; проекция на срединную плоскость; Steiner, Ricketts."""

import numpy as np
import pytest

from casedesigner import cephalometry as ce


def at(origin, angle_from_back_deg, length):
    """Точка от origin под углом к направлению «назад» (−y), вниз: (x, y, z) — вправо, вперёд, вверх."""
    a = np.radians(angle_from_back_deg)
    return np.asarray(origin, float) + length * np.array([0.0, -np.cos(a), -np.sin(a)])


def steiner_case(lateral=0.0):
    S, N = np.array([0.0, 0, 0]), np.array([0.0, 70, 0])
    A, B = at(N, 82, 45), at(N, 80, 90)
    nb = (B - N) / np.linalg.norm(B - N)
    forward = np.array([0.0, nb[2], -nb[1]])  # поперёк NB в срединной плоскости, вперёд
    forward *= np.sign(forward[1])
    pts = {"S": S, "N": N, "A": A, "B": B, "Pog": N + 100 * nb + 3 * forward, "Gn": np.array([0.0, 60, -110]),
           "Go_R": np.array([45.0, -10, -60]), "Go_L": np.array([-45.0, -10, -60]),
           "Po_R": np.array([50.0, -20, -20]), "Po_L": np.array([-50.0, -20, -20])}
    for k in ("A", "B", "Pog", "N"):  # срединные точки чуть вбок — асимметрия не влияет
        pts[k] = pts[k] + [lateral, 0, 0]
    return pts


def test_steiner_angles_and_signed_distance():
    res = ce.analyze(steiner_case(lateral=1.5), "steiner")
    got = {r["key"]: r for r in res["results"]}
    assert got["SNA"]["value"] == pytest.approx(82.0, abs=0.05) and got["SNA"]["text"] == "в норме"
    assert got["SNB"]["value"] == pytest.approx(80.0, abs=0.05)
    assert got["ANB"]["value"] == pytest.approx(2.0, abs=0.1)
    assert got["SN_GoGn"]["value"] == pytest.approx(np.degrees(np.arctan2(50, 70)), abs=0.05)
    assert got["Pog_NB_mm"]["value"] == pytest.approx(3.0, abs=0.05)  # впереди NB — плюс
    assert got["Pog_NB_mm"]["z"] == pytest.approx(1.0, abs=0.03)
    assert {s["key"] for s in res["skipped"]} == {"U1_NA_deg", "U1_NA_mm", "L1_NB_deg", "L1_NB_mm", "interincisal"}
    assert "U1_tip" in next(s for s in res["skipped"] if s["key"] == "U1_NA_mm")["missing"]


def test_incisors_and_interpretation():
    pts = steiner_case()
    N, A = pts["N"], pts["A"]
    na = (A - N) / np.linalg.norm(A - N)
    # верхний резец под 30° к NA (протрузия), режущий край на 2 мм позади NA
    rot = np.radians(30)
    axis = np.array([0.0, na[1] * np.cos(rot) + na[2] * np.sin(rot), -na[1] * np.sin(rot) + na[2] * np.cos(rot)])
    back = -np.array([0.0, na[2], -na[1]]) * np.sign(np.array([0.0, na[2], -na[1]])[1])
    pts["U1_tip"] = N + 40 * na + 2 * back
    pts["U1_apex"] = pts["U1_tip"] - 23 * axis
    got = {r["key"]: r for r in ce.analyze(pts, "steiner")["results"]}
    angle = got["U1_NA_deg"]["value"]
    assert angle == pytest.approx(30.0, abs=0.1) and got["U1_NA_deg"]["text"] == "протрузия верхнего резца"
    assert got["U1_NA_mm"]["value"] == pytest.approx(-2.0, abs=0.05)


def test_ricketts_facial_angle_from_frankfurt():
    Po, Or, N = np.array([0.0, -20, -20]), np.array([0.0, 50, -20]), np.array([0.0, 70, 0])
    pts = {"Po_R": Po + [50, 0, 0], "Po_L": Po - [50, 0, 0], "Or_R": Or + [32, 0, 0], "Or_L": Or - [32, 0, 0],
           "N": N, "Pog": at(N, 92, 120), "A": at(N, 85, 50)}
    res = ce.analyze(pts, "ricketts")
    got = {r["key"]: r for r in res["results"]}
    # угол между «назад» по FH и N→Pog: погонион чуть впереди вертикали от N — 92°, подбородок впереди
    assert got["facial_angle"]["value"] == pytest.approx(92.0, abs=0.05)
    assert got["facial_angle"]["text"] == "подбородок впереди"
    assert got["convexity"]["unit"] == " мм" and "9 лет" in res["note"]
    assert {s["key"] for s in res["skipped"]} == {"mandibular_plane", "L1_APog_mm", "L1_APog_deg"}


def test_midsagittal_needs_points():
    with pytest.raises(ValueError, match="срединная плоскость"):
        ce.midsagittal({"N": np.zeros(3)})
    origin, normal = ce.midsagittal({"Po_R": np.array([50.0, 0, 0]), "Po_L": np.array([-50.0, 0, 0]),
                                     "N": np.array([0.0, 70, 0])})
    assert abs(normal @ [1, 0, 0]) == pytest.approx(1.0)


def test_report_on_generator():
    from casedesigner import reports

    rep = ce.report([ce.analyze(steiner_case(), "steiner")], "Кейс 4 · 14.10.2026")
    assert rep.sections[0].title == "Steiner"
    assert ("SNA", "82,0° (норма 82,0 ± 2,0) — в норме") in rep.sections[0].rows
    assert any("нет точек U1_tip" in n or "нет точек U1_apex" in n for n in rep.sections[0].notes)
    assert reports.render(rep)[0].size == reports.PAGE
