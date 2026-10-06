"""Проверка на ваших реальных кейсах: HTML-экспорты exocad в своей папке, вне репозитория.

    set CCD_PRIVATE_CASES=D:\\cases        (Linux/macOS: export CCD_PRIVATE_CASES=...)
    set CCD_RECORD=1                       первый раз — запомнить результаты
    python -m pytest tests/test_private_cases.py
    set CCD_RECORD=                        дальше — сравнивать с запомненным

Запомненное лежит рядом с экспортами в ccd_expected.json: только числа, по
отпечатку (SHA-256) содержимого файла, без имён файлов и пациентов. Без
CCD_PRIVATE_CASES тест пропускается — в репозитории реальных данных нет.
"""

import glob
import hashlib
import json
import os

import pytest

from casedesigner import exocad_webview as ew

FOLDER = os.environ.get("CCD_PRIVATE_CASES", "")
RECORD = bool(os.environ.get("CCD_RECORD"))
EXPECTED = os.path.join(FOLDER, "ccd_expected.json")
FILES = sorted(glob.glob(os.path.join(FOLDER, "*.html"))) if FOLDER else []

ANGLE_TOL = 1.0  # градусы
MM_TOL = 0.1
FRACTION_TOL = 0.15  # доля пути, на которой начинается или кончается контакт

pytestmark = pytest.mark.skipif(not FILES, reason="нет папки CCD_PRIVATE_CASES с экспортами exocad")


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summary(path: str) -> dict:
    """Числа, по которым сверяются прогоны: углы, суставы, прикус, контакты по участкам."""
    try:
        report, _case, _anatomy = ew.analyze(ew.load(path))
    except ValueError as e:
        return {"error": str(e).split(" (")[0]}
    s = report["settings"]
    return {
        "guidance_deg": report["guidance_deg"],
        "sagittal_deg": [s["sagittal_right_deg"], s["sagittal_left_deg"]],
        "intercondylar_mm": report["intercondylar_mm"],
        "bite_penetration_mm": report["bite_penetration_mm"],
        "seating_deg": report["bite_seating"]["theta_deg"],
        "contacts": report["contacts"],
    }


def load_expected() -> dict:
    if os.path.isfile(EXPECTED):
        with open(EXPECTED, encoding="utf-8") as f:
            return json.load(f)
    return {}


def close(a, b, tol) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


@pytest.mark.parametrize("path", FILES, ids=[f"кейс{k + 1}" for k in range(len(FILES))])
def test_private_case(path):
    key = digest(path)
    got = summary(path)
    if RECORD:
        expected = load_expected()
        expected[key] = got
        with open(EXPECTED, "w", encoding="utf-8") as f:
            json.dump(expected, f, ensure_ascii=False, indent=1)
        return
    want = load_expected().get(key)
    if want is None:
        pytest.skip(f"{key[:10]}: результат не запомнен — запустите с CCD_RECORD=1")
    if "error" in want or "error" in got:
        assert got.get("error") == want.get("error")
        return
    problems = []
    for name, v in want["guidance_deg"].items():
        if not close(got["guidance_deg"].get(name), v, ANGLE_TOL):
            problems.append(f"ведение {name}: было {v}, стало {got['guidance_deg'].get(name)}")
    for k, (a, b) in enumerate(zip(got["sagittal_deg"], want["sagittal_deg"])):
        if not close(a, b, ANGLE_TOL):
            problems.append(f"ССП {'правый' if k == 0 else 'левый'}: было {b}, стало {a}")
    for name in ("intercondylar_mm", "bite_penetration_mm"):
        if not close(got[name], want[name], MM_TOL):
            problems.append(f"{name}: было {want[name]}, стало {got[name]}")
    if not close(got["seating_deg"], want["seating_deg"], 0.05):
        problems.append(f"посадка прикуса: было {want['seating_deg']}°, стало {got['seating_deg']}°")
    for move, sectors in want["contacts"].items():
        now = got["contacts"].get(move, {})
        if set(now) != set(sectors):
            problems.append(f"{move}: контакты были {sorted(sectors)}, стали {sorted(now)}")
            continue
        for sector, (lo, hi) in sectors.items():
            if abs(now[sector][0] - lo) > FRACTION_TOL or abs(now[sector][1] - hi) > FRACTION_TOL:
                problems.append(f"{move}, {sector}: путь {lo}–{hi} → {now[sector][0]}–{now[sector][1]}")
    assert not problems, f"{key[:10]}: " + "; ".join(problems)


@pytest.mark.parametrize("path", FILES, ids=lambda p: digest(p)[:8])
def test_bite_correction_returns_after_scanner_like_error(path):
    """Исправление прикуса: сбитый поворотом вокруг шарнирной оси и перекосом прикус возвращается (≤ 0,3 мм)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from casedesigner import bite
    from casedesigner import motion as mo
    from casedesigner.register import apply

    p = ew.load(path).parts()
    if p["upper_scan"] is None or p["lower_scan"] is None:
        pytest.skip("в сцене нет обеих челюстей")
    uv, uf, lv = p["upper_scan"].vertices, p["upper_scan"].faces, p["lower_scan"].vertices
    try:
        base = bite.correct_bite(uv, uf, lv).transform
    except ValueError:
        pytest.skip("сканы не в прикусе")
    anat = mo.anatomy_average(lv, uv)
    R, inc = anat.frame[:3, :3], anat.points["incisal"]
    hinge = (anat.points["condyle_right"] + anat.points["condyle_left"]) / 2
    probes = np.array([inc, inc + R.T @ [22, -25, 0], inc + R.T @ [-22, -25, 0]])
    M = np.eye(4)
    r = Rotation.from_rotvec(0.4 / np.linalg.norm(inc - hinge) * R[0]).as_matrix()  # раскрыт на 0,4 мм по резцам
    M[:3, :3], M[:3, 3] = r, hinge - r @ hinge
    T2 = np.eye(4)
    r2 = Rotation.from_rotvec(np.radians(0.2) * R[1]).as_matrix()  # перекос на сторону
    T2[:3, :3], T2[:3, 3] = r2, inc - r2 @ inc
    P = T2 @ M
    res = bite.correct_bite(uv, uf, apply(P, lv))
    err = np.linalg.norm(apply(res.transform @ P, probes) - apply(base, probes), axis=1)
    assert err.max() < 0.3, err
