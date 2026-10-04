"""Записи движения из разных систем: маркеры в XML, таблицы, JawTrackingSystem, пути мыщелков.

Форматы коммерческих систем открыто не описаны, поэтому выгрузки здесь
синтетические — по тому, что о них известно: XML с координатами маркеров по
кадрам и исходным (окклюзионным) расположением, таблицы с заголовками, CSV и
HDF5 открытого трекера JawTrackingSystem, ASCII с путями шарнирных точек.
"""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from casedesigner import kinematics as kin
from casedesigner import motion as mo
from casedesigner.cli import main
from casedesigner.register import apply, axis_angle, rigid
from test_motion import ANAT_TO_CASE, SETTINGS, recordings, truth

# Маркеры на нижней челюсти и на голове — координаты кейса в окклюзии, мм.
JAW = apply(ANAT_TO_CASE, np.array([[20, 95, -40.0], [-20, 95, -40], [0, 110, -55], [0, 100, -20]]))
HEAD = apply(ANAT_TO_CASE, np.array([[40, 120, 60.0], [-40, 120, 60], [0, 130, 90]]))
# Оси кондилографа: x — вперёд, y — влево, z — вниз (левая тройка); «вправо, вперёд, вверх» = «-y,x,-z».
DEVICE = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1.0]])


def head_sway(n: int) -> np.ndarray:
    """Голова в поле трекера слегка покачивается: поворот до 2°, сдвиг до 1.5 мм."""
    return np.array([rigid(axis_angle(np.array([0.2, 1.0, 0.3]), np.radians(2) * np.sin(k / 7)),
                           1.5 * np.sin(np.array([k / 9, k / 11, k / 13]))) for k in range(n)])


def markers_xml(recs, names, head=False, reference=True) -> bytes:
    """Как XML для CAD с маркерами: по кадру — координаты маркеров, отдельно — их положение в окклюзии."""
    def marks(points, ids):
        return "".join(f'<Marker id="{i}" x="{p[0]:.6f}" y="{p[1]:.6f}" z="{p[2]:.6f}"/>' for i, p in zip(ids, points))

    jaw_ids, head_ids = names
    out = ['<?xml version="1.0"?><JawMotionData Software="zebris JMAnalyser" Unit="mm">'
           '<Patient Name="Иванов Иван" Birth="1980-01-01"/>']
    if reference:
        out.append('<StaticReference Type="ICP">' + marks(JAW, jaw_ids) + (marks(HEAD, head_ids) if head else "")
                   + "</StaticReference>")
    for r in recs:
        H = head_sway(len(r.transforms)) if head else np.tile(np.eye(4), (len(r.transforms), 1, 1))
        out.append(f'<Movement name="{r.name}">')
        for t, M, h in zip(r.times, r.transforms, H):
            out.append(f'<Frame time="{t:.4f}">' + marks(apply(h @ M, JAW), jaw_ids)
                       + (marks(apply(h, HEAD), head_ids) if head else "") + "</Frame>")
        out.append("</Movement>")
    return ("".join(out) + "</JawMotionData>").encode("utf-8")


def read(tmp_path, name, data) -> mo.MotionCase:
    path = tmp_path / name
    path.write_bytes(data)
    return mo.read_case(str(path))


def cut(recs, start=5):
    """Записи не с окклюзии: без первых кадров."""
    return [mo.Recording(r.name, r.times[start:], r.transforms[start:], r.source) for r in recs]


# --- маркеры ---

def test_markers_with_occlusion_reference(tmp_path):
    recs = cut(recordings())
    case = read(tmp_path, "zebris.xml", markers_xml(recs, (["1", "2", "3", "4"], [])))
    assert case.systems == ["Zebris JMA"] and len(case.recordings) == len(recs) and not case.tracings
    for got, want in zip(case.recordings, recs):
        assert got.name == want.name and got.times == pytest.approx(want.times, abs=1e-4)
        assert np.abs(got.transforms - want.transforms).max() < 1e-4  # от окклюзии, а не от первого кадра
    assert not any("первого кадра" in n for n in case.notes)


def test_markers_without_reference_use_first_frame(tmp_path):
    recs = recordings()  # начинаются с окклюзии
    case = read(tmp_path, "m.xml", markers_xml(recs, (["a", "b", "c", "d"], []), reference=False))
    assert np.abs(case.recordings[2].transforms - recs[2].transforms).max() < 1e-4
    assert any("первого кадра первой записи" in n for n in case.notes)


@pytest.mark.parametrize("named", [False, True])
def test_head_and_jaw_markers_are_separated(tmp_path, named):
    """Маркеры головы и челюсти в одном кадре: тела — по постоянству расстояний, челюсть — относительно головы."""
    recs = recordings()[:4]
    ids = (["MP1", "MP2", "MP3", "MP4"], ["HP1", "HP2", "HP3"]) if named else (["4", "5", "6", "7"], ["1", "2", "3"])
    case = read(tmp_path, "m.xml", markers_xml(recs, ids, head=True))
    assert len(case.recordings) == 4 and not case.tracings
    for got, want in zip(case.recordings, recs):
        assert np.abs(got.transforms - want.transforms).max() < 1e-4
        assert "относительно маркеров головы" in got.source
    assert any("не подписаны" in n for n in case.notes) != named


def test_markers_report_like_poses(tmp_path):
    """По маркерам — тот же анализ, что и по положениям: шарнирная ось, углы суставного пути."""
    case = read(tmp_path, "m.xml", markers_xml(recordings(), (["1", "2", "3", "4"], [])))
    report = mo.analyze_case(case, truth())
    assert report["articulator"]["sagittal_right_deg"] == pytest.approx(SETTINGS.sagittal_right_deg, abs=0.2)
    assert report["articulator"]["bennett_left_deg"] == pytest.approx(SETTINGS.bennett_left_deg, abs=0.3)


def test_points_in_a_line_are_not_a_curve_or_a_body(tmp_path):
    curve = "".join(f'<P x="{k}" y="{k * k / 10}" z="0"/>' for k in range(12))  # линия без времени — не движение
    case = read(tmp_path, "curve.xml", f"<Spee>{curve}</Spee>".encode())
    assert not case.recordings and not case.tracings


# --- таблицы ---

def points_table(recs, unit="m") -> str:
    """Таблица как из программы трекера: строки метаданных, по блоку на движение, десятичная запятая."""
    a = truth()
    k = 1000.0 if unit == "m" else 1.0
    lines = ["Patient: Иванов Иван", "Datum: 01.02.2024"]
    titles = {"Протрузия": "Bewegung: Protrusion", "Латеротрузия вправо": "Bewegung: Laterotrusion rechts",
              "Латеротрузия влево": "Bewegung: Laterotrusion links"}
    for r in recs:
        lines += ["", titles.get(r.name, "Bewegung"),
                  f"Zeit [ms];Inzisal X [{unit}];Inzisal Y [{unit}];Inzisal Z [{unit}];"
                  f"Kondylus R X [{unit}];Kondylus R Y [{unit}];Kondylus R Z [{unit}];"
                  f"Kondylus L X [{unit}];Kondylus L Y [{unit}];Kondylus L Z [{unit}]"]
        tracks = [mo.track(r.transforms, a.points[p]) / k for p in ("incisal", "condyle_right", "condyle_left")]
        for i, t in enumerate(r.times):
            row = [t * 1000] + [v for tr in tracks for v in tr[i]]
            lines.append(";".join(f"{v:.9f}".replace(".", ",") for v in row))
    return "\n".join(lines)


def test_table_with_named_points(tmp_path):
    recs = recordings()[1:4]
    case = read(tmp_path, "Иванов_export.csv", points_table(recs).encode("cp1251"))
    assert [r.name for r in case.recordings] == ["протрузия", "латеротрузия вправо", "латеротрузия влево"]
    for got, want in zip(case.recordings, recs):
        assert np.abs(got.transforms - want.transforms).max() < 1e-4
        assert got.times == pytest.approx(want.times, abs=1e-6) and got.timed
    printed = "\n".join(mo.describe_table(points_table(recs)))
    assert "Kondylus R X" in printed and "блок 3: строк 41 по 10 чисел" in printed
    assert "Иванов" not in printed and "01.02" not in printed


def test_jts_csv(tmp_path):
    """CSV JawTrackingSystem: «tx, ty, tz, qw, qx, qy, qz» без заголовка; положения — от начала координат модели."""
    rec = recordings()[2]
    A = rigid(axis_angle(np.array([1.0, -2, 0.5]), np.radians(120)), np.array([30.0, -60, 15]))
    T = A @ rec.transforms  # большой постоянный поворот: по |w| порядок кватерниона не понять
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat(scalar_first=True)
    rows = np.c_[T[:, :3, 3], q]
    case = read(tmp_path, "jaw_motion.csv", "\n".join(",".join(f"{v:.9f}" for v in r) for r in rows).encode())
    assert np.abs(case.recordings[0].transforms - T).max() < 1e-5
    timed = np.c_[rec.times, T[:, :3, 3], q[:, [1, 2, 3, 0]]]  # со временем и скаляром последним
    near = rec.transforms[:, :3, :3]
    timed[:, 4:] = Rotation.from_matrix(near).as_quat()  # без большого поворота: скаляр узнаётся по |w| ≈ 1
    timed[:, 1:4] = rec.transforms[:, :3, 3]
    case = read(tmp_path, "b.txt", "\n".join(" ".join(f"{v:.9f}" for v in r) for r in timed).encode())
    got = case.recordings[0]
    assert got.timed and np.abs(got.transforms - rec.transforms).max() < 1e-5


def test_jts_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    rec = recordings()[1]
    path = tmp_path / "jaw_motion.h5"
    with h5py.File(path, "w") as f:
        f.attrs["jts_version"] = "1.0"
        for group, shift in (("T_model_origin_mand_landmark_t", 0.5), ("T_model_origin_mand_landmark_t_smooth", 0)):
            g = f.create_group(group)
            g.attrs["sample_rate"], g.attrs["unit"] = 200.0, "m"
            g.create_dataset("translations", data=(rec.transforms[:, :3, 3] + shift) / 1000)
            g.create_dataset("rotations", data=Rotation.from_matrix(rec.transforms[:, :3, :3]).as_quat(
                scalar_first=True))
    case = mo.read_case(str(path))
    assert case.systems == ["JawTrackingSystem (JTS)"] and len(case.recordings) == 1  # только сглаженная
    got = case.recordings[0]
    assert np.abs(got.transforms - rec.transforms).max() < 1e-6
    assert got.times[1] == pytest.approx(1 / 200)
    assert any("translations" in line for line in mo.describe_hdf5(path.read_bytes()))


# --- пути мыщелков (кондилограф) ---

def condyle_table(settings=SETTINGS) -> str:
    """ASCII кондилографа: пути правой и левой шарнирных точек в осях прибора, блок на движение."""
    a = truth()
    recs = kin.standard_movements(a, settings)[1:]
    lines = ["CADIAX export", "Patient: Иванов Иван"]
    for r, title in zip(recs, ("Protrusion", "Laterotrusion rechts", "Laterotrusion links")):
        P = mo.paths(r, a)
        right, left = P["condyle_right"] @ DEVICE.T, P["condyle_left"] @ DEVICE.T
        lines += [title, "Zeit [s]\tR X\tR Y\tR Z\tL X\tL Y\tL Z"]
        lines += ["\t".join(f"{v:.6f}" for v in (t, *right[i], *left[i])) for i, t in enumerate(r.times)]
    return "\n".join(lines)


def test_condyle_tracings_with_axes(tmp_path):
    case = read(tmp_path, "cadiax.txt", condyle_table().encode())
    assert case.systems == ["Gamma CADIAX"] and not case.recordings and len(case.tracings) == 3
    assert any("только пути" in n for n in case.notes)
    guessed = mo.analyze_tracings(case)  # путь круче ~34°: вперёд и вниз почти поровну — оси не угадать
    assert guessed["axes"] is None and any("задайте оси" in n for n in guessed["notes"])
    report = mo.analyze_tracings(case, mo.parse_axes("-y,x,-z"))
    s = report["articulator"]
    assert [r["kind"] for r in report["recordings"]] == ["protrusion", "laterotrusion_right", "laterotrusion_left"]
    assert s["sagittal_right_deg"] == pytest.approx(SETTINGS.sagittal_right_deg, abs=0.1)
    assert s["sagittal_left_deg"] == pytest.approx(SETTINGS.sagittal_left_deg, abs=0.1)
    assert s["bennett_right_deg"] == pytest.approx(SETTINGS.bennett_right_deg, abs=0.2)
    assert s["bennett_left_deg"] == pytest.approx(SETTINGS.bennett_left_deg, abs=0.2)
    assert s["side_shift_right_mm"] == pytest.approx(SETTINGS.side_shift_right_mm, abs=0.05)


def test_condyle_axes_guessed_on_flat_paths(tmp_path):
    flat = kin.Settings(20, 22, 8, 10, 0.5, 0.3)
    case = read(tmp_path, "t.txt", condyle_table(flat).encode())
    report = mo.analyze_tracings(case)
    assert np.allclose(report["axes"], mo.parse_axes("-y,x,-z"))
    assert report["articulator"]["sagittal_left_deg"] == pytest.approx(22, abs=0.1)
    assert any("левые" in n for n in report["notes"])


def test_cli_tracings_and_inspect(tmp_path, capsys):
    path = tmp_path / "cadiax.txt"
    path.write_text(condyle_table(), encoding="utf-8")
    out = tmp_path / "report"
    assert main(["motion", str(path), "--axes=-y,x,-z", "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Пути отдельных точек" in printed and "sagittal_right_deg = 33.0" in printed
    report = json.loads((out / "motion.json").read_text(encoding="utf-8"))
    assert report["tracings"]["articulator"]["bennett_left_deg"] == pytest.approx(13, abs=0.2)
    assert main(["motion", str(path), "--inspect"]) == 0
    printed = capsys.readouterr().out
    assert "R X | R Y" in printed and "Иванов" not in printed


def test_closed_format_and_systems(tmp_path):
    case = read(tmp_path, "case.jmtxd", b"\x00\x01binary")
    assert case.systems == ["SICAT JMT+"] and "открыто не описан" in case.files[0]["note"]
    assert mo.detect_system("export.xml", b"<Data Generator='MODJAW Tech in Motion'/>") == "Modjaw"
    assert mo.detect_system("x.csv", b"1;2;3") is None


# --- реальные данные: шум, невидимые маркеры, выбросы, сплошная запись ---

def test_hidden_and_bad_markers_and_spikes(tmp_path):
    """Трекер теряет маркеры, один маркер сбит бликом, один кадр — скачок: положения всё равно верные."""
    rec = recordings()[2]
    rng = np.random.default_rng(1)
    ids = ["1", "2", "3", "4", "5"]
    jaw = np.vstack([JAW, apply(ANAT_TO_CASE, np.array([[10, 105, -30.0]]))])
    out = ['<Data><Reference>' + "".join(f'<M id="{i}" x="{p[0]}" y="{p[1]}" z="{p[2]}"/>' for i, p in zip(ids, jaw))
           + '</Reference><Movement name="x">']
    for k, (t, M) in enumerate(zip(rec.times, rec.transforms)):
        pts = apply(M, jaw) + rng.normal(0, 0.02, jaw.shape)
        if k % 7 == 3:
            pts[k % 5] = np.nan  # маркер закрыт
        if k == 10:
            pts[1] += [3.0, 0, 0]  # блик сбил маркер
        if k in (20, 21):
            pts[:3] = np.nan  # видно только два маркера
        if k == 30:
            pts += [6.0, -4, 2]  # скачок трекера
        out.append(f'<Frame t="{t}">' + "".join(f'<M id="{i}" x="{p[0]}" y="{p[1]}" z="{p[2]}"/>'
                                                for i, p in zip(ids, pts) if not np.isnan(p).any()) + "</Frame>")
    case = read(tmp_path, "m.xml", ("".join(out) + "</Movement></Data>").encode())
    got = case.recordings[0]
    err = np.linalg.norm(np.einsum("nij,kj->nki", got.transforms[:, :3, :3] - rec.transforms[:, :3, :3], jaw)
                         + (got.transforms - rec.transforms)[:, None, :3, 3], axis=-1).max(1)  # на маркерах
    assert err.max() < 0.15, (err.argmax(), err.max())
    assert "со сбитым маркером 1" in got.source and "заполнено по соседним" in got.source
    assert any("выбросы трекера" in n for n in case.notes)


def test_table_with_empty_cells(tmp_path):
    recs = recordings()[1:2]
    text = points_table(recs, unit="mm").splitlines()
    data = [k for k, line in enumerate(text) if line[:1].isdigit()]
    for k in data[5:8]:  # резцовая точка не видна: пустые поля
        cells = text[k].split(";")
        cells[1:4] = ["", "NaN", "-"]
        text[k] = ";".join(cells)
    case = read(tmp_path, "t.csv", "\n".join(text).encode())
    assert len(case.recordings) == 1 and len(case.recordings[0].times) == len(recs[0].times)
    assert np.abs(case.recordings[0].transforms - recs[0].transforms).max() < 1e-3


def test_smoothing_removes_jitter():
    rec = recordings()[1]
    rng = np.random.default_rng(0)
    noisy = rec.transforms.copy()
    noisy[:, :3, 3] += rng.normal(0, 0.15, (len(noisy), 3))
    out = mo.smooth(mo.Recording("x", rec.times, noisy), window_s=0.25)
    before = np.abs(noisy[:, :3, 3] - rec.transforms[:, :3, 3]).mean()
    after = np.abs(out.transforms[:, :3, 3] - rec.transforms[:, :3, 3]).mean()
    assert after < 0.6 * before


def test_continuous_recording_is_split():
    """Одна лента: протрузия, пауза, латеротрузии, жевание — делится на движения, жевание остаётся целым."""
    recs = recordings()[1:]
    times, Ts, t0 = [], [], 0.0
    for r in recs:
        rest = np.tile(np.eye(4), (30, 1, 1))  # секунда покоя в окклюзии
        there = r.transforms if r.name == "Жевание" else np.concatenate([r.transforms, r.transforms[::-1]])
        for part, dt in ((rest, 1 / 30), (there, 1 / 30)):  # движение — туда и обратно в окклюзию
            ts = t0 + np.arange(len(part)) * dt
            times.append(ts)
            Ts.append(part)
            t0 = ts[-1] + 1 / 30
    tape = mo.Recording("лента", np.concatenate(times), np.concatenate(Ts))
    parts = mo.split_recording(tape)
    assert len(parts) == 4
    kinds = [mo.analyze_recording(p, truth(), np.eye(4))["kind"] for p in parts]
    assert kinds == ["protrusion", "laterotrusion_right", "laterotrusion_left", "chewing"]
