"""Кейс артикуляции — то, что вызывает интерфейс: монтаж, суставы, движения, контакты, отмена, сохранение."""

import numpy as np
import pytest

import phantom
from casedesigner import jawcase as jc
from casedesigner import kinematics as kin
from casedesigner import motion as mo
from casedesigner.register import apply


@pytest.fixture(scope="module")
def scans():
    """Фантомные сканы обеих челюстей в прикусе (верхняя — зеркало нижней), в произвольных координатах."""
    pose = phantom.scan_pose(6)
    lv, lf = phantom.make_scan("lower", resolution=0.3)
    uv, uf = phantom.make_scan("upper", resolution=0.3)
    return jc.Mesh(apply(pose, uv), uf, "верх"), jc.Mesh(apply(pose, lv), lf, "низ"), pose


def fresh(scans) -> jc.JawCase:
    case = jc.JawCase()
    case.set_scans(scans[0], scans[1])
    return case


def test_mount_generate_contacts_and_undo(scans):
    case = fresh(scans)
    with pytest.raises(ValueError, match="смонтируйте"):
        case.generate()
    state = case.mount()  # ни КТ, ни записи — средний артикулятор
    assert state["mounting"]["method"] == "average" and state["mounting"]["intercondylar_mm"] == 100.0
    up = scans[2][:3, :3] @ [0, 0, 1.0]  # верх фантома в координатах кейса
    assert np.degrees(np.arccos(case.anatomy.frame[2, :3] @ up)) < 10
    ids = case.generate(travel=4)
    assert len(ids) == 3 and all(not m["recorded"] for m in case.state()["movements"])
    guidance = case.analysis()["guidance_deg"]
    assert guidance["protrusion"] is not None
    c = case.contacts(ids[0])
    assert c["sectors"] and c["teeth"]  # протрузия: касаются передние зубы

    case.set_settings(sagittal_right_deg=45)
    assert case.settings_info()["sources"]["sagittal_right_deg"] == "вручную"
    assert case.undo() == "настройки суставов" and case.settings.sagittal_right_deg == 35.0
    assert case.redo() == "настройки суставов" and case.settings.sagittal_right_deg == 45.0
    case.undo()
    case.undo()  # движения артикулятора
    assert case.movements == {}


def test_seat_bite_and_save_load(scans, tmp_path):
    case = fresh(scans)
    case.mount("average")
    info = case.seat_bite()
    assert "theta_deg" in info
    case.generate(travel=3)
    case.set_settings(bennett_left_deg=15)
    path = tmp_path / "case.ccdjaw"
    case.save(str(path))
    back = jc.JawCase.load(str(path))
    assert back.state() == {**case.state(), "undo": None, "redo": None}
    assert np.allclose(back.lower_vertices, case.lower_vertices)
    for a, b in zip(back.movements.values(), case.movements.values()):
        assert np.allclose(a.recording.transforms, b.recording.transforms)
    assert back.analysis()["guidance_deg"] == case.analysis()["guidance_deg"]


def test_patient_recording_is_checked_and_mounts(scans, tmp_path):
    """Запись пациента загружается в кейс, сверяется со сканами и даёт монтаж по движению."""
    case = fresh(scans)
    case.mount("average")
    truth = case.anatomy
    recs = [kin.opening(truth, kin.Settings(30, 32)), kin.protrusion(truth, kin.Settings(30, 32), 5,
                                                                      case.occlusion)]
    xml = ['<JawMotion units="mm">']
    for r in recs:
        xml.append(f'<Movement name="{r.name}">' + "".join(
            f'<Frame time="{t:.4f}">{" ".join(f"{v:.9f}" for v in M.ravel())}</Frame>'
            for t, M in zip(r.times, r.transforms)) + "</Movement>")
    (tmp_path / "motion.xml").write_text("".join(xml) + "</JawMotion>")
    out = case.load_motion(str(tmp_path / "motion.xml"))
    assert out["recordings"] == 2
    check = out["scans_check"]
    assert check["recordings"][1]["max_depth_mm"] < kin.PENETRATION_MM  # ведение по зубам — без проникновения
    state = case.mount("motion")
    assert state["mounting"]["method"] == "motion" and case.settings.sources["sagittal_right_deg"] == "запись движения"
    # углы записи — от плоскости «шарнирная ось — резцовая точка»; запись построена средним артикулятором,
    # горизонталь которого (окклюзионная плоскость) наклонена к ней на угол Балквилла
    assert case.settings.sagittal_right_deg == pytest.approx(30 - mo.BALKWILL_DEG, abs=1.5)
    assert case.settings.frame is not None  # и остаются от своей горизонтали
    ids = case.generate(travel=3)
    assert len([m for m in case.movements.values() if m.recorded]) == 2 and len(ids) == 3
    analysis = mo.analyze_case(mo.MotionCase("x", [case.movements[ids[0]].recording]), case.anatomy)
    assert analysis["recordings"][0]["kind"] == "protrusion"


def test_fgp_from_case(scans):
    case = fresh(scans)
    case.mount("average")
    case.generate(travel=3)
    mesh = case.fgp("upper", cell=0.4)
    assert len(mesh.faces) > 100 and "верхней" in mesh.source
