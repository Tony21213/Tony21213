import numpy as np
import pytest

import phantom
from casedesigner.cli import main
from casedesigner.fusion import CaseCT, Scan
from casedesigner.learning import MIN_CASES, AlignmentMemory
from casedesigner.register import apply, axis_angle, rigid

DILATION = 0.12  # граница зубов на КТ «аппарата» снаружи настоящей


@pytest.fixture(scope="module")
def dilated_ct():
    return CaseCT(phantom.make_volume(tooth_dilation=DILATION))


def lower_scan(seed=2):
    verts, faces = phantom.make_scan("lower")
    pose = phantom.scan_pose(seed)
    return Scan("lower", apply(pose, verts), faces), verts, np.linalg.inv(pose)


def test_edge_shift_found_with_pose(dilated_ct):
    scan, truth, _ = lower_scan()
    reg = dilated_ct.register(scan, jaw="lower")
    # Фантом сам по себе даёт около −0.01 мм (размытие на изогнутой поверхности).
    assert reg.edge_shift == pytest.approx(DILATION, abs=0.02)
    assert np.linalg.norm(apply(reg.transform, scan.vertices) - truth, axis=1).max() < 0.02
    assert abs(reg.stats["signed_mean_mm"]) < 0.01


def test_memory_learns_from_accepted_and_corrected_cases(tmp_path, dilated_ct):
    memory = AlignmentMemory(str(tmp_path / "memory.jsonl"))
    assert memory.prior("Vatech X") == (0.0, 0.0)
    scan, truth, T_true = lower_scan()
    auto = dilated_ct.register(scan, jaw="lower")
    for _ in range(MIN_CASES - 1):
        memory.record("Vatech X", auto, auto)
    assert memory.prior("Vatech X") == (0.0, 0.0)  # мало кейсов — не учимся

    # Пользователь поставил скан вручную (здесь — в истинное положение): сдвиг восстанавливается из него.
    corrected = dilated_ct.evaluate(scan, T_true, jaw="lower")
    rec = memory.record("Vatech X", auto, corrected)
    assert rec["corrected_mm"] < 0.05 and rec["edge_shift_mm"] == pytest.approx(DILATION, abs=0.02)
    shift, weight = memory.prior("Vatech X")
    assert shift == pytest.approx(DILATION, abs=0.02) and weight > 0
    assert memory.prior("Другой аппарат") == (0.0, 0.0)

    # Плохо легший скан в обучение не идёт.
    bad = dilated_ct.evaluate(scan, rigid(np.eye(3), [0.4, 0, 0]) @ T_true, jaw="lower")
    rec = memory.record("Vatech X", auto, bad)
    assert rec["corrected_mm"] == pytest.approx(0.4, abs=0.01)
    assert memory.summary()["Vatech X"]["cases"] == MIN_CASES + 1
    assert memory.summary()["Vatech X"]["usable_for_learning"] == MIN_CASES
    assert "scan" not in (tmp_path / "memory.jsonl").read_text(encoding="utf-8").lower()


def test_refine_from_manual_position(dilated_ct):
    scan, truth, T_true = lower_scan()
    rough = rigid(axis_angle(np.array([1.0, 0.3, 0.2]), np.radians(3)), np.array([0.8, -0.5, 0.4])) @ T_true
    before = dilated_ct.evaluate(scan, rough, jaw="lower")
    after = dilated_ct.register(scan, jaw="lower", start=rough)
    assert before.stats["p90_mm"] > 0.3
    assert np.linalg.norm(apply(after.transform, scan.vertices) - truth, axis=1).max() < 0.02


def test_cli_memory(tmp_path):
    import SimpleITK as sitk
    import trimesh

    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(tmp_path / "ct.nii.gz"))
    verts, faces = phantom.make_scan("lower")
    trimesh.Trimesh(apply(phantom.scan_pose(2), verts), faces).export(tmp_path / "lower.stl")
    memory = tmp_path / "memory.jsonl"
    args = ["register", str(tmp_path / "ct.nii.gz"), "--scan", str(tmp_path / "lower.stl"), "--jaw", "lower=lower",
            "--memory", str(memory), "-o", str(tmp_path / "out")]
    assert main(args) == 0 and not memory.exists()  # без --accept ничего не запоминается
    assert main(args + ["--accept"]) == 0
    assert len(AlignmentMemory(str(memory)).records()) == 1
