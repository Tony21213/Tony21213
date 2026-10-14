"""Варианты выгрузки, независимые от подключения шага «Экспорт»."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from casedesigner import exocad_articulator_export as xa
from casedesigner.register import apply, axis_angle, rigid
from test_exocad_articulator_export import case, project, register, exocad_app  # noqa: F401


@pytest.mark.parametrize("with_mandible", [False, True], ids=["no-bones", "no-skull"])
def test_export_without_skull_uses_standard_inserts(case, project, register, tmp_path, with_mandible):
    """Отсутствие черепа не мешает выгрузке сканов; кость НЧ остаётся необязательной."""
    bone = case.bones["lower"]
    case.bones = {"lower": bone} if with_mandible else {}
    case.condylar_paths = None
    case.bites = []
    out = tmp_path / "case-002"
    result = xa.export(case, str(project), str(out), register=register,
                       exocad_app=exocad_app(tmp_path, custom=True))
    assert any("вставки ССП" in n and "стандартные" in n for n in result["notes"])
    assert result["files"].get("кости-каркасы", 0) == int(with_mandible)
    assert not list(out.glob("*-upperjaw-partialframework_cad.stl"))
    assert not list(out.glob("*-TotalJaw*.ply"))
    for jaw in ("upper", "lower"):
        mesh = trimesh.load_mesh(out / f"case-002-{jaw}jaw.ply", process=False)
        expected = apply(np.asarray(result["case_to_articulator"]), getattr(case, jaw).vertices)
        np.testing.assert_allclose(mesh.vertices, expected, atol=1e-3)
    if with_mandible:
        mesh = trimesh.load_mesh(out / "case-002-lowerjaw-partialframework_cad.stl", process=False)
        # STL дублирует вершины граней — сравниваем поверхность через ближайшие вершины.
        from scipy.spatial import cKDTree
        expected = apply(np.asarray(result["case_to_articulator"]), bone.vertices)
        assert cKDTree(mesh.vertices).query(expected)[0].max() < 1e-3
        assert (out / "case-002-lowerjaw.partialInfo").is_file()
    else:
        assert not list(out.glob("*.partialInfo"))
        assert not list(out.glob("*-partialframework_cad.stl"))
    art = out / xa.SUBFOLDER / "articulator" / result["articulator_folder"]
    settings = ET.parse(art / "articulatorparameters.xml").getroot()
    assert settings.find("ArticulatorMainParts/CondylarInserts") is None
    assert settings.find("ArticulatorMainParts/BennettInserts") is None
    for side in ("Left", "Right"):
        assert float(settings.findtext(f"RotationAxis{side}Param/Value")) == 0


def test_case_from_session_keeps_lower_bone_with_corrected_scan(case):
    """Прикус сканов и последующая поправка монтажа применяются к НЧ в одном порядке."""
    reg = rigid(axis_angle(np.array([0., 0., 1.]), .3), [8., -2., 4.])
    bite = rigid(axis_angle(np.array([1., 0., 0.]), -.08), [0., 1., -2.])
    fix = rigid(axis_angle(np.array([0., 1., 0.]), .04), [.1, 0., .2])
    items = {}
    for name in ("upper", "lower"):
        mesh = getattr(case, name)
        items[name] = {"role": "jaw", "path": None, "reg": SimpleNamespace(transform=reg),
                       "scan": SimpleNamespace(vertices=mesh.vertices, faces=mesh.faces)}
    # Кость задана в КТ, а скан — в своей системе: это одна и та же опорная геометрия.
    mandible = trimesh.Trimesh(apply(reg, case.lower.vertices), case.lower.faces, process=False)
    maxilla = trimesh.Trimesh(apply(reg, case.upper.vertices), case.upper.faces, process=False)
    session = SimpleNamespace(
        jaw=SimpleNamespace(anatomy=case.anatomy, settings=case.settings, lower_fix=fix,
                            movements={}, mounting="ct"),
        bite_motion=bite, scans=items, structures={"mandible": mandible, "maxilla": maxilla},
        landmarks={}, suggested=set(), landmark_sources={},
        _jaw_items=lambda: (items["upper"], items["lower"]))
    result = xa.case_from_session(session)
    expected = apply(fix, apply(bite, apply(reg, case.lower.vertices)))
    np.testing.assert_allclose(result.lower.vertices, expected)
    np.testing.assert_allclose(result.bones["lower"].vertices, expected)
    np.testing.assert_allclose(result.upper.vertices, maxilla.vertices)
    np.testing.assert_allclose(result.bones["upper"].vertices, maxilla.vertices)
    assert result.condylar_paths is None


def test_missing_project_does_not_create_export(case, register, tmp_path):
    out = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="нет проекта exocad"):
        xa.export(case, str(tmp_path), str(out), register=register,
                  exocad_app=exocad_app(tmp_path, custom=True))
    assert not out.exists()
