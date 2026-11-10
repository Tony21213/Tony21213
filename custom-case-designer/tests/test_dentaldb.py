"""Файловый обмен с DentalDB на синтетическом заказе; реальные данные не используются."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from casedesigner import dentaldb as db


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "Заказ 001"
    folder.mkdir()
    path = folder / "case-001.dentalProject"
    path.write_text('<Treatment><AntagonistType>MushBite</AntagonistType><MovementMarkerScan>false</MovementMarkerScan>'
                    '<Teeth><Tooth><Number>11</Number><Parameters/></Tooth></Teeth></Treatment>', encoding="utf-8")
    for side, z in (("upper", 4), ("lower", 0)):
        m = trimesh.creation.box((30, 30, 2))
        m.apply_translation((0, 0, z))
        m.export(folder / f"case-001-{side}jaw.ply")
    return path


def bundle(tmp_path, project):
    staged = tmp_path / "prepared"
    staged.mkdir()
    (staged / project.name).write_bytes(project.read_bytes())
    (staged / "case-001-upperjaw.ply").write_bytes(b"new scan")
    return staged


def test_exact_project_and_role_discovery(project):
    p = db.Project.open(str(project))
    assert set(p.sources) == {"upper", "lower"}
    assert all(np.array_equal(m, np.eye(4)) for m in p.matrices.values())
    assert len(p.case_id) == 8
    assert p.path == project.resolve()


def test_scaninfo_matrix_and_active_format(project):
    other = project.parent / "case-001-upperjaw.stl"
    trimesh.creation.box().export(other)
    with pytest.raises(ValueError, match="неоднозначные"):
        db.Project.open(str(project))
    root = ET.Element("ScanInfo")
    files = ET.SubElement(root, "ScanFiles")
    item = ET.SubElement(files, "ScanFile")
    ET.SubElement(item, "FileName").text = "case-001-upperjaw.ply"
    m = ET.SubElement(item, "TransformationMatrix")
    matrix = np.eye(4)
    matrix[3, 0] = -12
    for r in range(4):
        for c in range(4):
            ET.SubElement(m, f"_{r}{c}").text = str(matrix[r, c])
    project.with_suffix(".scanInfo").write_bytes(ET.tostring(root))
    p = db.Project.open(str(project))
    assert p.sources["upper"].suffix == ".ply"
    assert p.matrices["upper"][0, 3] == 12


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "a/../../outside"])
def test_files_cannot_escape_project(tmp_path, name):
    with pytest.raises(ValueError):
        db.relative(tmp_path, name)


def test_publication_and_repeat_keep_revision_and_backups(project, tmp_path):
    p = db.Project.open(str(project))
    original = (project.parent / "case-001-upperjaw.ply").read_bytes()
    result = p.publish(bundle(tmp_path, project))
    assert result["case_id"] == p.case_id
    assert (project.parent / "case-001-upperjaw.ply").read_bytes() == b"new scan"
    assert (project.parent / result["backup"] / "case-001-upperjaw.ply").read_bytes() == original
    assert not (project.parent / db.JOURNAL).exists()
    manifest = json.loads((project.parent / db.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["case_id"] == p.case_id and len(manifest["files"]) == 2
    p.verify()


def test_external_change_is_not_overwritten(project, tmp_path):
    p = db.Project.open(str(project))
    staged = bundle(tmp_path, project)
    project.write_bytes(project.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="изменены другой"):
        p.publish(staged)
    assert project.read_bytes().endswith(b"\n")


def test_foreign_output_is_preserved(project, tmp_path):
    foreign = project.parent / "bone_upper.stl"
    foreign.write_bytes(b"foreign")
    p = db.Project.open(str(project))
    staged = bundle(tmp_path, project)
    (staged / foreign.name).write_bytes(b"ours")
    with pytest.raises(ValueError, match="чужими"):
        p.publish(staged)
    assert foreign.read_bytes() == b"foreign"


def test_existing_design_requires_duplicate(project):
    project.with_suffix(".dentalCAD").write_bytes(b"design")
    p = db.Project.open(str(project))
    with pytest.raises(ValueError, match="Duplicate"):
        p.verify()


def test_publication_failure_rolls_back_all_files(project, tmp_path, monkeypatch):
    p = db.Project.open(str(project))
    original = {name: path.read_bytes() for name, path in [(project.name, project),
                ("case-001-upperjaw.ply", project.parent / "case-001-upperjaw.ply")]}
    staged = bundle(tmp_path, project)
    replace = db.os.replace
    calls = []

    def fail_after_first(a, b):
        if Path(b).parent == project.parent:
            calls.append(b)
            if len(calls) == 2:
                raise OSError("simulated disk failure")
        return replace(a, b)

    monkeypatch.setattr(db.os, "replace", fail_after_first)
    with pytest.raises(ValueError, match="не удалось записать"):
        p.publish(staged)
    for name, content in original.items():
        assert (project.parent / name).read_bytes() == content
    assert not (project.parent / db.JOURNAL).exists()
    assert not (project.parent / db.MANIFEST).exists()


def test_recovery_preserves_changes_after_crash(project):
    p = db.Project.open(str(project))
    backup = project.parent / db.SUB / "backups" / "test"
    backup.mkdir(parents=True)
    (backup / project.name).write_bytes(project.read_bytes())
    old = db.digest(project)
    project.write_bytes(b"published")
    record = {"backup": f"{db.SUB}/backups/test", "entries": {project.name: {"old": old, "new": db.digest(project)}}}
    db.write_json(project.parent / db.JOURNAL, record)
    project.write_bytes(b"other application changed this")
    with pytest.raises(ValueError, match="изменён после"):
        p.recover()
    assert project.read_bytes() == b"other application changed this"
    project.write_bytes(b"published")
    assert p.recover()
    assert db.digest(project) == old


def test_multiple_orders_import_exact_but_publish_refuses(project):
    other = project.parent / "case-002.dentalProject"
    other.write_bytes(project.read_bytes())
    p = db.Project.open(str(other))
    assert p.path == other.resolve() and not p.sources
    with pytest.raises(ValueError, match="один заказ"):
        p.verify()
