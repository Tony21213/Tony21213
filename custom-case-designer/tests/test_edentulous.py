"""Беззубая или почти беззубая челюсть (порядок работ, п. 15): совмещение по коронкам помечается ненадёжным,
скан ставится по другой челюсти (сканы одной выгрузки сканера) с проверкой толщины слизистой над костью.
Фантом: «беззубый» верхний скан — полный скан без вершин, легших на коронки КТ."""

import json
import urllib.request

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.app import server as app_server
from casedesigner.app.session import Session
from casedesigner.fusion import EDENTULOUS_SHARE
from casedesigner.register import apply


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("edentulous")
    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(d / "ct.nii.gz"))
    pose = phantom.scan_pose(2)
    for jaw in ("upper", "lower"):
        v, f = phantom.make_scan(jaw, palate=jaw == "upper")
        trimesh.Trimesh(apply(pose, v), f).export(d / f"{jaw}.stl")
    return d


def test_flag_and_place_by_other_jaw(files, tmp_path):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(files / "ct.nii.gz"))
    lo = s.add_scan(str(files / "lower.stl"))["id"]
    up = s.add_scan(str(files / "upper.stl"))["id"]
    assert s.register(lo, "lower")["stats"]["matched_fraction"] > EDENTULOUS_SHARE
    full = s.register(up, "upper")
    assert not full["edentulous"] and full["stats"]["matched_fraction"] > EDENTULOUS_SHARE
    T_full = np.asarray(s.scans[up]["reg"].transform)
    # «беззубый» скан: без вершин, легших на коронки КТ
    reg = s.scans[up]["reg"]
    mesh = trimesh.Trimesh(reg.scan.vertices, reg.scan.faces, process=False)
    keep = ~np.isfinite(reg.deviation)
    faces = mesh.faces[keep[mesh.faces].all(axis=1)]
    gum = trimesh.Trimesh(mesh.vertices, faces, process=False)
    gum.remove_unreferenced_vertices()
    gum.export(tmp_path / "upper_edentulous.stl")
    s.remove_scan(up)
    sid = s.add_scan(str(tmp_path / "upper_edentulous.stl"))["id"]
    info = s.register(sid, "upper")
    assert info["edentulous"] and any("беззубая" in w for w in info["warnings"])
    httpd = app_server.serve(s)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/api/scans/{sid}/other_jaw", data=b"{}",
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            placed = json.loads(r.read())
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert placed["placed_by"] == "other_jaw" and not placed["edentulous"] and placed["registered"]
    assert not any("беззубая" in w for w in placed["warnings"])
    T = np.asarray(s.scans[sid]["reg"].transform)
    v = s.scans[sid]["scan"].vertices
    assert np.linalg.norm(apply(T, v) - apply(T_full, v), axis=1).mean() < 0.5  # туда же, куда встал полный скан
    # кейс помнит отметку
    path = s.save_case(str(tmp_path / "case.ccdcase"))["path"]
    other = Session(memory_path=str(tmp_path / "m2.jsonl"))
    other.open_case(path)
    assert any(it.get("placed_by") == "other_jaw" for it in other.scans.values())


def test_other_jaw_needs_registered_opposite(files, tmp_path):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.load_ct(str(files / "ct.nii.gz"))
    up = s.add_scan(str(files / "upper.stl"))["id"]
    with pytest.raises(ValueError, match="нижней"):
        s.place_by_other_jaw(up)
