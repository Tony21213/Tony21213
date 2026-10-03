"""API приложения: сценарий врача от открытия КТ до экспорта, через HTTP, как в интерфейсе."""

import json
import time
import urllib.request

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.app.server import serve
from casedesigner.app.session import Session, group_of
from casedesigner.register import apply


@pytest.fixture(scope="module")
def case_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("case")
    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(d / "ct.nii.gz"))
    verts, faces = phantom.make_scan("lower")
    trimesh.Trimesh(apply(phantom.scan_pose(2), verts), faces).export(d / "lower.stl")
    return d, verts


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    session = Session(memory_path=str(tmp_path_factory.mktemp("mem") / "memory.jsonl"))
    srv = serve(session)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{base}/api/{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        raw = r.read()
        return json.loads(raw) if r.headers["Content-Type"].startswith("application/json") else raw


def wait(base, answer, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        job = call(base, f"jobs/{answer['job']}")
        if job["status"] == "done":
            return job["result"]
        assert job["status"] != "error", job["error"]
        time.sleep(0.2)
    raise TimeoutError


def test_doctor_workflow(server, case_files, tmp_path):
    d, truth = case_files
    page = urllib.request.urlopen(server + "/").read().decode()
    assert "Custom Case Designer" in page and "app.js" in page

    ct = wait(server, call(server, "ct", {"path": str(d / "ct.nii.gz")}))
    assert ct["shape"] and len(ct["focus"]) == 3
    geo = call(server, "ct/geometry?axis=axial")
    png = call(server, f"ct/slice?axis=axial&pos={ct['focus'][2]}")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(png)).size == (geo["width"], geo["height"])

    scan = call(server, "scans", {"path": str(d / "lower.stl")})
    raw = call(server, f"scans/{scan['id']}/mesh")
    nv, nf = np.frombuffer(raw[:8], np.uint32)
    assert nv == scan["vertices"] and len(raw) == 8 + nv * 12 + nf * 12

    reg = wait(server, call(server, f"scans/{scan['id']}/register", {}))
    T = np.array(reg["transform"])
    stl = trimesh.load_mesh(str(d / "lower.stl"))
    assert reg["registered"] and reg["jaw"] == "lower" and reg["stats"]["p90_mm"] < 0.1
    assert len(call(server, f"scans/{scan['id']}/colors")) == 3 * nv

    # Ручная коррекция: сдвиг ухудшает метрики, «уточнить» возвращает.
    moved = T.copy()
    moved[2, 3] += 0.5
    worse = wait(server, call(server, f"scans/{scan['id']}/evaluate", {"transform": moved.tolist()}))
    assert worse["stats"]["mean_mm"] > reg["stats"]["mean_mm"] + 0.1 and worse["corrected_mm"] == pytest.approx(0.5, abs=1e-3)
    back = wait(server, call(server, f"scans/{scan['id']}/register", {"start": moved.tolist()}))
    assert np.abs(np.array(back["transform"]) - T).max() < 0.01

    overlays = call(server, "overlays", {"axis": "axial", "pos": 7.0, "visible": [scan["id"]]})
    assert overlays and overlays[0]["id"] == scan["id"] and len(overlays[0]["segments"]) % 4 == 0

    accepted = call(server, f"scans/{scan['id']}/accept", {})
    assert accepted["scan"]["accepted"] and accepted["memory"]["cases"] == 1

    out = tmp_path / "out"
    res = wait(server, call(server, "export", {"out_dir": str(out), "bite": "scan", "frame": "exocad"}))
    assert {"lower.stl", "ct_teeth.stl", "case.json"} <= {p.name for p in out.iterdir()}
    assert "ct_teeth.stl" in res["files"]

    state = call(server, "state")
    assert state["ct"]["name"] == "ct.nii.gz" and state["scans"][0]["accepted"]


def test_errors_are_readable(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        call(server, "scans", {"path": "/нет/такого/скана.stl"})
    assert e.value.code == 400
    job = call(server, "ct", {"path": "/нет/такого/КТ"})
    for _ in range(50):
        state = call(server, f"jobs/{job['job']}")
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "error" and state["error"]


def test_groups():
    assert group_of("teeth/tooth_36") == "Зубы" and group_of("pulp/pulp_11") == "Зубы"
    assert group_of("mandibular_canal") == "Каналы" and group_of("teeth/implant") == "Ортопедия и импланты"
    assert group_of("nasal_cavity") == "Пазухи и дыхательные пути" and group_of("что-то") == "Другое"
