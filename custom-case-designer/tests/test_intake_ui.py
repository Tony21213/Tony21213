"""Кнопка «Входной контроль» (docs/handoff/intake.md): сессия, HTTP, отчёт PDF и PNG; кейс без КТ."""

import json
import time
import urllib.request

import pytest
import SimpleITK as sitk
import trimesh

import phantom
from casedesigner.app import server as app_server
from casedesigner.app.session import Session
from casedesigner.register import apply


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("intake")
    vol = phantom.make_volume()
    img = sitk.GetImageFromArray(vol.data)
    img.SetSpacing(vol.spacing.tolist())
    img.SetOrigin(vol.origin.tolist())
    img.SetDirection(vol.direction.ravel().tolist())
    sitk.WriteImage(img, str(d / "ct.nii.gz"))
    v, f = phantom.make_scan("lower")
    trimesh.Trimesh(apply(phantom.scan_pose(2), v), f).export(d / "lower.stl")
    return d


def test_session_intake_with_ct(files, tmp_path):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    with pytest.raises(ValueError, match="откройте КТ"):
        s.intake_check()
    s.load_ct(str(files / "ct.nii.gz"))
    s.add_scan(str(files / "lower.stl"))
    res = s.intake_check()
    by = {(f["section"], f["title"]): f for f in res["findings"]}
    assert by[("КТ", "Размер вокселя")]["level"] in ("ok", "warn", "error")
    assert by[("КТ", "Ориентиры")]["level"] == "warn"  # точек ещё нет — один пункт, а не шесть «не найдена»
    assert not any(t.startswith("Мыщелки") for _s, t in by)
    assert by[("Сканы", "Скан верхней челюсти")]["level"] == "error" and by[("Сканы", "Прикус")]["level"] == "warn"
    assert {f["section"] for f in res["findings"]} == {"КТ", "Сканы", "Фото"}
    saved = s.intake_save(str(tmp_path / "report"))
    assert saved["files"] == ["intake.png", "intake.pdf"]
    assert (tmp_path / "report" / "intake.pdf").read_bytes().startswith(b"%PDF")


def test_scans_only_case_has_no_ct_section(files, tmp_path):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.add_scan(str(files / "lower.stl"))
    sections = {f["section"] for f in s.intake_check()["findings"]}
    assert sections == {"Сканы", "Фото"}


def test_http_intake_check_and_save(files, tmp_path):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.add_scan(str(files / "lower.stl"))
    httpd = app_server.serve(s)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/api"

    def post(path, body=None):
        req = urllib.request.Request(f"{base}/{path}", data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    try:
        job = post("intake")["job"]
        for _ in range(200):
            with urllib.request.urlopen(f"{base}/jobs/{job}") as r:
                state = json.loads(r.read())
            if state["status"] != "running":
                break
            time.sleep(0.05)
        assert state["status"] == "done" and state["result"]["findings"]
        saved = post("intake/save", {"out_dir": str(tmp_path / "out")})
        assert (tmp_path / "out" / "intake.png").is_file() and saved["files"][1] == "intake.pdf"
    finally:
        httpd.shutdown()
        httpd.server_close()
