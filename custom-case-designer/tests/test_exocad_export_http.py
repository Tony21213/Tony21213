"""HTTP-контракт выгрузки в артикулятор после передачи серверного шага."""

import json
import time
import urllib.error
import urllib.request

import pytest

from casedesigner.app.server import serve


def _request(base, payload):
    request = urllib.request.Request(
        f"{base}/api/export/exocad-articulator",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _job(base, job_id):
    for _ in range(100):
        with urllib.request.urlopen(f"{base}/api/jobs/{job_id}") as response:
            value = json.loads(response.read())
        if value["status"] != "running":
            return value
        time.sleep(0.01)
    raise AssertionError("фоновая выгрузка не завершилась")


@pytest.mark.xfail(strict=True, reason="ждёт маршрут экспорта в шаге «Экспорт»")
def test_export_route_starts_job_and_returns_summary(tmp_path):
    class Session:
        def export_exocad_articulator(self, project_dir, out_dir=None, progress=None):
            assert project_dir == str(tmp_path)
            assert out_dir is None
            return {"files": {"верхний скан": "case-001-upperjaw.ply"}, "movements": 1}

    server = serve(Session())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, answer = _request(base, {"project": str(tmp_path), "out": None})
        assert status == 200 and set(answer) == {"job"}
        job = _job(base, answer["job"])
        assert job["status"] == "done"
        assert job["result"] == {"files": {"верхний скан": "case-001-upperjaw.ply"}, "movements": 1}
    finally:
        server.shutdown()


@pytest.mark.xfail(strict=True, reason="ждёт проверку обязательного пути проекта")
def test_export_route_rejects_missing_project_with_400():
    class Session:
        def export_exocad_articulator(self, *args, **kwargs):
            raise AssertionError("метод сессии нельзя вызывать без папки проекта")

    server = serve(Session())
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, answer = _request(base, {"out": None})
        assert status == 400
        assert "проект" in answer["error"].lower()
    finally:
        server.shutdown()
