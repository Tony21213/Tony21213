"""Неблокирующие маршруты локального сервера."""

import json
import time
import urllib.request

from casedesigner.app.server import serve


def _call(base, path, body):
    req = urllib.request.Request(
        f"{base}/api/{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def test_ct_series_returns_job(monkeypatch):
    class Session:
        def ct_series(self, path):
            time.sleep(0.02)
            return []

    srv = serve(Session())
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        answer = _call(base, "ct/series", {"path": "case-001"})
        assert set(answer) == {"job"}
        for _ in range(50):
            with urllib.request.urlopen(f"{base}/api/jobs/{answer['job']}") as response:
                job = json.loads(response.read())
            if job["status"] == "done":
                assert job["result"] == []
                break
            time.sleep(0.01)
        else:
            raise AssertionError("фоновой запрос не завершился")
    finally:
        srv.shutdown()


def test_articulation_analysis_returns_contact_report():
    class Session:
        def articulation_report(self):
            return {"analysis": {"recordings": []}, "contacts": []}

    srv = serve(Session())
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        answer = _call(base, "articulation/analysis", {})
        assert "job" in answer
        for _ in range(50):
            with urllib.request.urlopen(f"{base}/api/jobs/{answer['job']}") as response:
                job = json.loads(response.read())
            if job["status"] == "done":
                assert job["result"] == {"analysis": {"recordings": []}, "contacts": []}
                break
            time.sleep(0.01)
        else:
            raise AssertionError("анализ артикулятора не завершился")
    finally:
        srv.shutdown()
