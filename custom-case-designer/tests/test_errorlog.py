import json
import os
import urllib.request

import pytest

from casedesigner.app import errorlog
from casedesigner.app.server import serve
from casedesigner.app.session import Session


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(errorlog, "_private", {})
    monkeypatch.setattr(errorlog, "_folder", None)
    folder = errorlog.setup(str(tmp_path / "logs"), version="test")
    yield os.path.join(folder, "errors.log")
    for h in list(errorlog._logger.handlers):
        errorlog._logger.removeHandler(h)
        h.close()


def read(path):
    for h in errorlog._logger.handlers:
        h.flush()
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_scrub_removes_names_and_paths(log):
    errorlog.private(r"D:\Пациенты\Иванов Иван Иванович\КТ 2026", "КТ")
    errorlog.private(r"D:\Пациенты\Иванов Иван Иванович\Иванов-upperjaw.stl", "скан")
    errorlog.private("Иванов-upperjaw", "скан")
    text = errorlog.scrub(r"нет файла D:\Пациенты\Иванов Иван Иванович\КТ 2026\IM0001; скан Иванов-upperjaw; "
                          r"E:\Архив\Петров П.П\ct.zip; /home/someone/cases/Сидоров/scan.ply; 1/2 мм")
    assert "Иванов" not in text and "Петров" not in text and "Сидоров" not in text
    assert "<КТ>" in text and "<скан>" in text and "<путь>.zip" in text and "<путь>.ply" in text
    assert "1/2 мм" in text
    # файлы программы в стеке вызовов остаются — по ним разбирают ошибку
    here = os.path.abspath(errorlog.__file__)
    assert here.replace(os.path.expanduser("~"), "~") in errorlog.scrub(f'File "{here}", line 1')


def test_job_and_request_errors_are_logged_without_paths(log, tmp_path):
    secret = tmp_path / "Иванов Иван" / "КТ пациента"
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    srv = serve(s)
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    try:
        job = post("ct", {"path": str(secret)})["job"]
        for _ in range(100):
            state = json.loads(urllib.request.urlopen(base + f"jobs/{job}").read())
            if state["status"] != "running":
                break
            import time
            time.sleep(0.05)
        assert state["status"] == "error"
        post("log", {"message": f"TypeError в интерфейсе: {secret}", "stack": "at render (app.js:1)"})
    finally:
        srv.shutdown()
    text = read(log)
    assert "Иванов" not in text and str(secret) not in text
    assert "Открываю КТ" in text and "интерфейс" in text and "<КТ>" in text
