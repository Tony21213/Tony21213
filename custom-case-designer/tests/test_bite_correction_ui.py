"""Кнопка «Исправить прикус» (docs/handoff/archive/bite-correction.md): сессия, HTTP, поправка в файле кейса.

Синтетические дуги test_bite: нижняя — та же поверхность, что верхняя, с ошибкой сканера (раскрыт на шарнире);
сессия без КТ-объёма: case — заглушка, прикус сканов — общие координаты сканера (in_occlusion)."""

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from casedesigner.app import server as app_server
from casedesigner.app.session import Session
from casedesigner.fusion import Scan
from casedesigner.register import apply
from test_bite import arch_surface, hinge_turn


def session_with_bite(tmp_path, monkeypatch, open_deg=-0.35):
    upper, faces = arch_surface()
    lower = apply(hinge_turn(upper, open_deg), upper)  # прикус сканера раскрыт на шарнире
    monkeypatch.setattr("casedesigner.app.steps.scans.in_occlusion", lambda a, b: True)
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    s.case = SimpleNamespace(guided=set())
    for jaw, v in (("upper", upper), ("lower", lower)):
        s.scans[jaw] = {"scan": Scan(jaw, v, faces), "path": f"{jaw}.stl", "color": "#fff", "jaw": jaw, "role": "jaw",
                        "reg": SimpleNamespace(transform=np.eye(4), jaw=jaw), "auto": None, "transform": np.eye(4),
                        "accepted": True, "guided": True, "lower_bite": None}
    s._update_bite_motion()
    assert s.bite_info()["available"]
    return s


def test_correct_accept_reject_reset(tmp_path, monkeypatch):
    s = session_with_bite(tmp_path, monkeypatch)
    before = s.bite_info()
    res = s.bite_correct()
    json.dumps(res)  # отчёт — в JSON
    assert res["report"]["incisal_mm"] > 0.1 and res["bite_view"]["proposal"] and "Бонвилля" in " ".join(res["notes"])
    assert s.bite_reject()["proposal"] is False and not s.bite_info()["fixed"]  # отменили — прикус прежний
    s.bite_correct()
    info = s.bite_accept()
    assert info["fixed"] and info["fix_mm"] > 0.1 and not info["proposal"]
    assert np.allclose(s.bite_motion, s.bite_fix)  # прикус сканов = поправка ∘ прикус сканера (единичный)
    assert info["shift_mm"] != before["shift_mm"]
    with pytest.raises(ValueError, match="сначала"):
        s.bite_accept()
    assert not s.bite_reset()["fixed"] and np.allclose(s.bite_motion, np.eye(4))


def test_no_bite_and_no_support_refused(tmp_path, monkeypatch):
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    with pytest.raises(ValueError, match="прикуса сканов нет"):
        s.bite_correct()
    far = session_with_bite(tmp_path, monkeypatch, open_deg=-8.0)  # раскрыт так, что зубы не касаются
    with pytest.raises(ValueError, match="опоры|не в прикусе"):
        far.bite_correct()


def test_bite_fix_in_case_file(tmp_path, monkeypatch):
    s = session_with_bite(tmp_path, monkeypatch)
    s.bite_correct()
    s.bite_accept()
    saved = s.save_case(str(tmp_path / "bite.ccdcase"))
    with np.load(saved["path"], allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    assert np.allclose(meta["bite_fix"], s.bite_fix)
    reopened = Session(memory_path=str(tmp_path / "memory2.jsonl"))
    reopened.open_case(saved["path"])
    assert np.allclose(reopened.bite_fix, s.bite_fix)


def _post(base, path, body=None):
    req = urllib.request.Request(f"{base}/api/{path}", data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_correct_then_accept(tmp_path, monkeypatch):
    import time

    s = session_with_bite(tmp_path, monkeypatch)
    httpd = app_server.serve(s)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, answer = _post(base, "bite/correct")
        assert status == 200 and "job" in answer
        for _ in range(300):
            with urllib.request.urlopen(f"{base}/api/jobs/{answer['job']}") as r:
                job = json.loads(r.read())
            if job["status"] != "running":
                break
            time.sleep(0.05)
        assert job["status"] == "done", job
        status, info = _post(base, "bite/accept")
        assert status == 200 and info["fixed"]
        assert _post(base, "bite/accept")[0] == 400  # предложения уже нет — спокойный отказ
        assert _post(base, "bite/reset")[1]["fixed"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
