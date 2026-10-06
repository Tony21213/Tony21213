"""Программа без КТ, только со сканами: сканы стоят в координатах сканера (как в файлах), сканы прикуса —
на сканах челюстей, экспорт — в координатах сканера, кейс сохраняется и открывается без КТ."""

import json

import numpy as np
import pytest
import trimesh

import phantom
from casedesigner.app import session as app_session
from casedesigner.app.session import Session
from casedesigner.fusion import jaw_by_name
from casedesigner.register import apply
from test_bite_scans import bite_patch

POSE = phantom.scan_pose(4)  # координаты сканера: общие для сканов одной сессии


def write(path, v, f, T=POSE):
    trimesh.Trimesh(apply(T, v), f).export(path)
    return str(path)


def verts(path):
    return np.asarray(trimesh.load_mesh(str(path), process=False).vertices)


def test_jaw_by_name():
    for name, jaw in (("p-UpperJaw", "upper"), ("p-lowerjaw", "lower"), ("Maxillary", "upper"),
                      ("Mandibular scan", "lower"), ("Иванов ВЧ", "upper"), ("нижняя", "lower"),
                      ("OK-Modell", "upper"), ("UK", "lower")):
        assert jaw_by_name(name) == jaw, name
    for name in ("scan1", "book", "upper_lower", "TotalJaw0"):
        assert jaw_by_name(name) is None, name


def test_scans_only_case(tmp_path):
    files = {jaw: write(tmp_path / f"p-{jaw}jaw.stl", *phantom.make_scan(jaw)) for jaw in ("upper", "lower")}
    files["bite"] = write(tmp_path / "p-TotalJaw0.stl", *bite_patch(1))
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    ids = {k: s.add_scan(p)["id"] for k, p in files.items()}
    for k in ("upper", "lower", "bite"):
        info = s.register(ids[k])
        assert info["registered"] and info["jaw"] == k, k
        assert np.allclose(info["transform"], np.eye(4))  # как в файле: координаты сканера
        # Метрик совмещения с КТ нет; у скана прикуса — насколько он лёг на сканы челюстей.
        assert info["stats"] == {} if k != "bite" else info["stats"]["matched_fraction"] > 0.95
    assert s.scan_info(ids["upper"])["warnings"] == []  # что без КТ — сказано в шаге, не у каждого скана
    assert not s.bite_info()["available"]  # прикуса КТ нет — переключать нечего
    with pytest.raises(ValueError):
        s.accept(ids["upper"])
    with pytest.raises(ValueError):
        s.register(ids["upper"], start=np.eye(4).tolist())  # корректировать не по чему

    res = s.export(str(tmp_path / "out"), "ct", "dicom")  # без КТ — всегда координаты сканера, прикус сканов
    case = json.loads((tmp_path / "out" / "case.json").read_text(encoding="utf-8"))
    assert case["without_ct"] and case["bite"] == "scans" and "ct_teeth.stl" not in res["files"]
    assert "p-upperjaw_deviation.ply" not in res["files"]  # отклонений от КТ без КТ нет
    for k, p in files.items():
        got = verts(tmp_path / "out" / (s.scan_info(ids[k])["name"] + ".stl"))
        assert np.abs(got - verts(p)).max() < 1e-4, k  # как в файлах

    saved = s.save_case(str(tmp_path / "кейс"))
    again = Session(memory_path=str(tmp_path / "memory.jsonl"))
    state = again.open_case(saved["path"])
    assert state["ct"] is None
    placed = {i["name"]: (i["jaw"], i["registered"]) for i in state["scans"]}
    assert placed == {"p-upperjaw": ("upper", True), "p-lowerjaw": ("lower", True), "p-TotalJaw0": ("bite", True)}


def test_jaw_unknown_by_name(tmp_path):
    """Имя файла не говорит, какая челюсть: пользователь указывает её у первого скана, второй — другая."""
    a = write(tmp_path / "scan1.stl", *phantom.make_scan("upper"))
    b = write(tmp_path / "scan2.stl", *phantom.make_scan("lower"))
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    ia, ib = s.add_scan(a)["id"], s.add_scan(b)["id"]
    with pytest.raises(ValueError, match="укажите"):
        s.register(ia)
    assert s.register(ia, jaw="upper")["jaw"] == "upper"
    assert s.register(ib)["jaw"] == "lower"
    assert s.register(ib, jaw="bite")["role"] == "bite"  # и скан прикуса — тоже значком


def test_jaws_apart_without_bite_scans(tmp_path):
    """Сканы челюстей выгружены по отдельности, сканов прикуса нет: нижний — как в файле, с подсказкой."""
    up = write(tmp_path / "upperjaw.stl", *phantom.make_scan("upper"))
    lo = write(tmp_path / "lowerjaw.stl", *phantom.make_scan("lower"), T=phantom.scan_pose(9))
    s = Session(memory_path=str(tmp_path / "memory.jsonl"))
    iu, il = s.add_scan(up)["id"], s.add_scan(lo)["id"]
    s.register(iu)
    s.register(il)
    assert app_session.JAWS_APART in s.scan_info(il)["warnings"]
    assert app_session.JAWS_APART not in s.scan_info(iu)["warnings"]
    res = s.export(str(tmp_path / "out"), "scan", "exocad")
    assert any("стоит как в файле" in n for n in res["notes"])
