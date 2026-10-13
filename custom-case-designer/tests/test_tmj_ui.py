"""Интеграция визуализации суставных поверхностей и заметок ориентиров."""

import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from casedesigner import auto_landmarks as al
from casedesigner import guidance, landmarks
from casedesigner.app.server import serve
from casedesigner.app.session import Session


def _request(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{base}/api/{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request)


def _wait(base, job_id):
    for _ in range(100):
        with _request(base, f"jobs/{job_id}") as response:
            result = json.loads(response.read())
        if result["status"] == "done":
            return result["result"]
        assert result["status"] != "error", result["error"]
        time.sleep(0.01)
    raise AssertionError("фоновой поиск ориентиров не завершился")


def test_joint_parts_empty_without_segmentation(tmp_path):
    session = Session(memory_path=str(tmp_path / "memory.jsonl"))
    assert session.articulation_joint_parts() == {
        "right": {"head": None, "slope": None},
        "left": {"head": None, "slope": None},
    }


def test_heads_follow_bite_motion_but_slopes_stay_on_skull(tmp_path, monkeypatch):
    session = Session(memory_path=str(tmp_path / "memory.jsonl"))
    triangle = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
    session.structures = {"mandible": triangle.copy(), "skull": triangle.copy()}
    session.jaw.anatomy = SimpleNamespace(frame=np.eye(4))
    session._articulation_ready = True
    session.landmarks = {"Co_R": np.array([0., 0., 0.]), "Co_L": np.array([0., 0., 0.])}
    session.bite_motion = np.eye(4)
    session.bite_motion[0, 3] = 4.0
    session.bite_view = True
    monkeypatch.setattr(landmarks, "condylar_heads", lambda *_a, **_k: {
        "Co_R": {"vertices": np.array([0, 1, 2])},
        "Co_L": {"vertices": np.array([0, 1, 2])},
    })
    monkeypatch.setattr(guidance, "articular_surface", lambda *_a, **_k: np.array([0]))

    parts = session.articulation_joint_parts()
    assert np.allclose(parts["right"]["head"].vertices[:, 0], triangle.vertices[:, 0] + 4.0)
    assert np.allclose(parts["right"]["slope"].vertices, triangle.vertices)
    assert session.articulation_joint_parts() is parts  # кэш на ту же сегментацию и точки


def test_joint_api_lists_available_meshes_and_returns_binary(tmp_path):
    session = Session(memory_path=str(tmp_path / "memory.jsonl"))
    surface = trimesh.creation.icosphere(subdivisions=1)
    session.articulation_joint_parts = lambda: {
        "right": {"head": surface, "slope": None},
        "left": {"head": None, "slope": None},
    }
    server = serve(session)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with _request(base, "articulation/joint") as response:
            assert json.loads(response.read()) == {
                "right": {"head": True, "slope": False},
                "left": {"head": False, "slope": False},
            }
        with _request(base, "articulation/joint?side=right&part=head") as response:
            raw = response.read()
        nv, nf = np.frombuffer(raw[:8], np.uint32)
        assert len(raw) == 8 + nv * 12 + nf * 12
        with pytest.raises(urllib.error.HTTPError) as err:
            _request(base, "articulation/joint?side=middle&part=head")
        assert err.value.code == 400
    finally:
        server.shutdown()


def test_auto_landmark_replacement_note_is_exposed_in_state(tmp_path, monkeypatch):
    session = Session(memory_path=str(tmp_path / "memory.jsonl"))
    session.vol = object()
    session.case = SimpleNamespace()
    session._landmark_folder = lambda: str(tmp_path)
    session._mandible_condyles = lambda heads_only=False: {"Co_R": np.array([1., 2., 3.])}

    class Models:
        def __init__(self, _folder):
            pass

        @staticmethod
        def available():
            return ["available"]

    monkeypatch.setattr(al, "Models", Models)
    monkeypatch.setattr(al, "find", lambda *_a, **_k: {
        "Co_R": SimpleNamespace(point=np.array([10., 2., 3.]), note="")
    })
    server = serve(session)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with _request(base, "landmarks/auto", {"keys": ["Co_R"]}) as response:
            job = json.loads(response.read())
        result = _wait(base, job["job"])
        expected = "ALI-CBCT в 9.0 мм от верхушки головки по сегментации — взята головка"
        assert result["auto"]["Co_R"]["note"] == expected
        with _request(base, "landmarks") as response:
            state = json.loads(response.read())
        point = next(item for item in state["landmarks"] if item["key"] == "Co_R")
        assert point["note"] == expected
        assert session.articulation_condyles()["points"]["right"]["note"] == expected
    finally:
        server.shutdown()


def test_mounting_sagittal_warning_is_present_in_state(tmp_path):
    session = Session(memory_path=str(tmp_path / "memory.jsonl"))
    session.jaw.notes = ["ССП справа вне 5–75° — не применён"]
    assert session.jaw.state()["notes"] == ["ССП справа вне 5–75° — не применён"]
