"""Автоматические ориентиры: поиск агентом (как ALI-CBCT) на игрушечных сетях.

Вместо DenseNet — сеть, которая «смотрит» на яркое пятно в окне 64³: считает
центр яркости и велит шагать к нему. Так проверяются подготовка КТ, окно
агента, шаги по двум масштабам, итоговые пробы и перевод в мм пациента —
в том числе у КТ с осями не в LPS. Настоящие веса проверяются отдельно, если
лежат в CCD_LANDMARK_MODELS (тест пропускается без них).
"""

import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from casedesigner import auto_landmarks as al
from casedesigner.volume import Volume

FOV = al.FOV


def toward_bright(path):
    """Сеть-заглушка: Q = (смещение центра яркости окна от центра агента) по ±z, ±y, ±x."""
    idx = np.arange(FOV, dtype=np.float32) - FOV // 2  # агент — в вокселе FOV//2 окна
    grids = [np.broadcast_to(idx.reshape(s), (FOV,) * 3).astype(np.float32)
             for s in ((FOV, 1, 1), (1, FOV, 1), (1, 1, FOV))]
    inits = [numpy_helper.from_array(g.reshape(1, 1, FOV, FOV, FOV), n) for g, n in zip(grids, "ZYX")]
    inits += [numpy_helper.from_array(np.array(1.0, np.float32), "one"),
              numpy_helper.from_array(np.array(8.0, np.float32), "p"),
              numpy_helper.from_array(np.array([[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]],
                                               np.float32), "M")]
    nodes = [helper.make_node("Add", ["zone", "one"], ["w0"]),
             helper.make_node("Pow", ["w0", "p"], ["w"]),  # резче к самому яркому
             helper.make_node("ReduceSum", ["w"], ["total"], keepdims=0)]
    for a in "ZYX":
        nodes += [helper.make_node("Mul", ["w", a], [f"w{a}"]),
                  helper.make_node("ReduceSum", [f"w{a}"], [f"s{a}"], keepdims=0),
                  helper.make_node("Div", [f"s{a}", "total"], [f"c{a}"]),
                  helper.make_node("Reshape", [f"c{a}", "shape11"], [f"r{a}"])]
    inits.append(numpy_helper.from_array(np.array([1, 1], np.int64), "shape11"))
    nodes += [helper.make_node("Concat", ["rZ", "rY", "rX"], ["off"], axis=1),
              helper.make_node("MatMul", ["off", "M"], ["q"])]
    graph = helper.make_graph(nodes, "toward_bright",
                              [helper.make_tensor_value_info("zone", TensorProto.FLOAT, [1, 1, FOV, FOV, FOV])],
                              [helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 6])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)


@pytest.fixture(scope="module")
def toy_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("landmark_models")
    for ali in ("RPo", "N"):
        for scale in al.SCALE_KEYS:
            toward_bright(str(root / ali / f"{scale}.onnx"))
    return al.Models(str(root))


def blob_volume(centre_mm, direction=np.eye(3), size=70, spacing=0.8):
    """КТ size³ вокселей с ярким шаром (кость) радиусом 3 мм в точке centre_mm (мм LPS)."""
    origin = -np.asarray(direction) @ (np.full(3, (size - 1) / 2) * spacing)
    vol = Volume(np.zeros((size,) * 3, np.int16), np.full(3, spacing), origin, np.asarray(direction, float))
    zz, yy, xx = np.mgrid[:size, :size, :size]
    world = vol.to_world(np.stack([xx, yy, zz], -1).reshape(-1, 3)).reshape(size, size, size, 3)
    d = np.linalg.norm(world - np.asarray(centre_mm), axis=-1)
    vol.data[...] = np.where(d < 3.0, 2000, np.where(d < 12, 300 - 20 * d, -800)).astype(np.int16)
    return vol


def test_histogram_like_ali():
    a = np.random.default_rng(0).normal(200, 900, (40, 40, 40)).astype(np.float32)
    out = al.correct_histogram(a)
    assert out.dtype == np.int16
    assert out.min() >= -1500 and out.max() <= 4000
    lo, hi = np.percentile(a, [1, 99])
    lo, hi = max(lo, -1500), min(hi, 4000)  # границы ALI-CBCT
    assert abs(out.min() - lo) < 0.02 * np.ptp(a) and abs(out.max() - hi) < 0.02 * np.ptp(a)


@pytest.mark.parametrize("direction", [np.eye(3), np.diag([-1.0, -1.0, 1.0])], ids=["LPS", "RAS"])
def test_agent_finds_point_in_patient_mm(toy_models, direction):
    target = np.array([6.0, -9.0, 4.5])
    vol = blob_volume(target, direction)
    steps = []
    found = al.find(vol, toy_models, keys=["Po_R", "N"], progress=lambda f, m: steps.append(m))
    assert set(found) == {"Po_R", "N"} and steps[-1] == "Точки: 2 из 2"
    for f in found.values():
        assert f.point is not None, f.note
        assert np.linalg.norm(f.point - target) < 0.7, (f.key, f.point)  # точнее 0,3-мм масштаба ×2
        assert 0 < f.steps < al.MAX_STEPS


def test_same_result_for_any_number_of_workers(toy_models):
    vol = blob_volume(np.array([-5.0, 3.0, 0.0]))
    one = al.find(vol, toy_models, workers=1)
    two = al.find(vol, toy_models, workers=2)
    for k in one:
        assert np.allclose(one[k].point, two[k].point)


def test_models_folder_lists_points(toy_models, tmp_path):
    assert set(toy_models.available()) == {"Po_R", "N"}
    assert al.Models(str(tmp_path)).available() == []
    assert al.find(blob_volume(np.zeros(3)), al.Models(str(tmp_path))) == {}


@pytest.mark.skipif(not os.environ.get("CCD_LANDMARK_MODELS"), reason="нет настоящих весов (CCD_LANDMARK_MODELS)")
def test_real_models_on_open_cbct():
    """Открытый КЛКТ IC_0005 (тестовые данные ASO-CBCT): ANS, IF, PNS авторов — в пределах 2 мм."""
    from casedesigner.volume import load_volume

    ct = os.environ.get("CCD_LANDMARK_CT")
    if not ct:
        pytest.skip("нет КТ (CCD_LANDMARK_CT)")
    found = al.find(load_volume(ct), al.Models(os.environ["CCD_LANDMARK_MODELS"]), keys=["ANS", "PNS", "IP"])
    ref = {"ANS": [1.1, -49.8, 5.7], "IP": [2.4, -32.7, -8.5], "PNS": [-0.8, 7.6, -0.4]}
    for k, p in ref.items():
        assert np.linalg.norm(found[k].point - np.array(p)) < 2.0, (k, found[k].point)


def test_session_finds_landmarks_as_suggestions(toy_models, tmp_path, monkeypatch):
    """Через сессию: найденное — «проверьте», поставленное врачом не трогается."""
    from casedesigner.app.session import Session

    s = Session(memory_path=str(tmp_path / "memory.jsonl"), models_dir=str(tmp_path / "models"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    os.symlink(toy_models.folder, tmp_path / "models" / "landmarks")
    assert s.landmark_models_status()["usable"]
    target = np.array([6.0, -9.0, 4.5])
    vol = blob_volume(target)
    monkeypatch.setattr("casedesigner.app.session.CaseCT", lambda v, *a: type("C", (), {"levels": None})())
    s.vol, s.case = vol, object()
    s.landmarks["N"] = np.array([1.0, 2.0, 3.0])  # врач уже поставил
    info = s.auto_landmarks()
    by_key = {lm["key"]: lm for lm in info["landmarks"]}
    assert np.linalg.norm(np.array(by_key["Po_R"]["point"]) - target) < 0.7 and by_key["Po_R"]["suggested"]
    assert by_key["N"]["point"] == [1.0, 2.0, 3.0] and not by_key["N"]["suggested"]
    assert info["auto"]["Po_R"]["found"]
