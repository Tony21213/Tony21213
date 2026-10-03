"""Сети-заглушки ONNX для проверки движка сегментации без настоящих весов.

Выход каждого класса — линейная функция яркости (свёртка 1×1×1): так классы
делят шкалу яркости на интервалы, и результат легко предсказать.
"""

import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from casedesigner.structures import ANATOMY, TEETH

# Нормировка заглушек: яркость / 1000 — кость ≈ 1.3, зубы ≈ 2.5.
NORMALIZATION = {"clip": [-1000, 3000], "mean": 0.0, "std": 1000.0}


def linear_model(folder, scheme, slopes, biases, patch, extra=(), spacing=0.3):
    os.makedirs(folder, exist_ok=True)
    c = len(slopes)
    w = numpy_helper.from_array(np.array(slopes, np.float32).reshape(c, 1, 1, 1, 1), "W")
    b = numpy_helper.from_array(np.array(biases, np.float32), "B")
    node = helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[1, 1, 1])
    graph = helper.make_graph(
        [node], "toy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, "z", "y", "x"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c, "z", "y", "x"])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, os.path.join(folder, "model.onnx"))
    with open(os.path.join(folder, "model.json"), "w", encoding="utf-8") as f:
        json.dump({"scheme": scheme, "spacing": spacing, "patch": list(patch), "overlap": 0.5,
                   "normalization": NORMALIZATION, "extra_classes": list(extra)}, f)


def make_models(root, patch=(64, 96, 96)):
    """anatomy: кость → «нижняя челюсть», зубы → «зубы верхней челюсти»; teeth: зубы → «зуб 11»."""
    keys = [s.key for s in ANATOMY]
    slopes, biases = [0.0] * (len(ANATOMY) + 1), [0.0] + [-100.0] * len(ANATOMY)
    i = keys.index("mandible") + 1
    slopes[i], biases[i] = 1.0, -0.7  # выше 700
    i = keys.index("upper_teeth") + 1
    slopes[i], biases[i] = 2.0, -2.6  # выше 1900 (там зубы обгоняют кость)
    linear_model(os.path.join(root, "anatomy"), "anatomy", slopes, biases, patch)

    slopes, biases = [0.0] * (len(TEETH) + 2), [0.0] + [-100.0] * (len(TEETH) + 1)
    slopes[1], biases[1] = 2.0, -3.8  # зуб 11: выше 1900
    linear_model(os.path.join(root, "teeth"), "teeth", slopes, biases, patch, extra=["ignore"])
    return root
