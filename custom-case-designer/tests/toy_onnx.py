"""Сети-заглушки ONNX для проверки движка сегментации без настоящих весов.

Выход каждой метки — линейная функция яркости (свёртка 1×1×1): так метки
делят шкалу яркости на интервалы, и результат легко предсказать.
"""

import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Нормировка заглушек: яркость / 1000 — кость ≈ 1.3, зубы ≈ 2.5.
NORMALIZATION = {"scheme": "ct", "clip": [-1000, 3000], "mean": 0.0, "std": 1000.0}


def linear_model(folder, name, labels, slopes, biases, outputs, patch, region=None, priority=0,
                 spacing=(0.3, 0.3, 0.3), orientation="LPS", normalization=NORMALIZATION):
    os.makedirs(folder, exist_ok=True)
    c = len(labels)
    w = numpy_helper.from_array(np.array(slopes, np.float32).reshape(c, 1, 1, 1, 1), "W")
    b = numpy_helper.from_array(np.array(biases, np.float32), "B")
    node = helper.make_node("Conv", ["image", "W", "B"], ["logits"], kernel_shape=[1, 1, 1])
    graph = helper.make_graph(
        [node], "toy",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, "z", "y", "x"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, c, "z", "y", "x"])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, os.path.join(folder, "model.onnx"))
    with open(os.path.join(folder, "model.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "spacing": list(spacing), "patch": list(patch), "overlap": 0.5,
                   "orientation": orientation, "normalization": normalization, "labels": labels,
                   "outputs": outputs, "region": region or {"around": "whole"}, "priority": priority}, f)


def make_models(root, patch=(64, 96, 96)):
    """bones: кость → «нижняя челюсть», зубы → «зубы»; teeth (в области зубов, RAS): зубы → «зуб 11»."""
    linear_model(os.path.join(root, "bones"), "bones", ["background", "bone", "tooth"],
                 [0.0, 1.0, 2.0], [0.0, -0.7, -2.6],  # кость выше 700, зубы выше 1900
                 {"mandible": ["bone"], "upper_teeth": ["tooth"], "hard_tissue": ["bone", "tooth"]}, patch, priority=1)
    linear_model(os.path.join(root, "teeth"), "teeth", ["background", "tooth_11", "implant", "ignore"],
                 [0.0, 2.0, 0.0, 0.0], [0.0, -3.8, -100.0, -100.0],
                 {"teeth/tooth_11": ["tooth_11"], "teeth/implant": ["implant"], "upper_teeth": ["tooth_11"]},
                 patch, region={"around": "teeth", "margin_mm": [5, 5, 25]}, priority=2, spacing=(0.3, 0.25, 0.25),
                 orientation="RAS")
    return root
