"""Перевод сети зубов на внутриротовом скане DentalModelSeg в ONNX (один раз; нужны PyTorch и MONAI).

    python tools/prepare_scan_seg.py ВЕСА.pth ПАПКА_МОДЕЛЕЙ

ВЕСА — `07-21-22_val-loss0.169.pth` из релиза v3.0 https://github.com/DCBIA-OrthoLab/SlicerDentalModelSeg
(Apache-2.0; 2-е место MICCAI 3DTeethSeg'22; Leclercq M., Prieto J. C. и др., UNC и Univ. of Michigan). Сеть —
MONAI U-Net 2D: 4 канала (цвет по нормалям и глубина вида) → 34 класса (0 — фон, 1–32 — зубы Universal, 33 —
десна). Отрисовка видов и голосование — в casedesigner/scan_seg.py (без PyTorch3D).
"""

import hashlib
import json
import os
import sys

IMAGE = 320
CLASSES = 34


def main(argv=None):
    import numpy as np
    import onnxruntime as ort
    import torch
    from monai.networks.nets import UNet

    weights, out = (argv or sys.argv[1:])[:2]
    net = UNet(spatial_dims=2, in_channels=4, out_channels=CLASSES, channels=(16, 32, 64, 128, 256),
               strides=(2, 2, 2, 2), num_res_units=2)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    folder = os.path.join(out, "dentalmodelseg")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model.onnx")
    x = torch.rand(2, 4, IMAGE, IMAGE)
    torch.onnx.export(net, (x,), path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "views"}, "logits": {0: "views"}}, opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = net(x).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    meta = {"source": "SlicerDentalModelSeg v3.0 (Apache-2.0)", "weights_sha256": hashlib.sha256(open(weights, "rb").read()).hexdigest(),
            "image": IMAGE, "classes": CLASSES, "check_argmax_agreement": agree, "max_abs_diff": float(np.abs(ref - got).max())}
    with open(os.path.join(folder, "model.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"готово: {os.path.getsize(path)} байт, SHA-256 {sha}, совпадение с PyTorch {agree:.5f}")


if __name__ == "__main__":
    main()
