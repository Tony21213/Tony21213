"""Перевод нейросетей камеры фото в ONNX (один раз; нужны PyTorch и исходники моделей).

    python tools/prepare_camera_models.py geocalib ВЕСА.tar ИСХОДНИКИ_GEOCALIB ПАПКА_МОДЕЛЕЙ
    python tools/prepare_camera_models.py moge ВЕСА.pt ИСХОДНИКИ_MOGE ПАПКА_МОДЕЛЕЙ

* **GeoCalib** (Veicht A., Sarlin P.-E., Lindenberger P., Pollefeys M., ECCV 2024; код Apache-2.0, веса
  `geocalib-pinhole.tar` из релиза v1.0 https://github.com/cvg/GeoCalib — CC BY 4.0): в ONNX — только сеть (поля
  «вверх» и «широта» с уверенностью); подбор фокуса и вертикали по полям — в casedesigner/photo_camera.py.
* **MoGe-2 ViT-S** (Wang R. и др., Microsoft; MIT; `Ruicheng/moge-2-vits-normal` на Hugging Face): карта точек,
  нормали, маска и масштаб в метрах; восстановление фокуса и сдвига — в photo_camera.py.
"""

import hashlib
import json
import os
import sys


def _check(path, run_ref, feeds):
    import numpy as np
    import onnxruntime as ort

    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, feeds)
    ref = run_ref()
    return [float(np.abs(np.asarray(r) - np.asarray(g)).max()) for r, g in zip(ref, got)]


def geocalib(weights, src, out):
    import torch

    sys.path.insert(0, src)
    from geocalib.geocalib import GeoCalib

    model = GeoCalib()
    model.flexible_load(torch.load(weights, map_location="cpu")["model"])
    model.eval()

    class Net(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image):
            data = {"image": image}
            feats = {"hl": self.m.backbone(data)["features"], "ll": self.m.ll_enc(data)["features"]}
            o = self.m.perspective_decoder({"features": feats})
            return o["up_field"], o["up_confidence"], o["latitude_field"], o["latitude_confidence"]

    net = Net(model).eval()
    folder = os.path.join(out, "geocalib")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model.onnx")
    x = torch.rand(1, 3, 320, 480)
    names = ["up_field", "up_confidence", "latitude_field", "latitude_confidence"]
    torch.onnx.export(net, (x,), path, input_names=["image"], output_names=names, opset_version=17, dynamo=False,
                      dynamic_axes={"image": {2: "h", 3: "w"}, **{n: {-2: "h", -1: "w"} for n in names}})
    y = torch.rand(1, 3, 480, 320)
    with torch.no_grad():
        diff = _check(path, lambda: [t.numpy() for t in net(y)], {"image": y.numpy()})
    return path, {"source": "GeoCalib v1.0 pinhole (код Apache-2.0, веса CC BY 4.0)", "check_max_abs_diff": diff}


def moge(weights, src, out):
    import torch

    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, src)
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(weights)
    model.onnx_compatible_mode = True
    model.eval()
    folder = os.path.join(out, "moge2-vits")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model.onnx")
    names = ["points", "normal", "mask", "metric_scale"]
    torch.onnx.export(model, (torch.rand(1, 3, 518, 518), torch.tensor(1800)), path, input_names=["image", "num_tokens"],
                      output_names=names, dynamic_axes={"image": {0: "batch_size", 2: "height", 3: "width"}},
                      opset_version=17, dynamo=False)
    y = torch.rand(1, 3, 600, 400)
    tokens = torch.tensor(1800)
    with torch.no_grad():
        diff = _check(path, lambda: [model(y, tokens)[k].numpy() for k in names],
                      {"image": y.numpy(), "num_tokens": tokens.numpy()})
    return path, {"source": "MoGe-2 ViT-S normal (MIT)", "check_max_abs_diff": diff}


def main(argv=None):
    kind, weights, src, out = (argv or sys.argv[1:])[:4]
    path, meta = {"geocalib": geocalib, "moge": moge}[kind](weights, src, out)
    meta["weights_sha256"] = hashlib.sha256(open(weights, "rb").read()).hexdigest()
    with open(os.path.join(os.path.dirname(path), "model.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"{kind}: {os.path.getsize(path)} байт, SHA-256 {sha}, расхождение с PyTorch {meta['check_max_abs_diff']}")


if __name__ == "__main__":
    main()
