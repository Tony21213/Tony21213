"""Подготовка моделей автоматических ориентиров: веса ALI-CBCT → ONNX.

Запускается один раз (нужен PyTorch, MONAI и onnx: pip install torch monai onnx):

    python tools/prepare_landmarks.py ПАПКА [--pth ПАПКА_ВЕСОВ] [--release ПАПКА_РЕЛИЗА]

--release складывает файлы плоско, под именами релиза (ali-<точка>-<масштаб>.onnx),
для загрузки: gh release create landmarks-v1 ПАПКА_РЕЛИЗА/*.onnx

Из релиза SlicerAutomatedDentalTools (v0.1-v2.0_models) берутся только сети
нужных точек (casedesigner.auto_landmarks.POINTS) — из архивов по 1–2 ГБ
скачиваются нужные файлы по HTTP Range, не архивы целиком. Каждая точка —
две сети (воксель 1 и 0,3 мм), DenseNet по 58 МБ.

В ONNX веса хранятся в float16 и сразу после загрузки переводятся в float32
узлами Cast (ONNX Runtime сворачивает их при открытии): файл вдвое меньше,
считается так же в float32. Расхождение ответов с PyTorch проверяется на
случайных окнах.

В конце печатаются строки для model_store.LANDMARK_FILES (размер и SHA-256).
"""

import argparse
import glob
import hashlib
import os
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casedesigner.auto_landmarks import POINTS, SCALE_KEYS  # noqa: E402

RELEASE = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/"
ARCHIVES = ["Cranial_Base", "Upper_Bones_v2", "Lower_Bones_1", "Lower_Bones_2"]


def _range(url, a, b):
    return subprocess.run(["curl", "-sSL", "--retry", "4", "-r", f"{a}-{b}", url], capture_output=True, check=True).stdout


def _size(url):
    head = subprocess.run(["curl", "-sSL", "-r", "0-0", "-D", "-", "-o", os.devnull, url], capture_output=True,
                          text=True, check=True).stdout
    return int([ln for ln in head.splitlines() if ln.lower().startswith("content-range")][-1].split("/")[-1])


def _members(url):
    size = _size(url)
    tail = _range(url, max(size - 70000, 0), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    cd = _range(url, cd_off, cd_off + cd_size - 1)
    p = 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        crc, csize, usize = struct.unpack("<III", cd[p + 16:p + 28])
        n, e, c = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        yield cd[p + 46:p + 46 + n].decode(), method, crc, csize, usize, off
        p += 46 + n + e + c


def fetch_pth(folder):
    """Сети нужных точек: <folder>/<точка>/<масштаб>/<файл>.pth."""
    wanted = set(POINTS.values())
    for name in ARCHIVES:
        url = RELEASE + name + ".zip"
        for member, method, crc, csize, usize, off in _members(url):
            parts = member.strip("/").split("/")
            if len(parts) < 3 or parts[-3] not in wanted or not member.endswith(".pth"):
                continue
            dest = os.path.join(folder, parts[-3], parts[-2], parts[-1])
            if os.path.exists(dest) and os.path.getsize(dest) == usize:
                continue
            head = _range(url, off, off + 29)
            n, e = struct.unpack("<HH", head[26:30])
            data = _range(url, off + 30 + n + e, off + 30 + n + e + csize - 1)
            data = zlib.decompress(data, -15) if method == 8 else data
            if len(data) != usize or zlib.crc32(data) != crc:
                raise RuntimeError(f"{member}: файл повреждён при загрузке")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            print("скачано:", parts[-3], parts[-2], flush=True)


def _net():
    import torch.nn as nn
    import torch.nn.functional as F
    from monai.networks.nets.densenet import DenseNet

    class DN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc0, self.fc1 = nn.Linear(1024, 512), nn.Linear(512, 256)
            self.fc2, self.fc3 = nn.Linear(256, 128), nn.Linear(128, 6)

        def forward(self, x):
            return F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(F.relu(self.fc0(x))))))))

    class DNet(nn.Module):  # как ALI_CBCT_utils.brain.DNet
        def __init__(self):
            super().__init__()
            self.featNet = DenseNet(spatial_dims=3, in_channels=1, out_channels=1024, growth_rate=34,
                                    block_config=(6, 12, 24, 16))
            self.dens = DN()

        def forward(self, x):
            return self.dens(self.featNet(x))

    return DNet()


def _half_weights(path):
    """Веса float32 → float16 + Cast к float32 перед использованием."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(path)
    g = model.graph
    casts, keep = [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and len(init.dims) > 0:
            arr = numpy_helper.to_array(init).astype(np.float16)
            half = numpy_helper.from_array(arr, init.name + "_f16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def convert(pth_folder, out):
    import numpy as np
    import onnxruntime as ort
    import torch

    for pth in sorted(glob.glob(os.path.join(pth_folder, "*", "*", "*.pth"))):
        point, scale = pth.split(os.sep)[-3], pth.split(os.sep)[-2]
        if scale not in SCALE_KEYS:
            continue
        dest = os.path.join(out, point, f"{scale}.onnx")
        if os.path.exists(dest):
            continue
        net = _net()
        net.load_state_dict(torch.load(pth, map_location="cpu"))
        net.eval()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        torch.onnx.export(net, torch.zeros(1, 1, 64, 64, 64), dest, input_names=["zone"], output_names=["q"],
                          opset_version=17, dynamic_axes={"zone": {0: "n"}, "q": {0: "n"}}, dynamo=False)
        _half_weights(dest)
        x = torch.rand(8, 1, 64, 64, 64) * 2 - 1
        with torch.no_grad():
            ref = net(x).numpy()
        got = ort.InferenceSession(dest, providers=["CPUExecutionProvider"]).run(None, {"zone": x.numpy()})[0]
        same = float((ref.argmax(1) == got.argmax(1)).mean())
        print(f"{point} {scale}: ответ отличается до {np.abs(ref - got).max():.4f}, шаг тот же в {same:.0%}",
              flush=True)


def release(out, folder):
    import shutil

    os.makedirs(folder, exist_ok=True)
    for point in sorted(set(POINTS.values())):
        for scale in SCALE_KEYS:
            shutil.copyfile(os.path.join(out, point, f"{scale}.onnx"), os.path.join(folder, f"ali-{point}-{scale}.onnx"))
    print(f"файлы релиза: {folder}")


def table(out):
    for point in sorted({p for p in POINTS.values()}):
        for scale in SCALE_KEYS:
            path = os.path.join(out, point, f"{scale}.onnx")
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            print(f'    "{point}/{scale}.onnx": ({os.path.getsize(path)}, "{h}"),')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="папка моделей ориентиров (<точка>/<масштаб>.onnx)")
    ap.add_argument("--pth", help="папка для исходных весов .pth (по умолчанию <out>/_pth)")
    ap.add_argument("--release", help="сложить файлы для релиза landmarks-v1 в эту папку")
    a = ap.parse_args()
    pth = a.pth or os.path.join(a.out, "_pth")
    fetch_pth(pth)
    convert(pth, a.out)
    table(a.out)
    if a.release:
        release(a.out, a.release)
