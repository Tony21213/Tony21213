"""Подготовка моделей сегментации: скачать веса TotalSegmentator и перевести в ONNX.

Запускается один раз (нужны PyTorch и nnU-Net: pip install torch nnunetv2):

    python tools/prepare_models.py ПАПКА_МОДЕЛЕЙ [--cache ПАПКА_ДЛЯ_АРХИВОВ]

Получается папка моделей для приложения (--models): по подпапке на модель с
model.onnx и model.json. Что из какой модели берётся и под какой лицензией —
tools/models/*.json и docs/models.md.
"""

import argparse
import glob
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_nnunet_onnx import export  # noqa: E402

RELEASES = "https://github.com/wasserth/TotalSegmentator/releases/download"
MODELS = {
    # имя модели: (архив весов, конфигурация nnU-Net внутри архива)
    "teeth": ("v2.5.0-weights/Dataset113_ToothFairy3.zip", "nnUNetTrainer_onlyMirror01__nnUNetPlans__3d_lowres_high"),
    "craniofacial": ("v2.5.0-weights/Dataset115_mandible.zip", "nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres"),
    "cavities": ("v2.3.0-weights/Dataset775_head_glands_cavities_492subj.zip",
                 "nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres_high"),
}
SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# Точки лица для эстетической системы: MediaPipe Face Landmarker (Apache-2.0), готовая модель, без перевода в ONNX.
FACE_MODEL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
              "face_landmarker.task")


def fetch(archive: str, cache: str) -> str:
    """Скачать и распаковать архив весов (если ещё нет); вернуть папку, куда распакован."""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, os.path.basename(archive))
    if not os.path.isfile(path):
        print(f"Скачиваю {archive}…", flush=True)
        urllib.request.urlretrieve(f"{RELEASES}/{archive}", path + ".part")
        os.replace(path + ".part", path)
    folder = os.path.splitext(path)[0]
    if not glob.glob(os.path.join(folder, "*", "*", "fold_0", "checkpoint_final.pth")):
        with zipfile.ZipFile(path) as z:
            z.extractall(folder)
    return folder


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description="Скачать веса TotalSegmentator и перевести в ONNX")
    parser.add_argument("out", help="папка моделей для приложения")
    parser.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "casedesigner"),
                        help="куда складывать скачанные архивы")
    parser.add_argument("--only", nargs="+", choices=sorted(MODELS) + ["face"], help="только эти модели")
    args = parser.parse_args(argv)
    if not args.only or "face" in args.only:
        target = os.path.join(args.out, "face_landmarker.task")
        if not os.path.isfile(target):
            os.makedirs(args.out, exist_ok=True)
            print("Скачиваю модель точек лица (MediaPipe)…", flush=True)
            urllib.request.urlretrieve(FACE_MODEL, target + ".part")
            os.replace(target + ".part", target)
    for name in [m for m in (args.only or MODELS) if m != "face"]:
        archive, configuration = MODELS[name]
        folder = fetch(archive, args.cache)
        found = glob.glob(os.path.join(folder, "*", configuration))
        if not found:
            raise FileNotFoundError(f"в архиве {archive} нет конфигурации {configuration}")
        with open(os.path.join(SPECS, f"{name}.json"), encoding="utf-8") as f:
            spec = json.load(f)
        model = export(found[0], os.path.join(args.out, name), spec)
        print(f"{name}: готово, совпадение с PyTorch {model['check']['argmax_agreement']:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
