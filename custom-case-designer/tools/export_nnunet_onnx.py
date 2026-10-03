"""Перевод обученной модели nnU-Net v2 в ONNX для Custom Case Designer.

Нужен один раз на любой машине с PyTorch и nnU-Net (pip install torch nnunetv2);
приложению потом хватает ONNX Runtime. Скрипт собирает сеть по plans.json,
загружает веса, выгружает ONNX с произвольным размером окна, сверяет ответы
ONNX и PyTorch на случайном окне и пишет model.json для движка сегментации.

    python tools/export_nnunet_onnx.py ПАПКА_NNUNET ПАПКА_МОДЕЛИ --spec tools/models/teeth.json

ПАПКА_NNUNET — папка конфигурации nnU-Net (plans.json, dataset.json,
fold_0/checkpoint_final.pth), например из архива весов TotalSegmentator.
--spec — что модель выдаёт приложению: ориентация, область расчёта, какие
метки сети складываются в какие структуры, лицензия и авторы.
"""

import argparse
import json
import os
import sys

import numpy as np


def build(nnunet_dir: str):
    import torch
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    configuration = os.path.basename(os.path.normpath(nnunet_dir)).split("__")[-1]
    plans = PlansManager(os.path.join(nnunet_dir, "plans.json"))
    config = plans.get_configuration(configuration)
    with open(os.path.join(nnunet_dir, "dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)
    labels = plans.get_label_manager(dataset)
    net = get_network_from_plans(config.network_arch_class_name, config.network_arch_init_kwargs,
                                 config.network_arch_init_kwargs_req_import, 1, labels.num_segmentation_heads,
                                 allow_init=True, deep_supervision=False)
    checkpoint = torch.load(os.path.join(nnunet_dir, "fold_0", "checkpoint_final.pth"), map_location="cpu",
                            weights_only=False)
    net.load_state_dict(checkpoint["network_weights"])
    net.eval()
    if list(plans.transpose_forward) != [0, 1, 2]:
        raise ValueError(f"модель переставляет оси ({plans.transpose_forward}) — не поддерживается")

    scheme = config.normalization_schemes[0]
    if scheme == "ZScoreNormalization":
        normalization = {"scheme": "zscore"}
    elif scheme == "CTNormalization":
        p = plans.foreground_intensity_properties_per_channel["0"]
        normalization = {"scheme": "ct", "clip": [p["percentile_00_5"], p["percentile_99_5"]],
                         "mean": p["mean"], "std": p["std"]}
    else:
        raise ValueError(f"нормировка {scheme} не поддерживается")
    names = [name for name, _ in sorted(dataset["labels"].items(), key=lambda kv: kv[1])]
    return net, {
        "labels": names,
        "spacing": [float(s) for s in config.spacing],  # мм по осям массива (z, y, x)
        "patch": [int(p) for p in config.patch_size],
        "normalization": normalization,
        "source": {"configuration": configuration, "trainer": checkpoint.get("trainer_name"),
                   "channel": dataset.get("channel_names", {}).get("0"), "cases": dataset.get("numTraining")},
    }


def export(nnunet_dir: str, out_dir: str, spec: dict, opset: int = 17, check: bool = True) -> dict:
    import torch

    net, info = build(nnunet_dir)
    unknown = sorted({lab for labs in spec["outputs"].values() for lab in labs} - set(info["labels"]))
    if unknown:
        raise ValueError(f"в сети нет меток: {', '.join(unknown)}")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    dummy = torch.zeros(1, 1, *info["patch"])
    torch.onnx.export(net, dummy, onnx_path, opset_version=opset, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {2: "z", 3: "y", 4: "x"}, "logits": {2: "z", 3: "y", 4: "x"}},
                      do_constant_folding=True, dynamo=False)

    if check:
        import onnxruntime as ort

        x = torch.from_numpy(np.random.default_rng(0).normal(size=(1, 1, *info["patch"])).astype(np.float32))
        with torch.no_grad():
            ref = net(x).numpy()
        got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
        diff = float(np.abs(got - ref).max())
        same = float((got.argmax(1) == ref.argmax(1)).mean())
        info["check"] = {"max_abs_diff": diff, "argmax_agreement": same}
        if same < 0.999:
            raise RuntimeError(f"ONNX отвечает иначе, чем PyTorch: совпадение меток {same:.4f}")

    model = {
        "name": spec["name"],
        "title": spec.get("title", spec["name"]),
        "onnx": "model.onnx",
        "orientation": spec.get("orientation", "RAS"),
        "region": spec.get("region", {"around": "whole"}),
        "overlap": spec.get("overlap", 0.5),
        "outputs": spec["outputs"],
        "license": spec.get("license"),
        "attribution": spec.get("attribution"),
        **info,
    }
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description="nnU-Net v2 → ONNX для Custom Case Designer")
    parser.add_argument("nnunet_dir", help="папка конфигурации nnU-Net (plans.json, dataset.json, fold_0/)")
    parser.add_argument("out_dir", help="куда положить model.onnx и model.json")
    parser.add_argument("--spec", required=True, help="описание выходов модели (tools/models/*.json)")
    parser.add_argument("--no-check", action="store_true", help="не сверять ONNX с PyTorch")
    args = parser.parse_args(argv)
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    model = export(args.nnunet_dir, args.out_dir, spec, check=not args.no_check)
    print(f"{model['name']}: {len(model['labels'])} меток, окно {model['patch']}, шаг {model['spacing']} мм, "
          f"нормировка {model['normalization']['scheme']}, проверка {model.get('check')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
