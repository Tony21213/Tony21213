"""Командная строка Custom Case Designer.

    python -m casedesigner register КТ --scan upper.stl --scan lower.stl --models модели -o результат
    python -m casedesigner segment КТ --models модели -o результат
"""

import argparse
import os
import sys
import time

import numpy as np
import trimesh

from .fusion import JAWS, CaseCT, Scan, export_case
from .learning import AlignmentMemory
from .segment import Segmenter
from .volume import load_volume


def _parse_assignments(items, what):
    out = {}
    for item in items or ():
        name, sep, value = item.rpartition("=")
        if not sep:
            raise ValueError(f"{what}: ожидается СКАН=ЗНАЧЕНИЕ, получено {item!r}")
        out[name] = value
    return out


def load_pairs(path: str):
    """Пары точек: строки «x y z  X Y Z» — точка на скане и та же точка в КТ (мм)."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] != 6 or len(rows) < 3:
        raise ValueError(f"{path}: нужно минимум 3 строки по 6 чисел")
    return rows[:, :3], rows[:, 3:]


def _progress(model, done, total):
    print(f"\r  {model}: окно {done}/{total}", end="" if done < total else "\n", flush=True)


def _segment(vol, args) -> dict:
    segmenter = Segmenter(args.models, device=args.device, only=args.only)
    print("Сегментация: " + ", ".join(m.title for m in segmenter.models))
    result = segmenter.run(vol, smooth=args.smooth, progress=_progress)
    print(f"Найдено структур: {len(result.meshes)}")
    return result.meshes


def cmd_segment(args):
    started = time.perf_counter()
    meshes = _segment(load_volume(args.ct), args)
    for name, mesh in meshes.items():
        path = os.path.join(args.out, *name.split("/")) + ".stl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).export(path)
    print(f"Готово за {time.perf_counter() - started:.0f} с: {args.out} (координаты пациента DICOM, мм)")


def cmd_register(args):
    started = time.perf_counter()
    jaws = _parse_assignments(args.jaw, "--jaw")
    pairs = _parse_assignments(args.pairs, "--pairs")
    for jaw in jaws.values():
        if jaw not in JAWS:
            raise ValueError(f"--jaw: челюсть должна быть upper или lower, а не {jaw!r}")

    print("Читаю КТ…")
    vol = load_volume(args.ct)
    memory = AlignmentMemory(args.memory) if args.memory else None
    prior = memory.prior(vol.device) if memory else (0.0, 0.0)
    if prior[1]:
        print(f"Аппарат {vol.device or 'не указан'}: выученный сдвиг границы эмали {prior[0]:+.3f} мм")
    ct = CaseCT(vol, *prior)
    registrations = []
    for path in args.scan:
        scan = Scan.load(path)
        key = next((k for k in (path, scan.name) if k in jaws or k in pairs), None)
        reg = ct.register(scan, jaw=jaws.get(key), pairs=load_pairs(pairs[key]) if key in pairs else None)
        s = reg.stats
        print(f"{scan.name}: {reg.jaw} челюсть, на коронках {100 * s['matched_fraction']:.0f}% точек, "
              f"отклонение в среднем {s.get('mean_mm', float('nan')):.3f} мм, "
              f"90% точек ближе {s.get('p90_mm', float('nan')):.3f} мм")
        for warning in reg.warnings:
            print(f"  ВНИМАНИЕ: {warning}")
        if memory and args.accept:
            memory.record(vol.device, reg, reg)
        registrations.append(reg)
    meshes = {}
    if args.models:
        meshes.update(_segment(vol, args))
    if args.ct_surfaces:
        meshes.update(ct.surfaces())
    report = export_case(args.out, registrations, meshes, bite=args.bite, frame=args.frame, ct=ct)
    for note in report["notes"]:
        print(note)
    if report["ct_bite_vs_scans"]:
        b = report["ct_bite_vs_scans"]
        print(f"Прикус на КТ отличается от сканов: нижняя челюсть смещена в среднем на "
              f"{b['lower_jaw_on_ct_vs_scans_mean_mm']:.2f} мм, повёрнута на {b['rotation_deg']:.1f}°")
    print(f"Готово за {time.perf_counter() - started:.0f} с: {args.out}")


def _segment_options(parser, required):
    parser.add_argument("--models", required=required,
                        help="папка моделей сегментации (подпапки с model.onnx и model.json)")
    parser.add_argument("--only", nargs="+", metavar="МОДЕЛЬ", help="запустить только эти модели (по имени)")
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto", help="на чём считать (auto)")
    parser.add_argument("--smooth", type=int, default=10, help="итераций сглаживания поверхностей (10)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="casedesigner", description="Custom Case Designer")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="совместить сканы челюстей с КТ по зубам и экспортировать STL")
    reg.add_argument("ct", help="КТ: папка DICOM, .dcm, .zip, .nii.gz, .mha, .nrrd")
    reg.add_argument("--scan", action="append", required=True, help="скан челюсти (STL/PLY/OBJ), можно несколько")
    reg.add_argument("--jaw", action="append", metavar="СКАН=upper|lower",
                     help="какая это челюсть (по умолчанию определяется сама)")
    reg.add_argument("--pairs", action="append", metavar="СКАН=ФАЙЛ",
                     help="пары точек для начального положения, если автоматика не справилась")
    reg.add_argument("--bite", choices=("scan", "ct"), default="scan",
                     help="прикус: scan — сканов, структуры нижней челюсти из КТ переезжают к нижнему скану "
                          "(по умолчанию); ct — статическое наложение на КТ")
    reg.add_argument("--frame", choices=("exocad", "dicom"), default="exocad",
                     help="система координат: exocad — сканера, в них сканы открывает exocad (по умолчанию); "
                          "dicom — пациента из DICOM (только с --bite ct)")
    reg.add_argument("--ct-surfaces", action="store_true", help="также выгрузить зубы и кость из КТ (по порогам)")
    reg.add_argument("-o", "--out", required=True, help="папка результата")
    reg.add_argument("--memory", help="файл памяти совмещений: выученные поправки для аппаратов КТ")
    reg.add_argument("--accept", action="store_true",
                     help="результат проверен и принят — запомнить его в --memory для обучения")
    _segment_options(reg, required=False)
    reg.set_defaults(func=cmd_register)

    seg = sub.add_parser("segment", help="сегментировать КТ и выгрузить структуры в STL")
    seg.add_argument("ct", help="КТ: папка DICOM, .dcm, .zip, .nii.gz, .mha, .nrrd")
    seg.add_argument("-o", "--out", required=True, help="папка результата")
    _segment_options(seg, required=True)
    seg.set_defaults(func=cmd_segment)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
