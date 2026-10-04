"""Командная строка Custom Case Designer.

    python -m casedesigner register КТ --scan upper.stl --scan lower.stl --models модели -o результат
    python -m casedesigner segment КТ --models модели -o результат
    python -m casedesigner motion выгрузка_P-ART.zip -o отчёт
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

from . import motion
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


def cmd_motion(args):
    case = motion.read_case(args.case)
    print(f"Кейс: {case.source}, файлов: {len(case.files)}")
    for f in case.files:
        role = f" — {f['role']}" if f.get("role") else ""
        system = f" [{f['system']}]" if f.get("system") else ""
        note = f": {f['note']}" if f.get("note") else ""
        print(f"  [{f['kind']}] {f['path']}{role}{system}{note}")
    if args.inspect:  # структура без значений, имён и дат — ею можно делиться
        for rel, _size, read in motion._entries(args.case):
            ext = os.path.splitext(rel)[1].lower()
            if ext not in motion.MOTION_EXT + motion.HDF5_EXT + motion.MATRIX_EXT + motion.PROJECT_EXT:
                continue
            data = read()
            if ext in motion.HDF5_EXT:
                try:
                    lines = motion.describe_hdf5(data)
                except ImportError:
                    lines = ["для HDF5 установите h5py"]
            elif motion._is_xml(data):
                lines = motion.describe_xml(data)
            elif ext in (".csv", ".txt", ".tsv", ".asc", ".dat"):
                lines = motion.describe_table(motion._decode(data))
            else:
                continue
            print(f"\nСтруктура {ext}:")
            print("\n".join("  " + line for line in lines))
        return
    for note in case.notes:
        print(f"ВНИМАНИЕ: {note}")
    report = {}
    if case.recordings:
        lower = None
        if args.lower:
            lower = np.asarray(trimesh.load_mesh(args.lower, process=False).vertices, float)
        anatomy = motion.estimate_anatomy(case, lower=lower, incisal=args.incisal, icd=args.icd)
        report = motion.analyze_case(case, anatomy)
        print(f"\nСистема координат: {anatomy.source}")
        if anatomy.hinge_rms_mm is not None:
            print(f"Шарнирная ось: в начале открывания смещается в среднем на {anatomy.hinge_rms_mm:.2f} мм")
        _print_motion(report)
    if case.tracings:
        axes = motion.parse_axes(args.axes) if args.axes else None
        traced = motion.analyze_tracings(case, axes)
        print("\nПути отдельных точек (углы — к горизонтали системы записи):")
        _print_motion(traced)
        report = {**report, "tracings": traced} if report else {"case": case.source, "systems": case.systems,
                                                                 "tracings": traced}
    if args.out and report:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "motion.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if not case.recordings:
            print(f"Отчёт: {args.out} (motion.json)")
            return
        motion.write_paths_csv(os.path.join(args.out, "paths.csv"), case, anatomy)
        try:
            motion.plot(case, anatomy, os.path.join(args.out, "motion.png"), report)
            print(f"Отчёт: {args.out} (motion.json, paths.csv, motion.png)")
        except ImportError:
            print(f"Отчёт: {args.out} (motion.json, paths.csv); для картинки установите matplotlib")


def _print_motion(report: dict):
    for r in report["recordings"]:
        inc = r.get("incisal")
        if inc:
            extra = f", ведение {inc['guidance_deg']}°" if inc.get("guidance_deg") is not None else ""
            print(f"  {r['name']}: {r['kind_name']}, {r['frames']} кадров, резцовая точка до {inc['max_mm']} мм{extra}")
        else:
            paths = ", ".join(f"{'правый' if s == 'right' else 'левый'} {c['path_mm']} мм"
                              for s, c in r["condyles"].items())
            print(f"  {r['name']}: {r['kind_name']}, {r['frames']} кадров, мыщелки: {paths or '—'}")
    s = report["articulator"]
    if any(v is not None for v in s.values()):
        print("Для артикулятора: " + ", ".join(f"{k} = {v}" for k, v in s.items() if v is not None))
    for note in report["notes"]:
        print(f"Примечание: {note}")


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

    mot = sub.add_parser("motion", help="записи движений нижней челюсти (P-ART и др.): разобрать и проанализировать")
    mot.add_argument("case", help="выгрузка: папка, архив .zip или файл движения (.xml, .jawMotion, .csv, .txt, .h5)")
    mot.add_argument("--inspect", action="store_true",
                     help="только состав и структура файлов (XML, таблицы, HDF5) — без значений, имён и дат "
                          "(можно прислать для разбора формата)")
    mot.add_argument("--lower", help="модель нижней челюсти, если её нет в выгрузке или имя не распознано")
    mot.add_argument("--incisal", nargs=3, type=float, metavar=("X", "Y", "Z"),
                     help="резцовая точка в координатах моделей (по умолчанию ищется на модели)")
    mot.add_argument("--icd", type=float, default=motion.ICD_MM, help="межмыщелковое расстояние, мм (100)")
    mot.add_argument("--axes", help="для путей мыщелков: какие оси файла смотрят вправо пациента, вперёд и вверх, "
                                    "например --axes=-y,x,z (по умолчанию угадываются по путям)")
    mot.add_argument("-o", "--out", help="папка отчёта: motion.json, paths.csv, motion.png")
    mot.set_defaults(func=cmd_motion)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
