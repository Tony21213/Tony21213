"""Командная строка Custom Case Designer.

    python -m casedesigner register КТ --scan upper.stl --scan lower.stl -o результат
"""

import argparse
import sys
import time

import numpy as np

from .fusion import JAWS, CaseCT, Scan, export_case
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


def cmd_register(args):
    started = time.perf_counter()
    jaws = _parse_assignments(args.jaw, "--jaw")
    pairs = _parse_assignments(args.pairs, "--pairs")
    for jaw in jaws.values():
        if jaw not in JAWS:
            raise ValueError(f"--jaw: челюсть должна быть upper или lower, а не {jaw!r}")

    print("Читаю КТ…")
    ct = CaseCT(load_volume(args.ct))
    registrations = []
    for path in args.scan:
        scan = Scan.load(path)
        key = next((k for k in (path, scan.name) if k in jaws or k in pairs), None)
        reg = ct.register(scan, jaw=jaws.get(key), pairs=load_pairs(pairs[key]) if key in pairs else None)
        s = reg.stats
        print(f"{scan.name}: {reg.jaw} челюсть, на коронках {100 * s['matched_fraction']:.0f}% точек, "
              f"отклонение в среднем {s.get('mean_mm', float('nan')):.3f} мм, "
              f"90% точек ближе {s.get('p90_mm', float('nan')):.3f} мм")
        registrations.append(reg)
    surfaces = ct.surfaces() if args.ct_surfaces else None
    export_case(args.out, registrations, surfaces, frame=args.frame)
    print(f"Готово за {time.perf_counter() - started:.0f} с: {args.out}")


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
    reg.add_argument("--frame", choices=("ct", "scan"), default="ct",
                     help="система координат результата: ct — DICOM, scan — первого скана")
    reg.add_argument("--ct-surfaces", action="store_true", help="также выгрузить зубы и кость из КТ (по порогам)")
    reg.add_argument("-o", "--out", required=True, help="папка результата")
    reg.set_defaults(func=cmd_register)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
