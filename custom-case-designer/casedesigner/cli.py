"""Командная строка Custom Case Designer.

    python -m casedesigner register КТ --scan upper.stl --scan lower.stl --models модели -o результат
    python -m casedesigner segment КТ --models модели -o результат
    python -m casedesigner motion выгрузка_P-ART.zip -o отчёт
    python -m casedesigner landmarks КТ --models модели/landmarks -o точки.json
    python -m casedesigner bite верх.stl низ.stl -o низ_в_прикусе.stl
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

from . import motion
from .fusion import BITE, JAWS, CaseCT, Scan, export_case, is_bite_name, place_bites
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


def cmd_landmarks(args):
    from . import auto_landmarks as al
    from . import landmarks as lmk

    started = time.perf_counter()
    models = al.Models(args.models, device=args.device)
    if not models.available():
        raise ValueError(f"{args.models}: нет моделей ориентиров (tools/prepare_landmarks.py или кнопка в программе)")
    vol = load_volume(args.ct)
    found = al.find(vol, models, keys=args.only, workers=args.workers,
                    progress=lambda f, m: print(f"\r  {m}", end="" if f < 1 else "\n", flush=True))
    out = {}
    for key, f in found.items():
        name = lmk.BY_KEY[key].name if key in lmk.BY_KEY else key
        if f.point is None:
            print(f"  {name}: не найдена — {f.note}")
            continue
        out[key] = {"name": name, "point": f.point.round(2).tolist(), "note": f.note}
        print(f"  {name}: {', '.join(f'{v:.1f}' for v in f.point)} мм{' — ' + f.note if f.note else ''}")
    planes = {k: lmk.PLANES[k]["name"] for k in lmk.PLANES if lmk.plane({k: np.array(v["point"]) for k, v in out.items()}, k)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"landmarks": out, "coordinates": "мм пациента DICOM (LPS)", "planes": planes}, fh,
                      ensure_ascii=False, indent=1)
    print(f"Готово за {time.perf_counter() - started:.0f} с; плоскости: {', '.join(planes.values()) or 'нет'}")


def cmd_bite(args):
    from . import bite

    upper, lower = trimesh.load(args.upper, force="mesh"), trimesh.load(args.lower, force="mesh")
    condyles = None
    if args.condyles:
        c = np.array(args.condyles, float).reshape(2, 3)
        condyles = (c[0], c[1])
    res = bite.correct_bite(upper.vertices, upper.faces, lower.vertices, condyles=condyles)
    r = res.report
    s = r["incisal_shift_mm"]
    print(f"Резцы: {r['incisal_mm']:.2f} мм (вправо {s['right']:+.2f}, вперёд {s['forward']:+.2f}, "
          f"вверх {s['up']:+.2f}); моляры {r['molar_right_mm']:.2f} / {r['molar_left_mm']:.2f} мм; "
          f"поворот {r['turn_deg']:.2f}°")
    for when, name in (("before", "до"), ("after", "после")):
        c = r[when]
        print(f"Контакты {name}: {c['points']} точек (справа {c['sectors']['right']}, спереди {c['sectors']['front']}, "
              f"слева {c['sectors']['left']}), проникновение {c['penetration_mm']:.2f} мм")
    for note in r["notes"]:
        print(f"  {note}")
    if args.out:
        fixed = lower.copy()
        fixed.apply_transform(res.transform)
        fixed.export(args.out)
        print(f"Нижний скан в исправленном прикусе: {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"transform": res.transform.tolist(), "report": r}, f, ensure_ascii=False, indent=1)


def cmd_register(args):
    started = time.perf_counter()
    jaws = _parse_assignments(args.jaw, "--jaw")
    pairs = _parse_assignments(args.pairs, "--pairs")
    for jaw in jaws.values():
        if jaw not in JAWS + (BITE,):
            raise ValueError(f"--jaw: челюсть должна быть upper, lower или bite (скан прикуса), а не {jaw!r}")

    print("Читаю КТ…")
    vol = load_volume(args.ct)
    memory = AlignmentMemory(args.memory) if args.memory else None
    prior = memory.prior(vol.device) if memory else (0.0, 0.0)
    if prior[1]:
        print(f"Аппарат {vol.device or 'не указан'}: выученный сдвиг границы эмали {prior[0]:+.3f} мм")
    ct = CaseCT(vol, *prior)
    meshes = {}
    if args.models:  # сначала сегментация: зубы из неё — опора совмещения (CaseCT.use_teeth)
        meshes.update(_segment(vol, args))
        ct.use_teeth({jaw: meshes.get(f"{jaw}_teeth") for jaw in JAWS})
    else:
        print("Без --models зубы в КТ ищутся только по плотности: на снимке с большим полем скан может сесть "
              "со сдвигом вдоль дуги.")
    registrations, bites = [], []
    for path in args.scan:
        scan = Scan.load(path)
        key = next((k for k in (path, scan.name) if k in jaws or k in pairs), None)
        if jaws.get(key) == BITE or (key not in jaws and is_bite_name(scan.name)):
            bites.append(scan)  # скан прикуса — на сканы челюстей, после них
            continue
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
    if bites:
        by_jaw = {r.jaw: r for r in reversed(registrations)}
        placed = place_bites(bites, by_jaw.get("upper"), by_jaw.get("lower"), ct)
        for k, scan in enumerate(bites):
            reg = placed.regs[k]
            if reg is None:
                print(f"{scan.name}: скан прикуса не поставлен — {placed.failed[k]}")
                continue
            print(f"{scan.name}: скан прикуса, на сканах челюстей {100 * reg.stats.get('matched_fraction', 0):.0f}% точек")
            for warning in reg.warnings:
                print(f"  {warning}")
            registrations.append(reg)
        if placed.lower is not None:
            by_jaw["lower"].bite = placed.lower
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
    case = motion.read_case(args.case, split=args.split, smoothing=args.smooth)
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
        from . import articulator_fit

        _fitted, fitted = articulator_fit.fit(case, anatomy)
        report["articulator_fit"] = fitted
        print("Подбор артикулятора по всему начальному участку пути: " + ", ".join(
            f"{k} = {v}" for k, v in fitted["settings"].items() if fitted["sources"].get(k) == "подбор по записи"))
        for note in fitted["notes"]:
            print(f"  {note}")
        upper, lower_mesh = case.mesh("upper"), case.mesh("lower")
        if upper is not None and lower_mesh is not None:
            from . import kinematics

            try:
                check = kinematics.check_against_scans(case.recordings, upper[0], upper[1], lower_mesh[0],
                                                       ref=motion.reference_pose(case), anatomy=anatomy)
            except ValueError as e:
                check = {"verdict": "no_bite", "notes": [f"запись со сканами не сверить: {e}"]}
            report["scans_check"] = check
            print("\nЗапись и сканы:")
            for note in check["notes"]:
                print(f"  {note}")
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


def cmd_facebow(args):
    """Лицевая дуга для exocad: шарнир артикулятора — на мыщелках, гипсовка — по горизонтали монтажа."""
    import trimesh

    from . import exocad_facebow as ef
    from . import exocad_webview as ew

    def load(path):
        return None if not path else trimesh.load_mesh(path, process=False)

    upper, lower = load(args.upper), load(args.lower)
    anatomy, settings, eminence, notes = ew.mount(upper.vertices, lower.vertices, load(args.mandible), load(args.skull))
    folder = args.exocad or ef.find_exocad()
    if not folder:
        raise ValueError("укажите --exocad: папка DentalCADApp (нужна вилка Zebris SD из её библиотеки)")
    values = {}
    for side, name in (("right", "Right"), ("left", "Left")):
        values[f"TiltCondylarGuide{name}"] = round(settings.get("sagittal", side), 1)
        values[f"BennettAngle{name}"] = round(settings.get("bennett", side), 1)
        values[f"ImmediateSideshift{name}"] = round(settings.get("side_shift", side), 2)
    p = anatomy.points
    res = ef.export(args.out, anatomy.frame, p["condyle_right"], p["condyle_left"], p["incisal"],
                    ef.load_register(folder), settings=values)
    print(f"Монтаж: {anatomy.source}")
    print(f"Межмыщелковое расстояние {res['icd_mm']} мм; мыщелки от оси артикулятора: "
          f"справа {res['off_axis_mm']['right']} мм, слева {res['off_axis_mm']['left']} мм")
    print("Суставы (в артикуляторе — по умолчанию): " + ", ".join(f"{k} {v}" for k, v in values.items())
          + (f"; источник: {', '.join(sorted(set(settings.sources.values())))}" if settings.sources else
             "; источник: средние значения"))
    for note in notes + res["notes"]:
        print(f"  ! {note}")
    print(f"Файлы в {args.out}: " + ", ".join(f for f in res["files"] if "/" not in f)
          + f", папка артикулятора «{ef.ARTICULATOR_NAME}»")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="casedesigner", description="Custom Case Designer")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="совместить сканы челюстей с КТ по зубам и экспортировать STL")
    reg.add_argument("ct", help="КТ: папка DICOM, .dcm, .zip, .nii.gz, .mha, .nrrd")
    reg.add_argument("--scan", action="append", required=True, help="скан челюсти (STL/PLY/OBJ), можно несколько")
    reg.add_argument("--jaw", action="append", metavar="СКАН=upper|lower|bite",
                     help="какая это челюсть (по умолчанию определяется сама)")
    reg.add_argument("--pairs", action="append", metavar="СКАН=ФАЙЛ",
                     help="пары точек для начального положения, если автоматика не справилась")
    reg.add_argument("--bite", choices=("scan", "ct"), default="scan",
                     help="прикус: scan — сканов, структуры нижней челюсти из КТ переезжают к нижнему скану "
                          "(по умолчанию); ct — статическое наложение на КТ")
    reg.add_argument("--frame", choices=("exocad", "dicom"), default="exocad",
                     help="система координат: exocad — сканера, в них сканы открывает exocad (по умолчанию); "
                          "dicom — пациента из DICOM (с --bite scan верхняя челюсть — как на КТ, нижняя — "
                          "в прикусе сканов)")
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

    lm = sub.add_parser("landmarks", help="цефалометрические точки на КТ автоматически (ALI-CBCT)")
    lm.add_argument("ct", help="КТ: папка DICOM, архив, .nii.gz, .mha, .nrrd")
    lm.add_argument("--models", required=True, help="папка моделей ориентиров (<точка>/<масштаб>.onnx)")
    lm.add_argument("--only", nargs="+", help="только эти точки (Po_R Po_L Or_R Or_L Co_R Co_L N S ANS PNS Ba IP)")
    lm.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    lm.add_argument("--workers", type=int, default=2, help="точек одновременно (2)")
    lm.add_argument("-o", "--out", help="файл JSON с точками")
    lm.set_defaults(func=cmd_landmarks)

    bt = sub.add_parser("bite", help="исправить прикус сканов, как в Bite-Finder: нижний скан — в контакт с верхним")
    bt.add_argument("upper", help="верхний скан (STL, PLY, OBJ)")
    bt.add_argument("lower", help="нижний скан в прикусе сканера")
    bt.add_argument("--condyles", nargs=6, type=float, metavar=("XR", "YR", "ZR", "XL", "YL", "ZL"),
                    help="мыщелки правый и левый в координатах сканов (по КТ); без них — треугольник Бонвилля")
    bt.add_argument("-o", "--out", help="нижний скан в исправленном прикусе")
    bt.add_argument("--json", help="матрица и отчёт")
    bt.set_defaults(func=cmd_bite)

    mot = sub.add_parser("motion", help="записи движений нижней челюсти (P-ART и др.): разобрать и проанализировать")
    mot.add_argument("case", help="выгрузка: папка, архив .zip или файл движения (.xml, .jawMotion, .csv, .txt, .h5)")
    mot.add_argument("--inspect", action="store_true",
                     help="только состав и структура файлов (XML, таблицы, HDF5) — без значений, имён и дат "
                          "(можно прислать для разбора формата)")
    mot.add_argument("--lower", help="модель нижней челюсти, если её нет в выгрузке или имя не распознано")
    mot.add_argument("--incisal", nargs=3, type=float, metavar=("X", "Y", "Z"),
                     help="резцовая точка в координатах моделей (по умолчанию ищется на модели)")
    mot.add_argument("--icd", type=float, default=motion.ICD_MM, help="межмыщелковое расстояние, мм (100)")
    mot.add_argument("--split", action="store_true", help="сплошные записи — разделить на отдельные движения")
    mot.add_argument("--smooth", action="store_true", help="сгладить шум трекера (окно 0.1 с)")
    mot.add_argument("--axes", help="для путей мыщелков: какие оси файла смотрят вправо пациента, вперёд и вверх, "
                                    "например --axes=-y,x,z (по умолчанию угадываются по путям)")
    mot.add_argument("-o", "--out", help="папка отчёта: motion.json, paths.csv, motion.png")
    mot.set_defaults(func=cmd_motion)

    fb = sub.add_parser("facebow", help="лицевая дуга для exocad: модели в артикулятор — шарнир на мыщелках пациента")
    fb.add_argument("--upper", required=True, help="скан верхней челюсти (координаты, как в проекте exocad)")
    fb.add_argument("--lower", required=True, help="скан нижней челюсти в прикусе, в тех же координатах")
    fb.add_argument("--mandible", help="кость нижней челюсти из КТ в тех же координатах: мыщелки по ней")
    fb.add_argument("--skull", help="череп из КТ в тех же координатах: наклон суставных дорожек по бугоркам")
    fb.add_argument("--exocad", help="папка DentalCADApp exocad (по умолчанию ищется на дисках)")
    fb.add_argument("-o", "--out", required=True, help="папка результата")
    fb.set_defaults(func=cmd_facebow)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
