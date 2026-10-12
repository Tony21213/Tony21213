"""Выгрузка «Для exocad с артикулятором»: копия проекта exocad, где модели уже стоят в нашем артикуляторе.

Контракт (docs/handoff.md, «Передача: выгрузка для exocad с артикулятором»):

* **Вход** — ``ArticulatorCase``: сканы челюстей и прикуса и кости в координатах кейса (мм КТ; нижняя челюсть
  со своей костью — в прикусе монтажа, как в окне «Артикулятор»), монтаж (``mo.Anatomy``: система монтажа,
  мыщелки, резцовая точка), настройки суставов (``kin.Settings``), пути мыщелков по КТ для вставок ССП
  (``guidance.condylar_path(...)["path"]``) и движения (имя, положения нижней челюсти N×4×4 в координатах
  кейса, кадров в секунду).
* **Выход** — новая папка проекта рядом с исходной (исходная не меняется): копия ``.dentalProject`` с новым
  именем и флагами (тип антагониста — наш артикулятор, если его тип добавлен в DentalDB, иначе SAM 2P;
  скан-маркер движения челюсти), сканы, прикусы, маркер, файл движений и кости-каркасы — **в координатах
  артикулятора** (x вправо, y вперёд, z вверх, середина шарнирной оси (30, −80, 60)): так exocad ставит модели
  проекта «в артикуляторе» по координатам файлов (docs/exocad.md, «Лицевая дуга»). В подпапке
  ``KStomCaseDesigner`` — мыщелки для проверки, сводка ``articulator.json`` и папка артикулятора для
  ``library\\articulator`` (межмыщелковое и углы пациента, вставки ССП, плоскость гипсовки).
* ``.scanInfo`` и ``.matrix4`` исходного проекта не копируются: сцена = файл. Моделировки исходного проекта
  (``*_cad``) тоже — они стоят в старых координатах.
* Имена пациентов не выводятся: сводка называет файлы по ролям.
"""

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import trimesh

from . import exocad_facebow as ef
from . import exocad_project
from . import kinematics as kin
from . import motion as mo
from .fusion import BITE
from .register import apply

SUBFOLDER = "KStomCaseDesigner"
COPY_SUFFIX = "-KStom"
CUSTOM_TYPE, SAM_TYPE = "ArticulatorCustomCaseDesigner", "ArticulatorSendling"
SIDES = ("right", "left")


@dataclass
class JawScan:
    """Сетка для выгрузки в координатах кейса; colors — RGBA (uint8) на вершину, как в PLY сканера."""

    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None = None

    def moved(self, T: np.ndarray) -> trimesh.Trimesh:
        kw = {} if self.colors is None else {"vertex_colors": np.asarray(self.colors)}
        return trimesh.Trimesh(apply(T, np.asarray(self.vertices, float)), np.asarray(self.faces), process=False, **kw)


@dataclass
class ArticulatorCase:
    upper: JawScan
    lower: JawScan
    anatomy: mo.Anatomy
    settings: kin.Settings = field(default_factory=kin.Settings)
    bites: list = field(default_factory=list)  # JawScan: сканы прикуса (TotalJaw)
    bones: dict = field(default_factory=dict)  # "upper": череп и ВЧ, "lower": НЧ — JawScan
    condylar_paths: dict | None = None  # {"right", "left"}: N×2 (вперёд, вверх; система монтажа)
    movements: list = field(default_factory=list)  # (имя, N×4×4 в координатах кейса, кадров в секунду)
    mounting: str | None = "ct"  # JawCase.mounting: ct / aesthetic — мыщелки по КТ; average / motion — нет


def articulator_values(anatomy: mo.Anatomy, settings: kin.Settings) -> dict:
    """Настройки суставов → параметры артикулятора exocad, от горизонтали монтажа (горизонтали артикулятора).

    У настроек может быть своя горизонталь (``Settings.frame``: углы записи — от горизонтали записи, ССП по КТ —
    от Франкфурта): направления путей переводятся в систему монтажа (``kinematics._to_mount``) и углы
    измеряются заново."""
    V = kin._to_mount(anatomy, settings)
    out = {}
    for side in SIDES:
        name = side.capitalize()
        d = V @ kin._sagittal(settings.get("sagittal", side))
        out[f"TiltCondylarGuide{name}"] = round(float(np.degrees(np.arctan2(-d[2], d[1]))), 1)
        # балансирующий мыщелок этой стороны: вперёд 1, внутрь tg(Беннетта), вниз tg(ССП) — в системе углов
        inward = -1.0 if side == "right" else 1.0
        b = V @ np.array([inward * np.tan(np.radians(settings.get("bennett", side))), 1.0,
                          -np.tan(np.radians(settings.get("sagittal", side)))])
        out[f"BennettAngle{name}"] = round(float(np.degrees(np.arctan2(abs(b[0]), b[1]))), 1)
        out[f"ImmediateSideshift{name}"] = round(float(settings.get("side_shift", side)), 2)
    return out


def antagonist_type(exocad_app: str | None = None) -> tuple[str, str, str]:
    """Тип антагониста для копии проекта, пояснение и имя папки артикулятора.

    Наш тип — если он добавлен в конфигурацию exocad (tools/exocad_articulator_type.py); папка — та, на которую
    он там указывает (после установки под прежним названием это «Custom Case Designer»): exocad выбирает её
    сразу, папку выгрузки остаётся скопировать в library\\articulator под тем же именем. Иначе — SAM 2P (перевод
    из регистратора тот же) и папка ef.ARTICULATOR_NAME, которую выбирают в диалоге вручную."""
    app = exocad_app or ef.find_exocad()
    mapping = os.path.join(app, "config", "articulatormappings.xml") if app else None
    if mapping and os.path.isfile(mapping):
        with open(mapping, encoding="utf-8", errors="replace") as f:
            text = f.read()
        m = re.search(rf"<Type>{CUSTOM_TYPE}</Type>\s*<FolderName>([^<]+)</FolderName>", text)
        if m or CUSTOM_TYPE in text:
            folder = m.group(1).strip() if m else ef.ARTICULATOR_NAME
            return CUSTOM_TYPE, f"тип антагониста — наш артикулятор (папка «{folder}»)", folder
    return SAM_TYPE, (f"тип антагониста — SAM 2P (наш тип в DentalDB не добавлен): в диалоге «Вирт. артикулятор» "
                      f"выберите «{ef.ARTICULATOR_NAME}»"), ef.ARTICULATOR_NAME


def _project_file(project_dir: str) -> str:
    project = exocad_project.find(project_dir)
    if project is None:
        raise FileNotFoundError("в папке нет проекта exocad (.dentalProject)")
    return os.path.join(project_dir, project.stem + ".dentalProject")


def target(project_dir: str, out_dir: str | None = None) -> tuple[str, str]:
    """Папка и имя копии проекта: по умолчанию рядом с исходной, с суффиксом COPY_SUFFIX."""
    stem = os.path.splitext(os.path.basename(_project_file(project_dir)))[0]
    folder = out_dir or os.path.normpath(project_dir) + COPY_SUFFIX
    return folder, os.path.basename(os.path.normpath(folder)) if out_dir else stem + COPY_SUFFIX


def _set_tag(text: str, tag: str, value: str) -> tuple[str, bool]:
    pattern = re.compile(rf"<{tag}>[^<]*</{tag}>")
    if len(pattern.findall(text)) != 1:
        return text, False
    return pattern.sub(f"<{tag}>{value}</{tag}>", text), True


def copy_project(project_dir: str, folder: str, stem: str, antagonist: str) -> list[str]:
    """Копия .dentalProject в новую папку: новое имя, тип антагониста и скан-маркер движения. Возвращает заметки."""
    raw = open(_project_file(project_dir), "rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    notes = []
    for tag, value in (("AntagonistType", antagonist), ("MovementMarkerScan", "true")):
        text, ok = _set_tag(text, tag, value)
        if not ok:
            notes.append(f"в .dentalProject нет единственного <{tag}> — задайте это в DentalDB вручную")
    with open(os.path.join(folder, stem + ".dentalProject"), "wb") as f:
        f.write((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return notes


def export(case: ArticulatorCase, project_dir: str, out_dir: str | None = None, register: ef.Register | None = None,
           exocad_app: str | None = None) -> dict:
    """Копия проекта exocad с моделями в нашем артикуляторе (контракт — в описании модуля). Возвращает сводку
    (только JSON-типы, без имён пациентов)."""
    a = case.anatomy
    register = register or ef.load_register(exocad_app or ef.find_exocad())
    fb = ef.facebow(a.frame, a.points["condyle_right"], a.points["condyle_left"], a.points["incisal"], register)
    to_art = fb.case_to_articulator
    folder, stem = target(project_dir, out_dir)
    if os.path.exists(folder) and os.listdir(folder):
        raise FileExistsError("папка для копии проекта уже существует и не пуста — выберите другую")
    os.makedirs(folder, exist_ok=True)
    antagonist, type_note, art_name = antagonist_type(exocad_app)
    notes = copy_project(project_dir, folder, stem, antagonist) + [type_note] + fb.notes
    written = {"проект": f"{stem}.dentalProject"}

    def path(suffix):
        return os.path.join(folder, stem + suffix)

    case.upper.moved(to_art).export(path("-upperjaw.ply"))
    case.lower.moved(to_art).export(path("-lowerjaw.ply"))
    written.update({"верхний скан": f"{stem}-upperjaw.ply", "нижний скан": f"{stem}-lowerjaw.ply"})
    for i, bite in enumerate(case.bites):
        bite.moved(to_art).export(path(f"-TotalJaw{i}.ply"))
        written[f"скан прикуса {i + 1}"] = f"{stem}-TotalJaw{i}.ply"
    upper_art = trimesh.Trimesh(apply(to_art, np.asarray(case.upper.vertices, float)), np.asarray(case.upper.faces),
                                process=False)
    marker = trimesh.util.concatenate([upper_art, ef.marker_mesh(register, fb, None).apply_transform(to_art)])
    marker.export(path("-movementmarker.stl"))
    written["скан маркера"] = f"{stem}-movementmarker.stl"
    with open(path(".jawmotion"), "wb") as f:
        f.write(ef.jawmotion_xml(fb, f"{ef.ARTICULATOR_NAME}: шарнир на мыщелках пациента, гипсовка по горизонтали "
                                     "монтажа", case.movements or None))
    written["файл движений"] = f"{stem}.jawmotion"
    if case.bones:
        project = exocad_project.find(folder)
        files, bone_notes = exocad_project.write_bone_frameworks(
            project, {jaw: bone.moved(to_art) for jaw, bone in case.bones.items()})
        notes += bone_notes
        written["кости-каркасы"] = len([f for f in files if f.endswith(".stl")])

    sub = os.path.join(folder, SUBFOLDER)
    os.makedirs(sub, exist_ok=True)
    spheres = [trimesh.creation.icosphere(subdivisions=2, radius=2.5).apply_translation(p)
               for p in fb.condyles.values()]
    trimesh.util.concatenate(spheres).apply_transform(to_art).export(os.path.join(sub, "condyles.stl"))
    icd = round(float(np.linalg.norm(fb.condyles["right"] - fb.condyles["left"])), 1)
    values = {**articulator_values(a, case.settings), "TiltFrontplate": 40}
    incisal_art = apply(to_art, np.asarray(a.points["incisal"], float)[None])[0]
    art_dir = os.path.join(sub, "articulator", art_name)
    art_files = ef.write_articulator(art_dir, icd, values, inserts=case.condylar_paths,
                                     plane_height=round(float(incisal_art[2]), 2))
    if not case.condylar_paths:
        notes.append("путей мыщелков по КТ нет — вставки ССП в артикуляторе стандартные")
    if case.mounting not in ("ct", "aesthetic"):
        notes.append("монтаж не по мыщелкам КТ — шарнир артикулятора стоит не на мыщелках пациента")
    summary = {"files": written, "antagonist": antagonist, "articulator_folder": art_name, "notes": notes,
               "icd_mm": icd, "articulator": values, "off_axis_mm": fb.off_axis_mm,
               "plane_height_mm": round(float(incisal_art[2]), 2),
               "condyles_articulator": {s: apply(to_art, fb.condyles[s][None])[0].round(3).tolist() for s in SIDES},
               "case_to_articulator": to_art.round(6).tolist(), "movements": len(case.movements),
               "articulator_files": art_files}
    with open(os.path.join(sub, "articulator.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    summary["folder"] = folder
    return summary


def case_from_session(session) -> ArticulatorCase:
    """Собрать вход выгрузки из сессии приложения после монтажа (``Session.articulation_mount``): сканы — из
    исходных файлов с цветом (если файл доступен) в координатах кейса, нижняя челюсть и её кость — в прикусе
    монтажа (``bite_motion``, посадка прикуса ``JawCase.lower_fix``), пути мыщелков — по черепу КТ у точек кейса.

    Каркас ведущего: ведомый подключает это к ``Session`` (метод выгрузки) и к шагу «Экспорт»."""
    from . import condyles as cond
    from . import guidance

    jaw = session.jaw
    if jaw.anatomy is None:
        raise ValueError("сначала смонтируйте модели в шаге «Артикулятор»")
    up, lo = session._jaw_items()
    if up is None or lo is None or up.get("reg") is None or lo.get("reg") is None:
        raise ValueError("нужны совмещённые сканы обеих челюстей")
    bite = np.eye(4) if session.bite_motion is None else np.asarray(session.bite_motion, float)
    lower_T = jaw.lower_fix @ bite

    def scan(item, T):
        mesh = None
        path = item.get("path")
        if path and os.path.isfile(path):
            try:
                loaded = trimesh.load_mesh(path, process=False)
                if not isinstance(loaded, trimesh.Scene):
                    mesh = loaded
            except (OSError, ValueError):
                mesh = None
        if mesh is not None:
            colors = getattr(mesh.visual, "vertex_colors", None) if mesh.visual.kind == "vertex" else None
            return JawScan(apply(T, np.asarray(mesh.vertices)), np.asarray(mesh.faces), colors)
        return JawScan(apply(T, item["scan"].vertices), item["scan"].faces)

    upper = scan(up, np.asarray(up["reg"].transform, float))
    lower = scan(lo, lower_T @ np.asarray(lo["reg"].transform, float))
    bites = [scan(it, np.asarray(it["transform"], float)) for it in session.scans.values()
             if it.get("role") == BITE and it.get("transform") is not None]
    bones = {}
    upper_parts = [session.structures[k] for k in ("skull", "maxilla") if k in session.structures]
    if upper_parts:
        merged = trimesh.util.concatenate(upper_parts)
        bones["upper"] = JawScan(np.asarray(merged.vertices), np.asarray(merged.faces))
    if "mandible" in session.structures:
        m = session.structures["mandible"]
        bones["lower"] = JawScan(apply(lower_T, np.asarray(m.vertices)), np.asarray(m.faces))
    paths = None
    skull = session.structures.get("skull")
    points = cond.resolve(session.landmarks, session.suggested, getattr(session, "landmark_sources", {}))
    if skull is not None and all(p.point is not None for p in points.values()):
        paths = {}
        for side, p in points.items():
            try:
                y, z = guidance.eminence_profile(np.asarray(skull.vertices), np.asarray(skull.faces),
                                                 jaw.anatomy.frame, p.point)
                paths[side] = guidance.condylar_path(y, z)["path"]
            except ValueError:
                paths = None
                break
    movements = []
    for m in jaw.movements.values():
        rec = m.recording
        fps = (len(rec.transforms) - 1) / rec.duration if getattr(rec, "timed", False) and rec.duration else 60.0
        movements.append((rec.name, np.asarray(rec.transforms, float), float(fps)))
    return ArticulatorCase(upper, lower, jaw.anatomy, jaw.settings.copy(), bites, bones, paths, movements,
                           getattr(jaw, "mounting", None))
