"""Выгрузка «Для exocad с артикулятором»: копия проекта exocad, где модели уже стоят в нашем артикуляторе.

Контракт (docs/handoff/archive/exocad-articulator-export.md):

* **Вход** — ``ArticulatorCase``: сканы челюстей и прикуса и кости в координатах кейса (мм КТ; нижняя челюсть
  со своей костью — в прикусе монтажа, как в окне «Артикулятор»), монтаж (``mo.Anatomy``: система монтажа,
  мыщелки, резцовая точка), настройки суставов (``kin.Settings``), пути мыщелков по КТ для вставок ССП
  (``guidance.condylar_path(...)["path"]``) и движения (имя, положения нижней челюсти N×4×4 в координатах
  кейса, кадров в секунду).
* **Выход** — новая папка проекта рядом с исходной (исходная не меняется): копия ``.dentalProject`` с новым
  именем и флагами (тип антагониста — наш артикулятор, если его тип добавлен в DentalDB, иначе SAM 2P;
  скан-маркер движения челюсти), сканы, прикусы, маркер, файл движений и кости (bone_upper.stl, bone_lower.stl) — **в координатах
  артикулятора** (x вправо, y вперёд, z вверх, середина шарнирной оси (30, −80, 60)): так exocad ставит модели
  проекта «в артикуляторе» по координатам файлов (docs/exocad.md, «Лицевая дуга»). В подпапке
  ``KStomCaseDesigner`` — мыщелки для проверки, сводка ``articulator.json`` и папка общего артикулятора для
  ``library\\articulator`` (``exocad_facebow.write_library_articulator``: одна на все кейсы, ставится один раз).
* **Модульный артикулятор** (порядок работ владельца, п. 0.5): значения пациента — параметры зубов копии
  проекта (``project_parameters``: углы ССП, Беннетта, боковой сдвиг, ретрузия, резцовый столик; exocad берёт
  их при открытии — кейс 4, 13.10), межмыщелковое — там же при ``ICD_IN_PROJECT``. Вставка ССП — из набора
  общего артикулятора: по пути мыщелка подбирается ближайшая (``exocad_facebow.fit_insert``) — в сводке и
  заметке; в проекте — ССП прямой вставки (её exocad берёт по умолчанию), пока выбор вставки из файла не
  проверен (опыт 13).
* Есть эстетическая система (``ArticulatorCase.aesthetic``: плоскости «эстетическая» и «эстетическая через
  мыщелки») — в копию пишется набор изображений Smile Design (``<проект>.treatmentImages`` и фон-снимок,
  exocad_views): в exocad появляется кнопка «Вид Smile Design» — точный эстетический вид спереди.
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
from . import exocad_views as ev
from . import kinematics as kin
from . import motion as mo
from .fusion import BITE
from .register import apply

# Параметры артикулятора в проекте exocad (Tooth/Parameters/Parameter): пределы и шаг — как у exocad, когда он
# сам сохраняет их в проект (кейс 4, 13.10). Сторонам — Left/Right.
PROJECT_PARAMETERS = {"TiltCondylarGuide": (-20.0, 60.0, 0.1), "BennettAngle": (0.0, 30.0, 0.1),
                      "ImmediateSideshift": (0.0, 1.5, 0.1), "Retrusion": (0.0, 2.0, 0.1),
                      "TiltFrontplate": (0.0, 89.0, 1.0), "IntercondylarDistance": (80.0, 130.0, 0.1)}
# Межмыщелковое в проекте: в exocad владельца регулируется и выгружается значением в проекте (входящие 13.10),
# проверка значения из проекта — копия K4i. У exocad без этой доработки незнакомый тип параметра может не дать
# открыть проект, поэтому по умолчанию не пишется (значение — в сводке и заметке).
ICD_IN_PROJECT = False

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
    # Эстетическая система (строки «вправо, вперёд, вверх» в координатах кейса, JawCase.aesthetic_axes): есть —
    # в проект пишется набор изображений Smile Design, в exocad появляется «Вид Smile Design» (exocad_views).
    aesthetic: np.ndarray | None = None
    # Фото пациента («Эстетика», часть 1): [{"kind": "smile" | "retracted", "image": RGB, "camera": aesthetic.Camera}]
    # в координатах кейса — в набор Smile Design настоящие снимки по камере совмещения вместо снимка-сетки.
    photos: list = field(default_factory=list)
    # Корни зубов из КТ под коронками сканов и оси зубов (roots.py): {"upper" | "lower": JawScan} в координатах
    # кейса (нижние — в прикусе выгрузки, как кости); roots_info — по зубам и заметки для сводки.
    roots: dict = field(default_factory=dict)
    tooth_axes: dict = field(default_factory=dict)
    roots_info: dict = field(default_factory=dict)
    # Толщина десны над костью и корнями (gingiva.py): {"upper" | "lower": JawScan с цветами} — сканы в положении
    # выгрузки, цвета по толщине; пишутся в подпапку цветным PLY (в exocad — «Load mesh», как карта отклонений).
    gingiva: dict = field(default_factory=dict)


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


def project_parameters(values: dict, icd: float | None = None) -> tuple[list, list[str]]:
    """Значения артикулятора → параметры зубов проекта exocad: [(тип, значение, мин, макс, шаг)] и заметки.
    values — как у articulator_values (+ Retrusion, TiltFrontplate); icd — межмыщелковое, мм (None — не писать).
    Значение вне пределов exocad — к ближайшему пределу, с заметкой."""
    items = dict(values)
    if icd is not None:
        items["IntercondylarDistance"] = icd
    out, notes = [], []
    for kind, value in items.items():
        base = kind[:-4] if kind.endswith("Left") else kind[:-5] if kind.endswith("Right") else kind
        if base not in PROJECT_PARAMETERS:
            continue
        low, high, step = PROJECT_PARAMETERS[base]
        v = float(value)
        if not low <= v <= high:
            notes.append(f"{kind} = {v:g} вне пределов exocad ({low:g}…{high:g}) — в проекте {min(max(v, low), high):g}")
            v = min(max(v, low), high)
        out.append((kind, round(v, 2), low, high, step))
    return out, notes


def _param_xml(kind, value, low, high, step, indent: str, nl: str) -> str:
    inner = indent + "  "
    rows = [f"<Type>{kind}</Type>", f"<Value>{value:g}</Value>", f"<MinValue>{low:g}</MinValue>",
            f"<MaxValue>{high:g}</MaxValue>", f"<StepSize>{step:g}</StepSize>"]
    return f"<Parameter>{nl}" + "".join(f"{inner}{r}{nl}" for r in rows) + f"{indent}</Parameter>"


def set_tooth_parameters(text: str, params: list) -> tuple[str, int]:
    """Записать параметры (project_parameters) в каждый Tooth/Parameters проекта: такой же тип заменяется,
    нового — добавляется в конец; пустой <Parameters /> раскрывается. Возвращает текст и число зубов."""
    nl = "\r\n" if "\r\n" in text else "\n"
    teeth = 0

    def fill(block: str, indent: str) -> str:
        block = block.rstrip(" \t\r\n")
        for kind, value, low, high, step in params:
            node = _param_xml(kind, value, low, high, step, indent, nl)
            pattern = re.compile(rf"<Parameter>\s*<Type>{kind}</Type>.*?</Parameter>", re.S)
            if pattern.search(block):
                block = pattern.sub(lambda _m: node, block, count=1)
            else:
                block = block + nl + indent + node
        return block

    def tooth(m: re.Match) -> str:
        nonlocal teeth
        body = m.group(0)
        p = re.search(r"([ \t]*)<Parameters>(.*?)([ \t]*)</Parameters>", body, re.S)
        e = re.search(r"([ \t]*)<Parameters\s*/>", body)
        if p:
            outer = p.group(1)
            first = re.search(r"\n([ \t]*)<Parameter>", p.group(2))
            indent = first.group(1) if first else outer + "  "
            new = f"{outer}<Parameters>{fill(p.group(2), indent)}{nl}{outer}</Parameters>"
            body = body[:p.start()] + new + body[p.end():]
        elif e:
            outer = e.group(1)
            new = f"{outer}<Parameters>{fill('', outer + '  ')}{nl}{outer}</Parameters>"
            body = body[:e.start()] + new + body[e.end():]
        else:
            return body
        teeth += 1
        return body

    text = re.sub(r"<Tooth>.*?</Tooth>", tooth, text, flags=re.S)
    return text, teeth


def _set_tag(text: str, tag: str, value: str) -> tuple[str, bool]:
    pattern = re.compile(rf"<{tag}>[^<]*</{tag}>")
    if len(pattern.findall(text)) != 1:
        return text, False
    return pattern.sub(f"<{tag}>{value}</{tag}>", text), True


def copy_project(project_dir: str, folder: str, stem: str, antagonist: str, params: list | None = None) -> list[str]:
    """Копия .dentalProject в новую папку: новое имя, тип антагониста, скан-маркер движения и параметры
    артикулятора у зубов (project_parameters). Возвращает заметки."""
    raw = open(_project_file(project_dir), "rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    notes = []
    for tag, value in (("AntagonistType", antagonist), ("MovementMarkerScan", "true")):
        text, ok = _set_tag(text, tag, value)
        if not ok:
            notes.append(f"в .dentalProject нет единственного <{tag}> — задайте это в DentalDB вручную")
    if params:
        text, teeth = set_tooth_parameters(text, params)
        if not teeth:
            notes.append("в .dentalProject нет зубов с параметрами — значения артикулятора выставьте в exocad вручную")
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
    icd = round(float(np.linalg.norm(fb.condyles["right"] - fb.condyles["left"])), 1)
    values = {**articulator_values(a, case.settings), "Retrusion": 0.0, "TiltFrontplate": 40}
    params, param_notes = project_parameters(values, icd if ICD_IN_PROJECT else None)
    notes = copy_project(project_dir, folder, stem, antagonist, params) + [type_note] + fb.notes + param_notes
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
    if case.bones or case.roots:
        project = exocad_project.find(folder)
        files, bone_notes = exocad_project.write_bone_frameworks(
            project, {jaw: bone.moved(to_art) for jaw, bone in case.bones.items()},
            {jaw: r.moved(to_art) for jaw, r in case.roots.items()},
            {jaw: ax.moved(to_art) for jaw, ax in case.tooth_axes.items()})
        notes += bone_notes
        for role, prefix in (("кости", "bone_"), ("корни", "roots_"), ("оси зубов", "tooth_axes_")):
            names = [f for f in files if f.startswith(prefix) and f.endswith(".stl")]
            if names:
                written[role] = names
        if "корни" in written:
            notes.append(f"корни зубов из КТ ({len(case.roots_info.get('teeth', []))} зубов) и оси зубов — тем же "
                         "списком каркасов, что кости: двигаются с челюстями; коронки — со сканов")
    notes += list(case.roots_info.get("notes", []))

    sub = os.path.join(folder, SUBFOLDER)
    os.makedirs(sub, exist_ok=True)
    if case.gingiva:
        names = []
        for jaw, gum in case.gingiva.items():
            name = f"{stem}-{jaw}jaw-gingiva.ply"
            gum.moved(to_art).export(os.path.join(sub, name))
            names.append(f"{SUBFOLDER}/{name}")
        written["толщина десны"] = names
        notes.append(f"карта толщины десны над костью и корнями — цветные PLY в {SUBFOLDER} (в exocad — «Load mesh»): "
                     "красный — тоньше 1 мм, оранжевый — 1–2, зелёный — 2–3, синий — 3–6 мм")
    spheres = [trimesh.creation.icosphere(subdivisions=2, radius=2.5).apply_translation(p)
               for p in fb.condyles.values()]
    trimesh.util.concatenate(spheres).apply_transform(to_art).export(os.path.join(sub, "condyles.stl"))
    incisal_art = apply(to_art, np.asarray(a.points["incisal"], float)[None])[0]
    art_dir = os.path.join(sub, "articulator", art_name)
    art_files = ef.write_library_articulator(art_dir)
    app = exocad_app or ef.find_exocad()
    installed = ef.library_version(os.path.join(app, "library", "articulator", art_name)) if app else None
    if installed != ef.LIBRARY_VERSION:
        notes.append(f"общий артикулятор «{art_name}» в библиотеке exocad "
                     + ("не найден" if installed is None else f"устарел (версия {installed})")
                     + f" — скопируйте папку {SUBFOLDER}\\articulator\\{art_name} в library\\articulator один раз "
                       "(одна на все кейсы)")
    if not ICD_IN_PROJECT:
        notes.append(f"межмыщелковое пациента {icd:g} мм — в проект не записано (ждёт проверки в exocad), "
                     f"в общем артикуляторе {ef.LIBRARY_ICD_MM:g} мм: выставьте его в окне артикулятора")
    inserts = None
    if case.condylar_paths:
        inserts = {side: ef.fit_insert(case.condylar_paths[side]) for side in SIDES}
        text = ", ".join(f"{'справа' if s == 'right' else 'слева'} «{inserts[s]['insert']}» "
                         f"(ССП {inserts[s]['tilt_deg']:g}°, отклонение {inserts[s]['rms_mm']:g} мм)" for s in SIDES)
        notes.append(f"вставки ССП по пути мыщелка: {text}; в проекте — ССП для прямой вставки (exocad берёт её "
                     "по умолчанию), для точного пути выберите эти вставки и углы в окне артикулятора")
    else:
        notes.append("путей мыщелков по КТ нет — вставки ССП в артикуляторе стандартные")
    if case.photos:
        names = ev.write_photo_images(folder, stem, [{**p, "anchor": np.asarray(a.points["incisal"], float)}
                                                     for p in case.photos], to_art)
        written["эстетический вид"] = names
        notes.append("фото пациента — в наборе Smile Design по камере совмещения: «Вид Smile Design» в exocad "
                     "открывает фото с улыбкой, наложенное на модели")
    elif case.aesthetic is not None:
        axes_art = (to_art[:3, :3] @ np.asarray(case.aesthetic, float).T).T  # оси — строками, в мм артикулятора
        names = ev.write_treatment_images(folder, stem, axes_art, incisal_art)
        written["эстетический вид"] = names
        notes.append("эстетический вид спереди — кнопка «Вид Smile Design» в exocad (набор изображений проекта)")
    if case.mounting not in ("ct", "aesthetic"):
        notes.append("монтаж не по мыщелкам КТ — шарнир артикулятора стоит не на мыщелках пациента")
    summary = {"files": written, "antagonist": antagonist, "articulator_folder": art_name, "notes": notes,
               "icd_mm": icd, "articulator": values, "off_axis_mm": fb.off_axis_mm,
               "project_parameters": {kind: value for kind, value, *_r in params}, "inserts": inserts,
               "library_version": ef.LIBRARY_VERSION, "library_installed": installed,
               "plane_height_mm": round(float(incisal_art[2]), 2),
               "condyles_articulator": {s: apply(to_art, fb.condyles[s][None])[0].round(3).tolist() for s in SIDES},
               "case_to_articulator": to_art.round(6).tolist(), "movements": len(case.movements),
               "articulator_files": art_files, "roots": case.roots_info.get("teeth", [])}
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
        own = item["scan"]
        if getattr(own, "colors", None) is not None:  # цвет уже в скане (Scan.load) — файл перечитывать не нужно
            return JawScan(apply(T, own.vertices), own.faces, own.colors)
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
                info = guidance.condylar_path(y, z)
            except ValueError:
                paths = None
                break
            if not guidance.plausible_sagittal(info["sagittal_deg"]):  # профиль не с сустава — вставки стандартные
                paths = None
                break
            paths[side] = info["path"]
    recs = [m.recording for m in jaw.movements.values()]
    if recs and not any(m.recorded for m in jaw.movements.values()) \
            and not any(ef.movement_type(r.name) == "opening" for r in recs):
        # Открывание — поворот вокруг настоящей оси мыщелков: у пациента она бывает наклонена к горизонтали монтажа,
        # а шарнир артикулятора exocad всегда горизонтален, так что этот наклон передаёт только файл движений.
        recs.insert(0, kin.opening(jaw.anatomy, jaw.settings))
    movements = []
    for rec in recs:
        fps = (len(rec.transforms) - 1) / rec.duration if getattr(rec, "timed", False) and rec.duration else 60.0
        movements.append((rec.name, np.asarray(rec.transforms, float), float(fps)))
    case = ArticulatorCase(upper, lower, jaw.anatomy, jaw.settings.copy(), bites, bones, paths, movements,
                           getattr(jaw, "mounting", None), getattr(jaw, "aesthetic_axes", None))
    # корни зубов из КТ: разделены по сканам в соотношении КТ, нижние — в прикус выгрузки, как кость НЧ
    from . import roots as rt

    teeth = {k: m for k, m in session.structures.items() if k.startswith(rt.TOOTH_PREFIX)}
    if teeth:
        found, root_notes = rt.split_teeth(teeth, {
            "upper": apply(np.asarray(up["reg"].transform, float), up["scan"].vertices),
            "lower": apply(np.asarray(lo["reg"].transform, float), lo["scan"].vertices)})
        for jaw, T in (("upper", np.eye(4)), ("lower", lower_T)):
            mesh, axes = rt.jaw_mesh(found, jaw), rt.axes_mesh(found, jaw)
            if mesh is not None:
                case.roots[jaw] = JawScan(apply(T, mesh[0]), mesh[1])
            if axes is not None:
                case.tooth_axes[jaw] = JawScan(apply(T, axes[0]), axes[1])
        case.roots_info = {"teeth": rt.summary(found), "notes": root_notes}
    # толщина десны: карта — у сессии (кэш); сетка — скан челюсти из сессии в положении выгрузки
    gum_map = getattr(session, "gum_map", None)
    if gum_map is not None:
        for jaw, item, T in (("upper", up, np.asarray(up["reg"].transform, float)),
                             ("lower", lo, lower_T @ np.asarray(lo["reg"].transform, float))):
            sid = next((k for k, it in session.scans.items() if it is item), None)
            gum = None if sid is None else gum_map(sid)
            if gum is not None:
                case.gingiva[jaw] = JawScan(apply(T, item["scan"].vertices), item["scan"].faces, gum.colors())
    return case
