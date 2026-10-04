"""Реальные движения нижней челюсти: чтение выгрузок аксиографа и анализ.

Источник — кейсы из P-ART (цифровой аксиограф Prosystom / SDI Matrix): для
exocad он выгружает модели челюстей и XML с записанными движениями —
открывание, протрузия, латеротрузии, жевание. Описания формата нет, поэтому
чтение сделано «разведкой»: в XML ищутся серии одинаковых элементов, в каждом
из которых есть положение тела (матрица 4×4 или 3×4; смещение с кватернионом
или углами), и каждая серия — отдельная запись. Под образец выгрузки
добавится точное чтение, а пока `describe_xml` печатает структуру файла без
значений, имён и дат, чтобы ею можно было поделиться.

Другие системы записи (Zebris JMA, Modjaw, Medit, Gamma CADIAX и т.д.) тоже
читаются разведкой — их форматы открыто не описаны, известно только общее:

* XML для exocad: положения челюсти по кадрам или координаты маркеров по
  кадрам с исходным (окклюзионным) расположением. Маркеры (≥ 3 на тело)
  переводятся в положения челюсти по МНК (Кабш); если в кадре маркеры головы
  и челюсти, тела разделяются по постоянству расстояний между маркерами;
* таблицы CSV/ASCII с заголовками: время, смещение + кватернион или углы,
  ячейки матрицы или координаты точек (резцовая точка, мыщелки, маркеры);
* открытый трекер JawTrackingSystem: CSV «tx ty tz qw qx qy qz» и HDF5;
* меньше трёх точек (например, пути шарнирных точек мыщелков у
  кондилографа) — пути точек (`Tracing`): по ним считаются углы суставного
  пути, но двигать модели по ним нельзя.

Положения считаются положениями нижней челюсти относительно верхней в
координатах выгруженных моделей — так exocad двигает нижнюю модель. Если в
кадре два тела (верхняя и нижняя), берётся нижняя относительно верхней. Это
допущение проверяется на образце.

Анализ — в анатомической системе: x — вправо пациента, y — вперёд, z — вверх,
начало в середине шарнирной оси. Она берётся:

* из КТ (`anatomy_from_ct`): мыщелки и плоскость по ориентирам, после того как
  модели кейса совмещены с КТ по зубам; углы — от выбранной плоскости
  (Франкфурт, HIP, Кемпер);
* без КТ (`estimate_anatomy`): шарнирная ось — по началу открывания, резцовая
  точка — по модели нижней челюсти; горизонталь — плоскость «шарнирная ось —
  резцовая точка» (треугольник Бонвиля), поэтому сагиттальные углы меньше,
  чем от Франкфурта, на угол между этими плоскостями.

По каждой записи — тип движения, пути резцовой точки и мыщелков,
сагиттальный угол суставного пути, угол Беннетта и боковой сдвиг рабочего
мыщелка: те числа, которыми настраивают артикулятор.
"""

import csv
import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .landmarks import BY_KEY, reference_frame
from .register import apply, kabsch, rigid

MESH_EXT = (".stl", ".ply", ".obj", ".off")
MOTION_EXT = (".xml", ".jawmotion", ".csv", ".txt", ".tsv", ".asc", ".dat")
HDF5_EXT = (".h5", ".hdf5")
MATRIX_EXT = (".matrix4",)
PROJECT_EXT = (".dentalproject",)
# Форматы, описания которых нет в открытом доступе: прочитать нельзя, нужна выгрузка в открытом виде.
CLOSED_EXT = {".jmtxd": "SICAT JMT+"}

MIN_FRAMES = 5  # серия короче — не запись движения
MAX_BODIES = 4  # тел в одном кадре больше не бывает; больше — это уже серия кадров
HINGE_MAX_DEG = 10.0  # начало открывания до этого угла — чистое вращение на шарнирной оси
ICD_MM = 100.0  # межмыщелковое расстояние, если мыщелки не заданы (треугольник Бонвиля)
CHORD_MM = 5.0  # углы суставного пути — по хорде на этом участке пути мыщелка
ISS_FIT_MM = 1.5  # угол Беннетта — по участку после немедленного бокового сдвига
GUIDE_CHORD_MM = 3.0  # углы ведения зубами — по первым 3 мм пути резцовой точки
TOP_SHARE = 0.03  # резцовая точка ищется среди самых «верхних» 3% вершин модели
INCISAL_TILT_DEG = 30.0  # «верх» для поиска резцов наклонён вперёд: так впереди резцы, а не моляры
BONWILL_MM = 100.0  # сторона треугольника Бонвилля: мыщелок — мыщелок и мыщелок — резцовая точка
BALKWILL_DEG = 25.0  # угол Балквилла: между треугольником Бонвилля и окклюзионной плоскостью
CUSP_SHARE = 0.05  # окклюзионная плоскость — по самым высоким 5% точек нижних зубов
ARCH_BAND_MM = 6.0  # форма дуги — по коронкам не глубже 6 мм от окклюзионной плоскости
RIGID_STD_MM = 0.5  # маркеры одного тела: расстояние между ними за запись меняется меньше
MIN_SPREAD_MM = 1.0  # маркеры тела не на одной прямой: второй размер облака больше
METERS_SPREAD = 0.5  # маркеры, разнесённые меньше чем на 0.5 «единицы», — это метры
AXES_RATIO = 1.5  # оси путей мыщелков угадываются, только если вперёд они уходят заметно больше, чем вниз

KINDS = {"opening": "открывание", "protrusion": "протрузия", "laterotrusion_right": "латеротрузия вправо",
         "laterotrusion_left": "латеротрузия влево", "chewing": "жевание", "other": "другое"}
POINTS = {"incisal": "резцовая точка", "condyle_right": "правый мыщелок", "condyle_left": "левый мыщелок"}

# Роль модели по имени файла (английские, немецкие, русские обозначения и выгрузки для exocad:
# «…-UpperJaw.stl», «…-LowerJaw.stl», «…-TotalJaw0.stl»).
ROLE_KEYS = {
    "lower": ("lower", "mandib", "unterkiefer", "uk", "нижн", "нч"),
    "upper": ("upper", "maxill", "oberkiefer", "ok", "верхн", "вч"),
    "bite": ("total", "bite", "buccal", "прикус"),
}
# Из описания проекта exocad (.dentalProject) берутся только эти поля — без пациента и клиники.
PROJECT_FIELDS = ("AntagonistType", "MovementMarkerScan", "DentalDBProductName")
# Значения этих атрибутов describe_xml показывает (единицы, тип движения); прочие тексты скрыты.
SAFE_ATTRS = {"unit", "units", "version", "type", "kind", "format", "movement", "side", "jaw", "order"}
PRIVATE = ("patient", "birth", "person", "practice", "doctor", "name")

TIME_KEYS = ("time", "t", "timestamp", "ts", "sec", "seconds", "ms", "timems", "time_ms", "millis")
RATE_KEYS = ("fps", "rate", "framerate", "frequency", "hz", "samplerate", "samplingrate")
NAME_ATTRS = ("name", "title", "type", "label", "kind", "movement", "id")
T_PREFIXES = ("", "t", "p", "pos", "position", "translation", "trans", "offset", "origin")
Q_PREFIXES = ("q", "quat", "quaternion", "rotation", "rot", "orientation", "r", "")
E_PREFIXES = ("r", "rot", "rotation", "angle", "angles", "euler", "a")
E_NAMES = (("alpha", "beta", "gamma"), ("roll", "pitch", "yaw"))
FRAME_KEYS = ("frame", "frames", "sample", "samples", "index", "nr", "no", "n", "кадр", "номер")
POSE_WORDS = ("rot", "quat", "orient", "euler", "angle", "matrix")  # в кадре с ними — положение, а не точки
# Исходное расположение маркеров (окклюзия) в XML — по этим словам в имени элемента или его родителей.
REF_KEYS = ("reference", "referenz", "ref", "static", "statisch", "initial", "intercusp", "icp", "centric",
            "zentrik", "occlusion", "okklusion", "habitual", "baseline", "окклюз", "исходн", "опорн")
# Тела по именам маркеров: верхняя челюсть / голова и нижняя челюсть (в т.ч. обозначения JTS: HP — голова, MP — рот).
JAW_KEYS = {"upper": ROLE_KEYS["upper"] + ("head", "kopf", "cran", "skull", "hp", "cra", "голов", "череп"),
            "lower": ROLE_KEYS["lower"] + ("mp", "mta", "mouth", "мандиб")}
POINT_WORDS = {"incisal": ("incis", "inzis", "ip", "резц"),
               "condyle": ("condyl", "kondyl", "cond", "kond", "hinge", "scharnier", "achs", "axis", "мыщел",
                           "шарнир"),
               "right": ("right", "rechts", "r", "rt", "dx", "прав", "п"),
               "left": ("left", "links", "l", "lt", "sx", "лев", "л")}
# Слова движения в именах файлов и блоков: имя записи берётся только из них — имя пациента в отчёт не попадёт.
MOVE_WORDS = (("открывание", ("open", "öffn", "oeffn", "откр")), ("протрузия", ("protru", "vorschub", "протру")),
              ("ретрузия", ("retru", "ретру")), ("латеротрузия", ("latero", "латеро")),
              ("жевание", ("chew", "kau", "mastic", "жев")), ("закрывание", ("clos", "schlie", "закр")),
              ("вправо", ("right", "rechts", "прав")), ("влево", ("left", "links", "лев")))
UNIT_SCALE = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
# Системы записи — по словам в имени файла и в его начале (только для подписи в отчёте и подсказок).
SYSTEMS = (("Zebris JMA", ("zebris",)), ("Modjaw", ("modjaw",)), ("SDI Matrix / P-ART", ("prosystom", "p-art")),
           ("Gamma CADIAX", ("cadiax", "gamma dental")), ("Medit", ("medit",)), ("SICAT JMT+", ("sicat", ".jmtxd")),
           ("KaVo ARCUSdigma", ("arcusdigma", "arcus digma")), ("Planmeca 4D Jaw Motion", ("planmeca",)),
           ("Bioresearch JT-3D", ("bioresearch",)), ("OXO", ("oxo technologies",)), ("ITAKA", ("itaka",)),
           ("JawTrackingSystem (JTS)", ("jts_version", "t_model_origin_mand")), ("exocad", ("exocad",)))


@dataclass
class Recording:
    """Запись движения: положения нижней челюсти относительно верхней по кадрам."""

    name: str
    times: np.ndarray  # с; если время не записано — номера кадров (timed=False)
    transforms: np.ndarray  # (N, 4, 4), мм, координаты моделей кейса
    source: str = ""
    timed: bool = True

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0


@dataclass
class Tracing:
    """Пути отдельных точек, когда положения челюсти целиком нет (меньше трёх точек на тело).

    Так пишут кондилографы: пути шарнирных точек мыщелков. Координаты — в
    осях системы записи, мм; оси для анализа задаются явно или угадываются
    (`tracing_axes`).
    """

    name: str
    times: np.ndarray
    points: dict  # имя точки → (N, 3)
    source: str = ""
    timed: bool = True

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0


@dataclass
class MotionCase:
    """Выгруженный кейс: записи движений, модели и всё, что удалось понять из файлов."""

    source: str
    recordings: list = field(default_factory=list)
    tracings: list = field(default_factory=list)
    systems: list = field(default_factory=list)  # системы записи, узнанные по файлам
    meshes: dict = field(default_factory=dict)  # путь → (вершины, грани)
    roles: dict = field(default_factory=dict)  # upper / lower / bite → путь модели
    matrices: dict = field(default_factory=dict)  # путь .matrix4 → 4×4 (точка — столбец)
    project: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)  # прочие числа из XML: путь → значение
    files: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def mesh(self, role: str):
        path = self.roles.get(role)
        return self.meshes.get(path) if path else None


@dataclass
class Anatomy:
    """Анатомическая система и точки, по которым идёт анализ."""

    frame: np.ndarray  # 4×4: координаты кейса → x вправо, y вперёд, z вверх (мм)
    points: dict  # incisal, condyle_right, condyle_left — в координатах кейса
    source: str
    hinge_rms_mm: float | None = None  # насколько ось «плывёт» в начале открывания


# --- числа и положения в XML ---

_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_CELL = re.compile(r"^([a-z]{0,3})_?(\d)_?(\d)$")


def _numbers(text) -> list[float] | None:
    """Числа строки, если в ней нет ничего, кроме чисел и разделителей; десятичная запятая тоже понимается."""
    if not text or not (s := text.strip()):
        return None
    if "." not in s and re.search(r"\d,\d", s) and re.search(r"[\s;]", s):
        s = s.replace(",", ".")
    if _NUM.sub("", s).strip(" \t\r\n,;[]()"):
        return None
    return [float(v) for v in _NUM.findall(s)]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _orthonormal(M: np.ndarray) -> np.ndarray:
    U, _s, Vt = np.linalg.svd(M[:3, :3])
    out = M.copy()
    out[:3, :3] = U @ Vt
    return out


def _is_rotation(R: np.ndarray, tol: float = 2e-3) -> bool:
    return np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol


def _matrix(values) -> np.ndarray | None:
    """4×4 «точка — столбец» из 16 или 12 чисел; запись по строкам или по столбцам — та, что даёт поворот."""
    v = np.asarray(values, float)
    if len(v) == 16:
        candidates = [v.reshape(4, 4), v.reshape(4, 4).T]
    elif len(v) == 12:
        candidates = [np.vstack([v.reshape(3, 4), [0, 0, 0, 1]]),
                      rigid(v[:9].reshape(3, 3), v[9:]),
                      np.vstack([v.reshape(4, 3).T, [0, 0, 0, 1]])]
    else:
        return None
    for M in candidates:
        if np.allclose(M[3], [0, 0, 0, 1], atol=1e-6) and _is_rotation(M[:3, :3]):
            return _orthonormal(M)
    return None


def _fields(el) -> dict:
    """Числовые атрибуты и листья элемента (и атрибуты детей: <Translation x=…> → translationx)."""
    out = {}

    def put(name, text):
        n = _numbers(text)
        if n is not None and len(n) == 1:
            out.setdefault(name.lower(), n[0])

    for k, v in el.attrib.items():
        put(_local(k), v)
    for c in el:
        tag = _local(c.tag)
        if len(c) == 0:
            put(tag, c.text)
        for k, v in c.attrib.items():
            put(tag + _local(k), v)
    return out


def _triple(f: dict, prefixes, names=("x", "y", "z")):
    for p in prefixes:
        keys = [p + s for s in names]
        if all(k in f for k in keys):
            return np.array([f[k] for k in keys]), p
    return None, None


def _named_pose(f: dict) -> np.ndarray | None:
    """Положение из именованных чисел: ячейки матрицы (m00…, _00…), смещение + кватернион или углы."""
    cells, prefix = {}, set()
    for k, v in f.items():
        if m := _CELL.match(k):
            cells[(int(m[2]), int(m[3]))] = v
            prefix.add(m[1])
    if len(cells) >= 9 and len(prefix) == 1:
        base = min(min(ij) for ij in cells)
        size = max(max(ij) for ij in cells) - base + 1
        grid = np.full((size, size), np.nan)
        for (i, j), v in cells.items():
            grid[i - base, j - base] = v
        if size == 4 and not np.isnan(grid).any():
            return _matrix(grid.ravel())
        if size in (3, 4) and not np.isnan(grid[:3, :3]).any():
            t = grid[:3, 3] if size == 4 and not np.isnan(grid[:3, 3]).any() else _triple(f, T_PREFIXES[1:])[0]
            if t is not None and _is_rotation(grid[:3, :3]):
                return _orthonormal(rigid(grid[:3, :3], t))
    t, tp = _triple(f, T_PREFIXES)
    if t is None:
        return None
    for p in Q_PREFIXES:
        if p != tp and all(p + s in f for s in "wxyz"):
            w, x, y, z = (f[p + s] for s in "wxyz")
            q = np.array([x, y, z, w])
            if abs(np.linalg.norm(q) - 1) < 1e-2:
                return rigid(Rotation.from_quat(q / np.linalg.norm(q)).as_matrix(), t)
    for p in E_PREFIXES:
        if p != tp and all(p + s in f for s in "xyz"):
            return rigid(Rotation.from_euler("xyz", [f[p + s] for s in "xyz"], degrees=True).as_matrix(), t)
    for names in E_NAMES:
        if all(n in f for n in names):
            return rigid(Rotation.from_euler("xyz", [f[n] for n in names], degrees=True).as_matrix(), t)
    return None


def _pose(el, depth: int = 0) -> np.ndarray | None:
    """Одно положение в элементе: в тексте, в именованных числах или в единственном ребёнке."""
    n = _numbers(el.text)
    if n is not None and len(n) in (12, 16):
        return _matrix(n)
    M = _named_pose(_fields(el))
    if M is not None or depth >= 2:
        return M
    found = [p for c in el if (p := _pose(c, depth + 1)) is not None]
    return found[0] if len(found) == 1 else None


def _label(el) -> str:
    for a in NAME_ATTRS:
        if el.get(a):
            return el.get(a)
    return _local(el.tag)


class _Points(dict):
    """Кадр из отдельных точек: имя → xyz."""


def _xyz(el) -> np.ndarray | None:
    """Точка: три числа в тексте или x/y/z в атрибутах и листьях — без поворота."""
    n = _numbers(el.text)
    if n is not None and len(n) == 3 and not len(el):
        return np.array(n)
    f = _fields(el)
    t, _p = _triple(f, T_PREFIXES)
    return t if t is not None and _named_pose(f) is None else None


def _points(el) -> _Points | None:
    """Кадр из точек (маркеры, резцовая точка, мыщелки): 3·k чисел в тексте или дети с x/y/z."""
    n = _numbers(el.text)
    if n is not None and not len(el):
        if len(n) >= 9 and len(n) % 3 == 0 and not (len(n) == 9 and _is_rotation(np.reshape(n, (3, 3)))):
            return _Points({f"{k + 1}": np.array(n[3 * k:3 * k + 3]) for k in range(len(n) // 3)})
    if any(w in _local(c.tag).lower() for c in el for w in POSE_WORDS):
        return None
    found = [(_label(c), p) for c in el if (p := _xyz(c)) is not None]
    if not found:
        p = _xyz(el)  # кадр — сама точка: <P x=… y=… z=…/>
        return _Points({_label(el): p}) if p is not None else None
    labels = [k for k, _ in found]
    if len(set(labels)) < len(labels):  # одинаковые <Marker> без имён — по порядку
        labels = [f"{k}{i + 1}" for i, k in enumerate(labels)]
    return _Points(zip(labels, (p for _, p in found)))


def _bodies(el):
    """Кадр: одно положение — матрица; несколько тел — словарь «имя → матрица»; иначе — точки."""
    M = _pose(el)
    if M is not None:
        return M
    found = [(_label(c), p) for c in el if (p := _pose(c, 1)) is not None]
    labels = [k for k, _ in found]
    if 2 <= len(found) <= MAX_BODIES and len(set(labels)) == len(labels):
        return dict(found)
    return _points(el)


def _role(name: str) -> str | None:
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    tokens = [t for t in re.split(r"[^0-9a-zа-яё]+", stem) if t]
    for role, keys in ROLE_KEYS.items():
        if any(t == k or (len(k) >= 4 and t.startswith(k)) for t in tokens for k in keys):
            return role
    return None


def _times(parent, items) -> tuple[np.ndarray, bool]:
    fields = [_fields(c) for c in items]
    for key in TIME_KEYS:
        if all(key in f for f in fields):
            t = np.array([f[key] for f in fields])
            return (t / 1000 if "ms" in key or key == "millis" else t), True
    rate = _fields(parent)
    for key in RATE_KEYS:
        if rate.get(key, 0) > 0:
            return np.arange(len(items)) / rate[key], True
    return np.arange(len(items), dtype=float), False


def _recordings(parent, frames, source) -> list[Recording]:
    """Записи из серии кадров; если в кадре верхняя и нижняя челюсть — нижняя относительно верхней."""
    times, timed = _times(parent, [c for c, _ in frames])
    name = _label(parent)
    first = frames[0][1]
    if isinstance(first, dict):
        labels = [k for k in first if all(isinstance(b, dict) and k in b for _, b in frames)]
        roles = {_role(k): k for k in labels}
        if "upper" in roles and "lower" in roles:
            up, lo = roles["upper"], roles["lower"]
            T = np.array([np.linalg.inv(b[up]) @ b[lo] for _, b in frames])
            return [Recording(name, times, T, f"{source} ({lo} относительно {up})", timed)]
        return [Recording(f"{name} / {k}", times, np.array([b[k] for _, b in frames]), source, timed)
                for k in labels]
    return [Recording(name, times, np.array([b for _, b in frames]), source, timed)]


def _point_series(parent, frames, source) -> Tracing | None:
    """Серия кадров из точек: точки, что есть во всех кадрах (одиночные точки — с одинаковым именем)."""
    pts = [b for _, b in frames if isinstance(b, _Points)]
    if len(pts) < 0.9 * len(frames):
        return None
    common = set(pts[0]).intersection(*pts[1:])
    labels = [k for k in pts[0] if k in common]
    if not labels:
        return None  # у каждого «кадра» своя точка — это набор точек, а не запись
    keep = [(c, b) for c, b in frames if isinstance(b, _Points)]
    times, timed = _times(parent, [c for c, _ in keep])
    if len(labels) == 1 and not timed:
        return None  # ряд одиночных точек без времени — скорее линия или контур, чем движение
    return Tracing(_label(parent), times, {k: np.array([b[k] for _, b in keep], float) for k in labels},
                   source, timed)


def _tracks(root, source: str) -> tuple[list[Recording], list[Tracing], set]:
    """Серии кадров: родитель, у которого ≥ MIN_FRAMES одинаковых детей с положением или точками."""
    recs, series, used = [], [], set()
    for parent in root.iter():
        if id(parent) in used:
            continue
        groups = {}
        for c in parent:
            groups.setdefault(c.tag, []).append(c)
        for tag, items in groups.items():
            if len(items) < MIN_FRAMES:
                continue
            frames = [(c, b) for c in items if (b := _bodies(c)) is not None]
            if len(frames) < 0.9 * len(items):
                continue
            where = f"{source}: {_local(parent.tag)}/{_local(tag)}"
            if isinstance(frames[0][1], _Points):
                if (s := _point_series(parent, frames, where)) is None:
                    continue
                series.append(s)
            else:
                recs += _recordings(parent, [(c, b) for c, b in frames if not isinstance(b, _Points)], where)
            for c in items:
                used.update(id(d) for d in c.iter())
    return recs, series, used


def _reference_points(root, labels: set, used: set) -> dict | None:
    """Исходное расположение маркеров: элемент вне серий с теми же точками (имя — «reference», «ICP» и т.п.)."""
    parents = {c: p for p in root.iter() for c in p}
    others = []
    for el in root.iter():
        if id(el) in used or not (pts := _points(el)) or len(set(pts) & labels) < min(3, len(labels)):
            continue
        chain, e = [], el
        while e is not None and len(chain) < 4:
            chain.append(f"{_label(e)} {_local(e.tag)}".lower())
            e = parents.get(e)
        if any(k in " ".join(chain) for k in REF_KEYS):
            return dict(pts)
        others.append(dict(pts))
    return others[0] if len(others) == 1 else None  # без подписи — только если такой набор точек один


def _tokens(label: str) -> list[str]:
    return re.findall(r"[a-zа-яёäöüß]+", str(label).lower())


def _has(tokens, words) -> bool:
    return any(t == w or (len(w) >= 3 and t.startswith(w)) for t in tokens for w in words)


def _jaw(label: str) -> str | None:
    """Верхняя (голова) или нижняя челюсть по имени маркера: «UK1», «LowerJaw 2», «HP3»."""
    t = _tokens(label)
    up, low = _has(t, JAW_KEYS["upper"]), _has(t, JAW_KEYS["lower"])
    return "upper" if up and not low else "lower" if low and not up else None


def point_role(label: str, pair: bool = False) -> str | None:
    """Резцовая точка или мыщелок по имени точки; pair — точек две, и «R»/«L» без слова «мыщелок» — мыщелки."""
    t = _tokens(label)
    if _has(t, POINT_WORDS["incisal"]):
        return "incisal"
    side = "right" if _has(t, POINT_WORDS["right"]) else "left" if _has(t, POINT_WORDS["left"]) else None
    if side and (_has(t, POINT_WORDS["condyle"]) or pair):
        return f"condyle_{side}"
    return None


def _rigid_groups(P: np.ndarray) -> list[list[int]]:
    """Маркеры по телам: в теле расстояния между всеми маркерами за запись почти не меняются."""
    sd = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1).std(0)
    groups = []
    for i in range(P.shape[1]):
        for g in groups:
            if all(sd[i, j] < RIGID_STD_MM for j in g):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def _spread(points: np.ndarray) -> float:
    """Второй размер облака точек: у точек на одной прямой — около нуля."""
    if len(points) < 3:
        return 0.0
    return float(np.linalg.svd(points - points.mean(0), compute_uv=False)[1])


def _assign(bodies: list, labels: list, P: np.ndarray, notes: list) -> tuple:
    """Какое из тел — нижняя челюсть, какое — голова: по именам маркеров, иначе по размаху движения."""
    def jaw(g):
        votes = Counter(_jaw(labels[i]) for i in g)
        return "upper" if votes["upper"] > votes["lower"] else "lower" if votes["lower"] > votes["upper"] else None

    if not bodies:
        return None, None
    if len(bodies) == 1:
        return (None, bodies[0]) if jaw(bodies[0]) == "upper" else (bodies[0], None)
    a, b = sorted(bodies, key=len, reverse=True)[:2]
    if jaw(a) == "upper" or jaw(b) == "lower":
        return b, a
    if jaw(a) == "lower" or jaw(b) == "upper":
        return a, b
    moved = [float(np.linalg.norm(P[:, g] - P[0, g], axis=-1).mean()) for g in (a, b)]
    notes.append("маркеры головы и челюсти не подписаны — нижней челюстью считается тело, которое двигается больше")
    return (a, b) if moved[0] >= moved[1] else (b, a)


def _track_body(P: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Положения тела по маркерам от исходного расположения (Кабш) и отклонение от жёсткого движения, мм."""
    T = np.array([kabsch(ref, p) for p in P])
    moved = np.einsum("nij,kj->nki", T[:, :3, :3], ref) + T[:, None, :3, 3]
    return T, float(np.median(np.sqrt(((moved - P) ** 2).sum(-1).mean(-1))))


def _from_points(series: list, reference: dict | None = None, scale: float | None = None) -> tuple:
    """Записи по точкам: тело из ≥ 3 маркеров — положения челюсти, иначе — пути точек.

    Исходное расположение — из файла (окклюзия), иначе — первый кадр первой
    записи с теми же маркерами: так все записи кейса отсчитаны от одного
    положения. Если есть и маркеры головы, положение челюсти — относительно головы.
    """
    recs, tracings, notes = [], [], []
    if not series:
        return recs, tracings, notes
    if scale is None:
        first = np.array(list(series[0].points.values()))[:, 0]
        if len(first) >= 2 and np.ptp(first, axis=0).max() < METERS_SPREAD:
            scale = 1000.0
            notes.append("координаты точек похожи на метры — переведены в миллиметры")
    scale = scale or 1.0
    common, reference_used = {}, False
    for n, s in enumerate(series):
        labels = list(s.points)
        P = np.stack([s.points[k] for k in labels], 1) * scale
        big = [g for g in _rigid_groups(P) if len(g) >= 3]
        bodies = [g for g in big if _spread(P[0, g]) > MIN_SPREAD_MM]
        lower, upper = _assign(bodies, labels, P, notes)

        def ref_of(g):
            nonlocal reference_used
            names = tuple(labels[i] for i in g)
            if reference and all(n in reference for n in names):
                reference_used = True
                return np.array([reference[n] for n in names], float) * scale
            return common.setdefault(frozenset(names), P[0, g])

        T_up = None
        if upper is not None:
            T_up, _res = _track_body(P[:, upper], ref_of(upper))
        if lower is not None:
            T, res = _track_body(P[:, lower], ref_of(lower))
            if T_up is not None:
                T = np.linalg.inv(T_up) @ T
            body = f"маркеров {len(lower)}" + (" относительно маркеров головы" if T_up is not None else "")
            recs.append(Recording(s.name, s.times, T, f"{s.source} ({body}, отклонение от жёсткого тела "
                                                      f"{res:.2f} мм)", s.timed))
            continue
        if len(labels) >= 3 and upper is None:
            why = "лежат почти на одной прямой" if big else "не двигаются как одно тело"
            notes.append(f"запись {n + 1}: точки {why} — положения челюсти по ним нет, разобраны пути точек")
        keep = [i for i in range(len(labels)) if upper is None or i not in upper]
        if T_up is not None:  # пути точек относительно головы
            back = np.linalg.inv(T_up)
            P = np.einsum("nij,nkj->nki", back[:, :3, :3], P) + back[:, None, :3, 3]
        if keep:
            tracings.append(Tracing(s.name, s.times, {labels[i]: P[:, i] for i in keep}, s.source, s.timed))
    if recs and not reference_used:
        notes.append("исходного расположения маркеров в файле нет — положения отсчитаны от первого кадра первой записи")
    return recs, tracings, list(dict.fromkeys(notes))


def _values(root, used: set, limit: int = 300) -> dict:
    """Прочие числа файла (углы, точки, настройки) — по ним потом сверяется формат; личное пропускается."""
    out = {}

    def walk(el, path):
        if id(el) in used or len(out) >= limit:
            return
        p = f"{path}/{_local(el.tag)}" if path else _local(el.tag)
        if any(w in p.lower() for w in PRIVATE):
            return
        for k, v in el.attrib.items():
            n = _numbers(v)
            if n is not None and len(n) == 1 and not any(w in k.lower() for w in PRIVATE):
                out.setdefault(f"{p}@{_local(k)}", n[0])
        if len(el) == 0 and (n := _numbers(el.text)) is not None and len(n) == 1:
            out.setdefault(p, n[0])
        for c in el:
            walk(c, p)

    walk(root, "")
    return out


def _unit_scale(root) -> float | None:
    for el in root.iter():
        for k, v in el.attrib.items():
            if _local(k).lower() in ("unit", "units", "lengthunit"):
                v = v.strip().lower()
                if v in ("m", "meter", "metre", "meters", "metres"):
                    return 1000.0
                if v in ("cm", "centimeter", "centimetre"):
                    return 10.0
                if v in ("mm", "millimeter", "millimetre"):
                    return 1.0
    return None


def _is_xml(data: bytes) -> bool:
    head = data[:64].lstrip(b"\xef\xbb\xbf \t\r\n")
    return head.startswith(b"<") or data[:2] in (b"\xff\xfe", b"\xfe\xff")


def describe_xml(data: bytes, limit: int = 200) -> list[str]:
    """Структура XML без значений: теги, атрибуты, сколько раз повторяются, сколько чисел в тексте.

    Тексты и строковые атрибуты (имя пациента, даты, комментарии) не выводятся —
    только их наличие; показываются значения единиц и типов движения.
    """
    root = ET.fromstring(data)
    stats, order = {}, []

    def walk(el, path):
        key = path + (_local(el.tag),)
        if key not in stats:
            stats[key] = {"count": 0, "attrs": {}, "numbers": set(), "text": False}
            order.append(key)
        s = stats[key]
        s["count"] += 1
        for k, v in el.attrib.items():
            name, n = _local(k), _numbers(v)
            if n is not None:
                s["attrs"][name] = "число" if len(n) == 1 else f"{len(n)} чисел"
            elif name.lower() in SAFE_ATTRS and len(v) <= 24:
                s["attrs"][name] = repr(v)
            else:
                s["attrs"].setdefault(name, "текст")
        if (n := _numbers(el.text)) is not None:
            s["numbers"].add(len(n))
        elif el.text and el.text.strip():
            s["text"] = True
        for c in el:
            walk(c, key)

    walk(root, ())
    lines = []
    for key in order[:limit]:
        s = stats[key]
        attrs = ", ".join(f"{k}: {v}" for k, v in s["attrs"].items())
        bits = (["чисел в тексте: " + "/".join(map(str, sorted(s["numbers"])))] if s["numbers"] else []) + \
               (["текст"] if s["text"] else [])
        lines.append("  " * (len(key) - 1) + f"{key[-1]} ×{s['count']}" + (f" [{attrs}]" if attrs else "")
                     + (" — " + ", ".join(bits) if bits else ""))
    if len(order) > limit:
        lines.append(f"… ещё {len(order) - limit} видов элементов")
    return lines


def _safe_cell(text: str) -> str:
    text = text.strip().strip('"\'')
    return text if len(text) <= 24 and not any(w in text.lower() for w in PRIVATE) else "…"


def describe_table(text: str, limit: int = 30) -> list[str]:
    """Структура таблицы без значений: блоки чисел, сколько строк и столбцов, заголовки столбцов.

    Строки текста вне заголовков (пациент, дата, комментарии) не выводятся — только их число.
    """
    def column(c):  # похоже на имя столбца: время, номер кадра, ось x/y/z, кватернион, ячейка матрицы
        n = _cell(c)[1]
        return bool(n in TIME_KEYS or n in FRAME_KEYS or n.startswith(("time", "zeit", "время"))
                    or re.fullmatch(r"\w*[xyz]|[xyz]\w*|q?[wxyz]|m?\d\d|\d+", n))

    lines = []
    for k, (header, rows, texts) in enumerate(_blocks(text)[:limit]):
        if header and sum(map(column, header)) >= 0.5 * len(header):
            cols = " | ".join(_safe_cell(c) for c in header)
        else:
            cols = "заголовок не распознан — не показан" if header else "без заголовка"
        lines.append(f"блок {k + 1}: строк {len(rows)} по {rows.shape[1]} чисел; столбцы: {cols}"
                     + (f"; строк текста перед ним: {len(texts)}" if texts else ""))
    if m := _FILE_UNIT.search(text):
        lines.append(f"единица в тексте: {m[1]}")
    if m := _FILE_RATE.search(text):
        lines.append(f"частота в тексте: {m[1]}")
    return lines or ["чисел нет"]


def describe_hdf5(data: bytes) -> list[str]:
    """Структура HDF5 без значений: группы, наборы с размерами, имена атрибутов (значения — только unit и
    sample_rate)."""
    import h5py

    lines = []
    with h5py.File(io.BytesIO(data), "r") as f:
        lines.append("атрибуты файла: " + (", ".join(_safe_cell(k) for k in f.attrs) or "нет"))

        def visit(path, obj):
            name = _safe_cell(path.rsplit("/", 1)[-1])
            what = f"набор {obj.shape} {obj.dtype}" if isinstance(obj, h5py.Dataset) else "группа"
            attrs = ", ".join(f"{k}: {obj.attrs[k]!r}" if k in ("unit", "sample_rate") else _safe_cell(k)
                              for k in obj.attrs)
            lines.append("  " * path.count("/") + f"{name} — {what}" + (f" [{attrs}]" if attrs else ""))

        f.visititems(visit)
    return lines


# --- чтение кейса ---

def _add(case: MotionCase, recs: list, series: list, scale: float | None, reference: dict | None = None) -> str:
    """Записи файла — в кейс: положения челюсти (в т.ч. по маркерам) и пути точек; что получилось — строкой."""
    more, tracings, notes = _from_points(series, reference, scale)
    case.recordings += recs + more
    case.tracings += tracings
    case.notes += [n for n in notes if n not in case.notes]
    out = []
    if recs or more:
        out.append(f"записей движения: {len(recs) + len(more)}, кадров: {sum(len(r.transforms) for r in recs + more)}")
    if more:
        out.append(f"из них по маркерам: {len(more)}")
    if tracings:
        out.append(f"путей точек: {len(tracings)} ({', '.join(sorted({k for t in tracings for k in t.points}))})")
    return ", ".join(out)


def _read_xml(data: bytes, name: str, case: MotionCase) -> str:
    root = ET.fromstring(data)
    recs, series, used = _tracks(root, name)
    scale = _unit_scale(root)
    for rec in recs:
        if scale and scale != 1:
            rec.transforms[:, :3, 3] *= scale
    reference = _reference_points(root, set().union(*(s.points for s in series)), used) if series else None
    note = _add(case, recs, series, scale, reference)
    for k, v in _values(root, used).items():
        case.values[f"{name}:{k}"] = v
    if note:
        return note
    M = _pose(root)
    if M is not None:
        case.matrices[name] = M
        return "одно положение (матрица)"
    return "записей движения не найдено"


# --- таблицы CSV / ASCII ---

_UNIT = re.compile(r"[\[(]\s*(mm|cm|m|ms|s|deg|°|grad|rad)\s*[\])]", re.I)
_FILE_UNIT = re.compile(r"(?:unit|units|einheit|единиц\w*)\W{0,3}(mm|cm|m)\b", re.I)
_FILE_RATE = re.compile(r"(?:sample\s*rate|sampling\s*rate|frame\s*rate|frequency|frequenz|abtastrate|fps|"
                        r"частота)\D{0,12}?(\d+(?:[.,]\d+)?)", re.I)
_AXIS_LAST = re.compile(r"^(.*?)[\s_.\-/:]*([xyz])$", re.I)
_AXIS_FIRST = re.compile(r"^([xyz])[\s_.\-/:]*(.+)$", re.I)


def _cell(text: str) -> tuple[str, str, str | None]:
    """Заголовок столбца: (как есть без единиц, для сравнения — только буквы и цифры, единица)."""
    unit = None
    if m := _UNIT.search(text):
        unit = m[1].lower().replace("°", "deg").replace("grad", "deg")
        text = text[:m.start()] + text[m.end():]
    text = text.strip().strip('"\'').strip()
    return text, re.sub(r"[\W_]+", "", text.lower()), unit


def _split_header(line: str, width: int) -> list[str] | None:
    for sep in (";", "\t", ","):
        if sep in line:
            cells = [c.strip() for c in line.split(sep)]
            while cells and not cells[-1]:
                cells.pop()
            if len(cells) == width:
                return cells
    cells = re.sub(r"\s+([\[(])", r"\1", line.strip()).split()
    return cells if len(cells) == width else None


def _blocks(text: str) -> list[tuple]:
    """Числовые блоки таблицы: (заголовок или None, строки (N, k), строки текста перед блоком)."""
    blocks, rows, texts = [], [], []

    def close():
        if rows:
            header = _split_header(texts[-1], len(rows[0])) if texts else None
            blocks.append((header, np.array(rows), texts[:-1] if header else list(texts)))

    for line in text.splitlines():
        if not line.strip():
            continue
        n = _numbers(line.replace("\t", " "))
        if n:
            if rows and len(n) != len(rows[0]):
                close()
                rows, texts = [], []
            rows.append(n)
        else:
            if rows:
                close()
                rows, texts = [], []
            texts.append(line)
    close()
    return blocks


def _safe_name(text: str, fallback: str) -> str:
    """Имя записи только из слов движения (протрузия, вправо…): имени пациента из файла в отчёте не будет."""
    t = _tokens(text)
    words = [w for w, keys in MOVE_WORDS if _has(t, keys)]
    return " ".join(words) if words else fallback


def _quaternions(q: np.ndarray, scalar_first: bool | None = None) -> np.ndarray | None:
    """Повороты из кватернионов (N, 4). Скаляр — первым или последним: по данным (у него |w| ≈ 1),
    а если не понять — первым, как в JawTrackingSystem."""
    q = np.asarray(q, float)
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-2):
        return None
    if scalar_first is None:
        near = [(np.abs(q[:, k]) > 0.9).mean() for k in (0, 3)]
        scalar_first = not (near[1] > 0.9 and near[0] < 0.5)
    xyzw = q[:, [1, 2, 3, 0]] if scalar_first else q
    return Rotation.from_quat(xyzw / np.linalg.norm(xyzw, axis=1, keepdims=True)).as_matrix()


def _times_column(names: list, units: list, rows: np.ndarray, rate: float | None):
    for j, (n, u) in enumerate(zip(names, units)):
        if n in TIME_KEYS or n.startswith(("time", "zeit", "время")):
            t = rows[:, j]
            return (t / 1000 if u == "ms" or n in ("ms", "timems", "time_ms", "millis") else t), True, j
    if rate:
        return np.arange(len(rows)) / rate, True, None
    return np.arange(len(rows), dtype=float), False, None


def _headerless(rows: np.ndarray, name: str, source: str) -> tuple[list, str]:
    """Таблица без заголовка: 12/16 чисел матрицы или смещение + кватернион (7 чисел), впереди — время."""
    width = rows.shape[1]
    if width in (13, 17, 8) and np.all(np.diff(rows[:, 0]) >= 0) and np.ptp(rows[:, 0]) > 0:
        times, body, timed = rows[:, 0], rows[:, 1:], True
    else:
        times, body, timed = np.arange(len(rows), dtype=float), rows, False
    if body.shape[1] in (12, 16):
        mats = [_matrix(r) for r in body]
        ok = [k for k, M in enumerate(mats) if M is not None]
        if len(ok) < 0.9 * len(rows):
            return [], "числа не складываются в положения"
        return [Recording(name, times[ok], np.array([mats[k] for k in ok]), source, timed)], ""
    if body.shape[1] == 7:
        R = _quaternions(body[:, 3:])
        if R is None:
            return [], "в последних 4 столбцах не кватернионы"
        T = np.tile(np.eye(4), (len(body), 1, 1))
        T[:, :3, :3], T[:, :3, 3] = R, body[:, :3]
        return [Recording(name, times, T, f"{source} (смещение + кватернион)", timed)], ""
    return [], f"не похоже на движение: по {width} чисел в строке, заголовка нет"


def _with_header(header: list, rows: np.ndarray, name: str, source: str, rate: float | None,
                 unit: float | None) -> tuple[list, list, str]:
    """Таблица с заголовком: положения (смещение + кватернион/углы/ячейки матрицы) или точки x/y/z."""
    cells = [_cell(h) for h in header]
    names, units = [c[1] for c in cells], [c[2] for c in cells]
    times, timed, tj = _times_column(names, units, rows, rate)
    skip = {tj} | {j for j, n in enumerate(names) if n in FRAME_KEYS}
    cols = [j for j in range(len(names)) if j not in skip]
    vals = rows.astype(float).copy()
    for j in cols:
        if units[j] == "rad":
            vals[:, j] = np.degrees(vals[:, j])
    keys = [names[j] for j in cols]
    first = _named_pose(dict(zip(keys, vals[0, cols])))
    if first is not None:
        mats = [_named_pose(dict(zip(keys, r[cols]))) for r in vals]
        ok = [k for k, M in enumerate(mats) if M is not None]
        if len(ok) >= 0.9 * len(rows):
            T = np.array([mats[k] for k in ok])
            t_cols = _triple({k: j for k, j in zip(keys, cols)}, T_PREFIXES)[0]
            u = units[int(t_cols[0])] if t_cols is not None else None
            T[:, :3, 3] *= UNIT_SCALE.get(u, unit or 1.0)
            return [Recording(name, times[ok], T, source, timed)], [], ""
    groups = {}
    for j in cols:
        label = cells[j][0]
        m = _AXIS_LAST.match(label)
        if m and m[1].strip(" _.-/:"):
            point, axis = m[1].strip(" _.-/:"), m[2].lower()
        elif (m := _AXIS_FIRST.match(label)) and m[2].strip(" _.-/:"):
            point, axis = m[2].strip(" _.-/:"), m[1].lower()
        elif label.lower() in ("x", "y", "z"):
            point, axis = "точка", label.lower()
        else:
            continue
        groups.setdefault(point, {})[axis] = j
    points = {p: g for p, g in groups.items() if set(g) == {"x", "y", "z"}}
    if not points:
        return [], [], "столбцов с положением или точками x/y/z нет"
    pts = {}
    for p, g in points.items():
        scale = UNIT_SCALE.get(units[g["x"]], unit or 1.0)
        pts[p] = vals[:, [g["x"], g["y"], g["z"]]] * scale
    return [], [Tracing(name, times, pts, source, timed)], ""


def _read_table(text: str, name: str, case: MotionCase) -> str:
    """Таблица CSV/ASCII: блоки чисел по строке на кадр, с заголовком столбцов или без.

    Без заголовка — 12 или 16 чисел матрицы либо смещение + кватернион (7 чисел),
    впереди может стоять время. С заголовком — по именам столбцов: время,
    смещение + кватернион или углы, ячейки матрицы, точки x/y/z (резцовая точка,
    мыщелки, маркеры); единицы — из заголовка «[mm]» или строки «Unit: m».
    Разделитель — «;», табуляция, запятая или пробелы; десятичная запятая понимается.
    """
    blocks = [b for b in _blocks(text) if len(b[1]) >= MIN_FRAMES]
    if not blocks:
        return "чисел нет" if not _blocks(text) else "блоков чисел длиной с запись движения нет"
    unit = UNIT_SCALE[m[1].lower()] if (m := _FILE_UNIT.search(text)) else None
    rate = float(m[1].replace(",", ".")) if (m := _FILE_RATE.search(text)) else None
    stem = os.path.splitext(os.path.basename(name))[0]
    recs, series, problems = [], [], []
    for k, (header, rows, texts) in enumerate(blocks):
        title = _safe_name(" ".join(texts[-2:]), "") or _safe_name(stem, "")
        rec_name = title or (f"запись {len(case.recordings) + len(case.tracings) + k + 1}")
        source = f"{name}: блок {k + 1}" if len(blocks) > 1 else name
        if header is None:
            got, why = _headerless(rows, rec_name, source)
            for r in got:
                if unit:
                    r.transforms[:, :3, 3] *= unit
            recs += got
        else:
            got, pts, why = _with_header(header, rows, rec_name, source, rate, unit)
            recs += got
            series += pts
        if why:
            problems.append(why)
    known = unit is not None or any(_UNIT.search(" ".join(b[0])) for b in blocks if b[0])
    note = _add(case, recs, series, 1.0 if known else None)  # единицы известны — точки уже в мм
    return note or "; ".join(dict.fromkeys(problems))


# --- HDF5 (JawTrackingSystem и подобные) ---

def _read_hdf5(data: bytes, name: str, case: MotionCase) -> str:
    """Группы со смещениями (N, 3) и поворотами (кватернионы (N, 4), скаляр первым, или матрицы (N, 3, 3)),
    атрибуты sample_rate и unit — как сохраняет JawTrackingSystem; а также наборы (N, 4, 4)."""
    try:
        import h5py
    except ImportError:
        return "HDF5: для чтения установите h5py"
    recs = []
    with h5py.File(io.BytesIO(data), "r") as f:
        jts = "jts_version" in f.attrs
        found = []

        def visit(path, obj):
            if isinstance(obj, h5py.Group) and "translations" in obj and "rotations" in obj:
                found.append(path)
            elif isinstance(obj, h5py.Dataset) and obj.ndim == 3 and obj.shape[1:] == (4, 4):
                found.append(path)

        f.visititems(visit)
        smooth = {p[:-len("_smooth")] for p in found if p.endswith("_smooth")}
        found = [p for p in found if p not in smooth]  # есть сглаженная — сырая не нужна
        for k, path in enumerate(found):
            obj = f[path]
            attrs = obj.attrs if isinstance(obj, h5py.Group) else obj.parent.attrs
            if isinstance(obj, h5py.Group):
                t, r = np.asarray(obj["translations"], float), np.asarray(obj["rotations"], float)
                R = _quaternions(r, True if jts else None) if r.ndim == 2 and r.shape[1] == 4 else \
                    (r if r.ndim == 3 and r.shape[1:] == (3, 3) else None)
                if R is None or len(R) != len(t):
                    continue
                T = np.tile(np.eye(4), (len(t), 1, 1))
                T[:, :3, :3], T[:, :3, 3] = R, t
            else:
                T = np.asarray(obj, float)
            u = attrs.get("unit", "")
            u = u.decode() if isinstance(u, bytes) else str(u)
            T[:, :3, 3] *= UNIT_SCALE.get(u.strip().lower(), 1.0)
            rate = float(attrs.get("sample_rate", 0) or 0)
            times = np.arange(len(T)) / rate if rate > 0 else np.arange(len(T), dtype=float)
            rec_name = _safe_name(os.path.splitext(os.path.basename(name))[0], f"запись {k + 1}")
            recs.append(Recording(rec_name, times, T, f"{name}: {path}", rate > 0))
    return _add(case, recs, [], None) or "записей движения не найдено"


def detect_system(name: str, data: bytes) -> str | None:
    """Система записи по имени файла и его началу — только подпись; на чтение не влияет."""
    text = name.lower().encode("utf-8", "ignore") + b" " + data[:1 << 18].lower()
    for system, keys in SYSTEMS:
        if any(k.encode() in text for k in keys):
            return system
    return None


def _entries(path: str):
    """Файлы кейса: (относительный путь, размер, чтение) — из папки, архива или одного файла."""
    if os.path.isdir(path):
        for d, _dirs, names in os.walk(path):
            for n in sorted(names):
                full = os.path.join(d, n)
                yield os.path.relpath(full, path), os.path.getsize(full), (lambda f=full: open(f, "rb").read())
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir() and not info.filename.startswith("__MACOSX"):
                    yield info.filename, info.file_size, (lambda i=info: z.read(i))
    elif os.path.isfile(path):
        yield os.path.basename(path), os.path.getsize(path), (lambda: open(path, "rb").read())
    else:
        raise FileNotFoundError(f"нет такого файла или папки: {path}")


def _decode(data: bytes) -> str:
    """Текст таблицы: UTF-8 (с BOM или без), UTF-16 или кодировка Windows."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", "replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252" if not re.search(rb"[\xc0-\xff]{3}", data) else "cp1251", "replace")


def _kind(ext: str) -> str:
    for kind, exts in (("модель", MESH_EXT), ("движение", MOTION_EXT + HDF5_EXT + tuple(CLOSED_EXT)),
                       ("матрица", MATRIX_EXT),
                       ("проект exocad", PROJECT_EXT), ("изображение", (".png", ".jpg", ".jpeg", ".bmp")),
                       ("документ", (".pdf", ".html", ".htm"))):
        if ext in exts:
            return kind
    return "другое"


def read_case(path: str) -> MotionCase:
    """Прочитать выгрузку: папку, архив .zip или отдельный файл движения."""
    import trimesh

    case = MotionCase(source=os.path.basename(os.path.normpath(path)))
    for rel, size, read in _entries(path):
        ext = os.path.splitext(rel)[1].lower()
        entry = {"path": rel, "kind": _kind(ext), "size": size}
        try:
            if ext in MESH_EXT:
                mesh = trimesh.load(io.BytesIO(read()), file_type=ext[1:], force="mesh", process=False)
                case.meshes[rel] = (np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
                if (role := _role(rel)) and role not in case.roles:
                    case.roles[role] = rel
                    entry["role"] = role
            elif ext in MATRIX_EXT:
                M = _pose(ET.fromstring(read()))
                if M is not None:
                    case.matrices[rel] = M
            elif ext in PROJECT_EXT:
                data = read()
                if _is_xml(data):
                    root = ET.fromstring(data)
                    case.project.update({k: el.text.strip() for k in PROJECT_FIELDS
                                         if (el := root.find(f".//{k}")) is not None and el.text})
            elif ext in MOTION_EXT or ext in HDF5_EXT or ext in CLOSED_EXT:
                data = read()
                if system := detect_system(rel, data):
                    entry["system"] = system
                if ext in CLOSED_EXT:
                    entry["note"] = (f"формат {CLOSED_EXT[ext]} открыто не описан — нужна выгрузка движения "
                                     "в открытом виде (XML для exocad, CSV/ASCII)")
                elif ext in HDF5_EXT or data[:8] == b"\x89HDF\r\n\x1a\n":
                    entry["note"] = _read_hdf5(data, rel, case)
                elif _is_xml(data):
                    entry["note"] = _read_xml(data, rel, case)
                elif ext == ".jawmotion":
                    entry["note"] = "двоичный формат — нужен образец"
                else:
                    entry["note"] = _read_table(_decode(data), rel, case)
        except Exception as e:  # noqa: BLE001 — один плохой файл не мешает остальным
            entry["note"] = f"не прочитан: {type(e).__name__}: {e}"
        case.files.append(entry)
    case.systems = sorted({f["system"] for f in case.files if f.get("system")})

    if case.recordings:
        moved = max(float(np.abs(r.transforms[:, :3, 3]).max()) for r in case.recordings)
        turned = max(float(rotation_angles(r.transforms).max()) for r in case.recordings)
        if moved < 0.2 and turned > 5:
            for r in case.recordings:
                r.transforms[:, :3, 3] *= 1000
            case.notes.append("смещения похожи на метры — переведены в миллиметры")
        if not all(r.timed for r in case.recordings):
            case.notes.append("в части записей нет времени — вместо секунд номера кадров")
    elif not case.tracings:
        case.notes.append("записей движения не найдено — нужен образец выгрузки (motion --inspect)")
    if case.tracings and not case.recordings:
        case.notes.append("есть только пути отдельных точек: углы суставного пути считаются, "
                          "а двигать модели по ним нельзя")
    if case.meshes and "lower" not in case.roles:
        case.notes.append("модель нижней челюсти не определена по имени файла — укажите её (--lower)")
    return case


# --- геометрия движения ---

def rotation_angles(transforms: np.ndarray) -> np.ndarray:
    return np.degrees(np.linalg.norm(Rotation.from_matrix(transforms[:, :3, :3]).as_rotvec(), axis=1))


def track(transforms: np.ndarray, point) -> np.ndarray:
    """Положения точки нижней челюсти по кадрам."""
    return transforms[:, :3, :3] @ np.asarray(point, float) + transforms[:, :3, 3]


def reference_pose(case: MotionCase) -> np.ndarray | None:
    """Единичная матрица, если положения отсчитаны от окклюзии (она встречается в записях), иначе None."""
    for rec in case.recordings:
        near = (rotation_angles(rec.transforms) < 0.5) & (np.linalg.norm(rec.transforms[:, :3, 3], axis=1) < 0.3)
        if near.any():
            return np.eye(4)
    return None


def relative(rec: Recording, ref: np.ndarray | None) -> np.ndarray:
    """Смещения от исходного положения: от окклюзии или, если её нет, от первого кадра записи."""
    return rec.transforms @ np.linalg.inv(ref if ref is not None else rec.transforms[0])


def hinge_axis(transforms: np.ndarray, near=None, max_deg: float = HINGE_MAX_DEG):
    """Шарнирная ось по началу открывания: линия, точки которой почти не сдвигаются.

    Возвращает точку оси (ближайшую к near), направление (поворот вокруг него —
    открывание) и насколько точка оси в среднем сдвигается на этих кадрах (мм).
    """
    near = np.zeros(3) if near is None else np.asarray(near, float)
    rv = Rotation.from_matrix(transforms[:, :3, :3]).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    use = (ang > 1.0) & (ang <= max_deg)
    if use.sum() < 3:
        raise ValueError(f"для шарнирной оси нужно открывание: мало кадров с поворотом 1–{max_deg:.0f}°")
    T = transforms[use]
    D = T[:, :3, :3] - np.eye(3)
    q = np.linalg.lstsq(D.reshape(-1, 3), (-T[:, :3, 3] - D @ near).reshape(-1), rcond=1e-3)[0]
    axes = rv[use] / np.linalg.norm(rv[use], axis=1, keepdims=True)
    axis = (axes * np.sign(axes @ axes[-1])[:, None]).mean(0)
    axis /= np.linalg.norm(axis)
    point = near + q
    point += ((near - point) @ axis) * axis
    moved = np.linalg.norm(track(T, point) - point, axis=1)
    return point, axis, float(np.sqrt(np.mean(moved ** 2)))


def _frame(origin, x, y, z) -> np.ndarray:
    R = np.array([x, y, z])
    return rigid(R, -R @ origin)


def _incisal(vertices: np.ndarray, anterior, up, tilt_deg: float = 0.0) -> np.ndarray:
    """Резцовая точка модели нижней челюсти: самая передняя среди «верхних» вершин."""
    a = np.radians(tilt_deg)
    score = vertices @ (np.cos(a) * np.asarray(up) + np.sin(a) * np.asarray(anterior))
    band = vertices[score >= np.quantile(score, 1 - TOP_SHARE)]
    return band[np.argmax(band @ anterior)]


def _opening_record(case: MotionCase, ref) -> Recording:
    if not case.recordings:
        raise ValueError("в кейсе нет записей движения")
    return max(case.recordings, key=lambda r: rotation_angles(relative(r, ref)).max())


def estimate_anatomy(case: MotionCase, lower: np.ndarray | None = None, incisal=None,
                     icd: float = ICD_MM) -> Anatomy:
    """Анатомическая система без КТ: шарнирная ось по открыванию, резцовая точка по модели.

    Горизонталь — плоскость через шарнирную ось и резцовую точку; мыщелки — на
    оси, на icd/2 в стороны от середины. Право и лево следуют из движения: при
    открывании резцы уходят вниз, значит направление оси задаёт сторону.
    """
    ref = reference_pose(case)
    if lower is None and (m := case.mesh("lower")) is not None:
        lower = m[0]
    if incisal is None and lower is None:
        raise ValueError("нужна модель нижней челюсти или резцовая точка")
    near = np.asarray(incisal, float) if incisal is not None else lower.mean(0)
    point, axis, rms = hinge_axis(relative(_opening_record(case, ref), ref), near)
    source = "оценка по движению: шарнирная ось по открыванию"
    if incisal is None:
        anterior = lower.mean(0) - point
        anterior -= (anterior @ axis) * axis
        anterior /= np.linalg.norm(anterior)
        incisal = _incisal(lower, anterior, np.cross(anterior, axis))
        source += ", резцовая точка по модели"
    incisal = np.asarray(incisal, float)
    hinge = point + ((incisal - point) @ axis) * axis
    y = incisal - hinge
    if np.linalg.norm(y) < 10:
        raise ValueError("резцовая точка лежит почти на шарнирной оси — проверьте данные")
    y /= np.linalg.norm(y)
    z = np.cross(y, axis)
    x = np.cross(y, z)
    points = {"incisal": incisal, "condyle_right": hinge + icd / 2 * x, "condyle_left": hinge - icd / 2 * x}
    return Anatomy(_frame(hinge, x, y, z), points, source + "; горизонталь — шарнирная ось и резцовая точка",
                   rms)


def arch_axes(vertices: np.ndarray, up) -> tuple[np.ndarray, np.ndarray]:
    """Окклюзионная плоскость и направление вперёд по зубной дуге.

    Плоскость — по вершинам бугров (самые высокие точки вдоль up). Вперёд —
    к вершине дуги: в плоскости подбирается ось, вдоль которой коронки
    (точки не глубже ARCH_BAND_MM от плоскости) лучше всего ложатся на
    параболу, и её вершина — резцы. Форма дуги берётся по всем коронкам, а не
    только по вершинам бугров: при выраженной кривой Шпее самые высокие точки —
    одни моляры, и по ним дугу не понять.
    """
    up = np.asarray(up, float) / np.linalg.norm(up)
    h = vertices @ up
    tips = vertices[h >= np.quantile(h, 1 - CUSP_SHARE)]
    c = tips.mean(0)
    normal = np.linalg.svd(tips - c, full_matrices=False)[2][2]
    normal = normal if normal @ up > 0 else -normal
    u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    depth = (tips @ normal).mean() - vertices @ normal
    crowns = vertices[depth < ARCH_BAND_MM]
    if len(crowns) > 6000:
        crowns = crowns[np.random.default_rng(0).choice(len(crowns), 6000, replace=False)]
    p2 = np.c_[(crowns - c) @ u, (crowns - c) @ w]
    best = None
    for a in np.radians(np.arange(0.0, 180.0, 1.0)):
        d, e = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        x, y = p2 @ e, p2 @ d
        A = np.c_[np.ones_like(x), x, x ** 2]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        err = float(np.mean((A @ coef - y) ** 2))
        if best is None or err < best[0]:
            best = (err, d if coef[2] < 0 else -d)
    anterior = best[1][0] * u + best[1][1] * w
    return normal, anterior / np.linalg.norm(anterior)


def anatomy_average(lower: np.ndarray, upper: np.ndarray | None = None, lower_normals: np.ndarray | None = None,
                    incisal=None, side: float = BONWILL_MM, icd: float = BONWILL_MM,
                    balkwill_deg: float = BALKWILL_DEG) -> Anatomy:
    """Средний артикулятор по одним сканам: треугольник Бонвилля и угол Балквилла.

    Когда нет ни КТ, ни записи движений. Вверх — от нижних зубов к верхним
    (или по нормалям скана нижней челюсти, если верхнего нет); окклюзионная
    плоскость и направление вперёд — по зубной дуге; резцовая точка — на
    нижних резцах. Мыщелки — в вершинах равностороннего треугольника Бонвилля
    со стороной 100 мм, плоскость которого наклонена к окклюзионной на угол
    Балквилла (25°). Горизонталь анализа — окклюзионная плоскость.
    """
    lower = np.asarray(lower, float)
    if upper is not None:
        up = np.asarray(upper, float).mean(0) - lower.mean(0)
    elif lower_normals is not None:
        up = np.asarray(lower_normals, float).sum(0)
    else:
        raise ValueError("нужен скан верхней челюсти или нормали скана нижней, чтобы понять, где верх")
    z, y = arch_axes(lower, up)
    if incisal is None:
        incisal = _incisal(lower, y, z, INCISAL_TILT_DEG)
    incisal = np.asarray(incisal, float)
    x = np.cross(y, z)
    back = np.sqrt(side ** 2 - (icd / 2) ** 2)
    b = np.radians(balkwill_deg)
    hinge = incisal - y * back * np.cos(b) + z * back * np.sin(b)
    points = {"incisal": incisal, "condyle_right": hinge + icd / 2 * x, "condyle_left": hinge - icd / 2 * x}
    return Anatomy(_frame(hinge, x, y, z), points,
                   f"средние значения: треугольник Бонвилля {side:g} мм, угол Балквилла {balkwill_deg:g}°; "
                   "горизонталь — окклюзионная плоскость")


def anatomy_from_ct(landmarks: dict, plane: str, case_to_ct: np.ndarray, lower: np.ndarray | None = None,
                    incisal=None) -> Anatomy:
    """Анатомическая система по КТ: модели кейса совмещены с КТ (case_to_ct), ориентиры поставлены."""
    F = reference_frame(landmarks, plane) @ case_to_ct
    if incisal is None:
        if lower is None:
            raise ValueError("нужна модель нижней челюсти или резцовая точка")
        incisal = _incisal(lower, F[1, :3], F[2, :3], INCISAL_TILT_DEG)
    back = np.linalg.inv(case_to_ct)
    points = {"incisal": np.asarray(incisal, float),
              "condyle_right": apply(back, np.asarray(landmarks["Co_R"], float)[None])[0],
              "condyle_left": apply(back, np.asarray(landmarks["Co_L"], float)[None])[0]}
    names = {"frankfurt": "Франкфуртская горизонталь", "hip": "HIP", "camper": "плоскость Кемпера"}
    return Anatomy(F, points, f"КТ: {names.get(plane, plane)}; мыщелки — ориентиры "
                              f"«{BY_KEY['Co_R'].name}» и «{BY_KEY['Co_L'].name}»")


# --- анализ ---

def paths(rec: Recording, anatomy: Anatomy, ref: np.ndarray | None = None) -> dict:
    """Положения точек по кадрам в анатомической системе, мм."""
    rel = relative(rec, ref)
    return {k: apply(anatomy.frame, track(rel, p)) for k, p in anatomy.points.items()}


def _chord(path: np.ndarray, length: float):
    """Хорда пути от начала до места, где точка ушла на length мм (или до самого дальнего)."""
    d = path - path[0]
    dist = np.linalg.norm(d, axis=1)
    k = int(np.argmax(dist >= length)) if (dist >= length).any() else int(np.argmax(dist))
    return d[k], float(dist[k])


def _cycles(opening: np.ndarray) -> int:
    """Сколько раз рот открывался больше чем наполовину от максимума (с гистерезисом)."""
    top = float(opening.max())
    if top < 3:
        return 0
    count, is_open = 0, False
    for v in opening:
        if not is_open and v > 0.5 * top:
            count, is_open = count + 1, True
        elif is_open and v < 0.25 * top:
            is_open = False
    return count


def _classify(lat: float, ant: float, down: float, cycles: int) -> str:
    if cycles >= 3:
        return "chewing"
    if down > 2 * max(abs(lat), ant, 1.0):
        return "opening"
    if abs(lat) >= max(ant, 1.0):
        return "laterotrusion_right" if lat > 0 else "laterotrusion_left"
    if ant >= 1.0:
        return "protrusion"
    return "other"


def _r(v, digits=2):
    return None if v is None else round(float(v), digits)


def analyze_recording(rec: Recording, anatomy: Anatomy, ref: np.ndarray | None = None) -> dict:
    return _analyze_paths(rec.name, paths(rec, anatomy, ref), len(rec.transforms),
                          rec.duration if rec.timed else None)


def _classify_condyles(P: dict) -> str:
    """Тип движения по одним мыщелкам: оба уходят — протрузия (открывание от неё не отличить),
    один стоит — латеротрузия в его сторону."""
    d = {s: float(np.linalg.norm(P[f"condyle_{s}"] - P[f"condyle_{s}"][0], axis=1).max()) for s in ("right", "left")}
    if max(d.values()) < 1.0:
        return "other"
    if min(d.values()) > 0.5 * max(d.values()):
        return "protrusion"
    return "laterotrusion_right" if d["right"] < d["left"] else "laterotrusion_left"


def _analyze_paths(name: str, P: dict, frames: int, duration: float | None, kind: str | None = None) -> dict:
    """Углы и размах по путям точек в анатомической системе; точек может не хватать (пути мыщелков без резцов)."""
    out = {"name": name, "kind": kind, "frames": frames, "duration_s": _r(duration, 3), "cycles": 0}
    if "incisal" in P:
        inc = P["incisal"] - P["incisal"][0]
        k = int(np.argmax(np.linalg.norm(inc, axis=1)))
        out["cycles"] = _cycles(-inc[:, 2])
        kind = kind or _classify(inc[k, 0], inc[k, 1], -inc[k, 2], out["cycles"])
        out["incisal"] = {"max_mm": _r(np.linalg.norm(inc, axis=1).max()),
                          "opening_mm": _r(max(0, -inc[:, 2].min())), "protrusion_mm": _r(max(0, inc[:, 1].max())),
                          "right_mm": _r(max(0, inc[:, 0].max())), "left_mm": _r(max(0, -inc[:, 0].min()))}
        # Ведение зубами: наклон пути резцовой точки на первых миллиметрах (сбоку — при протрузии,
        # спереди — при боковых движениях).
        v, at = _chord(P["incisal"], GUIDE_CHORD_MM)
        if kind == "protrusion" and at >= 1.0 and v[1] > 0:
            out["incisal"]["guidance_deg"] = _r(np.degrees(np.arctan2(-v[2], v[1])), 1)
        elif kind.startswith("laterotrusion") and at >= 1.0:
            out["incisal"]["guidance_deg"] = _r(np.degrees(np.arctan2(-v[2], abs(v[0]))), 1)
    elif kind is None:
        kind = _classify_condyles(P) if "condyle_right" in P and "condyle_left" in P else "other"
    out["kind"], out["kind_name"] = kind, KINDS[kind]
    out["condyles"] = {}
    for side, sign in (("right", 1.0), ("left", -1.0)):
        if (c := P.get(f"condyle_{side}")) is None:
            continue
        v, at = _chord(c, CHORD_MM)
        info = {"path_mm": _r(np.linalg.norm(c - c[0], axis=1).max()), "measured_at_mm": _r(at)}
        if kind == f"laterotrusion_{side}":  # рабочий мыщелок: насколько ушёл наружу
            info["lateral_mm"] = _r(max(0.0, (sign * (c[:, 0] - c[0, 0])).max()))
        elif at >= 1.0 and v[1] > 0:
            info["sagittal_deg"] = _r(np.degrees(np.arctan2(-v[2], v[1])), 1)
            if kind.startswith("laterotrusion"):
                info.update(_bennett(c, sign))
        out["condyles"][side] = info
    if kind.startswith("laterotrusion"):
        other = "left" if kind.endswith("right") else "right"
        out["immediate_side_shift_mm"] = out["condyles"].get(other, {}).get("immediate_side_shift_mm")
    return out


def _bennett(c: np.ndarray, sign: float) -> dict:
    """Балансирующий мыщелок: угол Беннетта и немедленный боковой сдвиг.

    Сверху путь мыщелка — сначала сдвиг внутрь почти без движения вперёд
    (немедленный боковой сдвиг), потом прямая под углом Беннетта. Прямая
    подгоняется на участке от ISS_FIT_MM до CHORD_MM вперёд: её наклон — угол,
    отрезок при нулевом движении вперёд — немедленный сдвиг.
    """
    d = c - c[0]
    forward, medial = d[:, 1], -sign * d[:, 0]  # внутрь — к середине: для правого мыщелка это −x
    end = int(np.argmax(np.linalg.norm(d, axis=1) >= CHORD_MM)) if (np.linalg.norm(d, axis=1) >= CHORD_MM).any() \
        else len(d) - 1
    sel = np.arange(len(d)) <= end
    sel &= forward >= ISS_FIT_MM
    if sel.sum() >= 3 and np.ptp(forward[sel]) >= 1.0:
        slope, at_zero = np.polyfit(forward[sel], medial[sel], 1)
        return {"bennett_deg": _r(np.degrees(np.arctan(slope)), 1),
                "immediate_side_shift_mm": _r(max(0.0, at_zero))}
    v = d[end]
    return {"bennett_deg": _r(np.degrees(np.arctan2(-sign * v[0], v[1])), 1)}


def _mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 1) if values else None


def _summary(recs: list) -> tuple[dict, dict]:
    """Углы для настройки артикулятора и ведения зубами — средние по записям нужного типа."""
    def get(kind, side, key):
        return [r["condyles"].get(side, {}).get(key) for r in recs if r["kind"] == kind]

    settings = {
        "sagittal_right_deg": _mean(get("protrusion", "right", "sagittal_deg")),
        "sagittal_left_deg": _mean(get("protrusion", "left", "sagittal_deg")),
        "bennett_right_deg": _mean(get("laterotrusion_left", "right", "bennett_deg")),
        "bennett_left_deg": _mean(get("laterotrusion_right", "left", "bennett_deg")),
        "side_shift_right_mm": _mean(r.get("immediate_side_shift_mm") for r in recs
                                     if r["kind"] == "laterotrusion_right"),
        "side_shift_left_mm": _mean(r.get("immediate_side_shift_mm") for r in recs
                                    if r["kind"] == "laterotrusion_left"),
    }
    for side, other in (("right", "left"), ("left", "right")):
        nw, pro = _mean(get(f"laterotrusion_{other}", side, "sagittal_deg")), settings[f"sagittal_{side}_deg"]
        settings[f"fischer_{side}_deg"] = None if nw is None or pro is None else round(nw - pro, 1)
    guidance = {kind: _mean(r.get("incisal", {}).get("guidance_deg") for r in recs if r["kind"] == kind)
                for kind in ("protrusion", "laterotrusion_right", "laterotrusion_left")}
    return settings, guidance


def analyze_case(case: MotionCase, anatomy: Anatomy) -> dict:
    """Все записи кейса и сводка: углы для настройки артикулятора, наибольшие движения."""
    ref = reference_pose(case)
    recs = [analyze_recording(r, anatomy, ref) for r in case.recordings]
    settings, guidance = _summary(recs)
    notes = list(case.notes)
    if anatomy.source.startswith("оценка"):
        notes.append("углы отсчитаны от плоскости «шарнирная ось — резцовая точка», а не от Франкфурта: "
                     "сагиттальные меньше обычно на 10–20°; с КТ углы будут от выбранной плоскости")
    if ref is None:
        notes.append("окклюзии в записях нет — смещения отсчитаны от первого кадра каждой записи")
    return {
        "case": case.source, "systems": case.systems, "frame": anatomy.source,
        "hinge_rms_mm": _r(anatomy.hinge_rms_mm, 3),
        "reference": "окклюзия" if ref is not None else "первый кадр записи",
        "points": {k: [round(float(c), 2) for c in apply(anatomy.frame, p[None])[0]]
                   for k, p in anatomy.points.items()},
        "max_opening_mm": _r(max((r["incisal"]["opening_mm"] for r in recs), default=None)),
        "articulator": settings, "guidance_deg": guidance, "recordings": recs, "notes": notes,
    }


# --- пути отдельных точек (кондилографы) ---

def parse_axes(text: str) -> np.ndarray:
    """Оси системы записи: «-y,x,z» — какие оси файла смотрят вправо пациента, вперёд и вверх."""
    rows = []
    for part in text.replace(" ", "").lower().split(","):
        axis = part.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise ValueError(f"оси: «{text}» — нужно три оси файла через запятую, например «-y,x,z»")
        v = np.zeros(3)
        v["xyz".index(axis)] = -1.0 if part.startswith("-") else 1.0
        rows.append(v)
    A = np.array(rows)
    if A.shape != (3, 3) or abs(np.linalg.det(A)) < 0.5:
        raise ValueError(f"оси: «{text}» — три разные оси файла, например «-y,x,z»")
    return A


def _roles(tr: Tracing) -> dict:
    pair = len(tr.points) == 2
    out = {}
    for k, v in tr.points.items():
        if (role := point_role(k, pair)) and role not in out:
            out[role] = v
    return out


def tracing_axes(tracings: list) -> tuple[np.ndarray | None, str]:
    """Оси анализа путей мыщелков по самим путям — если это можно сделать надёжно.

    Вправо — от левого мыщелка к правому (ближайшая ось файла). Из двух
    оставшихся осей вперёд — та, вдоль которой мыщелки уходят больше, вверх —
    другая, со знаком «мыщелки уходят вниз»: мыщелок идёт вперёд и вниз по скату
    бугорка. При угле пути ближе к 45° (меньше чем в AXES_RATIO раз разницы)
    оси не угадать — тогда их нужно задать.
    """
    pairs = [r for t in tracings if "condyle_right" in (r := _roles(t)) and "condyle_left" in r]
    if not pairs:
        return None, "нет пары мыщелков с понятными именами — задайте оси и имена точек явно"
    x = np.mean([r["condyle_right"][0] - r["condyle_left"][0] for r in pairs], axis=0)
    i = int(np.argmax(np.abs(x)))
    if abs(x[i]) < np.cos(np.radians(20)) * np.linalg.norm(x):
        return None, "линия между мыщелками не идёт вдоль оси файла — задайте оси явно"
    E = np.eye(3)
    right = E[i] * np.sign(x[i])
    moves = np.array([c[np.argmax(np.linalg.norm(c - c[0], axis=1))] - c[0]
                      for r in pairs for c in (r["condyle_right"], r["condyle_left"])])
    rest = [j for j in range(3) if j != i]
    size = [float(np.abs(moves[:, j]).mean()) for j in rest]
    if max(size) < AXES_RATIO * min(size):
        return None, ("мыщелки уходят вперёд и вниз почти поровну — по путям не понять, какая ось вперёд; "
                      "задайте оси явно (например, --axes=-y,x,z)")
    j, k = rest[int(np.argmax(size))], rest[int(np.argmin(size))]
    forward = E[j] * np.sign(moves[:, j].sum())
    up = -E[k] * np.sign(moves[:, k].sum())  # по скату бугорка мыщелки уходят вниз
    axes = np.array([right, forward, up])
    note = "оси системы записи угаданы по путям мыщелков: " + ", ".join(
        f"{name} — {'-' if v.min() < 0 else ''}{'xyz'[int(np.argmax(np.abs(v)))]}"
        for name, v in zip(("вправо", "вперёд", "вверх"), axes))
    if np.linalg.det(axes) < 0:
        note += " (оси файла левые — так бывает, например «вперёд, влево, вниз»; если нет — проверьте стороны)"
    return axes, note


def _kind_from_name(name: str) -> str | None:
    t = _tokens(name)
    if _has(t, dict(MOVE_WORDS)["протрузия"]):
        return "protrusion"
    if _has(t, dict(MOVE_WORDS)["открывание"]):
        return "opening"
    if _has(t, dict(MOVE_WORDS)["жевание"]):
        return "chewing"
    if _has(t, dict(MOVE_WORDS)["латеротрузия"]):
        if _has(t, dict(MOVE_WORDS)["вправо"]):
            return "laterotrusion_right"
        if _has(t, dict(MOVE_WORDS)["влево"]):
            return "laterotrusion_left"
    return None


def analyze_tracings(case: MotionCase, axes: np.ndarray | None = None) -> dict:
    """Углы по путям отдельных точек: сагиттальный угол суставного пути, Беннетт, боковой сдвиг.

    Углы — к горизонтали системы записи (у кондилографа это обычно его опорная
    плоскость), а не к Франкфурту или окклюзионной плоскости. Тип движения — по
    имени записи, иначе по мыщелкам (открывание от протрузии по ним не отличить).
    """
    notes = []
    if axes is None:
        axes, note = tracing_axes(case.tracings)
        notes.append(note)
    else:
        notes.append("оси системы записи заданы явно")
    if axes is None:
        return {"axes": None, "recordings": [], "articulator": {}, "guidance_deg": {}, "notes": notes}
    recs = []
    for n, tr in enumerate(case.tracings):
        P = {k: v @ axes.T for k, v in _roles(tr).items()}
        if not P:
            notes.append(f"запись {n + 1}: точки не узнаны по именам ({len(tr.points)}) — пропущена")
            continue
        kind = _kind_from_name(tr.name)
        recs.append(_analyze_paths(tr.name, P, len(tr.times), tr.duration if tr.timed else None, kind))
        if kind is None and recs[-1]["kind"] == "protrusion" and "incisal" not in P:
            notes.append("симметричное движение мыщелков считается протрузией: открывание по путям мыщелков "
                         "от неё не отличить — подпишите записи")
    settings, guidance = _summary(recs)
    return {"axes": axes.tolist(), "recordings": recs, "articulator": settings, "guidance_deg": guidance,
            "notes": list(dict.fromkeys(notes))}


# --- выгрузка ---

def write_paths_csv(path: str, case: MotionCase, anatomy: Anatomy):
    """Траектории точек по кадрам в анатомической системе (мм) — для таблиц и графиков."""
    ref = reference_pose(case)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["запись", "время"] + [f"{k}_{a}" for k in POINTS for a in "xyz"])
        for rec in case.recordings:
            P = paths(rec, anatomy, ref)
            for i, t in enumerate(rec.times):
                w.writerow([rec.name, f"{t:.4f}"] + [f"{v:.3f}" for k in POINTS for v in P[k][i]])


KIND_COLORS = {"opening": "#5b8cff", "protrusion": "#3ecf8e", "laterotrusion_right": "#f5b84b",
               "laterotrusion_left": "#e8625f", "chewing": "#b48cff", "other": "#8a93a6"}


def plot(case: MotionCase, anatomy: Anatomy, path: str, analysis: dict | None = None):
    """Картинка: путь резцовой точки (сагиттально и спереди) и пути мыщелков (сбоку и сверху)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = analysis or analyze_case(case, anatomy)
    ref = reference_pose(case)
    start = {k: apply(anatomy.frame, p[None])[0] for k, p in anatomy.points.items()}
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11, 9.5))
    seen = set()
    for rec, res in zip(case.recordings, analysis["recordings"]):
        P = {k: v - start[k] for k, v in paths(rec, anatomy, ref).items()}
        color = KIND_COLORS[res["kind"]]
        label = res["kind_name"] if res["kind"] not in seen else None
        seen.add(res["kind"])
        a1.plot(P["incisal"][:, 1], P["incisal"][:, 2], color=color, label=label, lw=1.6)
        a2.plot(P["incisal"][:, 0], P["incisal"][:, 2], color=color, lw=1.6)
        for side, style, dx in (("right", "-", 12), ("left", "--", -12)):
            c = P[f"condyle_{side}"]
            a3.plot(c[:, 1], c[:, 2], style, color=color, lw=1.6)
            a4.plot(c[:, 0] + dx, c[:, 1], style, color=color, lw=1.6)
    a1.set(title="Резцовая точка — сбоку", xlabel="вперёд, мм", ylabel="вверх, мм")
    a2.set(title="Резцовая точка — спереди", xlabel="вправо пациента, мм", ylabel="вверх, мм")
    a2.invert_xaxis()  # как при взгляде на пациента: его правая сторона слева
    a3.set(title="Мыщелки — сбоку (сплошная — правый, пунктир — левый)", xlabel="вперёд, мм",
           ylabel="вверх, мм")
    a4.set(title="Мыщелки — сверху (разнесены на ±12 мм)", xlabel="вправо пациента, мм", ylabel="вперёд, мм")
    for a in (a1, a2, a3, a4):
        a.set_aspect("equal", adjustable="datalim")
        a.grid(alpha=0.3)
        a.axhline(0, color="#999", lw=0.6)
    a1.legend(loc="lower left", fontsize=9)
    s = analysis["articulator"]

    def fmt(v, unit):
        return "—" if v is None else f"{v:g}{unit}"

    fig.suptitle(f"Сагиттальный угол П {fmt(s['sagittal_right_deg'], '°')} / Л {fmt(s['sagittal_left_deg'], '°')}"
                 f"   Беннетт П {fmt(s['bennett_right_deg'], '°')} / Л {fmt(s['bennett_left_deg'], '°')}"
                 f"   Открывание {fmt(analysis['max_opening_mm'], ' мм')}\n{anatomy.source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
