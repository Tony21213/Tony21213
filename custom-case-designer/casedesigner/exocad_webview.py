"""Сцена из HTML-экспорта exocad («webview»): сканы, КТ и моделировки в координатах exocad.

Экспорт без пароля хранит сцену в base64 внутри HTML (DentalWebGL.m_Data):
описание сцены и по сетке OpenCTM на каждый объект дерева exocad с его
матрицей и путём в дереве («Сканы челюстей / Скан челюсти (17-...-27)»).
Чтение сцены и OpenCTM перенесено из crownai (проект DentalAI в этом же
репозитории, ветка claude/dental-ai-0me2qf) и проверено на реальных экспортах.

Разбор сцены для движений (`Scene.parts`):

* **скан челюсти** — самый протяжённый объект «Сканы челюстей» (или со «scan»
  в имени) для каждой челюсти — полная дуга, а не предпреп; скан без указания челюсти — на той, где есть
  моделировка, антагонист — на другой (без моделировки — по высоте);
* **КТ** — кости, импортированные в exocad: слот «Бюгельный каркас» или имена
  с bone/skull/череп/кост; нижняя челюсть — по lower/mandib/нижн;
* **моделировки** — «Полная анатомия», ваксапы по номерам зубов.

В ведении по зубам участвуют только сканы челюстей. Оси сцены exocad к
анатомии не привязаны (в разных экспортах «вперёд» — то +y, то −y), поэтому
вверх, вперёд и вправо определяются по самим зубам. Имена объектов (в них
бывает ФИО пациента) наружу не выводятся — только тип, челюсть и номера зубов.
"""

import base64
import lzma
import re
import struct
from dataclasses import dataclass, field

import numpy as np

from . import kinematics as kin
from . import motion as mo
from .guidance import condylar_path, eminence_profile, hanau_bennett
from .landmarks import suggest_condyles
from .register import rigid


# --- чтение сцены ---

class _Reader:
    def __init__(self, data: bytes):
        self.b = data
        self.o = 0

    def int(self) -> int:
        (v,) = struct.unpack_from("<i", self.b, self.o)
        self.o += 4
        return v

    def uint(self) -> int:
        (v,) = struct.unpack_from("<I", self.b, self.o)
        self.o += 4
        return v

    def float(self) -> float:
        (v,) = struct.unpack_from("<f", self.b, self.o)
        self.o += 4
        return v

    def bool(self) -> bool:
        return self.int() == 1

    def bytes(self, n: int, pad: bool = True) -> bytes:
        out = self.b[self.o:self.o + n]
        self.o += 4 * ((n + 3) // 4) if pad else n
        return out

    def string(self, pad: bool = True) -> str:
        n = self.int()
        return self.bytes(n, pad).decode("utf-8", errors="replace")

    def vec3(self):
        v = struct.unpack_from("<3f", self.b, self.o)
        self.o += 12
        return v

    def color(self):
        c = tuple(self.b[self.o:self.o + 3])
        self.o += 4
        return c

    def matrix(self) -> np.ndarray:
        m = np.frombuffer(self.b, dtype="<f4", count=16, offset=self.o).reshape(4, 4).T  # по столбцам
        self.o += 64
        return m.astype(np.float64)

    def image(self) -> None:
        n = self.int()
        if n > 0:
            self.string()
            self.bytes(n)

    def tree_paths(self) -> list:
        return [(self.string(), self.color()) for _ in range(self.int())]


def scene_bytes(html: str) -> bytes:
    m = re.search(r'DentalWebGL\.m_Data = \{"data": "([A-Za-z0-9+/=]+)"\}', html)
    if not m:
        raise ValueError("это не HTML-экспорт exocad (сцены внутри нет)")
    return base64.b64decode(m.group(1))


def _lzma_packed(r: _Reader, count: int, size: int, signed: bool = False) -> np.ndarray:
    packed = r.uint()
    props = r.bytes(5, pad=False)
    raw = r.bytes(packed, pad=False)
    d = props[0]
    lc, d = d % 9, d // 9
    lp, pb = d % 5, d // 5
    dict_size = struct.unpack("<I", props[1:5])[0]
    dec = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA1, "dict_size": dict_size, "lc": lc, "lp": lp, "pb": pb}])
    n = count * size * 4
    buf = dec.decompress(raw, max_length=n)
    if len(buf) < n:
        raise ValueError("OpenCTM: данные обрезаны")
    planes = np.frombuffer(buf, dtype=np.uint8).reshape(4, size, count)  # байтовая плоскость, компонента, элемент
    vals = ((planes[0].astype(np.uint32) << 24) | (planes[1].astype(np.uint32) << 16)
            | (planes[2].astype(np.uint32) << 8) | planes[3].astype(np.uint32)).T
    if signed:
        v = vals.astype(np.int64)
        return np.where(v & 1, -((v + 1) >> 1), v >> 1)
    return vals


def _restore_indices(idx: np.ndarray) -> np.ndarray:
    t = idx.astype(np.int64).copy()
    if len(t):
        t[0, 1] += t[0, 0]
        t[0, 2] += t[0, 0]
    for i in range(1, len(t)):
        t[i, 0] += t[i - 1, 0]
        t[i, 1] += t[i - 1, 1] if t[i, 0] == t[i - 1, 0] else t[i, 0]
        t[i, 2] += t[i, 0]
    return t & 0xFFFFFFFF  # OpenCTM считает это в uint32


def decode_ctm(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Сетка OpenCTM (RAW, MG1, MG2): вершины и треугольники."""
    r = _Reader(data)
    if r.bytes(4, pad=False) != b"OCTM":
        raise ValueError("это не OpenCTM")
    r.int()
    method = r.bytes(4, pad=False)
    nv, nt = r.int(), r.int()
    r.int(); r.int(); r.int()  # noqa: E702 — карты UV, атрибутов, флаги
    r.string(pad=False)
    if method == b"RAW\x00":
        r.int()
        tris = np.frombuffer(r.bytes(nt * 12, pad=False), "<u4").reshape(nt, 3)
        r.int()
        verts = np.frombuffer(r.bytes(nv * 12, pad=False), "<f4").reshape(nv, 3)
        return verts.astype(np.float64), tris.astype(np.int64)
    if method == b"MG1\x00":
        r.int()
        tris = _restore_indices(_lzma_packed(r, nt, 3))
        r.int()
        verts = _lzma_packed(r, nv * 3, 1).reshape(nv, 3).astype("<u4").view("<f4")
        return verts.astype(np.float64), tris
    if method == b"MG2\x00":
        r.int()
        prec = r.float()
        r.float()
        lo = np.array([r.float(), r.float(), r.float()])
        hi = np.array([r.float(), r.float(), r.float()])
        div = np.array([r.int(), r.int(), r.int()])
        size = (hi - lo) / div
        r.int()
        deltas = _lzma_packed(r, nv, 3).astype(np.int64)
        r.int()
        grid = np.cumsum(_lzma_packed(r, nv, 1)[:, 0].astype(np.int64))
        r.int()
        tris = _restore_indices(_lzma_packed(r, nt, 3))
        gz = grid // (div[0] * div[1])
        rem = grid - gz * div[0] * div[1]
        gy = rem // div[0]
        gx = rem - gy * div[0]
        dx = deltas[:, 0].copy()
        for i in range(1, nv):  # сдвиги по x накапливаются внутри одной ячейки сетки
            if grid[i] == grid[i - 1]:
                dx[i] += dx[i - 1]
        verts = np.column_stack([lo[0] + gx * size[0] + prec * dx,
                                 lo[1] + gy * size[1] + prec * deltas[:, 1],
                                 lo[2] + gz * size[2] + prec * deltas[:, 2]])
        return verts, tris
    raise ValueError(f"OpenCTM: неизвестный способ сжатия {method!r}")


def _weld(verts: np.ndarray, tris: np.ndarray, decimals: int = 6):
    """Склеить совпадающие вершины (у каждого треугольника могут быть свои)."""
    unique, inverse = np.unique(np.round(verts, decimals), axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[tris]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return unique, faces[keep]


# --- объекты сцены ---

_KINDS = (  # первое совпадение; подстроки пути в дереве, строчными
    ("image", ("2d-", "2d ", ".jpg", ".png", "img_", "bild")),
    ("gingiva", ("десн", "gingiva", "zahnfleisch", " gum")),
    ("prep", ("препарир", "prep", "stumpf", "die")),
    ("waxup", ("ваксап", "wax", "анатом", "anatom")),
    ("antagonist", ("антагонист", "antagonist", "gegenkiefer")),
    ("jaw", ("челюст", "jaw", "kiefer", "скан", "scan")),
)
_CT = ("bone", "skull", "череп", "кост", "бюгельный каркас")
_LOWER = ("ниж", "lower", "unterkiefer", "mandib")
_UPPER = ("верх", "upper", "oberkiefer", "maxill")
KIND_NAMES = {"ct": "КТ", "image": "изображение", "gingiva": "десна", "prep": "культя", "waxup": "моделировка",
              "antagonist": "антагонист", "jaw": "скан челюсти", "other": "прочее"}


@dataclass
class SceneObject:
    path: list
    vertices: np.ndarray  # мм, координаты exocad
    faces: np.ndarray
    visible: bool = True
    kind: str = "other"
    jaw: str | None = None
    teeth: list = field(default_factory=list)

    @property
    def is_scan(self) -> bool:
        low = " / ".join(self.path).lower()
        return self.kind == "jaw" and ("скан" in low or "scan" in low)

    def label(self) -> str:
        """Описание без имён: тип, челюсть, зубы."""
        jaw = {"upper": ", верх", "lower": ", низ"}.get(self.jaw, "")
        teeth = f", зубы {self.teeth}" if self.teeth else ""
        return f"{KIND_NAMES[self.kind]}{jaw}{teeth}"


def classify(path: list) -> tuple[str, str | None, list]:
    """Тип объекта, челюсть и номера зубов (FDI) по пути в дереве exocad."""
    name = " / ".join(path).lower()
    if any(k in name for k in _CT):
        kind = "ct"
    else:
        kind = next((k for k, keys in _KINDS if any(key in name for key in keys)), "other")
    leaf = path[-1] if path else ""
    own = re.match(r"\s*(\d\d)(?:-(\d\d))?\s*:", leaf)  # «33: …», «24-25: …»
    if own:
        a = int(own.group(1))
        b = int(own.group(2)) if own.group(2) else a
        teeth = list(range(min(a, b), max(a, b) + 1)) if a // 10 == b // 10 else [a, b]
    else:
        teeth = [int(t) for t in re.findall(r"(?<!\d)([1-4][1-8])(?!\d)", leaf)]
    teeth = [t for t in teeth if 1 <= t // 10 <= 4 and 1 <= t % 10 <= 8]
    if any(k in name for k in _UPPER):
        jaw = "upper"
    elif any(k in name for k in _LOWER):
        jaw = "lower"
    elif teeth:
        jaw = "upper" if teeth[0] // 10 in (1, 2) else "lower"
    else:
        jaw = None
    return kind, jaw, teeth


def parse_scene(data: bytes) -> list[SceneObject]:
    r = _Reader(data)
    version = r.int()
    if version > 1:
        r.string()  # язык интерфейса
    if version > 5:
        r.bool()
        r.string()
        r.string()
    if r.string():
        raise ValueError("экспорт защищён паролем — выгрузите его из exocad без пароля")
    if version > 1:
        r.string()
    r.string()
    for _ in range(3):  # освещение
        r.float()
    r.vec3(); r.bool(); r.vec3(); r.int()  # noqa: E702
    for _ in range(4):
        r.color()
    for _ in range(3):
        r.float()
    r.image()
    for _ in range(2):  # виды
        for _ in range(r.int()):
            r.string()
            r.matrix()
    for _ in range(r.int()):  # аннотации
        if version > 3:
            r.bytes(r.int())
        r.string(); r.vec3(); r.vec3(); r.color(); r.tree_paths()  # noqa: E702
        if version > 2:
            r.bool()
    objects = []
    for _ in range(r.int()):
        r.bool(); r.bool(); r.bool()  # noqa: E702
        r.color()
        r.color()
        r.color(); r.color()  # noqa: E702
        r.float(); r.float(); r.float()  # noqa: E702
        r.color()
        r.float()
        world = r.matrix()
        ctm = r.bytes(r.int())
        r.image()
        r.float()
        paths = [p for p, _c in r.tree_paths()]
        visible = r.bool() if version > 2 else True
        if version > 4:
            r.bool(); r.bool()  # noqa: E702
        verts, tris = decode_ctm(ctm)
        verts, tris = _weld(verts @ world[:3, :3].T + world[:3, 3], tris)
        path = [p for p in paths if p]
        kind, jaw, teeth = classify(path)
        objects.append(SceneObject(path, verts, tris, visible, kind, jaw, teeth))
    return objects


@dataclass
class Scene:
    objects: list

    def summary(self) -> list[str]:
        return [o.label() for o in self.objects]

    def parts(self) -> dict:
        """Сканы обеих челюстей, кость из КТ и моделировки по зубам."""
        scans = {"upper": [], "lower": []}
        unknown, antagonists, designs = [], [], {}
        skull, mandible = [], []
        for o in self.objects:
            if o.kind == "ct":
                (mandible if o.jaw == "lower" else skull).append(o)
            elif o.is_scan and o.jaw in scans:
                scans[o.jaw].append(o)
            elif o.is_scan:
                unknown.append(o)
            elif o.kind == "antagonist":
                antagonists.append(o)
            elif o.kind == "waxup" and len(o.teeth) == 1:
                designs[o.teeth[0]] = o
        if unknown or antagonists:
            designed = [o.jaw for o in designs.values()]
            if designed:
                prep = max(set(designed), key=designed.count)
            elif unknown and antagonists:  # без моделировки — по высоте вдоль оси z сцены
                prep = "upper" if unknown[0].vertices[:, 2].mean() > antagonists[0].vertices[:, 2].mean() else "lower"
            else:
                prep = "upper"
            scans[prep] += unknown
            scans["lower" if prep == "upper" else "upper"] += antagonists

        def biggest(items):  # полная дуга ~60 мм, предпреп или участок — около 10: выбирается по охвату
            return max(items, key=lambda o: float(np.linalg.norm(np.ptp(o.vertices, axis=0)))) if items else None

        return {"upper_scan": biggest(scans["upper"]), "lower_scan": biggest(scans["lower"]),
                "skull": _merged(skull), "mandible": _merged(mandible), "designs": designs}


def _merged(items: list) -> SceneObject | None:
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    offsets = np.cumsum([0] + [len(o.vertices) for o in items[:-1]])
    return SceneObject(items[0].path, np.vstack([o.vertices for o in items]),
                       np.vstack([o.faces + k for o, k in zip(items, offsets)]), True, items[0].kind, items[0].jaw)


def load(path: str) -> Scene:
    """Прочитать HTML-экспорт exocad."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return Scene(parse_scene(scene_bytes(f.read())))


# --- динамика по сцене ---

def analyze(scene: Scene, travel: float = 6.0) -> tuple[dict, mo.MotionCase, mo.Anatomy]:
    """Протрузия и латеротрузии по сканам челюстей сцены.

    Суставы — по КТ, если в сцене есть кость нижней челюсти (мыщелки) и черепа
    (ССП по суставному бугорку, угол Беннетта — по Ханау); иначе средний
    артикулятор (треугольник Бонвилля). Горизонталь — окклюзионная плоскость.
    """
    p = scene.parts()
    upper, lower = p["upper_scan"], p["lower_scan"]
    if upper is None or lower is None:
        raise ValueError("в сцене нужны сканы обеих челюстей (или скан и антагонист)")
    up, anterior = mo.arch_axes(lower.vertices, upper.vertices.mean(0) - lower.vertices.mean(0))
    right = np.cross(anterior, up)
    incisal = mo._incisal(lower.vertices, anterior, up, mo.INCISAL_TILT_DEG)
    notes, settings = [], kin.Settings()
    co = suggest_condyles(p["mandible"].vertices, up=up, left=-right, anterior=anterior) \
        if p["mandible"] is not None else {}
    if len(co) == 2:
        hinge = (co["Co_R"] + co["Co_L"]) / 2
        R = np.array([right, anterior, up])
        anatomy = mo.Anatomy(rigid(R, -R @ hinge),
                             {"incisal": incisal, "condyle_right": co["Co_R"], "condyle_left": co["Co_L"]},
                             "КТ: мыщелки по нижней челюсти; горизонталь — окклюзионная плоскость")
    else:
        anatomy = mo.anatomy_average(lower.vertices, upper.vertices, incisal=incisal)
        if p["mandible"] is not None:
            notes.append("мыщелки на кости нижней челюсти не найдены — средний артикулятор")
    eminence = {}
    if p["skull"] is not None and len(co) == 2:
        for side in ("right", "left"):
            try:
                info = condylar_path(*eminence_profile(p["skull"].vertices, p["skull"].faces, anatomy.frame,
                                                       anatomy.points[f"condyle_{side}"]))
            except ValueError as e:
                notes.append(f"{'правый' if side == 'right' else 'левый'} бугорок: {e}")
                continue
            eminence[side] = {k: v for k, v in info.items() if k != "path"}
            setattr(settings, f"sagittal_{side}_deg", info["sagittal_deg"])
            setattr(settings, f"bennett_{side}_deg", round(hanau_bennett(info["sagittal_deg"]), 1))
            settings.sources[f"sagittal_{side}_deg"] = "КТ: суставной бугорок"
            settings.sources[f"bennett_{side}_deg"] = "формула Ханау"
    try:
        occlusion = kin.Occlusion(upper.vertices, upper.faces, lower.vertices, anatomy.frame)
    except ValueError as e:
        raise ValueError(f"сканы челюстей не в прикусе ({e}); если челюсти беззубые, ведения по сканам нет") from e
    _seat, seating = kin.seat_bite(upper.vertices, upper.faces, lower.vertices, anatomy)  # только подсказка
    recs = [kin.protrusion(anatomy, settings, travel, occlusion),
            kin.laterotrusion(anatomy, settings, "right", travel, occlusion),
            kin.laterotrusion(anatomy, settings, "left", travel, occlusion)]
    case = mo.MotionCase("exocad", recs)
    analysis = mo.analyze_case(case, anatomy)
    report = {
        "scene": scene.summary(),
        "anatomy": anatomy.source,
        "intercondylar_mm": round(float(np.linalg.norm(anatomy.points["condyle_right"]
                                                        - anatomy.points["condyle_left"])), 1),
        "eminence": eminence,
        "settings": {k: v for k, v in vars(settings).items() if k not in ("sources", "frame")},
        "sources": dict(settings.sources),
        "bite_penetration_mm": round(occlusion.bite_penetration_mm, 3),
        "bite_seating": seating,  # как посадить прикус на шарнирной оси; прикус сканов не меняется
        "guidance_deg": analysis["guidance_deg"],
        "contacts": {r.name: kin.contact_sectors(r, anatomy, occlusion) for r in recs},
        "notes": notes,
    }
    return report, case, anatomy
