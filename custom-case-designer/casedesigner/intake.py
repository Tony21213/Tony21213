"""Входной контроль кейса (порядок работ владельца, п. 9; одобрено 13.10): при получении кейса программа
проверяет, чего не хватает, — список со скриншотами, чтобы сразу отправить клинике. Отчёт — ``reports.Report``
(тот же генератор, что у паспорта кейса).

Контракт:

* Каждая проверка — ``Finding``: уровень ``ok`` / ``warn`` / ``error``, раздел («КТ», «Сканы», «Фото»), что
  проверено, пояснение и, если есть, картинка (срез КТ у края поля, скан с подсвеченной дырой).
* КТ (``ct_findings``): размер вокселя — до VOXEL_OK_MM хорошо, до VOXEL_WARN_MM — предупреждение, больше — плохо
  для совмещения по коронкам; мыщелки, порионы (уши) и орбитали — точки кейса (ручные или ALI-CBCT) должны
  стоять не ближе FOV_MARGIN_MM к краю объёма: ближе — «у края поля, могли обрезаться» (срез через точку);
  точки нет — «не найдена: возможно, не попала в поле КТ». Смазанность от движения — позже.
* Сканы (``scan_findings``): сканы обеих челюстей; скан прикуса (или челюсти уже в прикусе — решает вызывающий);
  дыры — замкнутые края сетки длиннее HOLE_MIN_MM; внешний край каждого куска сетки и обрывки — не дыры.
* Проверять после поиска ориентиров (ALI-CBCT): без точек мыщелки, уши и глазницы — «не найдены».
* Фото (``photo_findings``): с улыбкой и с ретрактором — для «Эстетики».
* Имён пациентов нет: сканы называются по челюсти, подпись кейса задаёт вызывающий.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from . import reports

VOXEL_OK_MM, VOXEL_WARN_MM = 0.3, 0.4
FOV_MARGIN_MM = 5.0
HOLE_MIN_MM = 3.0
DEBRIS_FACES = 200  # кусок сетки меньше — обрывок, не смотрится
POINTS = {"мыщелки": ("Co_R", "Co_L"), "уши (порионы)": ("Po_R", "Po_L"), "глазницы (орбитали)": ("Or_R", "Or_L")}
SIDE = {"R": "правый", "L": "левый"}
JAW = {"upper": "верхней челюсти", "lower": "нижней челюсти"}


@dataclass
class Finding:
    level: str  # ok / warn / error
    section: str  # «КТ», «Сканы», «Фото»
    title: str
    text: str
    image: Image.Image | None = None


# --- КТ ----------------------------------------------------------------------------------------------------------

def _edge_mm(volume, point) -> float:
    """Расстояние от точки (мм пациента) до ближайшей грани объёма КТ, мм (отрицательное — снаружи)."""
    idx = volume.to_index(np.asarray(point, float)[None])[0]
    n = np.array(volume.data.shape[::-1], float)  # x, y, z
    return float(np.min(np.minimum(idx, n - 1 - idx) * np.asarray(volume.spacing, float)))


def ct_slice(volume, point, size: int = 520) -> Image.Image:
    """Сагиттальный срез КТ через точку (окно по кости), точка — красным кругом; верх картинки — верх объёма
    по третьей оси индекса."""
    x, y, z = np.round(volume.to_index(np.asarray(point, float)[None])[0]).astype(int)
    data = volume.data
    x = int(np.clip(x, 0, data.shape[2] - 1))
    plane = data[:, :, x].astype(float)  # (z, y)
    lo, hi = np.percentile(plane, 1), np.percentile(plane, 99.5)
    g = np.clip((plane - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    sy, sz = float(volume.spacing[1]), float(volume.spacing[2])
    im = Image.fromarray(g[::-1]).convert("RGB")  # z — вверх
    im = im.resize((max(1, round(im.width * sy)), max(1, round(im.height * sz))))  # пропорции в мм
    k = size / max(im.size)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))))
    px, pz = y * sy * k, (data.shape[0] - 1 - z) * sz * k
    d = ImageDraw.Draw(im)
    d.ellipse((px - 12, pz - 12, px + 12, pz + 12), outline=(230, 40, 40), width=3)
    return im


def ct_findings(volume, landmarks: dict) -> list[Finding]:
    """Воксель и попадание мыщелков, ушей и глазниц в поле КТ (точки кейса — мм пациента)."""
    out = []
    voxel = float(np.max(volume.spacing))
    level = "ok" if voxel <= VOXEL_OK_MM + 1e-6 else "warn" if voxel <= VOXEL_WARN_MM + 1e-6 else "error"
    out.append(Finding(level, "КТ", "Размер вокселя", f"{voxel:.2f} мм".replace(".", ",")
                       + ("" if level == "ok" else " — крупный: коронки на КТ видны хуже, совмещение сканов грубее")))
    for name, keys in POINTS.items():
        for key in keys:
            side = SIDE[key[-1]]
            point = landmarks.get(key)
            label = f"{name[:1].upper()}{name[1:]}: {side}"
            if point is None:
                out.append(Finding("warn", "КТ", label, "точка не найдена — возможно, не попала в поле КТ"))
                continue
            edge = _edge_mm(volume, point)
            if edge < FOV_MARGIN_MM:
                out.append(Finding("error", "КТ", label,
                                   f"у края поля КТ ({max(edge, 0):.1f} мм до края) — могла обрезаться".replace(".", ","),
                                   ct_slice(volume, point)))
            else:
                out.append(Finding("ok", "КТ", label, "в поле КТ"))
    return out


# --- сканы -------------------------------------------------------------------------------------------------------

def holes(vertices, faces) -> list[np.ndarray]:
    """Дыры скана: связные куски края сетки (рёбра одной грани) длиннее HOLE_MIN_MM. У каждого связного куска
    сетки самый длинный край — его внешний край, не дыра; куски меньше DEBRIS_FACES граней (обрывки) не смотрятся.
    Каждая дыра — рёбра (k×2, индексы вершин). Пороги подобраны на синтетике — уточнить на реальных сканах."""
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v, f = np.asarray(vertices, float), np.asarray(faces)
    n = len(v)
    mesh = trimesh.Trimesh(v, f, process=False)
    edges = mesh.edges_sorted
    _k, part = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)),
                                    directed=False)
    face_part = part[f[:, 0]]
    big = {int(p) for p, c in zip(*np.unique(face_part, return_counts=True)) if c >= DEBRIS_FACES}
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    border = unique[counts == 1]
    if not len(border):
        return []
    _k, label = connected_components(coo_matrix((np.ones(len(border)), (border[:, 0], border[:, 1])), shape=(n, n)),
                                     directed=False)
    comp = label[border[:, 0]]
    length = np.linalg.norm(v[border[:, 0]] - v[border[:, 1]], axis=1)
    loops = {}  # кусок сетки → [(длина, рёбра)]
    for c in np.unique(comp):
        sel = comp == c
        owner = int(part[border[sel][0, 0]])
        if owner in big:
            loops.setdefault(owner, []).append((float(length[sel].sum()), border[sel]))
    out = []
    for items in loops.values():
        items.sort(key=lambda x: x[0], reverse=True)
        out += [(total, e) for total, e in items[1:] if total >= HOLE_MIN_MM]
    return [e for _t, e in sorted(out, key=lambda x: x[0], reverse=True)]


def scan_image(vertices, faces, highlight=(), size=(640, 480)) -> Image.Image:
    """Скан сверху по наименьшей оси разброса: точки с затенением по нормали, highlight — рёбра дыр (holes)
    красным."""
    import trimesh

    v = np.asarray(vertices, float)
    mesh = trimesh.Trimesh(v, np.asarray(faces), process=False)
    c = v.mean(axis=0)
    _u, _s, vt = np.linalg.svd(v - c, full_matrices=False)
    axes = vt  # строки: две главные оси и «вид»
    p = (v - c) @ axes.T
    w, h = size
    scale = 0.9 * min(w / max(np.ptp(p[:, 0]), 1e-6), h / max(np.ptp(p[:, 1]), 1e-6))
    xy = np.c_[w / 2 + p[:, 0] * scale, h / 2 - p[:, 1] * scale]
    shade = np.abs(np.asarray(mesh.vertex_normals) @ axes[2])
    im = Image.new("RGB", size, (245, 246, 248))
    d = ImageDraw.Draw(im)
    for i in np.argsort(p[:, 2]):
        g = int(90 + 140 * shade[i])
        d.point((float(xy[i, 0]), float(xy[i, 1])), fill=(g, g, int(g * 0.95)))
    for edges in highlight:
        for a, b in edges:
            d.line((tuple(xy[a]), tuple(xy[b])), fill=(230, 40, 40), width=4)
    return im


def scan_findings(scans: dict, has_bite: bool) -> list[Finding]:
    """Сканы: scans — {"upper"/"lower": (вершины, грани)}; has_bite — есть скан прикуса или челюсти в прикусе."""
    out = []
    for jaw in ("upper", "lower"):
        if jaw not in scans:
            out.append(Finding("error", "Сканы", f"Скан {JAW[jaw]}", "нет скана"))
            continue
        v, f = scans[jaw]
        found = holes(v, f)
        if found:
            out.append(Finding("warn", "Сканы", f"Скан {JAW[jaw]}", f"дыр в скане: {len(found)} — проверьте зубы рядом",
                               scan_image(v, f, found)))
        else:
            out.append(Finding("ok", "Сканы", f"Скан {JAW[jaw]}", "без дыр"))
    out.append(Finding("ok" if has_bite else "warn", "Сканы", "Прикус",
                       "есть" if has_bite else "нет скана прикуса — прикус возьмётся с КТ, проверьте его"))
    return out


def photo_findings(smile: bool, retractor: bool) -> list[Finding]:
    return [Finding("ok" if smile else "warn", "Фото", "Фото с улыбкой", "есть" if smile else "нет — для «Эстетики»"),
            Finding("ok" if retractor else "warn", "Фото", "Фото с ретрактором",
                    "есть" if retractor else "нет — для «Эстетики»")]


# --- отчёт -------------------------------------------------------------------------------------------------------

def report(findings: list[Finding], case_label: str) -> reports.Report:
    """Отчёт входного контроля: сначала — чего не хватает (ошибки и предупреждения), затем разделы по порядку
    КТ, Сканы, Фото; картинки — к найденному."""
    rank = {"error": 2, "warn": 1, "ok": 0}
    mark = {"error": "нет / плохо", "warn": "проверить", "ok": "в порядке"}
    sections = []
    missing = [f for f in findings if f.level != "ok"]
    if missing:
        level = "error" if any(f.level == "error" for f in missing) else "warn"
        sections.append(reports.Section("Чего не хватает", notes=[f"{f.section} · {f.title}: {f.text}" for f in missing],
                                        level=level))
    for name in ("КТ", "Сканы", "Фото"):
        part = [f for f in findings if f.section == name]
        if not part:
            continue
        worst = max(part, key=lambda f: rank[f.level]).level
        sections.append(reports.Section(name, rows=[(f.title, f"{mark[f.level]} — {f.text}") for f in part],
                                        level={"ok": "info"}.get(worst, worst)))
    figures = [reports.Figure(f"{f.section} · {f.title}", f.image) for f in findings if f.image is not None]
    return reports.Report("Входной контроль кейса", case_label, sections, figures)
