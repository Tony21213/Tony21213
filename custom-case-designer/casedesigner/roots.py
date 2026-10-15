"""Корни зубов из КТ под коронками сканов — для exocad (решение владельца: «целый зуб»: коронка со скана + корень с
КТ, ось каждого зуба; техник видит наклон корней при моделировке коронок и виниров, пригодится и ортодонтии).

Зуб из сегментации КТ (структуры teeth/tooth_<FDI>) — целиком, коронка и корень. Коронку точнее показывает скан,
поэтому из зуба КТ берётся только то, чего на скане нет, — корень. Граница — линия десны со скана: вершины зуба КТ,
лежащие на скане (ближе COVER_MM), — видимая коронка; вдоль оси зуба по кругу (SECTORS секторов) в каждом секторе
берётся нижний край видимой коронки (квантиль MARGIN_QUANTILE, чтобы десна у шейки не утащила границу вниз), и всё,
что глубже него, — корень; с запасом OVERLAP_MM вверх, чтобы между корнем и коронкой скана не было щели.

Ось зуба — от верхушки корня к режущему краю или жевательной поверхности: сначала главная ось облака вершин зуба,
затем уточнение через центр видимой коронки и центр верхушки (у многокорневых — середина верхушек).

Всё в координатах КТ (мм пациента): сканы — в соотношении КТ (положение совмещения), не в прикусе выгрузки;
перенос в прикус и в координаты exocad — у выгрузки (exocad_articulator_export)."""

from dataclasses import dataclass, field

import numpy as np

TOOTH_PREFIX = "teeth/tooth_"
COVER_MM = 0.5  # вершина зуба КТ ближе к скану — на скане видна (КТ шире настоящей поверхности до ~0,3 мм)
MIN_COVERED = 40  # меньше вершин на скане — зуба на скане нет (не прорезался, удалён, вне скана): корень не отделить
SECTORS = 16  # секторов вокруг оси зуба: линия десны выше между зубами, ниже с губы и языка
MARGIN_QUANTILE = 0.1
OVERLAP_MM = 0.3  # корень заходит под коронку скана — без щели между ними
APEX_MM = 2.0  # верхушка — вершины не дальше этого от самой глубокой точки вдоль оси
FRAGMENT_MM = 2.0  # куски сетки зуба дальше от основного — чужие (утечка метки сегментации), ближе — обломки зуба
MIN_TOOTH_MM = 8.0  # зуб КТ короче вдоль оси — сегментирован не целиком: корень не отделить
MAX_TILT_DEG = 100.0  # ось зуба против дуги (коронка смотрит в другую челюсть) — номер или сегментация под вопросом


@dataclass
class Root:
    fdi: int
    vertices: np.ndarray  # корень (сетка зуба КТ ниже линии десны), координаты КТ
    faces: np.ndarray
    apex: np.ndarray  # верхушка корня (у многокорневых — середина верхушек)
    tip: np.ndarray  # режущий край или жевательная поверхность на оси зуба
    axis: np.ndarray  # единичный: от верхушки к коронке
    root_mm: float  # длина корня вдоль оси: от верхушки до линии десны (в среднем по кругу)
    crown_mm: float  # видимая коронка вдоль оси: от линии десны до края
    covered: float  # доля вершин зуба КТ, лежащих на скане
    notes: list = field(default_factory=list)


def _basis(axis):
    a = axis / np.linalg.norm(axis)
    e1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def _circular_fill(values: np.ndarray) -> np.ndarray:
    """Пустые сектора — линейно между соседними заполненными по кругу; затем медиана по трём соседям."""
    n = len(values)
    ok = np.flatnonzero(np.isfinite(values))
    idx = np.arange(n)
    ext = np.r_[ok - n, ok, ok + n]
    filled = np.interp(idx, ext, np.r_[values[ok], values[ok], values[ok]])
    return np.median(np.c_[np.roll(filled, 1), filled, np.roll(filled, -1)], axis=1)


def main_part(vertices, faces):
    """Сетка зуба без чужих кусков: самый большой связный кусок и обломки не дальше FRAGMENT_MM от него."""
    import trimesh
    from scipy.spatial import cKDTree

    m = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, int), process=False)
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return np.asarray(m.vertices), np.asarray(m.faces)
    parts = sorted(parts, key=lambda p: len(p.vertices), reverse=True)
    tree = cKDTree(parts[0].vertices)
    keep = [parts[0]] + [p for p in parts[1:] if tree.query(p.vertices)[0].min() <= FRAGMENT_MM]
    merged = trimesh.util.concatenate(keep)
    return np.asarray(merged.vertices), np.asarray(merged.faces)


def split_tooth(vertices, faces, scan_tree, fdi: int = 0, cover_mm: float = COVER_MM,
                overlap_mm: float = OVERLAP_MM) -> tuple[Root | None, str | None]:
    """Корень одного зуба: vertices, faces — сетка зуба КТ; scan_tree — cKDTree вершин скана его челюсти (в
    координатах КТ). Возвращает (Root, None) или (None, почему корня нет)."""
    if len(vertices) < 4 or not len(faces):
        return None, "пустая сетка"
    v, f = main_part(vertices, faces)
    dist = scan_tree.query(v, distance_upper_bound=cover_mm * 4)[0]
    covered = dist < cover_mm
    if covered.sum() < MIN_COVERED:
        return None, "коронки нет на скане"
    c = v.mean(0)
    axis = np.linalg.svd(v - c, full_matrices=False)[2][0]
    if (v[covered] - c).mean(0) @ axis < 0:
        axis = -axis
    for _ in range(2):  # уточнение: через центр видимой коронки и центр верхушки
        h = (v - c) @ axis
        apex = v[h < h.min() + APEX_MM].mean(0)
        crown = v[covered].mean(0)
        axis = (crown - apex) / np.linalg.norm(crown - apex)
    a, e1, e2 = _basis(axis)
    d = v - c
    h = d @ a
    if np.ptp(h) < MIN_TOOTH_MM:
        return None, "зуб КТ сегментирован не целиком"
    theta = np.arctan2(d @ e2, d @ e1)
    sector = np.minimum(((theta + np.pi) / (2 * np.pi) * SECTORS).astype(int), SECTORS - 1)
    margin = np.full(SECTORS, np.nan)
    for s in range(SECTORS):
        hs = h[covered & (sector == s)]
        if len(hs) >= 3:
            margin[s] = np.quantile(hs, MARGIN_QUANTILE)
    if np.isfinite(margin).sum() < 3:
        return None, "коронка на скане — с одной стороны зуба"
    margin = _circular_fill(margin)
    # граница у вершины — между центрами соседних секторов
    pos = (theta + np.pi) / (2 * np.pi) * SECTORS - 0.5
    lo = np.floor(pos).astype(int)
    w = pos - lo
    m_v = (1 - w) * margin[lo % SECTORS] + w * margin[(lo + 1) % SECTORS]
    is_root = h < m_v + overlap_mm
    keep = is_root[f].all(axis=1)
    if not keep.any():
        return None, "корня ниже десны нет"
    used, inverse = np.unique(f[keep].ravel(), return_inverse=True)
    apex = v[h < h.min() + APEX_MM].mean(0)
    apex = c + ((apex - c) @ a) * a  # на оси
    tip = c + h.max() * a
    m_mean = float(np.mean(margin))
    notes = []
    if m_mean - h.min() < 5.0:
        notes.append("корень короче 5 мм — проверьте сегментацию зуба или совмещение скана")
    return Root(fdi, v[used], inverse.reshape(-1, 3), apex, tip, a, round(m_mean - float(h.min()), 1),
                round(float(h.max()) - m_mean, 1), round(float(covered.mean()), 3), notes), None


def jaw_of_fdi(fdi: int) -> str:
    return "upper" if fdi < 30 else "lower"


def split_teeth(structures: dict, scans: dict, cover_mm: float = COVER_MM) -> tuple[dict, list]:
    """Корни всех зубов КТ: structures — {ключ: сетка (vertices, faces)} в координатах КТ (берутся teeth/tooth_*),
    scans — {"upper" | "lower": вершины скана челюсти в координатах КТ}. Возвращает ({FDI: Root}, заметки)."""
    from scipy.spatial import cKDTree

    trees = {jaw: cKDTree(np.asarray(v, float)) for jaw, v in scans.items() if v is not None and len(v)}
    # коронки нижних зубов смотрят к верхнему скану, верхних — к нижнему
    up = None
    if "upper" in trees and "lower" in trees:
        up = np.asarray(scans["upper"], float).mean(0) - np.asarray(scans["lower"], float).mean(0)
        up /= np.linalg.norm(up)
    out, skipped = {}, {}
    for key, mesh in structures.items():
        if not key.startswith(TOOTH_PREFIX):
            continue
        try:
            fdi = int(key[len(TOOTH_PREFIX):])
        except ValueError:
            continue
        tree = trees.get(jaw_of_fdi(fdi))
        if tree is None:
            continue
        root, why = split_tooth(mesh.vertices, mesh.faces, tree, fdi, cover_mm)
        if root is not None and up is not None:
            expected = up if jaw_of_fdi(fdi) == "lower" else -up
            if np.degrees(np.arccos(np.clip(root.axis @ expected, -1, 1))) > MAX_TILT_DEG:
                root, why = None, "ось зуба против дуги — проверьте номер зуба в сегментации"
        if root is None:
            skipped.setdefault(why, []).append(fdi)
        else:
            out[fdi] = root
    notes = [f"корни не отделены ({why}): {', '.join(map(str, sorted(fdis)))}" for why, fdis in skipped.items()]
    return dict(sorted(out.items())), notes


def jaw_mesh(roots: dict, jaw: str):
    """Корни челюсти одной сеткой (vertices, faces) или None."""
    parts = [r for r in roots.values() if jaw_of_fdi(r.fdi) == jaw]
    if not parts:
        return None
    vs, fs, n = [], [], 0
    for r in parts:
        vs.append(r.vertices)
        fs.append(r.faces + n)
        n += len(r.vertices)
    return np.vstack(vs), np.vstack(fs)


def axes_mesh(roots: dict, jaw: str, radius: float = 0.3, beyond_mm: float = 3.0):
    """Оси зубов челюсти — тонкие цилиндры от верхушки до края коронки и дальше на beyond_mm (над коронкой
    видно направление), одной сеткой (vertices, faces) или None."""
    import trimesh

    parts = []
    for r in roots.values():
        if jaw_of_fdi(r.fdi) != jaw:
            continue
        end = r.tip + beyond_mm * r.axis
        parts.append(trimesh.creation.cylinder(radius=radius, segment=np.array([r.apex, end]), sections=10))
    if not parts:
        return None
    m = trimesh.util.concatenate(parts)
    return np.asarray(m.vertices), np.asarray(m.faces)


def summary(roots: dict) -> list[dict]:
    """По зубам для сводки и паспорта: FDI, корень и коронка вдоль оси, мм; наклон оси — у выгрузки (система
    монтажа)."""
    return [{"fdi": r.fdi, "root_mm": r.root_mm, "crown_mm": r.crown_mm, "covered": r.covered, "notes": r.notes}
            for r in roots.values()]
