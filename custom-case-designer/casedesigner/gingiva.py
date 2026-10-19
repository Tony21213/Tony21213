"""Толщина десны над костью и корнями по КТ — цветной картой на скане (решение владельца: под промежуточные части
мостов, коронки на имплантах, биотип и риск рецессии; в exocad — цветным PLY, как карта отклонений).

Для каждой вершины скана, которая не лежит на коронке КТ, — расстояние до ближайшей твёрдой ткани КТ: кости
челюсти или корня зуба (под десной у зубов — корень; тонкую кость над корнями сегментация часто теряет, тогда
до корня — это десна вместе с ней). Поверхности кости и зубов берутся точками с шагом SAMPLE_MM (ошибка
расстояния — около десятой мм) только рядом со сканом. Вершина скана внутри кости или корня (скан лёг глубже
твёрдой ткани — совмещение или сегментация) — толщина 0 и отдельный цвет.

Скан и структуры — в координатах КТ (скан — в положении совмещения, не в прикусе)."""

from dataclasses import dataclass, field

import numpy as np

SAMPLE_MM = 0.25  # шаг точек на поверхности кости и зубов
CROWN_MM = 0.35  # вершина скана у самой поверхности зуба — это коронка на скане, а не десна
FAR_MM = 6.0  # дальше — не десна над костью (губа, щека, дно преддверия): без цвета
MARGIN_MM = 8.0  # кость и зубы берутся в рамке скана с таким запасом
THIN_MM = 1.0  # тонкий биотип: десна над корнем тоньше 1 мм — риск рецессии
# Цвета (RGBA) по толщине, мм: (до, цвет, подпись)
BINS = [(THIN_MM, (220, 60, 50, 255), "< 1 мм — тонкая"), (2.0, (240, 165, 40, 255), "1–2 мм"),
        (3.0, (80, 180, 90, 255), "2–3 мм"), (FAR_MM, (70, 130, 210, 255), "3–6 мм — толстая")]
INSIDE = (170, 40, 140, 255)  # скан глубже кости или корня
NONE = (170, 170, 170, 255)  # коронка, далеко от кости или нет данных

BONE, ROOT = 1, 2  # что под десной


@dataclass
class GumMap:
    thickness: np.ndarray  # мм на вершину скана; nan — не десна над костью или корнем (коронка, далеко)
    under: np.ndarray  # 0 — ничего, BONE — кость, ROOT — корень зуба
    tooth: np.ndarray  # номер зуба FDI под десной (0 — нет или кость)
    inside: np.ndarray  # вершина скана глубже кости или корня
    notes: list = field(default_factory=list)

    def colors(self) -> np.ndarray:
        """RGBA на вершину: по BINS, внутри твёрдой ткани — INSIDE, без толщины — NONE."""
        out = np.tile(np.array(NONE, np.uint8), (len(self.thickness), 1))
        t = self.thickness
        lo = 0.0
        for hi, color, _label in BINS:
            out[np.isfinite(t) & (t >= lo) & (t < hi)] = color
            lo = hi
        out[self.inside] = INSIDE
        return out

    def by_tooth(self) -> list[dict]:
        """Десна над корнем каждого зуба: медиана толщины, мм, и биотип (тонкий — медиана меньше THIN_MM). Вершины
        глубже корня (совмещение или сегментация) в медиану не входят — их доля отдельно; больше половины — биотип
        не определяется (thin = None)."""
        out = []
        for fdi in sorted(set(self.tooth[self.tooth > 0].tolist())):
            sel = (self.tooth == fdi) & np.isfinite(self.thickness)
            t = self.thickness[sel & ~self.inside]
            inside = float(self.inside[sel].mean()) if sel.any() else 0.0
            if sel.sum() < 20:
                continue
            med = float(np.median(t)) if len(t) else None
            ok = med is not None and inside <= 0.5
            out.append({"fdi": int(fdi), "median_mm": None if med is None else round(med, 2),
                        "thin": (med < THIN_MM) if ok else None,
                        "thin_share": round(float((t < THIN_MM).mean()), 2) if len(t) else None,
                        "inside_share": round(inside, 2)})
        return out

    def summary(self) -> dict:
        t = self.thickness[np.isfinite(self.thickness)]
        shares = {}
        lo = 0.0
        for hi, _color, label in BINS:
            shares[label] = round(float(((t >= lo) & (t < hi)).mean()), 3) if len(t) else 0.0
            lo = hi
        return {"vertices": int(len(t)), "median_mm": round(float(np.median(t)), 2) if len(t) else None,
                "shares": shares, "inside_share": round(float(self.inside.mean()), 4), "teeth": self.by_tooth(),
                "notes": list(self.notes)}


def _sample(meshes: list, lo: np.ndarray, hi: np.ndarray, step: float):
    """Точки на поверхностях в рамке [lo, hi] с шагом около step: (точки, нормали граней, номер сетки)."""
    import trimesh

    pts, normals, owner = [], [], []
    for k, mesh in enumerate(meshes):
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.faces, np.int64)
        if not len(f):
            continue
        tri = v[f]
        keep = ((tri.min(axis=1) <= hi) & (tri.max(axis=1) >= lo)).all(axis=1)  # рамка грани пересекает рамку
        if not keep.any():
            continue
        # наружу или внутрь смотрят нормали сетки — по всей сетке: грани от центра или к центру
        whole = trimesh.Trimesh(v, f, process=False)
        out = np.einsum("ij,ij->i", whole.triangles_center - v.mean(0), whole.face_normals) @ whole.area_faces
        part = trimesh.Trimesh(v, f[keep], process=False)
        count = int(min(max(part.area / step ** 2, 100), 3_000_000))
        p, face = trimesh.sample.sample_surface(part, count, seed=0)
        pts.append(np.asarray(p))
        normals.append(np.asarray(part.face_normals)[face] * (1.0 if out >= 0 else -1.0))
        owner.append(np.full(len(p), k))
    if not pts:
        return None, None, None
    return np.vstack(pts), np.vstack(normals), np.concatenate(owner)


def thickness(scan_vertices, crown_mask, bones: list, teeth: dict, step: float = SAMPLE_MM) -> GumMap:
    """Толщина десны на скане челюсти: scan_vertices — вершины скана в координатах КТ; crown_mask — вершины на
    коронках КТ (Registration.deviation не nan); bones — сетки кости этой челюсти; teeth — {FDI: сетка зуба} (0 —
    зубы одной сеткой, без номеров)."""
    from scipy.spatial import cKDTree

    v = np.asarray(scan_vertices, float)
    n = len(v)
    empty = GumMap(np.full(n, np.nan), np.zeros(n, np.int8), np.zeros(n, np.int16), np.zeros(n, bool))
    crown = np.zeros(n, bool) if crown_mask is None else np.asarray(crown_mask, bool)
    lo, hi = v.min(0) - MARGIN_MM, v.max(0) + MARGIN_MM
    tooth_ids = list(teeth)
    pts, normals, owner = _sample(list(bones) + [teeth[k] for k in tooth_ids], lo, hi, step)
    if pts is None:
        empty.notes.append("нет кости и зубов КТ рядом со сканом — толщину десны не посчитать")
        return empty
    kind = np.where(owner < len(bones), BONE, ROOT).astype(np.int8)
    fdi = np.zeros(len(owner), np.int16)
    is_tooth = owner >= len(bones)
    fdi[is_tooth] = np.asarray(tooth_ids, np.int16)[owner[is_tooth] - len(bones)]
    gum = ~crown
    d, idx = cKDTree(pts).query(v[gum], distance_upper_bound=FAR_MM)
    found = np.isfinite(d)
    idx = np.where(found, idx, 0)
    signed = np.einsum("ij,ij->i", v[gum] - pts[idx], normals[idx])
    # точнее точек выборки: до плоскости грани ближайшей точки, если вершина почти над ней (не в стороне)
    lateral = np.sqrt(np.maximum(np.where(found, d, 0.0) ** 2 - signed ** 2, 0.0))
    d = np.where(found & (lateral < 2 * step), np.abs(signed), d)
    under = np.where(found, kind[idx], 0).astype(np.int8)
    tooth = np.where(found, fdi[idx], 0).astype(np.int16)
    t = np.where(found, d, np.nan)
    inside = found & (signed < -0.05)
    t[inside] = 0.0
    # у самой поверхности зуба снаружи — коронка на скане (не легла на коронку КТ: контакты, пломбы), не десна
    on_crown = found & (under == ROOT) & ~inside & (d < CROWN_MM)
    t[on_crown], under[on_crown], tooth[on_crown] = np.nan, 0, 0
    out = GumMap(np.full(n, np.nan), np.zeros(n, np.int8), np.zeros(n, np.int16), np.zeros(n, bool))
    out.thickness[gum], out.under[gum], out.tooth[gum], out.inside[gum] = t, under, tooth, inside
    if out.inside.mean() > 0.02:
        out.notes.append(f"{out.inside.mean():.0%} вершин скана — глубже кости или корня: проверьте совмещение скана "
                         "и сегментацию кости")
    return out
