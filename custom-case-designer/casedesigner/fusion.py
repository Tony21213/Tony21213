"""Совмещение сканов челюстей с КТ и экспорт всего в единых координатах."""

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .quality import Segment, check_scan
from .register import Target, apply, edge_offset, fit_score, icp, kabsch, occlusal_init
from .scan_teeth import crown_candidates
from .structures import jaw_of
from .segment import Mesh
from .surface import Surface, extract_surface
from .teeth import OCCLUSAL, Levels, crown_surface, split_jaws

JAWS = ("upper", "lower")
# Точка скана считается лежащей на коронке, если до неё ближе этого расстояния.
MATCH_MM = 0.5
FINE_SCHEDULE = ((1.0, 15), (0.5, 15), (0.3, 20), (0.2, 20))
# С сегментацией коронки КТ — только у зубов: дальше этого от поверхности зуба — кость или артефакт.
TEETH_NEAR_MM = 1.0
# Опора начального поиска по сегментации — точки на поверхности зубов: мм² на точку и пределы их числа.
TEETH_MM2_PER_POINT = 0.05
TEETH_POINTS = (20_000, 400_000)


@dataclass
class Scan:
    name: str
    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def load(cls, path: str) -> "Scan":
        mesh = trimesh.load_mesh(path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        name = os.path.splitext(os.path.basename(path))[0]
        return cls(name, np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64))

    @property
    def normals(self) -> np.ndarray:
        return np.asarray(trimesh.Trimesh(self.vertices, self.faces, process=False).vertex_normals)

    @property
    def crowns(self) -> np.ndarray:
        """Маска вершин, которые могут быть коронками (scan_teeth.crown_candidates)."""
        if getattr(self, "_crowns", None) is None:
            self._crowns = crown_candidates(self.vertices, self.normals)
        return self._crowns


@dataclass
class Registration:
    scan: Scan
    jaw: str
    transform: np.ndarray  # координаты скана → мм пациента КТ (DICOM)
    deviation: np.ndarray  # отклонение вершин от коронок КТ, мм: > 0 — снаружи; nan — не на коронке
    stats: dict = field(default_factory=dict)
    edge_shift: float = 0.0  # сдвиг границы эмали на КТ наружу, мм, при котором скан лёг именно так
    segments: list[Segment] = field(default_factory=list)  # участки вдоль дуги (quality.check_scan)
    warnings: list[str] = field(default_factory=list)  # что показать пользователю
    # Скан → КТ в прикусе со скана прикуса (врача), если сканы челюстей выгружены не в прикусе (place_bite).
    bite: np.ndarray | None = None


class CaseCT:
    """КТ кейса с подготовленными коронками обеих челюстей.

    edge_prior, prior_weight — выученный для аппарата сдвиг границы эмали
    (мм наружу) и его вес (см. learning.AlignmentMemory.prior). Сдвиг
    уточняется в каждом совмещении, а выученное значение его стабилизирует.
    """

    def __init__(self, vol, edge_prior: float = 0.0, prior_weight: float = 0.0):
        self.vol = vol
        self.edge_prior = float(edge_prior)
        self.prior_weight = float(prior_weight)
        self.levels = Levels.of(vol)
        crowns = crown_surface(vol, self.levels, step=2)
        if len(crowns.points) < 100:
            raise ValueError("в КТ не найдены коронки зубов")
        upper, lower = split_jaws(crowns)
        self.coarse = {"upper": Target(upper.points, upper.normals), "lower": Target(lower.points, lower.normals)}
        self.guided: set[str] = set()  # челюсти, у которых опора — зубы из сегментации (use_teeth)

    def use_teeth(self, meshes: dict) -> None:
        """Зубы из сегментации — опора совмещения: {"upper": Mesh, "lower": Mesh}.

        По одним порогам плотности «коронками» на КЛКТ с большим полем
        оказывается и тонкая кость у воздуха (носовые раковины, стенки пазух):
        начальный поиск по ним сажает скан со сдвигом вдоль дуги, а метрики
        отклонения этого не видят (на реальном кейсе — 4 мм). С сегментацией
        выбор челюсти и начальный поиск идут по поверхности её зубов, а точная
        подгонка — по границе эмали только у этих зубов.
        """
        for jaw, mesh in meshes.items():
            if jaw not in JAWS or mesh is None or not len(mesh.faces):
                continue
            teeth = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
            count = int(np.clip(teeth.area / TEETH_MM2_PER_POINT, *TEETH_POINTS))
            points, face = trimesh.sample.sample_surface(teeth, count, seed=0)
            outward = 1.0 if teeth.volume >= 0 else -1.0  # нормали — наружу, как у коронок по порогам
            self.coarse[jaw] = Target(points, outward * teeth.face_normals[face])
            self.guided.add(jaw)

    def fine_crowns(self, points: np.ndarray, margin: float = 3.0) -> Target:
        roi = (points.min(axis=0) - margin, points.max(axis=0) + margin)
        crowns = crown_surface(self.vol, self.levels, roi=roi, refine=True)
        if self.guided and len(crowns.points):
            near = np.min([self.coarse[j].tree.query(crowns.points)[0] for j in self.guided], axis=0)
            crowns = Surface(crowns.points[near <= TEETH_NEAR_MM], crowns.normals[near <= TEETH_NEAR_MM])
        if len(crowns.points) < 100:
            raise ValueError("рядом со сканом в КТ нет коронок — проверьте положение скана")
        return Target(crowns.points, crowns.normals)

    def _nearest_jaw(self, scan: Scan, T: np.ndarray, jaws=JAWS) -> str:
        scores = {j: fit_score(scan.vertices[scan.crowns], self.coarse[j], T, tol=1.5) for j in jaws}
        return max(scores, key=scores.get)

    def register(self, scan: Scan, jaw: str | None = None, pairs=None, start: np.ndarray | None = None) -> Registration:
        """Совмещает скан с КТ по коронкам зубов.

        jaw — "upper"/"lower", если известно; иначе пробуются обе челюсти.
        pairs — (точки на скане, те же точки в КТ), минимум 3: начальное
        положение по ним вместо автоматического поиска.
        start — матрица «скан → КТ», с которой начать уточнение (например,
        положение, которое пользователь поправил вручную).
        """
        if jaw is not None and jaw not in JAWS:
            raise ValueError(f"челюсть должна быть upper или lower, а не {jaw!r}")
        jaws = (jaw,) if jaw else JAWS
        crowns = scan.vertices[scan.crowns]
        if start is not None:
            T0 = np.asarray(start, float)
            jaw = jaw or self._nearest_jaw(scan, T0)
        elif pairs is not None:
            T0 = kabsch(*pairs)
            jaw = self._nearest_jaw(scan, T0, jaws)
        else:
            normals = scan.normals[scan.crowns]
            found = {j: occlusal_init(crowns, normals, self.coarse[j], OCCLUSAL[j]) for j in jaws}
            jaw = max(found, key=lambda j: found[j][1])
            T0 = found[jaw][0]

        # Совмещаются только коронки: кандидаты на скане и коронки на КТ.
        target = self.fine_crowns(apply(T0, crowns))
        T, _used = icp(crowns, target, T0, schedule=FINE_SCHEDULE, keep=0.9)
        # Последний шаг: положение вместе со сдвигом границы эмали (см. register.edge_offset).
        shift, T = edge_offset(crowns, target, T, self.edge_prior, self.prior_weight)
        return self._result(scan, jaw, T, target, shift)

    def evaluate(self, scan: Scan, transform: np.ndarray, jaw: str | None = None) -> Registration:
        """Точность положения как есть, без уточнения: для ручной коррекции.

        Положение считается верным, поэтому сдвиг границы эмали — тот, что
        объясняет именно его: выученный сдвиг плюс оставшееся среднее отклонение.
        """
        T = np.asarray(transform, float)
        jaw = jaw or self._nearest_jaw(scan, T)
        target = self.fine_crowns(apply(T, scan.vertices[scan.crowns]))
        reg = self._result(scan, jaw, T, target, self.edge_prior)
        reg.edge_shift = self.edge_prior - reg.stats.get("signed_mean_mm", 0.0)
        return reg

    def _result(self, scan, jaw, T, target, shift) -> Registration:
        """Отклонения скана от коронок с учётом сдвига границы эмали."""
        corrected = Target(target.points - shift * target.normals, target.normals)
        deviation = surface_deviation(apply(T, scan.vertices), corrected)
        # Участки дуги проверяются только по коронкам: нёбо и глубокая десна их не портят.
        segments, warnings = check_scan(scan.vertices[scan.crowns], T, deviation[scan.crowns], corrected, jaw)
        return Registration(scan, jaw, T, deviation, deviation_stats(deviation), float(shift), segments, warnings)

    def surfaces(self, step: int = 1) -> dict[str, Mesh]:
        """Поверхности КТ по порогам плотности: зубы и кость — когда нет моделей сегментации."""
        teeth = extract_surface(self.vol, self.levels.dense, step=step)
        bone = extract_surface(self.vol, self.levels.hard, step=max(step, 2))
        return {"ct_teeth": Mesh(teeth.points, teeth.faces), "ct_bone": Mesh(bone.points, bone.faces)}


def surface_deviation(points: np.ndarray, target: Target) -> np.ndarray:
    """Расстояние со знаком от каждой точки до поверхности коронок: > 0 — точка снаружи.

    Считается до касательной плоскости ближайшей точки коронки; дальше
    MATCH_MM — nan (точка не на коронке: десна, артефакт).
    """
    dist, j = target.tree.query(points, distance_upper_bound=MATCH_MM)
    out = np.full(len(points), np.nan)
    ok = np.isfinite(dist)
    out[ok] = np.einsum("ij,ij->i", points[ok] - target.points[j[ok]], target.normals[j[ok]])
    return out


def deviation_stats(deviation: np.ndarray) -> dict:
    signed = deviation[np.isfinite(deviation)]
    d = np.abs(signed)
    if not len(d):
        return {"matched_fraction": 0.0}
    return {
        "matched_fraction": round(float(len(d) / len(deviation)), 3),
        # Среднее со знаком: систематический сдвиг скана наружу (+) или внутрь (−) границы эмали.
        "signed_mean_mm": round(float(signed.mean()), 4),
        "mean_mm": round(float(d.mean()), 4),
        "rms_mm": round(float(np.sqrt((d ** 2).mean())), 4),
        "p90_mm": round(float(np.quantile(d, 0.9)), 4),
        "max_mm": round(float(d.max()), 4),
        "within_0_1_mm": round(float((d <= 0.1).mean()), 3),
        "within_0_2_mm": round(float((d <= 0.2).mean()), 3),
    }


def deviation_colors(deviation: np.ndarray) -> np.ndarray:
    """Цвета карты отклонений: зелёный ≤ 0.1 мм, жёлтый ≤ 0.2, красный больше, серый — не на коронке."""
    colors = np.tile(np.array([170, 170, 170, 255], np.uint8), (len(deviation), 1))
    d = np.nan_to_num(np.abs(deviation), nan=-1)
    colors[(d >= 0) & (d <= 0.1)] = (40, 170, 70, 255)
    colors[(d > 0.1) & (d <= 0.2)] = (230, 190, 30, 255)
    colors[d > 0.2] = (210, 50, 40, 255)
    return colors


# Сканы одной сессии сканера стоят в прикусе: столько точек коронок верхнего скана
# должно оказаться не дальше CONTACT_MM от нижнего, чтобы считать их сомкнутыми.
CONTACT_MM = 1.5
CONTACT_SHARE = 0.003
# Часть сетки целиком относится к одной челюсти, если к ней ближе такая доля вершин.
JAW_MAJORITY = 0.85


def in_occlusion(upper: Scan, lower: Scan) -> bool:
    """Сканы в общих координатах и сомкнуты (прикус со сканера)?

    Сканы одной сессии сканера лежат в прикусе: бугры верхних зубов касаются
    нижних. Сканы, выгруженные по отдельности, стоят где попало и не касаются.
    """
    from scipy.spatial import cKDTree

    a, b = upper.vertices[upper.crowns], lower.vertices[lower.crowns]
    dist, _ = cKDTree(b).query(a, distance_upper_bound=CONTACT_MM)
    return bool(np.isfinite(dist).mean() >= CONTACT_SHARE)


# Скан прикуса (щёчный, «bite»; у выгрузок сканера для exocad — TotalJaw) ставится на сканы
# челюстей, с КТ не совмещается и задаёт прикус, снятый врачом.
BITE = "bite"
BITE_NAME = re.compile(r"bite|buccal|occlu|totaljaw|прикус|вестиб", re.IGNORECASE)
SHARED_MM = 0.3  # вершина скана прикуса лежит на скане челюсти, если ближе этого
SHARED_SHARE = 0.15  # столько вершин на скане челюсти — значит, у них общие координаты сканера
OFF_UPPER_MM = 0.5  # точки скана прикуса дальше этого от верхнего скана — нижние зубы
BITE_SCHEDULE = ((6.0, 15), (3.0, 20), (1.5, 20), (0.8, 20), (0.4, 20), (0.25, 20))
BITE_POINTS = 4000
OWN_FIT = 0.3  # подогнанный сам скан прикуса: столько точек должно лечь на скан челюсти
OWN_MATCHED = 0.6  # скан прикуса в своих координатах принимается, если лёг на челюсти не хуже
MAX_BITE_SHIFT_MM = 8.0  # прикус по сканам прикуса дальше этого от прикуса на КТ — ошибка


def is_bite_name(name: str) -> bool:
    """Скан прикуса по имени файла."""
    return bool(BITE_NAME.search(name or ""))


# Челюсть по имени файла (exocad, 3Shape, Medit и др.): нужна без КТ, когда совмещать не с чем.
_LETTER = r"[^\W\d_]"
UPPER_NAME = re.compile(rf"upper|maxill|oberkiefer|верх|(?<!{_LETTER})(?:ok|вч)(?!{_LETTER})", re.IGNORECASE)
LOWER_NAME = re.compile(rf"lower|mandib|unterkiefer|нижн|низ|(?<!{_LETTER})(?:uk|нч)(?!{_LETTER})", re.IGNORECASE)


def jaw_by_name(name: str) -> str | None:
    """upper или lower по имени файла скана; None — по имени не понять."""
    up, lo = bool(UPPER_NAME.search(name or "")), bool(LOWER_NAME.search(name or ""))
    return "upper" if up and not lo else "lower" if lo and not up else None


def shared_share(scan: Scan, other: Scan, T: np.ndarray | None = None) -> float:
    """Доля вершин scan (T — куда их перенести; без T — как в файле), лежащих на other ближе SHARED_MM."""
    from scipy.spatial import cKDTree

    pts = scan.vertices if T is None else apply(T, scan.vertices)
    dist, _ = cKDTree(other.vertices).query(pts, distance_upper_bound=SHARED_MM)
    return float(np.isfinite(dist).mean())


def _surface(scan: Scan, T: np.ndarray) -> Target:
    """Поверхность скана, перенесённая матрицей T (точки и нормали)."""
    return Target(apply(T, scan.vertices), scan.normals @ T[:3, :3].T)


def _sample(points: np.ndarray, n: int = BITE_POINTS) -> np.ndarray:
    pick = np.random.default_rng(0).choice(len(points), min(len(points), n), replace=False)
    return points[pick]


@dataclass
class BitePlacement:
    regs: list  # по сканам прикуса: Registration или None — поставить не удалось
    lower: np.ndarray | None  # нижний скан → КТ в прикусе со сканов прикуса; None — прикус со сканера или КТ
    failed: dict  # номер скана прикуса → почему не поставлен


def place_bites(bites: list[Scan], upper: Registration | None, lower: Registration | None,
                ct: "CaseCT | None" = None, on_ct: bool = True) -> BitePlacement:
    """Сканы прикуса — на сканы челюстей (не на КТ), и прикус врача по ним.

    transform скана прикуса — скан прикуса → КТ в системе верхнего скана.

    * Скан прикуса в общих координатах со сканами челюстей (выгрузка сканера, в том
      числе для exocad) — стоит вместе с ними, точно.
    * В своих координатах — начальное положение по КТ (или по нижнему скану, если у
      них общие координаты), затем подгонка к верхнему скану по поверхности. Лёг
      плохо — не ставится (небольшой щёчный участок по КТ не найти; надёжно — когда
      сканы выгружены одной сессией сканера).
    * Сканы челюстей не в прикусе — нижний скан подгоняется сразу ко всем нижним
      зубам сканов прикуса (обычно справа и слева): прикус врача, а не прикус на КТ.
      Слишком далёкий от прикуса на КТ или плохо легший прикус не принимается.

    on_ct=False — без КТ: положения челюстей — координаты их файлов, прикуса КТ нет.
    """
    from scipy.spatial import cKDTree

    if upper is None and lower is None:
        raise ValueError("скан прикуса ставится на сканы челюстей: сначала добавьте и совместите их")
    scanner_bite = upper is not None and lower is not None and (
        in_occlusion(upper.scan, lower.scan) or in_occlusion(lower.scan, upper.scan))
    placed, failed, lower_bite, best, own = {}, {}, None, -1.0, set()
    for i, bite in enumerate(bites):
        on_upper = shared_share(bite, upper.scan) if upper else 0.0
        on_lower = shared_share(bite, lower.scan) if lower else 0.0
        if upper is not None and on_upper >= SHARED_SHARE:
            placed[i] = (upper.transform, ["Скан прикуса в координатах сканов челюстей — стоит вместе с ними; "
                                           "с КТ не совмещается."])
            continue
        if lower is not None and on_lower >= SHARED_SHARE and (upper is None or scanner_bite):
            placed[i] = ((upper or lower).transform, ["Скан прикуса в координатах сканов челюстей — стоит вместе "
                                                      "с ними; с КТ не совмещается."])
            continue
        host = upper or lower
        with_lower = lower is not None and on_lower >= SHARED_SHARE
        start = lower.transform if with_lower else (ct.register(bite).transform if ct is not None else host.transform)
        points = _sample(bite.vertices)
        T, _ = icp(points, _surface(host.scan, host.transform), start, schedule=BITE_SCHEDULE, keep=0.5)
        host_tree = cKDTree(apply(host.transform, host.scan.vertices))
        fit = float(np.isfinite(host_tree.query(apply(T, points), distance_upper_bound=SHARED_MM)[0]).mean())
        if fit < OWN_FIT:
            failed[i] = ("Скан прикуса в своих координатах, и поставить его на сканы челюстей не удалось. "
                         "Выгрузите сканы челюстей и прикуса из одной сессии сканера (в общих координатах).")
            continue
        placed[i] = (T, [f"Скан прикуса в своих координатах — совмещён с {'верхним' if host is upper else 'нижним'} "
                         "сканом по поверхности; с КТ не совмещается."])
        own.add(i)
        if upper is not None and with_lower and fit > best:
            lower_bite, best = T, fit  # у нижнего скана и этого скана прикуса общие координаты

    notes = []
    fallback = "прикус взят с КТ" if on_ct else "нижний скан стоит как в файле"
    if upper is not None and lower is not None and not scanner_bite and placed:
        crowns = lower.scan.vertices[lower.scan.crowns]
        if lower_bite is None:
            on_upper_tree = cKDTree(apply(upper.transform, upper.scan.vertices))
            teeth = []
            for i, (T, _n) in placed.items():  # нижние зубы на сканах прикуса — то, что не лежит на верхнем скане
                pts = apply(T, bites[i].vertices)
                teeth.append(pts[on_upper_tree.query(pts)[0] > OFF_UPPER_MM])
            teeth = np.vstack(teeth)
            if len(teeth) >= 100:
                sample = _sample(teeth)
                target = _surface(lower.scan, lower.transform)
                X, _ = icp(sample, target, np.eye(4), schedule=BITE_SCHEDULE, keep=0.8)
                lies = float(np.isfinite(target.tree.query(apply(X, sample), distance_upper_bound=SHARED_MM)[0]).mean())
                lower_bite = np.linalg.inv(X) @ lower.transform if lies >= 0.3 else None
                if lower_bite is None:
                    notes.append(f"Нижний скан не лёг на нижние зубы сканов прикуса — {fallback}.")
            else:
                notes.append(f"На сканах прикуса не нашлось нижних зубов — {fallback}.")
        if lower_bite is not None and not on_ct:
            notes.append("Нижний скан поставлен в прикус по сканам прикуса.")
        elif lower_bite is not None:
            moved = float(np.linalg.norm(apply(lower_bite, crowns) - apply(lower.transform, crowns), axis=1).mean())
            if moved > MAX_BITE_SHIFT_MM:
                notes.append(f"Прикус по сканам прикуса на {moved:.0f} мм дальше прикуса на КТ — не похоже на правду, "
                             "прикус взят с КТ. Проверьте сканы прикуса.")
                lower_bite = None
            else:
                notes.append(f"Нижний скан поставлен в прикус по сканам прикуса (от прикуса на КТ — {moved:.1f} мм).")

    # Насколько каждый скан прикуса лёг на сканы челюстей в итоговом прикусе.
    surfaces = []
    if upper is not None:
        surfaces.append(apply(upper.transform, upper.scan.vertices))
    if lower is not None:
        lower_at = lower_bite if lower_bite is not None else (upper.transform if scanner_bite else lower.transform)
        surfaces.append(apply(lower_at, lower.scan.vertices))
    jaws_tree = cKDTree(np.vstack(surfaces))
    regs = [None] * len(bites)
    for i, (T, texts) in placed.items():
        dist, _ = jaws_tree.query(apply(T, bites[i].vertices), distance_upper_bound=MATCH_MM)
        deviation = np.where(np.isfinite(dist), dist, np.nan)
        stats = deviation_stats(deviation)
        if i in own and stats.get("matched_fraction", 0) < OWN_MATCHED:  # подогнанный сам — лёг плохо: не верим
            failed[i] = ("Скан прикуса в своих координатах лёг на сканы челюстей плохо — не поставлен. "
                         "Выгрузите сканы челюстей и прикуса из одной сессии сканера (в общих координатах).")
            continue
        if stats.get("matched_fraction", 0) < 0.3:
            texts = texts + ["Скан прикуса плохо лёг на сканы челюстей — проверьте, тот ли это пациент и тот ли прикус."]
        regs[i] = Registration(bites[i], BITE, T, deviation, stats, warnings=texts + notes)
    return BitePlacement(regs, lower_bite, failed)


def split_by_jaw(mesh: Mesh, ct: "CaseCT") -> dict[str, Mesh]:
    """Делит сетку из КТ на части верхней и нижней челюсти (импланты, поверхности по порогам).

    Каждая вершина относится к челюсти, чьи коронки ближе. Связная часть
    сетки целиком уходит к челюсти большинства своих вершин; если явного
    большинства нет (например, сомкнутые зубы слились), делится по граням.
    """
    d_up, _ = ct.coarse["upper"].tree.query(mesh.vertices)
    d_lo, _ = ct.coarse["lower"].tree.query(mesh.vertices)
    lower_v = d_lo < d_up
    parts = trimesh.graph.connected_component_labels(
        trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).face_adjacency, node_count=len(mesh.faces))
    face_lower = lower_v[mesh.faces].mean(axis=1) > 0.5
    for part in np.unique(parts):
        sel = parts == part
        share = lower_v[np.unique(mesh.faces[sel])].mean()
        if share >= JAW_MAJORITY or share <= 1 - JAW_MAJORITY:
            face_lower[sel] = share > 0.5
    out = {}
    for jaw, mask in (("upper", ~face_lower), ("lower", face_lower)):
        if mask.any():
            faces = mesh.faces[mask]
            used = np.unique(faces)
            remap = np.full(len(mesh.vertices), -1)
            remap[used] = np.arange(len(used))
            out[jaw] = Mesh(mesh.vertices[used], remap[faces])
    return out


def _rotation_deg(T: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


def export_case(out_dir: str, registrations: list[Registration], ct_meshes: dict[str, Mesh] | None = None,
                bite: str = "scan", frame: str = "exocad", ct: "CaseCT | None" = None,
                reference: np.ndarray | None = None, reference_name: str = "",
                scene: np.ndarray | None = None, without_ct: bool = False) -> dict:
    """Пишет все сетки в одной системе координат и файл с матрицами.

    ct_meshes — сетки из КТ (структуры сегментации или поверхности по порогам),
    ключ — путь файла без расширения, например "mandible" или "teeth/tooth_36".
    С какой челюстью двигается структура — structures.jaw_of; смешанные
    (импланты, поверхности по порогам) делятся по челюстям (split_by_jaw, нужен ct).

    bite — чей прикус:
      "scan" (по умолчанию) — прикус сканов. Сканы стоят как пришли со
      сканера, структуры каждой челюсти из КТ переезжают к скану своей
      челюсти: нижняя челюсть, нижние зубы, канал — к нижнему скану,
      остальное — к верхнему. Прикус на КТ (часто с приоткрытым ртом) не важен;
      «ct» — статическое наложение на КТ: всё стоит как на КТ, сканы
      переносятся на свои челюсти в КТ.
    frame — система координат: "exocad" — сканера (в них сканы открывает
    exocad; опорный скан — верхний, если есть), "dicom" — пациента из DICOM
    (с bite="scan" верхняя челюсть — как на КТ, нижняя со своими структурами —
    в прикусе сканов относительно неё), "reference" — система референтной плоскости или
    артикулятора: reference — матрица 4×4 «мм пациента (КТ) → эта система»
    (landmarks.reference_frame, articulators.articulator_frame). Прикус при
    этом любой: верхняя челюсть ставится по КТ, нижняя — по выбранному прикусу.
    scene — с frame="exocad": матрица «координаты файла опорного скана → сцена
    проекта exocad» (exocad_project): всё выгружается в координатах его сцены.
    without_ct — кейс без КТ: transform сканов — координаты их файлов (в системе
    опорного скана); только координаты сканера и прикус сканов.
    """
    if bite not in ("scan", "ct"):
        raise ValueError("bite должен быть scan или ct")
    if frame not in ("exocad", "dicom", "reference"):
        raise ValueError("frame должен быть exocad, dicom или reference")
    if frame == "reference":
        if reference is None:
            raise ValueError("для frame=reference нужна матрица системы (референтная плоскость или артикулятор)")
        # Считаем как обычно (прикус сканов — в координатах сканера, прикус КТ — в DICOM),
        # а в конце переводим всё в систему плоскости.
        base = "exocad" if bite == "scan" else "dicom"
        result = export_case(out_dir, registrations, ct_meshes, bite=bite, frame=base, ct=ct,
                             reference=reference, reference_name=reference_name)
        return result
    if without_ct and (frame != "exocad" or bite != "scan"):
        raise ValueError("без КТ — только координаты сканера и прикус сканов")
    if not registrations:
        raise ValueError("нет совмещённых сканов")

    by_jaw = {}
    for reg in registrations:
        if reg.jaw != BITE:
            by_jaw.setdefault(reg.jaw, reg)
    if not by_jaw:
        raise ValueError("нет совмещённых сканов челюстей: скан прикуса ставится на них")
    ref = by_jaw.get("upper") or next(iter(by_jaw.values()))
    # Прикус сканов и координаты exocad считаются в координатах сканера опорного скана. В DICOM прикус сканов
    # переводится на КТ по опорному скану: верхняя челюсть — как на КТ, нижняя — в прикусе сканов.
    scanner = frame == "exocad" or bite == "scan"
    # Перевод из базовой системы в систему плоскости/артикулятора, сцену exocad или DICOM.
    post = np.eye(4)
    if reference is not None:
        post = np.asarray(reference, float) @ (ref.transform if scanner else np.eye(4))
    elif scene is not None and frame == "exocad":
        post = np.asarray(scene, float)
    elif frame == "dicom" and scanner:
        post = ref.transform
    to_out = np.linalg.inv(ref.transform) if scanner else np.eye(4)
    notes = []

    # Где стоит каждый скан: в прикусе сканов — где пришёл со сканера или по скану прикуса (врача),
    # иначе — на своей челюсти в КТ. Скан прикуса стоит на сканах челюстей (place_bite).
    placement, placed_by = {}, {}
    for reg in registrations:
        placement[id(reg)], placed_by[id(reg)] = to_out @ reg.transform, "ct"
        if reg.jaw == BITE:
            placed_by[id(reg)] = "jaws"
        elif scanner and reg is ref:
            placement[id(reg)], placed_by[id(reg)] = np.eye(4), "scanner"
        elif bite == "scan" and reg.bite is not None:
            placement[id(reg)], placed_by[id(reg)] = to_out @ reg.bite, "bite scan"
        elif bite == "scan":
            if in_occlusion(ref.scan, reg.scan) or in_occlusion(reg.scan, ref.scan):
                placement[id(reg)], placed_by[id(reg)] = np.eye(4), "scanner"
            elif without_ct:
                notes.append(f"{reg.scan.name}: скан не в прикусе с {ref.scan.name} (выгружен отдельно), "
                             "сканов прикуса нет — стоит как в файле.")
            else:
                notes.append(f"{reg.scan.name}: скан не в прикусе с {ref.scan.name} (выгружен отдельно) — "
                             "поставлен по КТ, прикус взят с КТ.")

    # Структуры челюсти из КТ ставятся туда же, куда поставлен скан этой челюсти.
    jaw_transform = {jaw: placement[id(r)] @ np.linalg.inv(r.transform) for jaw, r in by_jaw.items()}
    for jaw in JAWS:
        jaw_transform.setdefault(jaw, to_out)

    bite_report = None
    if "upper" in by_jaw and "lower" in by_jaw and scanner and not without_ct:
        lower = by_jaw["lower"]
        # Насколько нижняя челюсть на КТ стоит иначе, чем на сканах (относительно верхней).
        diff = np.linalg.inv(placement[id(lower)]) @ (to_out @ lower.transform)
        moved = np.linalg.norm(apply(diff, lower.scan.vertices[lower.scan.crowns]) -
                               lower.scan.vertices[lower.scan.crowns], axis=1)
        bite_report = {"lower_jaw_on_ct_vs_scans_mean_mm": round(float(moved.mean()), 3),
                       "max_mm": round(float(moved.max()), 3), "rotation_deg": round(_rotation_deg(diff), 2)}

    placement = {k: post @ M for k, M in placement.items()}
    jaw_transform = {k: post @ M for k, M in jaw_transform.items()}

    os.makedirs(out_dir, exist_ok=True)
    written = {}

    def write(name, parts, colors=None, suffix=".stl"):
        """parts — [(вершины, грани, матрица)]: части одной сетки могут двигаться по-разному."""
        verts, faces, offset = [], [], 0
        for v, f, M in parts:
            verts.append(apply(M, v))
            faces.append(f + offset)
            offset += len(v)
        path = os.path.join(out_dir, *name.split("/")) + suffix
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mesh = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
        if colors is not None:
            mesh.visual.vertex_colors = colors
        mesh.export(path)
        written[name + suffix] = [M.round(9).tolist() for _v, _f, M in parts]

    scans = {}
    for reg in registrations:
        M = placement[id(reg)]
        write(reg.scan.name, [(reg.scan.vertices, reg.scan.faces, M)])
        if not without_ct or reg.jaw == BITE:  # карта отклонений от КТ (у скана прикуса — от сканов челюстей)
            write(reg.scan.name + "_deviation", [(reg.scan.vertices, reg.scan.faces, M)],
                  deviation_colors(reg.deviation), suffix=".ply")
        scans[reg.scan.name] = {
            "jaw": reg.jaw, "scan_to_ct": reg.transform.round(9).tolist(),
            "placement": placed_by[id(reg)],
            "fit": reg.stats, "edge_shift_mm": round(reg.edge_shift, 4), "warnings": list(reg.warnings),
            "segments": [vars(s) for s in reg.segments]}

    for name, mesh in (ct_meshes or {}).items():
        if not len(mesh.faces):
            continue
        jaw = jaw_of(name)
        if jaw is not None or bite == "ct" or ct is None:
            write(name, [(mesh.vertices, mesh.faces, jaw_transform[jaw or "upper"])])
        else:
            write(name, [(m.vertices, m.faces, jaw_transform[j]) for j, m in split_by_jaw(mesh, ct).items()])

    report = {
        "bite": "scans" if bite == "scan" else "ct",
        "frame": (f"{reference_name or 'reference frame'}: origin at the hinge axis centre, X right, Y forward, Z up, mm"
                  if reference is not None else
                  f"exocad project scene: scanner coordinates of {ref.scan.name} × project matrix, mm"
                  if frame == "exocad" and scene is not None else
                  f"scanner coordinates of {ref.scan.name} (as opened in exocad), mm" if frame == "exocad"
                  else f"DICOM patient coordinates, mm (LPS); {ref.scan.name} as on the CT" if scanner
                  else "DICOM patient coordinates, mm (LPS)"),
        **({"scanner_to_output": post.round(9).tolist()} if scene is not None and frame == "exocad" else {}),
        **({"without_ct": True} if without_ct else
           {"ct_to_output": {jaw: M.round(9).tolist() for jaw, M in jaw_transform.items()}}),
        "ct_bite_vs_scans": bite_report,
        "notes": notes,
        "files": {name: {"source_to_output": Ms} for name, Ms in written.items()},
        "scans": scans,
    }
    with open(os.path.join(out_dir, "case.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
