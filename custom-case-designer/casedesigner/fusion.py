"""Совмещение сканов челюстей с КТ и экспорт всего в единых координатах."""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .quality import Segment, check_scan
from .register import Target, apply, edge_offset, fit_score, icp, kabsch, occlusal_init
from .segment import Mesh
from .surface import extract_surface
from .teeth import OCCLUSAL, Levels, crown_surface, split_jaws

JAWS = ("upper", "lower")
# Точка скана считается лежащей на коронке, если до неё ближе этого расстояния.
MATCH_MM = 0.5
FINE_SCHEDULE = ((1.0, 15), (0.5, 15), (0.3, 20), (0.2, 20))


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

    def fine_crowns(self, points: np.ndarray, margin: float = 3.0) -> Target:
        roi = (points.min(axis=0) - margin, points.max(axis=0) + margin)
        crowns = crown_surface(self.vol, self.levels, roi=roi, refine=True)
        if len(crowns.points) < 100:
            raise ValueError("рядом со сканом в КТ нет коронок — проверьте положение скана")
        return Target(crowns.points, crowns.normals)

    def _nearest_jaw(self, scan: Scan, T: np.ndarray, jaws=JAWS) -> str:
        scores = {j: fit_score(scan.vertices, self.coarse[j], T, tol=1.5) for j in jaws}
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
        if start is not None:
            T0 = np.asarray(start, float)
            jaw = jaw or self._nearest_jaw(scan, T0)
        elif pairs is not None:
            T0 = kabsch(*pairs)
            jaw = self._nearest_jaw(scan, T0, jaws)
        else:
            normals = scan.normals
            found = {j: occlusal_init(scan.vertices, normals, self.coarse[j], OCCLUSAL[j]) for j in jaws}
            jaw = max(found, key=lambda j: found[j][1])
            T0 = found[jaw][0]

        target = self.fine_crowns(apply(T0, scan.vertices))
        T, _used = icp(scan.vertices, target, T0, schedule=FINE_SCHEDULE, keep=0.9)
        # Последний шаг: положение вместе со сдвигом границы эмали (см. register.edge_offset).
        shift, T = edge_offset(scan.vertices, target, T, self.edge_prior, self.prior_weight)
        return self._result(scan, jaw, T, target, shift)

    def evaluate(self, scan: Scan, transform: np.ndarray, jaw: str | None = None) -> Registration:
        """Точность положения как есть, без уточнения: для ручной коррекции.

        Положение считается верным, поэтому сдвиг границы эмали — тот, что
        объясняет именно его: выученный сдвиг плюс оставшееся среднее отклонение.
        """
        T = np.asarray(transform, float)
        jaw = jaw or self._nearest_jaw(scan, T)
        target = self.fine_crowns(apply(T, scan.vertices))
        reg = self._result(scan, jaw, T, target, self.edge_prior)
        reg.edge_shift = self.edge_prior - reg.stats.get("signed_mean_mm", 0.0)
        return reg

    def _result(self, scan, jaw, T, target, shift) -> Registration:
        """Отклонения скана от коронок с учётом сдвига границы эмали."""
        corrected = Target(target.points - shift * target.normals, target.normals)
        deviation = surface_deviation(apply(T, scan.vertices), corrected)
        segments, warnings = check_scan(scan.vertices, T, deviation, corrected, jaw)
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


def export_case(out_dir: str, registrations: list[Registration], ct_meshes: dict[str, Mesh] | None = None,
                frame: str = "ct") -> dict:
    """Пишет все сетки в одной системе координат и файл с матрицами.

    ct_meshes — сетки из КТ (структуры сегментации или поверхности по порогам),
    ключ — путь файла без расширения, например "mandible" или "teeth/tooth_36".

    frame="ct" — координаты пациента из DICOM (мм, LPS);
    frame="scan" — координаты первого скана: удобно, когда дальше работа
    идёт в CAD, где этот скан уже открыт.
    """
    if frame not in ("ct", "scan"):
        raise ValueError("frame должен быть ct или scan")
    if not registrations:
        raise ValueError("нет совмещённых сканов")
    os.makedirs(out_dir, exist_ok=True)
    to_out = np.eye(4) if frame == "ct" else np.linalg.inv(registrations[0].transform)

    written = {}

    def write(name, vertices, faces, source_to_out, colors=None, suffix=".stl"):
        path = os.path.join(out_dir, *name.split("/")) + suffix
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mesh = trimesh.Trimesh(apply(source_to_out, vertices), faces, process=False)
        if colors is not None:
            mesh.visual.vertex_colors = colors
        mesh.export(path)
        written[name + suffix] = source_to_out

    for reg in registrations:
        M = to_out @ reg.transform
        write(reg.scan.name, reg.scan.vertices, reg.scan.faces, M)
        write(reg.scan.name + "_deviation", reg.scan.vertices, reg.scan.faces, M,
              deviation_colors(reg.deviation), suffix=".ply")
    for name, mesh in (ct_meshes or {}).items():
        if len(mesh.faces):
            write(name, mesh.vertices, mesh.faces, to_out)

    report = {
        "frame": "DICOM patient coordinates, mm (LPS)" if frame == "ct"
                 else f"coordinates of scan {registrations[0].scan.name}, mm",
        "files": {name: {"source_to_output": M.round(9).tolist()} for name, M in written.items()},
        "scans": {
            reg.scan.name: {"jaw": reg.jaw, "scan_to_ct": reg.transform.round(9).tolist(), "fit": reg.stats,
                            "edge_shift_mm": round(reg.edge_shift, 4), "warnings": reg.warnings,
                            "segments": [vars(s) for s in reg.segments]}
            for reg in registrations
        },
    }
    with open(os.path.join(out_dir, "case.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
