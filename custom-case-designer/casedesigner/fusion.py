"""Совмещение сканов челюстей с КТ и экспорт всего в единых координатах."""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .register import Target, apply, fit_score, icp, kabsch, occlusal_init
from .surface import Surface, extract_surface
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
    deviation: np.ndarray  # отклонение каждой вершины от коронок КТ, мм; nan — не на коронке
    stats: dict = field(default_factory=dict)


class CaseCT:
    """КТ кейса с подготовленными коронками обеих челюстей."""

    def __init__(self, vol):
        self.vol = vol
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
            raise ValueError("рядом со сканом в КТ нет коронок — проверьте грубое совмещение")
        return Target(crowns.points, crowns.normals)

    def register(self, scan: Scan, jaw: str | None = None, pairs=None) -> Registration:
        """Совмещает скан с КТ по коронкам зубов.

        jaw — "upper"/"lower", если известно; иначе пробуются обе челюсти.
        pairs — (точки на скане, те же точки в КТ), минимум 3: начальное
        положение по ним вместо автоматического поиска.
        """
        if jaw is not None and jaw not in JAWS:
            raise ValueError(f"челюсть должна быть upper или lower, а не {jaw!r}")
        jaws = (jaw,) if jaw else JAWS
        if pairs is not None:
            T0 = kabsch(*pairs)
            scores = {j: fit_score(scan.vertices, self.coarse[j], T0, tol=1.5) for j in jaws}
            jaw = max(scores, key=scores.get)
        else:
            normals = scan.normals
            found = {j: occlusal_init(scan.vertices, normals, self.coarse[j], OCCLUSAL[j]) for j in jaws}
            jaw = max(found, key=lambda j: found[j][1])
            T0 = found[jaw][0]

        target = self.fine_crowns(apply(T0, scan.vertices))
        T, _used = icp(scan.vertices, target, T0, schedule=FINE_SCHEDULE, keep=0.9)
        deviation = surface_deviation(apply(T, scan.vertices), target)
        return Registration(scan, jaw, T, deviation, deviation_stats(deviation))

    def surfaces(self, step: int = 1) -> dict:
        """Поверхности КТ по порогам: зубы и кость (до появления сегментации)."""
        return {
            "ct_teeth": extract_surface(self.vol, self.levels.dense, step=step),
            "ct_bone": extract_surface(self.vol, self.levels.hard, step=max(step, 2)),
        }


def surface_deviation(points: np.ndarray, target: Target) -> np.ndarray:
    """Расстояние от каждой точки до поверхности коронок (по касательной плоскости ближайшей точки)."""
    dist, j = target.tree.query(points, distance_upper_bound=MATCH_MM)
    out = np.full(len(points), np.nan)
    ok = np.isfinite(dist)
    out[ok] = np.abs(np.einsum("ij,ij->i", points[ok] - target.points[j[ok]], target.normals[j[ok]]))
    return out


def deviation_stats(deviation: np.ndarray) -> dict:
    d = deviation[np.isfinite(deviation)]
    if not len(d):
        return {"matched_fraction": 0.0}
    return {
        "matched_fraction": round(float(len(d) / len(deviation)), 3),
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
    d = np.nan_to_num(deviation, nan=-1)
    colors[(d >= 0) & (d <= 0.1)] = (40, 170, 70, 255)
    colors[(d > 0.1) & (d <= 0.2)] = (230, 190, 30, 255)
    colors[d > 0.2] = (210, 50, 40, 255)
    return colors


def export_case(out_dir: str, registrations: list[Registration], ct_surfaces: dict | None = None,
                frame: str = "ct") -> dict:
    """Пишет все сетки в одной системе координат и файл с матрицами.

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
        path = os.path.join(out_dir, name + suffix)
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
    for name, surf in (ct_surfaces or {}).items():
        if isinstance(surf, Surface) and surf.faces is not None and len(surf.faces):
            write(name, surf.points, surf.faces, to_out)

    report = {
        "frame": "DICOM patient coordinates, mm (LPS)" if frame == "ct"
                 else f"coordinates of scan {registrations[0].scan.name}, mm",
        "files": {name: {"source_to_output": M.round(9).tolist()} for name, M in written.items()},
        "scans": {
            reg.scan.name: {"jaw": reg.jaw, "scan_to_ct": reg.transform.round(9).tolist(), "fit": reg.stats}
            for reg in registrations
        },
    }
    with open(os.path.join(out_dir, "case.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
