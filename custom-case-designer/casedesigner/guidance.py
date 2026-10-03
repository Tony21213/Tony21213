"""Углы суставных путей для виртуального артикулятора: из записи, из КТ, из зубов.

Источники по убыванию надёжности:

1. **запись реальных движений** (P-ART) — motion.analyze_case;
2. **полночерепное КТ при правильном монтаже** — форма суставного бугорка.
   В сагиттальном сечении через мыщелок строится профиль суставной
   поверхности височной кости (нижняя граница кости над мыщелком и перед
   ним). По профилю катится головка мыщелка — окружность, касающаяся крыши
   ямки; путь её центра и есть суставной путь, его наклон к выбранной
   плоскости — ССП. Отдельно считается наклон бугорка по линии «крыша ямки —
   вершина бугорка», как в работах по КЛКТ ВНЧС. Угол Беннетта по КТ надёжно
   не измерить, поэтому, пока нет записи, — по формуле Ханау L = H/8 + 12;
3. **средние значения** (Settings по умолчанию).

Ведение зубами (резцовое, клыковое) не задаётся числом, а получается из самих
моделей в прикусе — kinematics.Occlusion: нижние зубы скользят по верхним с
их реальной формой и стёртостью.

Записи P-ART — эталон: на кейсах, где есть и запись, и КТ, видно, насколько
ССП по бугорку расходится с реальным путём; по накоплении кейсов эта
поправка уточняется так же, как память совмещений.
"""

import numpy as np

from .kinematics import Settings
from .landmarks import reference_frame
from .register import apply

CONDYLE_RADIUS_MM = 4.0  # от верхней точки мыщелка (Co) до центра головки
PROFILE_STEP_MM = 0.25
PROFILE_REACH_MM = (-6.0, 22.0)  # по y от мыщелка: от задней части ямки до ската за вершиной бугорка
PROFILE_WINDOW_MM = (-16.0, 10.0)  # по z от мыщелка: ниже — шиловидный отросток и прочее, выше — свод
PATH_LENGTH_MM = 12.0
CHORD_MM = 5.0  # как в motion: ССП — по хорде на первых 5 мм пути


def hanau_bennett(sagittal_deg: float) -> float:
    """Угол Беннетта по формуле Ханау: L = H/8 + 12."""
    return sagittal_deg / 8 + 12


def eminence_profile(vertices: np.ndarray, faces: np.ndarray, frame: np.ndarray, condyle: np.ndarray):
    """Профиль суставной поверхности в сагиттальном сечении через мыщелок.

    vertices/faces — кость без нижней челюсти (череп из сегментации), в мм КТ;
    frame — анатомическая система (landmarks.reference_frame); condyle — Co в мм КТ.
    Возвращает y и z относительно мыщелка (мм, анатомическая система); где кости нет — NaN.
    """
    import trimesh

    mesh = trimesh.Trimesh(apply(frame, vertices), np.asarray(faces), process=False)
    c = apply(frame, np.asarray(condyle, float)[None])[0]
    segments = trimesh.intersections.mesh_plane(mesh, plane_normal=[1.0, 0, 0], plane_origin=c)
    y = np.arange(PROFILE_REACH_MM[0], PROFILE_REACH_MM[1] + 1e-9, PROFILE_STEP_MM)
    z = np.full(len(y), np.nan)
    if len(segments) == 0:
        return y, z
    t = np.linspace(0, 1, 5)[:, None, None]
    pts = (segments[None, :, 0] * (1 - t) + segments[None, :, 1] * t).reshape(-1, 3) - c
    pts = pts[(pts[:, 2] >= PROFILE_WINDOW_MM[0]) & (pts[:, 2] <= PROFILE_WINDOW_MM[1])]
    idx = np.round((pts[:, 1] - y[0]) / PROFILE_STEP_MM).astype(int)
    ok = (idx >= 0) & (idx < len(y))
    for i, zi in zip(idx[ok], pts[ok, 2]):
        if np.isnan(z[i]) or zi < z[i]:
            z[i] = zi  # нижняя граница кости в столбце — суставная поверхность
    return y, z


def condylar_path(y: np.ndarray, z: np.ndarray, radius: float = CONDYLE_RADIUS_MM) -> dict:
    """Путь центра головки мыщелка, катящейся по профилю, и углы.

    Центр головки — на radius ниже Co; окружность радиусом «центр — ближайшая
    точка профиля» касается крыши ямки и при движении вперёд скользит по скату
    бугорка, не входя в кость.
    """
    ok = ~np.isnan(z)
    if ok.sum() < 10:
        raise ValueError("в сечении через мыщелок нет суставной поверхности — проверьте мыщелок и сегментацию")
    py, pz = y[ok], z[ok]
    centre = np.array([0.0, -radius])
    r = float(np.min(np.hypot(py - centre[0], pz - centre[1])))
    cy = np.arange(0.0, PATH_LENGTH_MM + 1e-9, PROFILE_STEP_MM)
    cz = np.full(len(cy), np.nan)
    for k, yc in enumerate(cy):
        near = np.abs(py - yc) < r
        if near.any():
            cz[k] = np.min(pz[near] - np.sqrt(r ** 2 - (py[near] - yc) ** 2))
    good = ~np.isnan(cz)
    cy, cz = cy[good], cz[good]
    d = np.hypot(cy - cy[0], cz - cz[0])
    k = int(np.argmax(d >= CHORD_MM)) if (d >= CHORD_MM).any() else len(d) - 1
    sagittal = float(np.degrees(np.arctan2(cz[0] - cz[k], cy[k] - cy[0])))

    roof = (py >= -4) & (py <= 4)
    if not roof.any():
        raise ValueError("над мыщелком нет крыши суставной ямки — проверьте положение мыщелка")
    roof_idx = np.flatnonzero(roof)
    top = roof_idx[pz[roof_idx] >= pz[roof_idx].max() - 0.1][-1]  # при плоской крыше — её передний край
    ahead = py >= py[top] + 4
    eminence = fossa_depth = None
    if ahead.any():
        crest = np.flatnonzero(ahead)[np.argmin(pz[ahead])]
        eminence = float(np.degrees(np.arctan2(pz[top] - pz[crest], py[crest] - py[top])))
        fossa_depth = float(pz[top] - pz[crest])
    return {"sagittal_deg": round(sagittal, 1), "measured_at_mm": round(float(d[k]), 2),
            "eminence_deg": None if eminence is None else round(eminence, 1),
            "fossa_depth_mm": None if fossa_depth is None else round(fossa_depth, 2),
            "rolling_radius_mm": round(r, 2), "path": np.c_[cy, cz]}


def ct_settings(bone_vertices: np.ndarray, bone_faces: np.ndarray, landmarks: dict, plane: str = "frankfurt",
                base: Settings | None = None, case_to_ct: np.ndarray | None = None) -> tuple[Settings, dict]:
    """ССП обеих сторон по КТ, угол Беннетта по Ханау; прочее — из base.

    Углы отсчитаны от плоскости plane по КТ и так и остаются при любом монтаже:
    Settings.frame = «координаты кейса → система плоскости» (case_to_ct — как
    модели кейса стоят в КТ; по умолчанию кейс и есть КТ).
    """
    frame = reference_frame(landmarks, plane)
    s = (base or Settings()).copy()
    s.frame = frame @ (np.eye(4) if case_to_ct is None else np.asarray(case_to_ct, float))
    details = {}
    for side, key in (("right", "Co_R"), ("left", "Co_L")):
        y, z = eminence_profile(bone_vertices, bone_faces, frame, landmarks[key])
        info = condylar_path(y, z)
        details[side] = {k: v for k, v in info.items() if k != "path"}
        setattr(s, f"sagittal_{side}_deg", info["sagittal_deg"])
        s.sources[f"sagittal_{side}_deg"] = f"КТ: суставной бугорок ({plane})"
    for side in ("right", "left"):
        setattr(s, f"bennett_{side}_deg", round(hanau_bennett(s.get("sagittal", side)), 1))
        s.sources[f"bennett_{side}_deg"] = "формула Ханау по ССП из КТ"
    return s, details
