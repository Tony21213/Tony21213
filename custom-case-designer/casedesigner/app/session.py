"""Состояние открытого кейса и операции над ним — то, что вызывает интерфейс.

Всё в координатах пациента из DICOM (мм, LPS): КТ, структуры, сканы (через
текущую матрицу «скан → КТ»). Интерфейс двигает скан, присылает новую
матрицу, а сессия оценивает точность и при желании уточняет положение.
"""

from .common import (  # noqa: F401 — прежние имена session.py для импортёров
    AlignmentMemory,
    BITE,
    CASE_EXT,
    CaseCT,
    DEV_COLORS,
    EXOCAD_SUBFOLDER,
    GROUPS,
    GUIDE_KEYS,
    JAWS_APART,
    JawCase,
    JawMesh,
    LOWER_BONES,
    MAX_SLICE_PIXELS,
    MOUNT_PLANE_SETTING,
    RECENT,
    Registration,
    SCAN_COLORS,
    SEGMENT_PARTS,
    SHARED_SHARE,
    SLICES,
    SLICE_PIXELS,
    Scan,
    Sectioner,
    Segmenter,
    UNGUIDED,
    UPPER_BONES,
    _axis_aligned,
    apply,
    arts,
    cond,
    dataclasses,
    deviation_colors,
    errorlog,
    exocad_project,
    export_case,
    gd,
    group_of,
    in_occlusion,
    io,
    is_bite_name,
    jaw_by_name,
    jaw_of,
    lmk,
    load_volume,
    mesh_bytes,
    mo,
    model_store,
    mplanes,
    np,
    open_folder,
    os,
    part_of,
    place_bites,
    rigid,
    scanner_registration,
    shared_share,
    split_by_jaw,
    structure,
    threading,
    trimesh,
    uuid,
)
from .steps.ct import CtSteps
from .steps.scans import ScanSteps
from .steps.articulation import ArticulationSteps
from .steps.landmarks import LandmarkSteps
from .steps.export import ExportSteps
from .steps.casefile import CaseFileSteps


class Session(CtSteps, ScanSteps, ArticulationSteps, LandmarkSteps, ExportSteps, CaseFileSteps):
    def __init__(self, memory_path: str | None = None, models_dir: str | None = None):
        self.lock = threading.RLock()
        self.memory = AlignmentMemory(memory_path or os.path.join(os.path.expanduser("~"), ".casedesigner",
                                                                    "memory.jsonl"))
        self.models_dir = models_dir
        self.ct_path = None
        self.vol = None
        self.case: CaseCT | None = None
        self.window = (400.0, 3000.0)
        self.scans: dict[str, dict] = {}
        # Прикус сканов в окне: нижняя челюсть со всеми её структурами — в прикусе сканов (врача), а не как на КТ.
        self.bite_view = False
        self.bite_motion = None  # движение нижней челюсти с КТ в прикус сканов (мм КТ → мм КТ); None — прикуса нет
        # «Исправить прикус» (bite.correct_bite): поправка прикуса сканов, мм КТ → мм КТ, после прикуса сканера или
        # сканов прикуса; сбрасывается, когда меняется совмещение сканов. Предложение — до «Принять» / «Отменить».
        self.bite_fix = np.eye(4)
        self._bite_proposal = None
        self._bite_meshes = {}
        self.structures: dict[str, trimesh.Trimesh] = {}
        self._sections = {}
        self.landmarks: dict[str, np.ndarray] = {}
        self.suggested: set[str] = set()  # предложены программой и ещё не подтверждены врачом
        self.landmark_sources: dict[str, str] = {}  # откуда точка: condyles.SOURCES и др. (сохраняется с кейсом)
        self.landmark_notes: dict[str, str] = {}  # пояснения к автоматической замене ориентиров
        self._joint_parts_cache_key = None
        self._joint_parts_cache = None
        self._mounted_condyles = None  # точки мыщелков, с которыми сделан текущий монтаж (condyles.state: applied)
        self.articulators_path = os.path.join(os.path.dirname(self.memory.path), "articulators.json")
        self.settings_path = os.path.join(os.path.dirname(self.memory.path), "settings.json")
        self.settings = self._load_settings()
        self.case_path: str | None = None  # файл кейса, куда сохранять
        self.guide_teeth: dict = {}  # зубы из сегментации — опора совмещения (сохраняются с кейсом)
        self.jaw = JawCase()  # артикуляция: монтаж, суставы, движения, контакты (интерфейс — позже)
        self._articulation_ready = False
        self._case_mount_plane: str | None = None

    # --- настройки (рядом с памятью совмещений) -----------------------------
    def _load_settings(self) -> dict:
        import json

        known = [p for p, _t, _k in SEGMENT_PARTS]
        settings = {"segment_parts": known, "recent": [], "incognito": False,
                    "mount_plane": MOUNT_PLANE_SETTING}
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                saved = json.load(f)
            settings["segment_parts"] = [p for p in known if p in saved.get("segment_parts", [])]
            settings["recent"] = [r for r in saved.get("recent", []) if isinstance(r, str)][:RECENT]
            settings["incognito"] = saved.get("incognito") is True
            plane = saved.get("mount_plane", MOUNT_PLANE_SETTING)
            settings["mount_plane"] = plane if plane in mplanes.PLANES else MOUNT_PLANE_SETTING
        except (OSError, ValueError, AttributeError):  # нет файла или он испорчен — всё по умолчанию
            pass
        return settings

    def set_segment_parts(self, parts: list[str]) -> dict:
        known = [p for p, _t, _k in SEGMENT_PARTS]
        unknown = sorted(set(parts) - set(known))
        if unknown:
            raise ValueError(f"нет таких частей: {', '.join(unknown)}")
        self.settings["segment_parts"] = [p for p in known if p in parts]
        self._save_settings()
        return self.segment_parts_info()

    def _save_settings(self):
        import json

        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=1)

    def set_incognito(self, on: bool) -> dict:
        """Инкогнито: интерфейс не показывает имён пациентов и путей (для показа программы). Помнится между
        запусками — программа, открытая на демонстрации, сразу скрывает недавние кейсы."""
        self.settings["incognito"] = bool(on)
        self._save_settings()
        return {"incognito": self.settings["incognito"]}

    def set_default_mount_plane(self, key: str) -> dict:
        """Сохранить предпочтительную плоскость монтажа; недоступная для кейса позже разрешится по FALLBACK."""
        if key not in mplanes.PLANES:
            raise ValueError(f"нет такой плоскости монтажа: {key}")
        self.settings["mount_plane"] = key
        self._save_settings()
        return {"mount_plane": key}

    def set_articulation_plane(self, key: str) -> dict:
        """Выбрать плоскость текущего кейса; недоступность проверяется при монтаже."""
        if key not in mplanes.PLANES:
            raise ValueError(f"нет такой плоскости монтажа: {key}")
        self._case_mount_plane = key
        return self.articulation_planes()

    def segment_parts_info(self) -> dict:
        return {"parts": [{"id": p, "title": t} for p, t, _k in SEGMENT_PARTS],
                "selected": list(self.settings["segment_parts"])}
