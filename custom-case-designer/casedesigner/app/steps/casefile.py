"""Кейс: сохранить и открыть, недавние, состояние для интерфейса — методы Session (разделено из session.py; контракты — там же, где были)."""

from ..common import (  # noqa: F401
    BITE,
    CASE_EXT,
    RECENT,
    Scan,
    errorlog,
    is_bite_name,
    lmk,
    mplanes,
    np,
    os,
    scanner_registration,
    trimesh,
    uuid,
)


class CaseFileSteps:
    # --- кейс: сохранить и открыть ----------------------------------------------
    def save_case(self, path: str | None = None) -> dict:
        """Кейс в один файл: путь к КТ, сканы (сетки — внутри, чтобы не зависеть от исходных файлов) с
        положениями, автоматические положения (для обучения), структуры сегментации, окно КТ."""
        import json

        if self.case is None and not self.scans:
            raise ValueError("нечего сохранять: откройте КТ или добавьте сканы")
        path = path or self.case_path
        errorlog.private(path, "кейс")
        if not path:
            raise ValueError("укажите, куда сохранить кейс")
        if not path.lower().endswith(CASE_EXT):
            path += CASE_EXT
        arrays = {}
        meta = {"version": 1, "ct_path": self.ct_path, "ct_series": getattr(self, "ct_series_id", None),
                "mount_plane": self._case_mount_plane,
                "landmarks": {k: np.asarray(v, float).tolist() for k, v in self.landmarks.items()},
                "suggested": sorted(self.suggested),
                "landmark_sources": {k: v for k, v in self.landmark_sources.items() if k in self.landmarks},
                "landmark_notes": {k: v for k, v in self.landmark_notes.items() if k in self.landmarks},
                "window": list(self.window), "scans": [], "structures": [],
                "guide": [], "bite_fix": np.asarray(self.bite_fix, float).tolist(),
                "photos": self._photos_meta(), "rehab": getattr(self, "rehab", None)}
        for i, (sid, item) in enumerate(self.scans.items()):
            scan = item["scan"]
            arrays[f"scan{i}_v"], arrays[f"scan{i}_f"] = scan.vertices, scan.faces
            if getattr(scan, "colors", None) is not None:
                arrays[f"scan{i}_c"] = scan.colors
            auto = item["auto"]
            meta["scans"].append({
                "name": scan.name, "path": item["path"], "color": item["color"], "jaw": item["jaw"],
                "reg_jaw": item["reg"].jaw if item["reg"] is not None else None,
                "transform": None if item["transform"] is None else np.asarray(item["transform"]).tolist(),
                "auto": None if auto is None else np.asarray(auto.transform).tolist(), "accepted": item["accepted"],
                "role": item["role"], "placed_by": item.get("placed_by")})
        for k, (key, mesh) in enumerate(self.structures.items()):
            arrays[f"st{k}_v"], arrays[f"st{k}_f"] = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            meta["structures"].append(key)
        for jaw, mesh in self.guide_teeth.items():
            if mesh is not None and len(mesh.faces):
                arrays[f"guide_{jaw}_v"], arrays[f"guide_{jaw}_f"] = np.asarray(mesh.vertices), np.asarray(mesh.faces)
                meta["guide"].append(jaw)
        arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)  # прерванная запись не портит прежний файл
        self.case_path = path
        self._remember(path)
        return {"path": path, "name": os.path.basename(path)}

    def open_case(self, path: str, progress=None, ct_path: str | None = None) -> dict:
        """Открыть сохранённый кейс: КТ по пути из кейса (или ct_path, если КТ переносили), сканы на своих местах,
        метрики пересчитываются, структуры и опора совмещения — из файла, без повторной сегментации."""
        import json
        from types import SimpleNamespace

        from ...segment import Mesh

        errorlog.private(path, "кейс")
        errorlog.private(ct_path, "КТ")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        errorlog.private(meta.get("ct_path"), "КТ")
        for sm in meta.get("scans", []):
            errorlog.private(sm.get("path"), "скан")
            errorlog.private(sm.get("name"), "скан")
        for ph in (meta.get("photos") or {}).values():
            errorlog.private(ph.get("path") if isinstance(ph, dict) else None, "фото")
        ct = ct_path or meta["ct_path"]
        if ct is None:  # кейс только со сканами
            self.close_ct()
        elif not os.path.exists(ct):
            raise FileNotFoundError(f"КТ этого кейса не найден: {ct} — его переместили или удалили")
        else:
            self.load_ct(ct, progress, meta.get("ct_series"))
        with self.lock:
            self.window = tuple(meta.get("window", self.window))
            saved_plane = meta.get("mount_plane")
            self._case_mount_plane = saved_plane if saved_plane in mplanes.PLANES else None
            self.structures = {key: trimesh.Trimesh(arrays[f"st{k}_v"], arrays[f"st{k}_f"], process=False)
                               for k, key in enumerate(meta["structures"])}
            self.scans = {}
        self.guide_teeth = {jaw: Mesh(arrays[f"guide_{jaw}_v"], arrays[f"guide_{jaw}_f"]) for jaw in meta["guide"]}
        with self.lock:
            self.landmarks = {k: np.asarray(v, float) for k, v in meta.get("landmarks", {}).items() if k in lmk.BY_KEY}
            self.suggested = set(meta.get("suggested", [])) & set(self.landmarks)
            self.landmark_sources = {k: v for k, v in meta.get("landmark_sources", {}).items() if k in self.landmarks}
            self.landmark_notes = {k: v for k, v in meta.get("landmark_notes", {}).items() if k in self.landmarks}
            self._joint_parts_cache_key = self._joint_parts_cache = None
            self._mounted_condyles = None
        if self.guide_teeth and self.case is not None:
            self.case.use_teeth(self.guide_teeth)
        for i, sm in enumerate(meta["scans"]):
            if progress:
                progress(0.6 + 0.4 * i / max(len(meta["scans"]), 1), f"Сканы: {sm['name']}")
            scan = Scan(sm["name"], arrays[f"scan{i}_v"], arrays[f"scan{i}_f"], arrays.get(f"scan{i}_c"))
            sid = uuid.uuid4().hex[:8]
            role = sm.get("role") or (BITE if is_bite_name(scan.name) else "jaw")
            item = {"scan": scan, "path": sm["path"], "color": sm["color"], "jaw": sm["jaw"], "reg": None, "auto": None,
                    "role": role, "lower_bite": None, "placed_by": sm.get("placed_by"),
                    "transform": None, "accepted": False, "guided": False}
            if sm["transform"] is not None and role != BITE and self.case is None:  # без КТ — как в файле
                reg = scanner_registration(scan, sm["reg_jaw"])
                item.update(reg=reg, auto=reg, transform=reg.transform, guided=True)
            elif sm["transform"] is not None and role != BITE:  # сканы прикуса ставятся заново на челюсти
                reg = self.case.evaluate(scan, np.asarray(sm["transform"], float), jaw=sm["reg_jaw"])
                if sm.get("placed_by") == "other_jaw":  # поставлен по другой челюсти: метрики коронок не о нём
                    reg.warnings = []
                item.update(reg=reg, transform=reg.transform, guided=reg.jaw in self.case.guided)
                auto = np.asarray(sm["auto"], float) if sm["auto"] is not None else reg.transform
                item["auto"] = reg if np.allclose(auto, reg.transform) else SimpleNamespace(transform=auto, jaw=reg.jaw)
                item["accepted"] = sm["accepted"]
            with self.lock:
                self.scans[sid] = item
        self._place_bites()  # сканы прикуса — на сканы челюстей, как при совмещении
        if meta.get("bite_fix") is not None:  # «Исправить прикус»; старые кейсы — без поправки
            self.bite_fix = np.asarray(meta["bite_fix"], float)
            self._update_bite_motion()
        self._load_photos_meta(meta.get("photos"))  # «Эстетика»: фото — по путям, пары — в координатах скана
        saved = meta.get("rehab")  # полная реабилитация: подъём прикуса и шаблон
        self.rehab = None if not isinstance(saved, dict) else {
            "raise_mm": float(saved.get("raise_mm") or 0.0), "template": bool(saved.get("template")),
            "radius": float(saved.get("radius") or 101.6)}
        self.case_path = path
        self._remember(path)
        return self.state()

    def _remember(self, path: str):
        recent = [p for p in self.settings.get("recent", []) if os.path.normcase(p) != os.path.normcase(path)]
        self.settings["recent"] = [path] + recent[: RECENT - 1]
        self._save_settings()

    def recent_cases(self) -> list[dict]:
        return [{"path": p, "name": os.path.splitext(os.path.basename(p))[0], "exists": os.path.isfile(p)}
                for p in self.settings.get("recent", [])]

    def state(self) -> dict:
        from ... import __version__

        return {"version": __version__, "ct": self.ct_info(), "scans": [self.scan_info(s) for s in self.scans],
                "case": None if not self.case_path else {"path": self.case_path,
                                                         "name": os.path.splitext(os.path.basename(self.case_path))[0]},
                "recent": self.recent_cases(),
                "models_dir": self.models_dir, "models": self.models_status(), **self.structures_info(),
                "segment_parts": self.segment_parts_info(), "bite_view": self.bite_info(),
                "mount_plane_default": self.settings["mount_plane"],
                "incognito": self.settings["incognito"],
                **self.landmarks_info(),
                "articulation": self.jaw.state(), "articulation_guides": self.articulation_guides()}


def route(h, s, p, method, q):
    """HTTP-маршруты шага (вызывает server.Handler.api): h — обработчик запроса, s — сессия; True — запрос обработан."""
    if p == ["case", "save"] and method == "POST":
        return h.json(s.save_case(h.body().get("path")))
    if p == ["case", "open"] and method == "POST":
        b = h.body()
        return h.job("Открываю кейс", lambda progress: s.open_case(b["path"], progress, b.get("ct_path")))
