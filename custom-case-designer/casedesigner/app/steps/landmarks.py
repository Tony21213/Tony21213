"""Ориентиры и плоскости: точки, поиск ALI-CBCT, плоскости и артикуляторы — методы Session (разделено из session.py; контракты — там же, где были)."""

from ..common import (arts, cond, lmk, model_store, np, os, threading)  # noqa: F401


class LandmarkSteps:
    # --- ориентиры и плоскости ----------------------------------------------
    def landmarks_info(self) -> dict:
        planes = []
        for key, p in lmk.PLANES.items():
            missing = lmk.missing(self.landmarks, key)
            planes.append({"key": key, "name": p["name"], "ready": not missing, "missing": missing})
        return {
            "landmarks": [{"key": l.key, "name": l.name, "hint": l.hint,
                           "point": self.landmarks[l.key].round(2).tolist() if l.key in self.landmarks else None,
                           "suggested": l.key in self.suggested,
                           "source": self.landmark_sources.get(l.key),
                           "note": self.landmark_notes.get(l.key)} for l in lmk.LANDMARKS],
            "planes": planes, "angles": lmk.plane_angles(self.landmarks),
            "articulators": [{"key": a.key, "name": a.name, "maker": a.maker, "plane": a.plane,
                              "calibrated": a.calibrated} for a in arts.load(self.articulators_path)],
        }

    def set_landmark(self, key: str, point) -> dict:
        if key not in lmk.BY_KEY:
            raise ValueError(f"нет такого ориентира: {key}")
        with self.lock:
            if point is None:
                self.landmarks.pop(key, None)
                self.landmark_sources.pop(key, None)
            else:
                self.landmarks[key] = np.asarray(point, float)
                self.landmark_sources[key] = cond.SOURCE_USER
            self.suggested.discard(key)
        return self.landmarks_info()

    def _landmark_folder(self) -> str | None:
        """Папка моделей ориентиров: рядом с моделями сегментации или в папке пользователя."""
        from ...auto_landmarks import Models

        places = [model_store.landmarks_dir(d) for d in (self.models_dir, model_store.default_dir()) if d]
        usable = [p for p in places if os.path.isdir(p) and Models(p).available()]
        return (usable or places or [None])[0]

    def landmark_models_status(self) -> dict:
        from ...auto_landmarks import Models

        folder = self._landmark_folder()
        st = model_store.landmarks_status(os.path.dirname(folder) if folder else None)
        st["available"] = Models(folder).available() if folder and os.path.isdir(folder) else []
        st["usable"] = bool(st["available"])
        return st

    def download_landmark_models(self, progress=None) -> dict:
        """Скачать модели ориентиров (докачка, проверка SHA-256)."""
        folder = self.models_dir or model_store.default_dir()
        self._download_cancel = threading.Event()
        model_store.download_landmarks(folder, progress, self._download_cancel)
        return self.landmark_models_status()

    def auto_landmarks(self, progress=None, keys=None) -> dict:
        """Найти ориентиры на КТ нейросетью (ALI-CBCT). Поставленное врачом не трогается; найденное — «проверьте»."""
        from ... import auto_landmarks as al

        self._require_ct()
        folder = self._landmark_folder()
        models = al.Models(folder) if folder and os.path.isdir(folder) else None
        if models is None or not models.available():
            raise ValueError("нет моделей ориентиров: скачайте их в программе")
        self._landmark_cancel = threading.Event()
        found = al.find(self.vol, models, keys, progress, cancel=self._landmark_cancel)
        heads = (self._mandible_condyles(heads_only=True)
                 if any(k in found for k in cond.SIDES.values()) else {})
        notes = {}
        with self.lock:
            for key, f in found.items():
                if f.point is not None and key in lmk.BY_KEY and (key not in self.landmarks or key in self.suggested):
                    point, source = np.asarray(f.point, float), cond.SOURCE_ALI
                    if key in cond.SIDES.values():  # сверка с головкой, отрезанной от нижней челюсти
                        point, source, notes[key] = cond.prefer_head(point, heads.get(key))
                    self.landmarks[key] = point
                    self.suggested.add(key)
                    self.landmark_sources[key] = source
                    if key in cond.SIDES.values():
                        if notes.get(key):
                            self.landmark_notes[key] = notes[key]
                        else:
                            self.landmark_notes.pop(key, None)
                    self._joint_parts_cache_key = None
        info = self.landmarks_info()
        info["auto"] = {k: {"found": f.point is not None, "note": notes.get(k) or f.note} for k, f in found.items()}
        return info

    def suggest_landmarks(self) -> dict:
        """Предложить мыщелки и порионы по сегментации (не трогая поставленные врачом)."""
        found = {}
        if "mandible" in self.structures:
            found.update(self._mandible_condyles())  # головки — без венечного отростка
        mid_x = float(np.median(self.structures["mandible"].vertices[:, 0])) if "mandible" in self.structures else 0.0
        for side, key in (("right", "Po_R"), ("left", "Po_L")):
            canal = self.structures.get(f"auditory_canal_{side}")
            if canal is not None:
                found.update(lmk.suggest_porion(canal.vertices, key, mid_x))
        if not found:
            raise ValueError("нечего предложить: сначала сегментируйте КТ (нужны нижняя челюсть и слуховые проходы)")
        with self.lock:
            for key, point in found.items():
                if key not in self.landmarks or key in self.suggested:
                    self.landmarks[key] = np.asarray(point, float)
                    self.suggested.add(key)
                    self.landmark_sources[key] = cond.SOURCE_MANDIBLE if key in cond.SIDES.values() else "segmentation"
        return self.landmarks_info()

    def reference(self, kind: str) -> tuple[np.ndarray, str]:
        """Матрица «КТ → система» и её название: plane:<плоскость> или articulator:<ключ>."""
        what, _, key = kind.partition(":")
        if what == "plane":
            return lmk.reference_frame(self.landmarks, key), lmk.PLANES[key]["name"]
        if what == "articulator":
            art = next((a for a in arts.load(self.articulators_path) if a.key == key), None)
            if art is None:
                raise ValueError(f"нет такого артикулятора: {key}")
            return arts.articulator_frame(self.landmarks, art), f"{art.maker} {art.name}"
        raise ValueError(f"неизвестная система координат: {kind}")
