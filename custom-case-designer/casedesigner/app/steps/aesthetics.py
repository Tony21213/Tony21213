"""Шаг «Эстетика»: фото анфас с улыбкой и с ретрактором — методы Session (docs/handoff/archive/aesthetics-photo.md).

Каждое фото совмещается со сканом верхней челюсти, как в exocad Smile Design: 2+ пары точек «фото — скан» дают
грубую камеру (aesthetic.fit_camera, расстояние до камеры — задано), доводит модель — маски зубов на фото по SAM 2.1
и подгонка камеры по контурам каждого зуба (photo_teeth.align; номера зубов — из сегментации КТ); поправить можно
мышью поверх фото (поворот скана — ПКМ, сдвиг и масштаб — с Shift) — тогда камера запоминается как есть.
Горизонт, средняя линия лица и точки лица отмечаются на фото. Камера фото с улыбкой даёт эстетическую систему
монтажа (шаг «Артикулятор»: photo_aesthetic), оба фото с камерами — набор Smile Design exocad (шаг «Экспорт»)."""

from ..common import (  # noqa: F401
    apply,
    errorlog,
    io,
    mo,
    np,
    os,
    rigid,
    trimesh,
)

PHOTO_KINDS = {"smile": "с улыбкой", "retracted": "с ретрактором"}
FACE_KEYS = ("pupil", "endocanthion", "canthus", "ala", "commissure")  # пары _right / _left — aesthetic.FACE_WIDTHS
CONTOUR_STEP = 2  # контуры зубов в окне — каждая вторая клетка
OCCLUDER_REACH_MM = 6.0  # нижние зубы, что загораживают верхние на фото с ретрактором: не дальше этого от верхнего скана
OCCLUDER_POINTS = 150_000
FACE_MODEL = "face_landmarker.task"  # MediaPipe Face Landmarker (Google, Apache-2.0) — в папке моделей
EYE_LINE_MAX_DEG = 10.0  # линия глаз круче — не горизонт (лицо повёрнуто или точки сбились)
ON_SCAN_MM = 2.0  # точка пары — на скане верхней челюсти: не дальше от его вершин
FIT_WARN_SHARE = 0.004  # ошибка совмещения больше этой доли диагонали фото — пары стоит проверить
PREVIEW_PX = 1600  # длинная сторона снимка для окна
OVERLAY_POINTS = 6000  # точек скана в наложении на фото
# Инкогнито: лицо размыто, резким остаётся рот — рамка пар точек с запасом (доля длинной стороны); пар меньше
# двух — средняя нижняя часть кадра, где рот на портрете анфас.
BLUR_KEEP_MARGIN = 0.06
BLUR_KEEP_DEFAULT = (0.25, 0.5, 0.75, 0.82)  # доли кадра: слева, сверху, справа, снизу


def _kind(kind: str) -> str:
    if kind not in PHOTO_KINDS:
        raise ValueError(f"нет такого фото: {kind}")
    return kind


def _xy(value, size, what: str) -> list[float]:
    try:
        u, v = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{what}: нужна точка на фото (x, y)") from None
    if not (np.isfinite(u) and np.isfinite(v) and 0 <= u <= size[0] and 0 <= v <= size[1]):
        raise ValueError(f"{what}: точка вне фото")
    return [round(u, 2), round(v, 2)]


class AestheticsSteps:
    # --- фото -------------------------------------------------------------------------------------------------------
    def _photo(self, kind: str) -> dict:
        item = self.photos.get(_kind(kind))
        if item is None:
            raise ValueError(f"фото {PHOTO_KINDS[kind]} ещё не загружено")
        return item

    def aesthetic_add_photo(self, kind: str, path: str) -> dict:
        """Загрузить фото (поворот и фокус — по EXIF): пары, горизонт и средняя линия — заново."""
        from ... import aesthetic as ae

        _kind(kind)
        errorlog.private(path, "фото")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("фото не найдено")
        try:
            photo = ae.load_photo(path)
        except OSError:
            raise ValueError("не удалось открыть фото: нужен JPEG или PNG") from None
        self.photos[kind] = self._new_photo(path, photo)
        self._auto_face(self.photos[kind])
        return self.aesthetics_info()

    def _face_model(self) -> str | None:
        folder = getattr(self, "models_dir", None)
        path = os.path.join(folder, FACE_MODEL) if folder else None
        if not path or not os.path.isfile(path):
            return None
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            return None
        return path

    def _auto_face(self, item: dict, force: bool = False) -> str | None:
        """Лицо на фото — автоматически (MediaPipe), как exocad Smile Creator: зрачки, углы глаз, нос, углы рта,
        средняя линия, контур губ; горизонт — по линии глаз. Поставленное врачом не трогает (force — заменить
        всё). Возвращает причину, если не вышло."""
        from ... import aesthetic as ae

        model = self._face_model()
        if model is None:
            return "нет модели MediaPipe (models/face_landmarker.task) или пакета mediapipe"
        try:
            hints = ae.face_hints(item["photo"].image, model)
        except (ValueError, RuntimeError) as e:
            item["face_note"] = str(e)
            return str(e)
        size = item["photo"].size
        inside = lambda q: 0 <= q[0] <= size[0] and 0 <= q[1] <= size[1]  # noqa: E731
        for key in (f"{k}_{s_}" for k in FACE_KEYS for s_ in ("right", "left")):
            if (force or key not in item["face"]) and key in hints and inside(hints[key]):
                item["face"][key] = [round(float(c), 2) for c in hints[key]]
        if force or not item["midline_px"]:
            item["midline_px"] = [[round(float(c), 2) for c in q] for q in hints["midline"] if inside(q)][:8]
        item["lips"] = {k: [[round(float(c), 1) for c in q] for q in hints[f"lips_{k}"]] for k in ("inner", "outer")}
        eye = float(hints["eye_line_deg"])
        if (force or item.get("horizon_source") in (None, "глаза")) and abs(eye) <= EYE_LINE_MAX_DEG:
            item["horizon_deg"], item["horizon_source"] = round(eye, 2), "глаза"
        item["face_note"] = None
        item["face_auto"] = True
        return None

    def aesthetic_face(self, kind: str) -> dict:
        """«Найти лицо»: точки лица, губы и горизонт по линии глаз — заново (MediaPipe)."""
        item = self._photo(kind)
        why = self._auto_face(item, force=True)
        if why:
            raise ValueError(f"лицо не найдено: {why}")
        return self.aesthetics_info()

    def _new_photo(self, path: str, photo) -> dict:
        self._photo_version = getattr(self, "_photo_version", 0) + 1
        return {"path": path, "photo": photo, "pairs": [], "horizon_deg": 0.0, "midline_px": [], "face": {},
                "upper_n": None, "version": self._photo_version, "_cam_key": None, "_cam": None, "_overlay": None,
                "distance_mm": None, "auto": True, "override": None, "_auto": None, "_refined": None,
                "lips": None, "horizon_source": None, "face_auto": False, "face_note": None}

    def aesthetic_remove_photo(self, kind: str) -> dict:
        self.photos.pop(_kind(kind), None)
        return self.aesthetics_info()

    # --- пары точек «фото — скан верхней челюсти» ---------------------------------------------------------------------
    def _upper(self):
        up, _lo = self._jaw_items()
        if up is None:
            raise ValueError("нужен совмещённый скан верхней челюсти (шаг «Сканы»)")
        return up

    def _upper_ct(self):
        """Скан верхней челюсти, его положение и вершины в координатах КТ (кэш: настоящие сканы — сотни тысяч
        вершин, а окно спрашивает после каждого щелчка)."""
        up = self._upper()
        T = np.asarray(up["reg"].transform, float)
        key = (id(up["scan"]), len(up["scan"].vertices), T.tobytes())
        cache = getattr(self, "_aes_upper", None)
        if cache is None or cache["key"] != key:
            cache = self._aes_upper = {"key": key, "v": apply(T, up["scan"].vertices), "tree": None, "base": None}
        return up, T, cache

    def _pairs_valid(self, item: dict) -> bool:
        """Пары — в координатах скана верхней челюсти: скан заменили (другое число вершин) — пары не годятся."""
        up = self._jaw_items()[0]
        return up is not None and item["upper_n"] == len(up["scan"].vertices)

    def aesthetic_pair(self, kind: str, px, point) -> dict:
        """Пара: пиксель на фото и та же точка на скане верхней челюсти (координаты окна 3D = КТ, мм)."""
        from scipy.spatial import cKDTree

        item = self._photo(kind)
        uv = _xy(px, item["photo"].size, "пара")
        up, T, cache = self._upper_ct()
        p = np.asarray(point, float).reshape(3)
        if not np.isfinite(p).all():
            raise ValueError("пара: нужна точка на скане")
        if cache["tree"] is None:
            cache["tree"] = cKDTree(cache["v"])
        if cache["tree"].query(p)[0] > ON_SCAN_MM:
            raise ValueError("точка пары — не на скане верхней челюсти")
        if not self._pairs_valid(item):
            item["pairs"], item["upper_n"] = [], len(up["scan"].vertices)
        local = apply(np.linalg.inv(T), p[None])[0]
        item["pairs"].append({"px": uv, "local": [round(float(c), 4) for c in local]})
        item["override"] = None
        return self.aesthetics_info()

    def aesthetic_unpair(self, kind: str, index: int | None = None) -> dict:
        """Удалить пару (index) или все пары фото."""
        item = self._photo(kind)
        if index is None:
            item["pairs"] = []
        else:
            if not 0 <= int(index) < len(item["pairs"]):
                raise ValueError("нет такой пары")
            del item["pairs"][int(index)]
        item["override"] = None
        return self.aesthetics_info()

    def aesthetic_set(self, kind: str, horizon_deg=None, midline_px=None, face=None, distance_mm=None,
                      auto=None) -> dict:
        """Горизонт (угол истинной горизонтали в кадре, градусы), точки средней линии лица, точки лица
        ({"pupil_right": [x, y] | None, …} — меняются только названные), расстояние до камеры (мм; 0 — по
        умолчанию) и доводка моделью (auto)."""
        from ... import aesthetic as ae

        item = self._photo(kind)
        size = item["photo"].size
        if distance_mm is not None:
            d = float(distance_mm)
            if d and not ae.DISTANCE_RANGE_MM[0] <= d <= ae.DISTANCE_RANGE_MM[1]:
                lo, hi = ae.DISTANCE_RANGE_MM
                raise ValueError(f"расстояние до камеры — от {lo:g} до {hi:g} мм")
            item["distance_mm"], item["override"] = (round(d) if d else None), None
        if auto is not None:
            item["auto"], item["override"] = bool(auto), None
        if horizon_deg is not None:
            h = float(horizon_deg)
            if not np.isfinite(h) or abs(h) > 45:
                raise ValueError("горизонт: угол от −45° до 45°")
            item["horizon_deg"], item["horizon_source"] = round(h, 2), ("кадр" if h == 0 else "вручную")
        if midline_px is not None:
            item["midline_px"] = [_xy(p, size, "средняя линия") for p in midline_px][:8]
        if face is not None:
            known = {f"{k}_{s}" for k in FACE_KEYS for s in ("right", "left")}
            for key, value in dict(face).items():
                if key not in known:
                    raise ValueError(f"нет такой точки лица: {key}")
                if value is None:
                    item["face"].pop(key, None)
                else:
                    item["face"][key] = _xy(value, size, "точка лица")
        return self.aesthetics_info()

    # --- камера и эстетическая система --------------------------------------------------------------------------------
    def _base_frame(self) -> np.ndarray:
        """Функциональная система для начального положения камеры и эстетической системы: монтаж шага
        «Артикулятор», а до него — по сканам (окклюзионная плоскость и дуга)."""
        if self.jaw.anatomy is not None:
            return np.asarray(self.jaw.anatomy.frame, float)
        up, lo = self._jaw_items()
        if up is None or lo is None:
            raise ValueError("нужны совмещённые сканы обеих челюстей")
        _up, _T, cache = self._upper_ct()
        T_lo = np.asarray(lo["reg"].transform, float)
        key = (id(lo["scan"]), T_lo.tobytes())
        if cache["base"] is None or cache["base"][0] != key:
            uv = cache["v"]
            lv = apply(T_lo, lo["scan"].vertices)
            z, y = mo.arch_axes(uv, mo.occlusal_up(uv, lv))
            y = y - (y @ z) * z
            y /= np.linalg.norm(y)
            R = np.array([np.cross(y, z), y, z])
            cache["base"] = (key, rigid(R, -R @ uv.mean(0)))
        return cache["base"][1]

    def _need_pairs(self, item: dict) -> int:
        from ... import aesthetic as ae

        return ae.MIN_PAIRS

    def _distance(self, item) -> float:
        from ... import aesthetic as ae

        return float(item.get("distance_mm") or ae.DEFAULT_DISTANCE_MM)

    def _aes_teeth(self, compute: bool = True):
        """Зубы верхнего скана для доводки: номера вершин (сегментация КТ), нормали — кэш; None — номеров нет
        (или ещё не посчитаны: compute=False — не считать, для окна)."""
        from ... import photo_teeth as pt

        up, _T, cache = self._upper_ct()
        teeth = {int(k.rsplit("_", 1)[1]): np.asarray(m.vertices) for k, m in self.structures.items()
                 if k.startswith("teeth/tooth_") and k.rsplit("_", 1)[1].isdigit() and int(k.rsplit("_", 1)[1]) // 10 in (1, 2, 5, 6)}
        key = (cache["key"], tuple(sorted((k, len(v)) for k, v in teeth.items())))
        if cache.get("teeth_key") != key and not compute:
            return None, None
        if cache.get("teeth_key") != key:
            labels = normals = None
            if teeth:
                # по близости к зубам КТ (до photo_teeth.REACH_MM): признак «на коронке» по совмещению
                # (Registration.deviation) бывает почти пустым — тогда от зубов оставались обрывки контуров
                labels = pt.crown_labels(cache["v"], teeth)
                if (labels > 0).sum() < 200:
                    labels = None
                else:
                    normals = np.asarray(trimesh.Trimesh(cache["v"], up["scan"].faces, process=False).vertex_normals)
            cache["teeth_key"], cache["teeth"] = key, (labels, normals)
        return cache["teeth"]

    def _aes_occluders(self):
        """Нижние зубы в прикусе (координаты КТ) — загораживают верхние на фото с ретрактором: (точки, нормали)."""
        from scipy.spatial import cKDTree

        _up, lo = self._jaw_items()
        if lo is None or lo.get("reg") is None:
            return None, None
        T = np.asarray(lo["reg"].transform, float)
        if getattr(self, "bite_motion", None) is not None:
            T = np.asarray(self.bite_motion, float) @ T
        _u, _T, cache = self._upper_ct()
        key = (id(lo["scan"]), T.tobytes(), cache["key"])
        if cache.get("occ_key") != key:
            v = apply(T, lo["scan"].vertices)
            n = np.asarray(trimesh.Trimesh(v, lo["scan"].faces, process=False).vertex_normals)
            near = np.isfinite(cKDTree(cache["v"]).query(v, distance_upper_bound=OCCLUDER_REACH_MM)[0])
            v, n = v[near], n[near]
            if len(v) > OCCLUDER_POINTS:
                k = np.random.default_rng(0).choice(len(v), OCCLUDER_POINTS, replace=False)
                v, n = v[k], n[k]
            cache["occ_key"], cache["occ"] = key, (v, n)
        return cache["occ"]

    def _sam(self):
        from ... import photo_teeth as pt

        folder = getattr(self, "models_dir", None)
        if not pt.Sam2.available(folder):
            return None
        sam = getattr(self, "_sam2", None)
        if sam is None or sam.folder != os.path.join(folder, pt.MODEL_DIR):
            sam = self._sam2 = pt.Sam2(folder)
        return sam

    def _auto_refine(self, kind: str, item: dict, camera):
        """Доводка моделью: маски зубов на фото (SAM) и подгонка камеры по контурам; (Refined | None, причина)."""
        from ... import photo_teeth as pt

        sam = self._sam()
        if sam is None:
            return None, "нет нейросети SAM 2.1 (models/sam2.1-tiny)"
        labels, normals = self._aes_teeth()
        if labels is None:
            return None, "нет номеров зубов: нужна сегментация зубов на КТ"
        occ, occ_n = self._aes_occluders() if kind == "retracted" else (None, None)
        anchor = self._upper_incisal(self._base_frame())
        try:
            r = pt.align(sam, item["photo"].image, camera, self._upper_ct()[2]["v"], labels, anchor,
                         self._distance(item), normals, occ, occ_n)
        except ValueError as e:
            return None, str(e)
        if not r.teeth:
            return None, "зубы на фото не совпали со сканом — проверьте пары точек"
        return r, None

    def _camera(self, kind: str, refine: bool = False):
        """Камера совмещения фото (aesthetic.Camera, координаты КТ) или None; вторым — почему её нет.
        Запомненная (ручная поправка или «Уточнить по зубам») — как есть; иначе — по парам, а если доводка уже
        посчитана для этих пар — доведённая. Доводку (десятки секунд: нейросеть) считает только refine=True —
        фоновая задача окна (aesthetic_auto) и выгрузка; окно, монтаж и плоскости её не ждут."""
        from ... import aesthetic as ae

        item = self.photos.get(kind)
        if item is None:
            return None, "нет фото"
        if item.get("override") is not None:
            o = item["override"]
            return ae.Camera(np.asarray(o["K"], float), np.asarray(o["R"], float), np.asarray(o["t"], float), 0.0,
                             bool(o.get("mirrored")), tuple(item["photo"].size)), None
        if not self._pairs_valid(item) or len(item["pairs"]) < self._need_pairs(item):
            n = len(item["pairs"]) if self._pairs_valid(item) else 0
            return None, f"пар точек {n} из {self._need_pairs(item)}"
        _up, T, _cache = self._upper_ct()
        px = np.array([p["px"] for p in item["pairs"]], float)
        pts = apply(T, np.array([p["local"] for p in item["pairs"]], float))
        key = (T.tobytes(), px.tobytes(), pts.tobytes(), self._distance(item), bool(item["auto"]),
               self._sam() is not None, len(self.structures))
        if item["_cam_key"] != key:
            base = self._base_frame()
            try:
                cam = ae.fit_camera(px, pts, item["photo"].size, base, distance_mm=self._distance(item),
                                    anchor=self._upper_incisal(base))
            except (ValueError, np.linalg.LinAlgError) as e:
                return None, str(e)
            item["_auto"], item["_refined"], item["_auto_key"] = None, None, None
            item["_cam_key"], item["_cam"], item["_coarse"], item["_overlay"] = key, cam, cam, None
        if refine and item["auto"] and item.get("_auto_key") != key:
            with self._aes_lock():
                if item.get("_auto_key") != key:  # пока ждали — посчитал другой запрос
                    r, why = self._auto_refine(kind, item, item["_coarse"])
                    cam = item["_coarse"]
                    if r is not None and np.mean(list(r.iou_after.values()) or [0]) >= np.mean(list(r.iou_before.values()) or [0]):
                        cam = r.camera
                        item["_refined"] = r
                    item["_auto"] = self._auto_report(r, why)
                    item["_cam"], item["_overlay"], item["_auto_key"] = cam, None, key
        return item["_cam"], None

    def _aes_lock(self):
        import threading

        lock = getattr(self, "_aes_refine_lock", None)
        if lock is None:
            lock = self._aes_refine_lock = threading.Lock()
        return lock

    def _auto_pending(self, item) -> bool:
        """Доводку для этих пар ещё не считали (окно запустит фоновую задачу)."""
        return bool(item.get("auto") and item.get("override") is None and item.get("_cam") is not None
                    and item.get("_auto_key") != item.get("_cam_key") and self._sam() is not None)

    def aesthetic_auto(self, kind: str | None = None) -> dict:
        """Фоновая задача окна: доводка нейросетью для фото, где она ещё не посчитана."""
        for k in ([kind] if kind else list(PHOTO_KINDS)):
            if self.photos.get(k) is not None:
                self._camera(k, refine=True)
        return self.aesthetics_info()

    @staticmethod
    def _auto_report(r, why=None) -> dict:
        if r is None:
            return {"used": False, "error": why}
        mean = lambda d: round(float(np.mean([d[n] for n in r.teeth])), 3) if r.teeth else None  # noqa: E731
        return {"used": True, "error": None, "teeth": list(r.teeth), "iou_before": mean(r.iou_before),
                "iou_after": mean(r.iou_after), "edge_px": None if r.edge_px is None else round(r.edge_px, 1)}

    def _remember_camera(self, item: dict, cam) -> None:
        item["override"] = {"K": cam.K.round(6).tolist(), "R": cam.R.round(9).tolist(), "t": cam.t.round(6).tolist(),
                            "mirrored": bool(cam.mirrored)}
        item["_overlay"] = None

    def aesthetic_adjust(self, kind: str, manual: dict) -> dict:
        """Ручная поправка совмещения поверх фото (aesthetic.adjust_camera): поворот скана, сдвиг, масштаб —
        от текущей камеры; камера запоминается как есть."""
        from ... import aesthetic as ae

        item = self._photo(kind)
        cam, why = self._camera(kind)
        if cam is None:
            raise ValueError(f"поправлять нечего: {why}")
        delta = {k: float(v) for k, v in dict(manual or {}).items() if k in ae.MANUAL_KEYS}
        anchor = self._upper_incisal(self._base_frame())
        self._remember_camera(item, ae.adjust_camera(cam, delta, anchor))
        item["_refined"] = None
        item["_auto"] = {"used": False, "error": None, "manual": True}
        return self.aesthetics_info()

    def aesthetic_refine(self, kind: str) -> dict:
        """«Уточнить по зубам»: доводка моделью от текущей камеры (после ручной поправки — от неё)."""
        item = self._photo(kind)
        cam, why = self._camera(kind)
        if cam is None:
            raise ValueError(f"уточнять нечего: {why}")
        with self._aes_lock():  # не вместе с фоновой доводкой
            r, why = self._auto_refine(kind, item, cam)
        if r is None:
            raise ValueError(why)
        self._remember_camera(item, r.camera)
        item["_refined"], item["_auto"] = r, self._auto_report(r)
        return self.aesthetics_info()

    def aesthetic_reset_camera(self, kind: str) -> dict:
        """Забыть поправку: камера — снова по парам и доводке."""
        item = self._photo(kind)
        item["override"], item["_cam_key"] = None, None
        return self.aesthetics_info()

    def _upper_incisal(self, base: np.ndarray) -> np.ndarray:
        """Точка между центральными резцами верхней челюсти (координаты КТ): самая передняя у режущих краёв."""
        R = np.asarray(base, float)[:3, :3]
        cache = self._upper_ct()[2]
        key = R.round(9).tobytes()
        if cache.get("incisal_key") != key:
            cache["incisal_key"], cache["incisal"] = key, mo._incisal(cache["v"], R[1], -R[2], mo.INCISAL_TILT_DEG)
        return cache["incisal"]

    def _mount_kind(self) -> str | None:
        """Фото для монтажа: с улыбкой, без него — с ретрактором (запасной вариант)."""
        photos = getattr(self, "photos", {})  # сессия, собранная без __init__ (тесты шагов), — без фото
        for kind in PHOTO_KINDS:
            if photos.get(kind) is not None and self._camera(kind)[0] is not None:
                return kind
        return None

    def photo_aesthetic(self):
        """Эстетическая система по фото для JawCase.mount(photo_aesthetic=...): функция «система монтажа →
        aesthetic.Aesthetic» (камера, горизонт, средняя линия, резцы); нет совмещённого фото — None."""
        from ... import aesthetic as ae

        kind = self._mount_kind()
        if kind is None:
            return None
        item, cam = self.photos[kind], self._camera(kind)[0]
        midline = item["midline_px"] if len(item["midline_px"]) >= 2 else None
        face = dict(item["face"]) or None

        def build(base):
            return ae.aesthetic_frame(base, cam, item["horizon_deg"], midline, self._upper_incisal(base), face=face)
        return build

    def _overlay(self, kind: str, cam) -> dict:
        """Наложение скана на фото: контуры зубов скана — точками скана (3D, координаты КТ: окно перепроецирует их,
        пока скан двигают мышью), контуры масок зубов на фото (доводка) — пиксели; без номеров зубов — видимые
        точки скана (3D)."""
        from ... import photo_teeth as pt

        item = self.photos[kind]
        if item["_overlay"] is None:
            up, _T, cache = self._upper_ct()
            labels, normals = self._aes_teeth(compute=False)
            out = {"scan": [], "photo": [], "points": []}
            if labels is not None:
                on = labels > 0
                P, L, N = cache["v"][on], labels[on], normals[on]
                if len(P) > pt.MAX_POINTS:
                    k = np.random.default_rng(0).choice(len(P), pt.MAX_POINTS, replace=False)
                    P, L, N = P[k], L[k], N[k]
                try:
                    window = pt.teeth_window(cam, P, item["photo"].size)
                    grid = pt.grid_for(cam, P, L, window, N)
                    occ, occ_n = self._aes_occluders() if kind == "retracted" else (None, None)
                    view = pt.project(cam, P, L, window, grid, N, occ, occ_n)
                    out["scan"] = [{"tooth": c["tooth"], "points": c["points"].round(3).tolist()}
                                   for c in pt.scan_contours(view, P, CONTOUR_STEP)]
                except ValueError:
                    pass
            r = item.get("_refined")
            if r is not None and r.masks:
                out["photo"] = [{"tooth": c["tooth"], "px": c["px"].round(1).tolist()}
                                for c in pt.mask_contours(r.masks, r.window, r.grid, CONTOUR_STEP)]
            if not out["scan"]:
                mesh = trimesh.Trimesh(cache["v"], up["scan"].faces, process=False)
                v, n = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
                v = v[np.einsum("ij,ij->i", n, cam.centre - v) > 0]
                if len(v) > OVERLAY_POINTS:
                    v = v[np.random.default_rng(0).choice(len(v), OVERLAY_POINTS, replace=False)]
                out["points"] = v.round(3).tolist()
            item["_overlay"] = out
        return item["_overlay"]

    def _export_photos(self) -> list:
        """Фото с камерами — для ArticulatorCase.photos (набор Smile Design exocad)."""
        out, photos = [], getattr(self, "photos", {})  # сессия без шага «Эстетика» (тесты выгрузки) — фото нет
        for kind in PHOTO_KINDS:
            cam = self._camera(kind, refine=True)[0] if photos.get(kind) is not None else None
            if cam is not None:
                out.append({"kind": kind, "image": self.photos[kind]["photo"].image, "camera": cam})
        return out

    # --- для окна --------------------------------------------------------------------------------------------------
    def aesthetics_info(self) -> dict:
        from ... import aesthetic as ae

        up = self._jaw_items()[0]
        photos = {}
        for kind, label in PHOTO_KINDS.items():
            item = self.photos.get(kind)
            if item is None:
                photos[kind] = None
                continue
            w, h = item["photo"].size
            valid = self._pairs_valid(item)
            T = None if up is None else np.asarray(up["reg"].transform, float)
            pairs = [{"px": p["px"], "point": apply(T, np.array([p["local"]], float))[0].round(3).tolist()}
                     for p in item["pairs"]] if valid and T is not None else []
            cam, why = self._camera(kind)
            info = {"label": label, "size": [w, h], "focal": item["photo"].focal_px is not None,
                    "notes": list(item["photo"].notes), "version": item["version"], "pairs": pairs,
                    "need": self._need_pairs(item), "fit_error": why, "camera": None,
                    "distance_mm": self._distance(item), "distance_set": item.get("distance_mm") is not None,
                    "auto": bool(item.get("auto", True)), "auto_info": item.get("_auto"), "auto_pending": False,
                    "auto_ready": self._sam() is not None,
                    "horizon_deg": item["horizon_deg"], "midline_px": item["midline_px"], "face": item["face"],
                    "horizon_source": item.get("horizon_source"), "lips": item.get("lips"),
                    "face_auto": bool(item.get("face_auto")), "face_note": item.get("face_note"),
                    "face_ready": self._face_model() is not None,
                    "overlay": {"scan": [], "photo": [], "points": []}, "widths": {}, "guide": [], "aesthetic": None}
            if item["pairs"] and not valid:
                info["notes"].append("скан верхней челюсти заменён — пары точек нужно поставить заново")
            info["auto_pending"] = self._auto_pending(item)
            if cam is not None:
                diag = float(np.hypot(w, h))
                px_ = np.array([p["px"] for p in item["pairs"]], float) if valid and item["pairs"] else np.zeros((0, 2))
                pts_ = apply(T, np.array([p["local"] for p in item["pairs"]], float)) if len(px_) else np.zeros((0, 3))
                residual = float(np.sqrt(np.mean(np.sum((cam.project(pts_) - px_) ** 2, 1)))) if len(px_) else 0.0
                anchor = self._upper_incisal(self._base_frame())
                info["camera"] = {"residual_px": round(residual, 1), "mirrored": bool(cam.mirrored),
                                  "focal_35mm": round(float(cam.K[0, 0]) * ae.FULL_FRAME_DIAGONAL_MM / diag, 1),
                                  "distance_mm": round(float((anchor - cam.centre) @ cam.R[2])),
                                  "check": residual > FIT_WARN_SHARE * diag and not item.get("override"),
                                  "K": cam.K.round(6).tolist(), "R": cam.R.round(9).tolist(), "t": cam.t.round(6).tolist(),
                                  "anchor": anchor.round(4).tolist(), "fixed": item.get("override") is not None}
                info["overlay"] = self._overlay(kind, cam)
                try:
                    base = self._base_frame()
                    inc = self._upper_incisal(base)
                    widths = ae.face_widths_mm(cam, item["face"], inc)
                    info["widths"] = widths
                    info["guide"] = [{"name": k, "mm": v, "why": why_}
                                     for k, (v, why_) in ae.tooth_guide(widths).items()]
                    midline = item["midline_px"] if len(item["midline_px"]) >= 2 else None
                    a = ae.aesthetic_frame(base, cam, item["horizon_deg"], midline, inc, face=dict(item["face"]) or None)
                    info["aesthetic"] = {"roll_deg": a.roll_deg, "midline_offset_mm": a.midline_offset_mm,
                                         "facial_midline_tilt_deg": a.facial_midline_tilt_deg, "hints": a.hints,
                                         "notes": a.notes}
                except (ValueError, np.linalg.LinAlgError):
                    pass
            photos[kind] = info
        source = getattr(self.jaw.anatomy, "source", "") or ""
        return {"photos": photos, "upper": up is not None, "mount_kind": self._mount_kind(),
                "mounted_by_photo": "(фото" in source or "по фото" in source,
                "missing": list(getattr(self, "photo_missing", []))}

    def aesthetic_image(self, kind: str) -> bytes:
        """Снимок для окна (JPEG, длинная сторона до PREVIEW_PX); в инкогнито лицо размыто, рот — резкий."""
        from PIL import Image, ImageFilter

        item = self._photo(kind)
        image = Image.fromarray(np.ascontiguousarray(item["photo"].image))
        w, h = image.size
        scale = min(1.0, PREVIEW_PX / max(w, h))
        if scale < 1:
            image = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        if self.settings.get("incognito"):
            pw, ph = image.size
            if self._pairs_valid(item) and len(item["pairs"]) >= 2:
                px = np.array([p["px"] for p in item["pairs"]], float) * scale
                m = BLUR_KEEP_MARGIN * max(pw, ph)
                box = (px[:, 0].min() - m, px[:, 1].min() - m, px[:, 0].max() + m, px[:, 1].max() + m)
            else:
                box = (BLUR_KEEP_DEFAULT[0] * pw, BLUR_KEEP_DEFAULT[1] * ph, BLUR_KEEP_DEFAULT[2] * pw,
                       BLUR_KEEP_DEFAULT[3] * ph)
            box = tuple(int(round(np.clip(c, 0, lim))) for c, lim in zip(box, (pw, ph, pw, ph)))
            sharp = image.crop(box) if box[2] > box[0] and box[3] > box[1] else None
            image = image.filter(ImageFilter.GaussianBlur(max(pw, ph) / 40))
            if sharp is not None:
                image.paste(sharp, box[:2])
        out = io.BytesIO()
        image.convert("RGB").save(out, "JPEG", quality=88)
        return out.getvalue()

    # --- кейс ---------------------------------------------------------------------------------------------------------
    def _photos_meta(self) -> dict:
        return {kind: {"path": item["path"], "pairs": item["pairs"], "upper_n": item["upper_n"],
                       "horizon_deg": item["horizon_deg"], "midline_px": item["midline_px"], "face": item["face"],
                       "distance_mm": item.get("distance_mm"), "auto": bool(item.get("auto", True)),
                       "override": item.get("override"), "lips": item.get("lips"),
                       "horizon_source": item.get("horizon_source"), "face_auto": bool(item.get("face_auto"))}
                for kind, item in self.photos.items()}

    def _load_photos_meta(self, meta: dict | None):
        """Фото кейса — по путям (как сканы): нет файла — фото пропускается, в окне — пометка."""
        from ... import aesthetic as ae

        self.photos, self.photo_missing = {}, []
        for kind, saved in (meta or {}).items():
            if kind not in PHOTO_KINDS or not isinstance(saved, dict):
                continue
            path = saved.get("path")
            errorlog.private(path, "фото")
            try:
                photo = ae.load_photo(path) if path and os.path.isfile(path) else None
            except OSError:
                photo = None
            if photo is None:
                self.photo_missing.append(PHOTO_KINDS[kind])
                continue
            item = self._new_photo(path, photo)
            item.update(pairs=list(saved.get("pairs") or []), upper_n=saved.get("upper_n"),
                        horizon_deg=float(saved.get("horizon_deg") or 0.0),
                        midline_px=list(saved.get("midline_px") or []), face=dict(saved.get("face") or {}),
                        distance_mm=saved.get("distance_mm"), auto=bool(saved.get("auto", True)),
                        override=saved.get("override"), lips=saved.get("lips"),
                        horizon_source=saved.get("horizon_source"), face_auto=bool(saved.get("face_auto")))
            self.photos[kind] = item


def route(h, s, p, method, q):
    """HTTP-маршруты шага «Эстетика» (вызывает server.Handler.api)."""
    if not p or p[0] != "aesthetics":
        return False
    if p == ["aesthetics"] and method == "GET":
        return h.json(s.aesthetics_info())
    if len(p) == 3 and p[2] == "image" and method == "GET":
        return h.send(s.aesthetic_image(p[1]), "image/jpeg")
    if method != "POST":
        return False
    b = h.body()
    if p == ["aesthetics", "photo"]:
        return h.json(s.aesthetic_add_photo(b.get("kind"), b.get("path")))
    if p == ["aesthetics", "photo", "remove"]:
        return h.json(s.aesthetic_remove_photo(b.get("kind")))
    if p == ["aesthetics", "pair"]:
        return h.json(s.aesthetic_pair(b.get("kind"), b.get("px"), b.get("point")))
    if p == ["aesthetics", "pair", "remove"]:
        return h.json(s.aesthetic_unpair(b.get("kind"), b.get("index")))
    if p == ["aesthetics", "set"]:
        return h.json(s.aesthetic_set(b.get("kind"), b.get("horizon_deg"), b.get("midline_px"), b.get("face"),
                                      b.get("distance_mm"), b.get("auto")))
    if p == ["aesthetics", "adjust"]:  # ручная поправка поверх фото (ПКМ — поворот скана, Shift — сдвиг и масштаб)
        return h.json(s.aesthetic_adjust(b.get("kind"), b.get("manual") or {}))
    if p == ["aesthetics", "face"]:  # «Найти лицо» — MediaPipe: точки лица, губы, горизонт по линии глаз
        return h.json(s.aesthetic_face(b.get("kind")))
    if p == ["aesthetics", "auto"]:  # доводка нейросетью для новых пар — фоновой задачей окна
        return h.job("Доводка по зубам", lambda progress: s.aesthetic_auto(b.get("kind")))
    if p == ["aesthetics", "refine"]:  # «Уточнить по зубам» — доводка моделью от текущей камеры
        return h.job("Уточнение по зубам", lambda progress: s.aesthetic_refine(b.get("kind")))
    if p == ["aesthetics", "camera", "reset"]:
        return h.json(s.aesthetic_reset_camera(b.get("kind")))
    return False
