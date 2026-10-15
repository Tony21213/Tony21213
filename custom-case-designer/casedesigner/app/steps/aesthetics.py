"""Шаг «Эстетика»: фото анфас с улыбкой и с ретрактором — методы Session (docs/handoff/archive/aesthetics-photo.md).

Каждое фото совмещается со сканом верхней челюсти по своим парам точек «фото — скан» (aesthetic.fit_camera);
горизонт, средняя линия лица и точки лица отмечаются на фото. Камера фото с улыбкой даёт эстетическую систему
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
PAIRS_WITH_FOCAL, PAIRS_WITHOUT_FOCAL = 4, 6  # aesthetic.fit_camera: фокус из EXIF или подбирается
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
        return self.aesthetics_info()

    def _new_photo(self, path: str, photo) -> dict:
        self._photo_version = getattr(self, "_photo_version", 0) + 1
        return {"path": path, "photo": photo, "pairs": [], "horizon_deg": 0.0, "midline_px": [], "face": {},
                "upper_n": None, "version": self._photo_version, "_cam_key": None, "_cam": None, "_overlay": None}

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
        return self.aesthetics_info()

    def aesthetic_set(self, kind: str, horizon_deg=None, midline_px=None, face=None) -> dict:
        """Горизонт (угол истинной горизонтали в кадре, градусы), точки средней линии лица, точки лица
        ({"pupil_right": [x, y] | None, …} — меняются только названные)."""
        item = self._photo(kind)
        size = item["photo"].size
        if horizon_deg is not None:
            h = float(horizon_deg)
            if not np.isfinite(h) or abs(h) > 45:
                raise ValueError("горизонт: угол от −45° до 45°")
            item["horizon_deg"] = round(h, 2)
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
        return PAIRS_WITH_FOCAL if item["photo"].focal_px is not None else PAIRS_WITHOUT_FOCAL

    def _camera(self, kind: str):
        """Камера совмещения фото (aesthetic.Camera, координаты КТ) или None; вторым — почему её нет."""
        from ... import aesthetic as ae

        item = self.photos.get(kind)
        if item is None:
            return None, "нет фото"
        if not self._pairs_valid(item) or len(item["pairs"]) < self._need_pairs(item):
            n = len(item["pairs"]) if self._pairs_valid(item) else 0
            return None, f"пар точек {n} из {self._need_pairs(item)}"
        _up, T, _cache = self._upper_ct()
        px = np.array([p["px"] for p in item["pairs"]], float)
        pts = apply(T, np.array([p["local"] for p in item["pairs"]], float))
        key = (T.tobytes(), px.tobytes(), pts.tobytes())
        if item["_cam_key"] != key:
            try:
                cam = ae.fit_camera(px, pts, item["photo"].size, self._base_frame(), item["photo"].focal_px)
            except (ValueError, np.linalg.LinAlgError) as e:
                return None, str(e)
            item["_cam_key"], item["_cam"], item["_overlay"] = key, cam, None
        return item["_cam"], None

    def _upper_incisal(self, base: np.ndarray) -> np.ndarray:
        """Точка между центральными резцами верхней челюсти (координаты КТ): самая передняя у режущих краёв."""
        R = np.asarray(base, float)[:3, :3]
        return mo._incisal(self._upper_ct()[2]["v"], R[1], -R[2], mo.INCISAL_TILT_DEG)

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

    def _overlay(self, kind: str, cam) -> list:
        """Скан верхней челюсти на фото по камере: вершины, обращённые к камере, — пиксели фото."""
        item = self.photos[kind]
        if item["_overlay"] is None:
            up, _T, cache = self._upper_ct()
            mesh = trimesh.Trimesh(cache["v"], up["scan"].faces, process=False)
            v, n = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
            centre = -cam.R.T @ cam.t
            facing = np.einsum("ij,ij->i", n, centre - v) > 0
            v = v[facing]
            if len(v) > OVERLAY_POINTS:
                v = v[np.random.default_rng(0).choice(len(v), OVERLAY_POINTS, replace=False)]
            uv = cam.project(v) if len(v) else np.zeros((0, 2))
            w, h = item["photo"].size
            inside = (uv[:, 0] >= 0) & (uv[:, 0] <= w) & (uv[:, 1] >= 0) & (uv[:, 1] <= h)
            item["_overlay"] = np.round(uv[inside], 1).tolist()
        return item["_overlay"]

    def _export_photos(self) -> list:
        """Фото с камерами — для ArticulatorCase.photos (набор Smile Design exocad)."""
        out, photos = [], getattr(self, "photos", {})  # сессия без шага «Эстетика» (тесты выгрузки) — фото нет
        for kind in PHOTO_KINDS:
            cam = self._camera(kind)[0] if photos.get(kind) is not None else None
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
                    "horizon_deg": item["horizon_deg"], "midline_px": item["midline_px"], "face": item["face"],
                    "overlay": [], "widths": {}, "guide": [], "aesthetic": None}
            if item["pairs"] and not valid:
                info["notes"].append("скан верхней челюсти заменён — пары точек нужно поставить заново")
            if cam is not None:
                diag = float(np.hypot(w, h))
                info["camera"] = {"residual_px": round(cam.residual_px, 1), "mirrored": bool(cam.mirrored),
                                  "focal_35mm": round(float(cam.K[0, 0]) * ae.FULL_FRAME_DIAGONAL_MM / diag, 1),
                                  "check": cam.residual_px > FIT_WARN_SHARE * diag}
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
                       "horizon_deg": item["horizon_deg"], "midline_px": item["midline_px"], "face": item["face"]}
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
                        midline_px=list(saved.get("midline_px") or []), face=dict(saved.get("face") or {}))
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
        return h.json(s.aesthetic_set(b.get("kind"), b.get("horizon_deg"), b.get("midline_px"), b.get("face")))
    return False
