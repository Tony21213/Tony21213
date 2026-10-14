"""КТ: загрузка, серии, срезы и сечения, модели сегментации, структуры — методы Session (разделено из session.py; контракты — там же, где были)."""

from ..common import (  # noqa: F401
    CaseCT,
    DEV_COLORS,
    GUIDE_KEYS,
    MAX_SLICE_PIXELS,
    SLICES,
    SLICE_PIXELS,
    Sectioner,
    Segmenter,
    _axis_aligned,
    apply,
    errorlog,
    group_of,
    io,
    load_volume,
    mesh_bytes,
    model_store,
    np,
    os,
    part_of,
    structure,
    threading,
    trimesh,
)


class CtSteps:
    # --- КТ -----------------------------------------------------------------
    def ct_series(self, path: str) -> list[dict]:
        """Серии DICOM в папке или архиве — чтобы выбрать, какую открыть (у файла КТ — пусто)."""
        from ...volume import series_of

        errorlog.private(path, "КТ")
        return series_of(path)

    def load_ct(self, path: str, progress=None, series: str | None = None) -> dict:
        errorlog.private(path, "КТ")
        vol = load_volume(path, series)
        if progress:
            progress(0.5, "Ищу коронки зубов")
        prior = self.memory.prior(vol.device)
        case = CaseCT(vol, *prior)
        with self.lock:
            self.ct_path, self.vol, self.case, self.ct_series_id = path, vol, case, series
            self.case_path, self.guide_teeth = None, {}
            lo, hi = case.levels.hard, case.levels.dense
            self.window = ((lo + hi) / 2, max(hi - lo, 1.0) * 2.5)
            self.structures, self._sections = {}, {}
            self.bite_view, self.bite_motion, self._bite_meshes = False, None, {}
            self.bite_fix, self._bite_proposal = np.eye(4), None
            self.landmarks, self.suggested, self.landmark_sources = {}, set(), {}
            self.landmark_notes = {}
            self._joint_parts_cache_key = self._joint_parts_cache = None
            self._mounted_condyles = None
            for item in self.scans.values():
                item.update(reg=None, auto=None, transform=None, guided=False, lower_bite=None)
        return self.ct_info()

    def close_ct(self):
        """Без КТ: только сканы (кейс, сохранённый без КТ)."""
        with self.lock:
            self.ct_path, self.vol, self.case, self.ct_series_id = None, None, None, None
            self.case_path, self.guide_teeth = None, {}
            self.structures, self._sections = {}, {}
            self.bite_view, self.bite_motion, self._bite_meshes = False, None, {}
            self.bite_fix, self._bite_proposal = np.eye(4), None
            self.landmarks, self.suggested, self.landmark_sources = {}, set(), {}
            self.landmark_notes = {}
            self._joint_parts_cache_key = self._joint_parts_cache = None
            self._mounted_condyles = None
            for item in self.scans.values():
                item.update(reg=None, auto=None, transform=None, guided=False, lower_bite=None)

    def ct_info(self) -> dict | None:
        if self.vol is None:
            return None
        lo, hi = self.bounds()
        prior = self.memory.prior(self.vol.device)
        crowns = np.vstack([t.points for t in self.case.coarse.values()])
        return {"path": self.ct_path, "name": os.path.basename(os.path.normpath(self.ct_path)),
                "focus": crowns.mean(axis=0).round(2).tolist(),  # срезы открываются на зубах
                "shape": list(self.vol.data.shape[::-1]), "spacing": self.vol.spacing.round(4).tolist(),
                "device": self.vol.device, "bounds": [lo.tolist(), hi.tolist()],
                "window": [round(self.window[0]), round(self.window[1])],
                "levels": {"hard": round(float(self.case.levels.hard)), "dense": round(float(self.case.levels.dense)),
                           "min": round(float(self._cval))},
                "learned_edge_shift_mm": round(prior[0], 3) if prior[1] else None}

    def bounds(self):
        n = np.array(self.vol.data.shape[::-1]) - 1
        corners = np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])], float)
        w = self.vol.to_world(corners)
        return w.min(axis=0), w.max(axis=0)

    def slice_geometry(self, axis: str) -> dict:
        """Сетка картинки среза: мм пациента пикселя (i, j) = (u0 + i·du, v0 + j·dv) по осям u, v."""
        s = SLICES[axis]
        lo, hi = self.bounds()
        step = max(hi[s["u"]] - lo[s["u"]], hi[s["v"]] - lo[s["v"]]) / SLICE_PIXELS
        width = int((hi[s["u"]] - lo[s["u"]]) / step) + 1
        height = int((hi[s["v"]] - lo[s["v"]]) / step) + 1
        v0, dv = (float(lo[s["v"]]), step) if s["v_down"] else (float(lo[s["v"]] + (height - 1) * step), -step)
        return {"axis": axis, "u_axis": s["u"], "v_axis": s["v"], "normal_axis": s["normal"],
                "u0": float(lo[s["u"]]), "du": step, "v0": v0, "dv": dv, "width": width, "height": height,
                "range": [float(lo[s["normal"]]), float(hi[s["normal"]])], "step": step}

    def slice_values(self, axis: str, pos: float, region=None, size=None) -> np.ndarray:
        """Яркость среза (строки × столбцы); region=(ua, ub, va, vb) — только эта часть (мм по осям среза,
        края слева-направо и сверху-вниз), size=(cols, rows) — в таком разрешении."""
        g = self.slice_geometry(axis)
        s = SLICES[axis]
        if region is None:
            us, vs = g["u0"] + g["du"] * np.arange(g["width"]), g["v0"] + g["dv"] * np.arange(g["height"])
        else:
            (ua, ub, va, vb), (cols, rows) = region, (int(np.clip(n, 1, MAX_SLICE_PIXELS)) for n in size)
            us, vs = ua + (np.arange(cols) + 0.5) * (ub - ua) / cols, va + (np.arange(rows) + 0.5) * (vb - va) / rows
        if _axis_aligned(self.vol.direction):
            return self._plane(s, pos, us, vs)
        uu, vv = np.meshgrid(us, vs)  # КТ повёрнут относительно осей пациента — общий путь
        pts = np.zeros((uu.size, 3))
        pts[:, s["u"]], pts[:, s["v"]], pts[:, s["normal"]] = uu.ravel(), vv.ravel(), pos
        return self.vol.sample(pts).reshape(uu.shape)

    def _plane(self, s: dict, pos: float, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """Срез КТ, лежащего по осям пациента: плоскость берётся из массива (линейно между соседними
        слоями), в плоскости — билинейно по отдельности вдоль строк и столбцов. В десятки раз быстрее
        трёхмерной интерполяции и даёт то же самое."""
        vol = self.vol
        data = vol.data
        cval = self._cval
        to_index = vol.to_index
        k = {a: int(np.argmax(np.abs(vol.direction[a]))) for a in range(3)}  # ось пациента → ось индекса x,y,z

        def along(a, values):  # индекс вдоль оси пациента a для значений values (остальные — любые)
            pts = np.zeros((len(values), 3))
            pts[:, s["normal"]] = pos
            pts[:, a] = values
            return to_index(pts)[:, k[a]]

        def axis_of(index_axis):  # ось массива data ([z, y, x]) для оси индекса x/y/z
            return 2 - index_axis

        n_idx = float(along(s["normal"], np.array([pos]))[0])
        n0 = int(np.floor(n_idx))
        t = n_idx - n0
        n_axis = axis_of(k[s["normal"]])
        size_n = data.shape[n_axis]

        def layer(i):
            if 0 <= i < size_n:
                return np.take(data, i, axis=n_axis).astype(np.float32)
            return None

        a, b = layer(n0), layer(n0 + 1)
        if a is None and b is None:
            return np.full((len(vs), len(us)), cval, np.float32)
        a = a if a is not None else np.full_like(b, cval)
        b = b if b is not None else np.full_like(a, cval)
        plane = a * (1 - t) + b * t if t > 1e-6 else a
        rest = [ax for ax in (0, 1, 2) if ax != n_axis]  # оси массива, оставшиеся в плоскости
        u_ax, v_ax = rest.index(axis_of(k[s["u"]])), rest.index(axis_of(k[s["v"]]))
        plane = plane if (v_ax, u_ax) == (0, 1) else plane.T  # строки — v, столбцы — u

        def weights(idx, n):
            i0 = np.floor(idx).astype(np.int64)
            f = (idx - i0).astype(np.float32)
            ok0, ok1 = (i0 >= 0) & (i0 < n), (i0 + 1 >= 0) & (i0 + 1 < n)
            return np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1), f, ok0, ok1

        iu0, iu1, fu, ou0, ou1 = weights(along(s["u"], us), plane.shape[1])
        iv0, iv1, fv, ov0, ov1 = weights(along(s["v"], vs), plane.shape[0])
        p = np.where(ov0[:, None], plane[iv0], cval)  # строки
        q = np.where(ov1[:, None], plane[iv1], cval)
        rows = p * (1 - fv)[:, None] + q * fv[:, None]
        left = np.where(ou0[None], rows[:, iu0], cval)
        right = np.where(ou1[None], rows[:, iu1], cval)
        return left * (1 - fu)[None] + right * fu[None]

    @property
    def _cval(self) -> float:
        if getattr(self, "_cval_for", None) is not self.vol:
            self._cval_for, self._cval_value = self.vol, float(self.vol.data.min())
        return self._cval_value

    def _windowed(self, values: np.ndarray, level, width) -> np.ndarray:
        level = self.window[0] if level is None else level
        width = self.window[1] if width is None else width
        return np.clip((values - (level - width / 2)) * (255.0 / width), 0, 255).astype(np.uint8)

    def slice_png(self, axis: str, pos: float, level: float | None = None, width: float | None = None,
                  region=None, size=None) -> bytes:
        """Срез картинкой PNG (оттенки серого в окне level/width)."""
        from PIL import Image

        img = self._windowed(self.slice_values(axis, pos, region, size), level, width)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    def slice_raw(self, axis: str, pos: float, level: float | None = None, width: float | None = None,
                  region=None, size=None) -> bytes:
        """Срез без сжатия — для интерфейса: столбцы и строки (uint32), затем байты серого построчно.
        Сжатие PNG и его разбор в браузере дольше, чем передать байты по локальному соединению."""
        img = self._windowed(self.slice_values(axis, pos, region, size), level, width)
        return np.array([img.shape[1], img.shape[0]], np.uint32).tobytes() + np.ascontiguousarray(img).tobytes()

    # --- сечения сеток плоскостью среза --------------------------------------
    def _sectioner(self, key: str, vertices: np.ndarray, faces: np.ndarray, version) -> "Sectioner":
        cache = self.__dict__.setdefault("_sectioners", {})
        item = cache.get(key)
        if item is None or item[0] != version:
            item = (version, Sectioner(vertices, faces))
            cache[key] = item
        return item[1]

    def overlays(self, axis: str, pos: float, visible: list[str], heat: bool = False) -> list[dict]:
        out = []
        n = SLICES[axis]["normal"]
        uv = [SLICES[axis]["u"], SLICES[axis]["v"]]
        with self.lock:
            for sid, item in self.scans.items():
                if item["transform"] is None or sid not in visible:
                    continue
                T = self.shown_transform(item)
                sec = self._sectioner(f"scan:{sid}", apply(T, item["scan"].vertices), item["scan"].faces,
                                      (id(item["scan"]), T.tobytes()))
                if not heat or item["reg"] is None:
                    out.append({"id": sid, "color": item["color"], "width": 1.6, "segments": sec.cut(n, pos, uv)})
                    continue
                # Контур по цвету отклонения от КТ: где скан лёг на эмаль, а где нет.
                segs, faces = sec.cut(n, pos, uv, with_faces=True)
                if not len(faces):
                    continue
                dev = np.abs(item["reg"].deviation)
                d = np.nanmax(np.where(np.isnan(dev[item["scan"].faces[faces]]), -1, dev[item["scan"].faces[faces]]), 1)
                segs = np.asarray(segs).reshape(-1, 4)
                for color, mask in ((DEV_COLORS[0], (d >= 0) & (d <= 0.1)), (DEV_COLORS[1], (d > 0.1) & (d <= 0.2)),
                                    (DEV_COLORS[2], d > 0.2), (DEV_COLORS[3], d < 0)):
                    if mask.any():
                        out.append({"id": sid, "color": color, "width": 1.6 if color != DEV_COLORS[3] else 1.0,
                                    "segments": segs[mask].reshape(-1).tolist()})
            for key, mesh in self.structures.items():
                if key in visible:
                    sec = self._sectioner(f"structure:{key}", *self._shown_structure(key))
                    out.append({"id": key, "color": structure(key).color, "width": 1.2,
                                "segments": sec.cut(n, pos, uv)})
        return [o for o in out if o["segments"]]

    # --- модели сегментации --------------------------------------------------
    def models_status(self) -> dict:
        return model_store.status(self.models_dir or model_store.default_dir())

    def download_models(self, progress=None) -> dict:
        """Скачать недостающие модели сегментации (докачка, проверка SHA-256); после — сегментация доступна."""
        folder = self.models_dir or model_store.default_dir()
        self._download_cancel = threading.Event()
        result = model_store.download(folder, progress, self._download_cancel)
        self.models_dir = folder
        return result

    def cancel_download(self):
        cancel = getattr(self, "_download_cancel", None)
        if cancel is not None:
            cancel.set()

    # --- структуры ----------------------------------------------------------
    def segment(self, models_dir: str | None = None, device: str = "auto", progress=None) -> dict:
        """Сегментировать выбранные в настройке части КТ (segment_parts); зубы для совмещения — всегда."""
        self._require_ct()
        models_dir = models_dir or self.models_dir
        if not models_dir:
            raise ValueError("укажите папку моделей сегментации")
        parts = set(self.settings["segment_parts"])
        if not parts:
            raise ValueError("не выбрано, что сегментировать: откройте «Что сегментировать…»")
        shown = lambda key: part_of(key) in parts or part_of(key) is None  # noqa: E731
        segmenter = Segmenter(models_dir, device=device)
        names = {m.name: m.title for m in segmenter.models}
        planned = {m.name for m in segmenter.plan(lambda k: shown(k) or k in GUIDE_KEYS)}
        done = {}

        def report(model, n, total):
            done[model] = n / total
            if progress:
                progress(sum(done.values()) / len(planned), f"{names[model]}: окно {n}/{total}")

        result = segmenter.run(self.vol, progress=report, want=lambda k: shown(k) or k in GUIDE_KEYS)
        with self.lock:
            self.models_dir = models_dir
            self.structures = {k: trimesh.Trimesh(m.vertices, m.faces, process=False)
                               for k, m in result.meshes.items() if shown(k)}
        self.guide_teeth = {jaw: result.meshes.get(f"{jaw}_teeth") for jaw in ("upper", "lower")}
        self.case.use_teeth(self.guide_teeth)
        # Сканы, совмещённые до сегментации по одной плотности, — заново по зубам.
        # Принятые и поправленные вручную остаются как есть.
        for sid, item in list(self.scans.items()):
            if item["reg"] is not None and not item["guided"] and not item["accepted"] and item["reg"] is item["auto"]:
                if progress:
                    progress(1.0, f"Совмещаю заново по зубам: {item['scan'].name}")
                try:
                    self.register(sid, item["jaw"])
                except (ValueError, RuntimeError):  # не вышло — скан остаётся где был, с подсказкой
                    pass
        self._place_bites()  # челюсти могли переехать — сканы прикуса за ними
        return {**self.structures_info(), "scans": [self.scan_info(s) for s in self.scans]}

    def structures_info(self) -> dict:
        def order(key):  # сначала целые структуры, затем зубы и пульпа по номеру FDI
            kind = 1 if key.startswith("teeth/tooth_") else 2 if key.startswith("pulp/") else 0
            digits = "".join(ch for ch in key if ch.isdigit())
            return kind, int(digits) if kind and digits else 0, key

        items = [{"key": k, "name": structure(k).name, "color": structure(k).color, "group": group_of(k),
                  "jaw": structure(k).jaw} for k in sorted(self.structures, key=order)]
        return {"structures": items}

    def structure_mesh(self, key: str) -> bytes:
        v, f, _version = self._shown_structure(key)
        return mesh_bytes(v, f)


def route(h, s, p, method, q):
    """HTTP-маршруты шага (вызывает server.Handler.api): h — обработчик запроса, s — сессия; True — запрос обработан."""
    if p == ["ct"] and method == "POST":
        b = h.body()
        return h.job("Открываю КТ", lambda progress: s.load_ct(b["path"], progress, b.get("series")))
    if p == ["ct", "series"] and method == "POST":
        # DICOM folders and archives can contain thousands of files. Keep
        # the UI responsive while SimpleITK inspects them.
        return h.job("Читаю серии КТ", lambda progress: s.ct_series(h.body()["path"]))
    if p == ["ct", "geometry"]:
        return h.json(s.slice_geometry(q["axis"]))
    if p == ["ct", "slice"]:
        level = float(q["level"]) if "level" in q else None
        width = float(q["width"]) if "width" in q else None
        region = tuple(float(q[k]) for k in ("ua", "ub", "va", "vb")) if "ua" in q else None
        size = (int(q["cols"]), int(q["rows"])) if region else None
        if q.get("format") == "raw":
            return h.binary(s.slice_raw(q["axis"], float(q["pos"]), level, width, region, size))
        return h.send(s.slice_png(q["axis"], float(q["pos"]), level, width, region, size), "image/png")
    if p == ["ct", "surface"]:
        return h.binary(s.ct_surface(q.get("part", "teeth")))
    if p == ["overlays"] and method == "POST":
        b = h.body()
        return h.json(s.overlays(b["axis"], float(b["pos"]), b.get("visible", []), bool(b.get("heat"))))
    if p == ["models"]:
        return h.json(s.models_status())
    if p == ["models", "download"] and method == "POST":
        return h.job("Загрузка моделей", lambda progress: s.download_models(progress))
    if p == ["models", "cancel"] and method == "POST":
        s.cancel_download()
        return h.json({"ok": True})
    if p == ["segment"] and method == "POST":
        b = h.body()
        return h.job("Сегментация", lambda progress: s.segment(b.get("models_dir"), b.get("device", "auto"),
                                                                   progress))
    if len(p) >= 3 and p[0] == "structures" and p[-1] == "mesh":
        return h.binary(s.structure_mesh("/".join(p[1:-1])))
