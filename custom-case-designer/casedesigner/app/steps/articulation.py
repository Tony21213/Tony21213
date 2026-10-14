"""Артикулятор: монтаж, плоскость, движения, мыщелки, суставные поверхности — методы Session (разделено из session.py; контракты — там же, где были)."""

from ..common import (JawMesh, apply, cond, gd, lmk, mesh_bytes, mo, mplanes, np, rigid, trimesh)  # noqa: F401


class ArticulationSteps:
    # --- предварительная артикуляция ---------------------------------------
    def _prepare_jawcase(self):
        """Передать совмещённые сканы и кости в ядро артикулятора.

        Монтаж всегда выполняется в положении прикуса сканов, если такой
        прикус найден. Это даёт пользователю ту же картину, которую он увидит
        при экспорте, ещё до создания файлов exocad.
        """
        up, lo = self._jaw_items()
        if up is None or lo is None or up.get("reg") is None or lo.get("reg") is None:
            raise ValueError("сначала поставьте верхний и нижний сканы")

        def vertices(item, bite=True):
            T = item["reg"].transform
            if bite and item["reg"].jaw == "lower" and self.bite_motion is not None:
                T = self.bite_motion @ T
            return apply(T, item["scan"].vertices)

        self.jaw.set_scans(JawMesh(vertices(up), up["scan"].faces, "скан верхней челюсти"),
                           JawMesh(vertices(lo), lo["scan"].faces, "скан нижней челюсти"))
        if self.case is not None:
            for key, attr in (("mandible", "mandible"), ("skull", "skull")):
                source = self.structures.get(key)
                if source is not None:
                    setattr(self.jaw, attr, JawMesh(np.asarray(source.vertices), np.asarray(source.faces), key))
        self._articulation_ready = True

    def articulation_prepare(self) -> dict:
        """Подготовить окно визуальной проверки артикуляции."""
        self._prepare_jawcase()
        return self.jaw.state()

    def _face_points(self):
        parts = [np.asarray(self.structures[key].vertices, float) for key in ("skull", "maxilla")
                 if key in self.structures]
        if parts:
            return np.vstack(parts)
        if self.jaw.skull is not None:
            return np.asarray(self.jaw.skull.vertices, float)
        return None

    def _mount_plane_choice(self, method: str, plane: str | None):
        if method != "ct":
            if plane is not None and plane != mplanes.DEFAULT:
                raise ValueError("выбор плоскости доступен только для монтажа по КТ")
            return mplanes.DEFAULT, None, None
        if plane is not None and plane not in mplanes.PLANES:
            raise ValueError(f"нет такой плоскости монтажа: {plane}")
        preferred = plane or self._case_mount_plane or (self.jaw.plane if self.jaw.mounting == "ct"
                                                         else self.settings["mount_plane"])
        hinge, _ct_points, why = self._mount_condyles()
        if hinge is None:
            raise ValueError(f"монтаж по КТ невозможен: {why}")
        face = self._face_points()
        available = mplanes.available(self.landmarks, hinge, face is not None)
        if plane is not None:
            info = available[plane]
            if not info["available"]:
                raise ValueError(f"плоскость «{info['name']}» недоступна: не хватает {', '.join(info['missing'])}")
            return plane, None, hinge
        resolved, note = mplanes.resolve(preferred, self.landmarks, hinge, face is not None)
        return resolved, note, hinge

    def validate_articulation_mount(self, method: str = "auto", plane: str | None = None) -> dict:
        """Проверить выбор плоскости до запуска фонового HTTP-задания."""
        if method not in ("auto", "ct", "average", "motion"):
            raise ValueError(f"неизвестный способ монтажа: {method}")
        if method in ("auto", "ct"):
            if self.case is None:
                if method == "ct":
                    raise ValueError("монтаж по КТ недоступен: нет КТ")
                return {"method": method, "plane": mplanes.DEFAULT}
            if method == "auto" and self._mount_condyles()[0] is None:
                return {"method": "motion" if self.jaw._recorded() else "average", "plane": mplanes.DEFAULT}
            selected_plane, _note, hinge = self._mount_plane_choice("ct", plane)
            return {"method": "ct", "plane": selected_plane, "condyles": hinge}
        if method == "average" and plane not in (None, mplanes.DEFAULT):
            raise ValueError("выбор плоскости доступен только для монтажа по КТ")
        return {"method": method, "plane": mplanes.DEFAULT}

    def articulation_planes(self) -> dict:
        """Доступность плоскостей, активная и предпочтительная плоскости и наклон к окклюзионной."""
        hinge = None
        if self.case is not None:
            hinge, _ct_points, _why = self._mount_condyles()
        face = self._face_points()
        planes = mplanes.available(self.landmarks, hinge, face is not None)
        tilt = {}
        if self._articulation_ready and self.jaw.upper is not None and self.jaw.lower_vertices is not None:
            up = mo.occlusal_up(self.jaw.upper.vertices, self.jaw.lower_vertices)
            _occlusal_up, anterior = mo.arch_axes(self.jaw.upper.vertices, up)
            for key, info in planes.items():
                if not info["available"]:
                    continue
                try:
                    if key in ("aesthetic", "aesthetic_hinge"):
                        if face is None:
                            continue
                        base = mplanes.base_for_aesthetic(self.landmarks, hinge)
                        base_R = mplanes.rotation(base, up, anterior, self.landmarks, hinge)
                        from ... import aesthetic as ae
                        aesthetic_frame = ae.aesthetic_frame_ct(rigid(base_R, np.zeros(3)), face).frame
                        if key == "aesthetic":
                            rotation = aesthetic_frame[:3, :3]
                        else:
                            rotation = mplanes.rotation(key, up, anterior, self.landmarks, hinge,
                                                        face_up=aesthetic_frame[:3, :3][2])
                    else:
                        rotation = mplanes.rotation(key, up, anterior, self.landmarks, hinge)
                    tilt[key] = mplanes.relative_angles(rotation, up, anterior)
                except (ValueError, np.linalg.LinAlgError):
                    tilt[key] = None
        selected = self._case_mount_plane or (self.jaw.plane if self.jaw.mounting == "ct" else None)
        return {"planes": planes, "current": self.jaw.plane if self.jaw.mounting else None,
                "selected": selected or self.settings["mount_plane"],
                "default": self.settings["mount_plane"], "tilt": tilt}

    @staticmethod
    def _compact_mesh(vertices, faces):
        """Сократить сетку до используемых вершин; пустая поверхность — None."""
        vertices = np.asarray(vertices, float).reshape(-1, 3)
        faces = np.asarray(faces, int).reshape(-1, 3)
        if not len(vertices) or not len(faces):
            return None
        used, inverse = np.unique(faces.ravel(), return_inverse=True)
        if len(used) < 3:
            return None
        return trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False)

    def articulation_joint_parts(self) -> dict:
        """Головки мыщелков и суставные скаты для окна артикулятора.

        Головки следуют за нижней челюстью в прикусе сканов; скаты остаются на черепе.
        Все сетки возвращаются в координатах 3D-окна. Скат выбирается, как в ядре (JawCase.mount →
        eminence_profile): череп и точка мыщелка — в соотношении КТ, система — монтажа (anatomy.frame).
        """
        empty = {side: {"head": None, "slope": None} for side in cond.SIDES}
        if not self._articulation_ready or self.jaw.anatomy is None:
            return empty
        mandible, skull = self.structures.get("mandible"), self.structures.get("skull")
        points = self._condyle_points()
        motion = np.eye(4) if self.bite_motion is None else np.asarray(self.bite_motion, float)
        frame = np.asarray(self.jaw.anatomy.frame, float)
        signature = (id(mandible), id(skull),
                     tuple(None if points[s].point is None else tuple(np.round(points[s].point, 4))
                           for s in cond.SIDES), motion.tobytes(), frame.tobytes())
        if signature == self._joint_parts_cache_key:
            return self._joint_parts_cache

        parts = {side: {"head": None, "slope": None} for side in cond.SIDES}
        if mandible is not None and getattr(mandible, "faces", None) is not None:
            mv, mf = np.asarray(mandible.vertices), np.asarray(mandible.faces, int)
            axes = {"up": (0.0, 0.0, 1.0), "left": (1.0, 0.0, 0.0),
                    "anterior": (0.0, -1.0, 0.0)}
            for key, head in lmk.condylar_heads(mv, mf, **axes).items():
                side = "right" if key == "Co_R" else "left"
                selected = np.isin(mf, head["vertices"]).all(axis=1)
                mesh = self._compact_mesh(mv, mf[selected])
                if mesh is not None and self.bite_view and self.bite_motion is not None:
                    mesh.vertices = cond.to_view(mesh.vertices, self.bite_motion)
                parts[side]["head"] = mesh

        if skull is not None and getattr(skull, "faces", None) is not None:
            sv, sf = np.asarray(skull.vertices), np.asarray(skull.faces, int)
            for side in cond.SIDES:
                point = points[side].point
                if point is None:
                    continue
                patch = gd.articular_surface(sv, sf, frame, point)
                if patch is not None:
                    parts[side]["slope"] = self._compact_mesh(sv, sf[np.asarray(patch, int)])
        self._joint_parts_cache_key, self._joint_parts_cache = signature, parts
        return parts

    def articulation_mount(self, method: str = "auto", plane: str | None = None) -> dict:
        """Смонтировать модели для предварительного просмотра.

        Мыщелки — точки кейса Co_R, Co_L (condyles.py: подтверждённые врачом выше автоматических; недостающие
        предлагаются по нижней челюсти). Шарнир — в прикусе монтажа (в прикусе сканов точки едут с нижней
        челюстью), бугорок — в соотношении КТ. Точки не готовы: auto монтирует без КТ, ct — ошибка.
        auto с готовыми точками — монтаж по КТ по выбранной плоскости (как validate_articulation_mount)."""
        if not self._articulation_ready:
            self._prepare_jawcase()
        hinge, ct_points, why = self._mount_condyles()
        if hinge is None and method in ("auto", "ct"):
            if method == "ct":
                raise ValueError(f"монтаж по КТ невозможен: {why}")
            method = "motion" if self.jaw._recorded() else "average"
        elif method == "auto":
            method = "ct"
        selected_plane, fallback_note, _selected_hinge = self._mount_plane_choice(method, plane)
        if method == "ct":
            self._case_mount_plane = selected_plane
        face = self._face_points() if method == "ct" else None
        result = self.jaw.mount(method, condyles=hinge, ct_condyles=ct_points, plane=selected_plane,
                                landmarks=self.landmarks, face_points=face)
        if fallback_note:
            self.jaw.notes = [*self.jaw.notes, fallback_note]
            result = self.jaw.state()
        self._mounted_condyles = None if hinge is None else self._condyle_signature()
        return result

    # --- мыщелки в шаге «Артикулятор» (контракт — casedesigner/condyles.py, docs/handoff/archive/condyle-adjustment.md) --------------
    def _condyle_points(self) -> dict:
        return cond.resolve(self.landmarks, self.suggested, self.landmark_sources)

    def _condyle_signature(self):
        """Точки мыщелков как неизменяемое значение: монтаж сделан с ними или нет."""
        return tuple(None if p.point is None else tuple(np.round(p.point, 4))
                     for p in self._condyle_points().values())

    def _condyle_checks(self, points) -> list:
        mandible = self.structures.get("mandible")
        if self.jaw.anatomy is None:
            # DICOM LPS → оси монтажа: x вправо, y вперёд, z вверх.
            frame = np.diag([-1.0, -1.0, 1.0, 1.0])
        else:
            # Anatomy.frame ждёт координаты окна; хранимые в КТ точки сначала едут
            # вместе с нижней челюстью в прикус сканов.
            motion = np.eye(4) if self.bite_motion is None else np.asarray(self.bite_motion, float)
            frame = np.asarray(self.jaw.anatomy.frame, float) @ motion
        return cond.checks(points,
                           None if mandible is None else cond.cached_surface(mandible), frame)

    def _mandible_condyles(self, heads_only: bool = False) -> dict:
        """Мыщелки по сетке нижней челюсти (мм КТ, LPS: вверх +z, влево +x, вперёд −y): верхушки головок,
        отрезанных от челюсти (landmarks.condylar_heads); heads_only=False — где не разделились, задняя доля
        ветви (suggest_condyles)."""
        mandible = self.structures.get("mandible")
        if mandible is None:
            return {}
        v, f = np.asarray(mandible.vertices), getattr(mandible, "faces", None)
        axes = {"up": (0.0, 0.0, 1.0), "left": (1.0, 0.0, 0.0), "anterior": (0.0, -1.0, 0.0)}
        if heads_only:
            return {} if f is None else {k: h["top"] for k, h in lmk.condylar_heads(v, f, **axes).items()}
        return lmk.suggest_condyles(v, faces=f, **axes)

    def _suggest_missing_condyles(self):
        """Недостающие точки — предложение по нижней челюсти (как «Предложить» в ориентирах); подтверждённые
        врачом и найденные нейросетью не трогаются."""
        mandible = self.structures.get("mandible")
        if mandible is None or all(k in self.landmarks for k in cond.SIDES.values()):
            return
        found = self._mandible_condyles()
        with self.lock:
            for key, point in found.items():
                if key not in self.landmarks:
                    self.landmarks[key] = np.asarray(point, float)
                    self.suggested.add(key)
                    self.landmark_sources[key] = cond.SOURCE_MANDIBLE

    def _mount_condyles(self):
        """(шарнирные точки в координатах монтажа, точки в соотношении КТ, причина) или (None, None, причина)."""
        if self.case is None:
            return None, None, "нет КТ"
        self._suggest_missing_condyles()
        points = self._condyle_points()
        found = self._condyle_checks(points)
        if not cond.ready(points, found):
            missing = [s for s, p in points.items() if p.point is None]
            errors = [c.text for c in found if c.level == "error"]
            why = "; ".join(([f"нет точки: {', '.join(missing)}"] if missing else []) + errors)
            return None, None, why or "точки мыщелков не готовы"
        hinge = {side: cond.to_view(p.point, self.bite_motion) for side, p in points.items()}
        return hinge, {side: p.point.copy() for side, p in points.items()}, ""

    def articulation_condyles(self) -> dict:
        """Состояние точек мыщелков для шага «Артикулятор» (condyles.state)."""
        self._suggest_missing_condyles()
        points = self._condyle_points()
        found = self._condyle_checks(points)
        applied = (self.jaw.anatomy is not None and self._mounted_condyles is not None
                   and self._mounted_condyles == self._condyle_signature())
        result = cond.state(points, found, available=self.case is not None, bite_motion=self.bite_motion,
                            applied=applied, mandible="mandible" in self.structures, skull="skull" in self.structures)
        for side, key in cond.SIDES.items():
            result["points"][side]["note"] = self.landmark_notes.get(key)
        return result

    @staticmethod
    def _sides(side: str) -> list[str]:
        if side == "both":
            return list(cond.SIDES)
        if side not in cond.SIDES:
            raise ValueError(f"сторона мыщелка: right, left или both, а не {side}")
        return [side]

    def set_condyle(self, side: str, point, space: str = "ct") -> dict:
        """Поставить точку мыщелка вручную (источник «врач», подтверждена). space: ct — мм КТ (как на срезах),
        view — координаты окна «Артикулятор» (в прикусе сканов точка переводится к нижней челюсти как на КТ)."""
        self._require_ct()
        if side not in cond.SIDES:
            raise ValueError(f"сторона мыщелка: right или left, а не {side}")
        p = np.asarray(point, float).reshape(3)
        if space == "view":
            p = cond.from_view(p, self.bite_motion)
        elif space != "ct":
            raise ValueError("система точки: ct или view")
        key = cond.SIDES[side]
        with self.lock:
            self.landmarks[key] = p
            self.suggested.discard(key)
            self.landmark_sources[key] = cond.SOURCE_USER
            self.landmark_notes.pop(key, None)
            self._mounted_condyles = None
            self._joint_parts_cache_key = None
        return self.articulation_condyles()

    def confirm_condyle(self, side: str = "both") -> dict:
        """Подтвердить предложенную точку без сдвига: дальше её не меняет автоматика."""
        for s in self._sides(side):
            key = cond.SIDES[s]
            if key not in self.landmarks:
                raise ValueError(f"нет точки мыщелка: {s}")
            with self.lock:
                self.suggested.discard(key)
                self._mounted_condyles = None
        return self.articulation_condyles()

    def reset_condyle(self, side: str = "both") -> dict:
        """Сбросить точку к автоматической: предложение по нижней челюсти (нейросеть ALI-CBCT заново не
        запускается); без сегментации нижней челюсти точки не будет."""
        for s in self._sides(side):
            key = cond.SIDES[s]
            with self.lock:
                self.landmarks.pop(key, None)
                self.suggested.discard(key)
                self.landmark_sources.pop(key, None)
                self.landmark_notes.pop(key, None)
                self._mounted_condyles = None
                self._joint_parts_cache_key = None
        self._suggest_missing_condyles()
        return self.articulation_condyles()

    def articulation_generate(self, travel: float = 6.0) -> dict:
        """Построить виртуальные движения для просмотра до экспорта."""
        if not self._articulation_ready:
            self._prepare_jawcase()
        self.jaw.generate(travel=float(travel), guided=True)
        return self.jaw.state()

    def articulation_analysis(self) -> dict:
        """Рассчитать сводку суставных путей и контактов."""
        if not self._articulation_ready:
            self._prepare_jawcase()
        return self.jaw.analysis()

    def articulation_report(self) -> dict:
        """Рассчитать траектории и карту контактов по каждому движению."""
        if not self._articulation_ready:
            self._prepare_jawcase()
        analysis = self.jaw.analysis()
        contacts = []
        for movement in self.jaw.movements.values():
            result = self.jaw.contacts(movement.id)
            contacts.append({"id": movement.id, "name": movement.recording.name,
                             "sectors": result["sectors"], "teeth": result["teeth"],
                             "pairs": len(result["teeth"])})
        return {"analysis": analysis, "contacts": contacts}

    def articulation_guides(self) -> dict:
        """Точки и линии, объясняющие в 3D основу построения динамики."""
        fallback_axes = {"right": [-1.0, 0.0, 0.0], "forward": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0]}
        def ct_axes(axes):
            rotation = np.eye(3) if self.bite_motion is None else np.linalg.inv(self.bite_motion)[:3, :3]
            return {key: (rotation @ value).round(6).tolist() for key, value in axes.items()}

        if not self._articulation_ready or self.jaw.anatomy is None:
            return {"available": False, "points": {}, "paths": [], "axes": fallback_axes,
                    "ct_axes": ct_axes(fallback_axes)}
        a = self.jaw.anatomy
        axes = {name: np.asarray(a.frame, float)[i, :3].round(6).tolist()
                for i, name in enumerate(("right", "forward", "up"))}
        points = {k: np.asarray(v, float).round(3).tolist() for k, v in a.points.items()
                  if k in ("condyle_right", "condyle_left", "incisal")}
        paths = []
        for movement in self.jaw.movements.values():
            T = np.asarray(movement.recording.transforms, float)  # кадры N×4×4

            def track(point):  # путь точки по кадрам: N×3
                return (np.einsum("nij,j->ni", T[:, :3, :3], np.asarray(point, float)) + T[:, :3, 3]).round(3).tolist()

            paths.append({"id": movement.id, "name": movement.recording.name,
                          "source": "запись движения" if movement.recorded else "виртуальный артикулятор",
                          "condyle_right": track(a.points["condyle_right"]),
                          "condyle_left": track(a.points["condyle_left"]),
                          "incisal": track(a.points["incisal"])})
        return {"available": True, "points": points, "hinge": [points.get("condyle_right"), points.get("condyle_left")],
                "paths": paths, "axes": axes, "ct_axes": ct_axes(axes),
                "basis": "мыщелки пациента + резцовое ведение по сканам челюстей"}

    def articulation_motion(self, mid: str) -> dict:
        """Отдать кадры одного движения для визуального проигрывания."""
        if mid not in self.jaw.movements:
            raise ValueError("движение не найдено")
        rec = self.jaw.movements[mid].recording
        # Для интерактивного окна достаточно 61 кадров, исходная запись
        # остаётся полной и экспортируется отдельно.
        ids = np.linspace(0, len(rec.transforms) - 1, min(61, len(rec.transforms))).round().astype(int)
        return {"id": mid, "name": rec.name, "frames": rec.transforms[ids].round(6).tolist(),
                "duration_s": round(rec.duration, 2) if rec.timed else None}

    def ct_surface(self, part: str = "teeth") -> bytes:
        """Поверхность зубов или кости по плотности, всегда в координатах КТ."""
        self._require_ct()
        if part not in ("teeth", "bone"):
            raise ValueError("неизвестная поверхность КТ")
        mesh = self.case.surfaces(step=2)["ct_" + part]
        return mesh_bytes(mesh.vertices, mesh.faces)


from ..common import mesh_bytes  # noqa: E402
def route(h, s, p, method, q):
    """HTTP-маршруты шага (вызывает server.Handler.api): h — обработчик запроса, s — сессия; True — запрос обработан."""
    if p == ["articulation", "prepare"] and method == "POST":
        return h.json(s.articulation_prepare())
    if p == ["articulation", "planes"]:
        if method == "POST":
            return h.json(s.set_articulation_plane(h.body().get("plane")))
        return h.json(s.articulation_planes())
    if p == ["articulation", "mount"] and method == "POST":
        b = h.body()
        s.validate_articulation_mount(b.get("method", "auto"), b.get("plane"))
        return h.job("Монтаж моделей", lambda progress: s.articulation_mount(
            b.get("method", "auto"), b.get("plane")))
    if p == ["articulation", "generate"] and method == "POST":
        return h.job("Расчёт движений", lambda progress: s.articulation_generate(h.body().get("travel", 6.0)))
    if p == ["articulation", "analysis"] and method == "POST":
        return h.job("Анализ контактов и движений", lambda progress: s.articulation_report())
    if p == ["articulation", "guides"]:
        return h.json(s.articulation_guides())
    if p == ["articulation", "joint"]:
        parts = s.articulation_joint_parts()
        side, part = q.get("side"), q.get("part")
        if side is None and part is None:
            return h.json({s: {k: v is not None for k, v in sides.items()}
                              for s, sides in parts.items()})
        if side not in ("right", "left") or part not in ("head", "slope"):
            raise ValueError("суставная поверхность: side=right|left, part=head|slope")
        mesh = parts[side][part]
        if mesh is None:
            raise ValueError("эта суставная поверхность недоступна: проверьте сегментацию и точки мыщелков")
        return h.binary(mesh_bytes(mesh.vertices, mesh.faces))
    # мыщелки (контракт — casedesigner/condyles.py, docs/handoff/archive/condyle-adjustment.md): состояние, поставить, подтвердить, сбросить
    if p == ["articulation", "condyles"]:
        if method == "POST":
            b = h.body()
            return h.json(s.set_condyle(b["side"], b["point"], b.get("space", "ct")))
        return h.json(s.articulation_condyles())
    if p == ["articulation", "condyles", "confirm"] and method == "POST":
        return h.json(s.confirm_condyle(h.body().get("side", "both")))
    if p == ["articulation", "condyles", "reset"] and method == "POST":
        return h.json(s.reset_condyle(h.body().get("side", "both")))
    if len(p) == 3 and p[:2] == ["articulation", "motion"]:
        return h.json(s.articulation_motion(p[2]))
