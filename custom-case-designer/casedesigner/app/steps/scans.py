"""Сканы: загрузка, совмещение с КТ, сканы прикуса, прикус сканов — методы Session (разделено из session.py; контракты — там же, где были)."""

from ..common import (  # noqa: F401
    BITE,
    JAWS_APART,
    Registration,
    SCAN_COLORS,
    SHARED_SHARE,
    Scan,
    UNGUIDED,
    apply,
    deviation_colors,
    errorlog,
    in_occlusion,
    is_bite_name,
    jaw_by_name,
    jaw_of,
    mesh_bytes,
    np,
    place_bites,
    scanner_registration,
    shared_share,
    split_by_jaw,
    uuid,
)


class ScanSteps:
    # --- сканы -------------------------------------------------------------
    def add_scan(self, path: str) -> dict:
        errorlog.private(path, "скан")
        scan = Scan.load(path)
        errorlog.private(scan.name, "скан")
        sid = uuid.uuid4().hex[:8]
        with self.lock:
            color = SCAN_COLORS[len(self.scans) % len(SCAN_COLORS)]
            self.scans[sid] = {"scan": scan, "path": path, "color": color, "jaw": None,
                               "reg": None, "auto": None, "transform": None, "accepted": False, "guided": False,
                               "role": BITE if is_bite_name(scan.name) else "jaw", "lower_bite": None}
        return self.scan_info(sid)

    def remove_scan(self, sid: str):
        with self.lock:
            self.scans.pop(sid, None)
        self._place_bites()

    # --- сканы прикуса: стоят на сканах челюстей и задают прикус врача ---------------------
    def _jaw_items(self, skip: str | None = None):
        """Совмещённые сканы челюстей (не прикуса): верхний и нижний, первые по порядку."""
        found = {}
        for sid, item in self.scans.items():
            if sid != skip and item["role"] != BITE and item["reg"] is not None:
                found.setdefault(item["reg"].jaw, item)
        return found.get("upper"), found.get("lower")

    def _on_both_jaws(self, sid: str) -> bool:
        """Скан лежит в координатах файла и на верхнем, и на нижнем скане — значит, это скан прикуса."""
        up, lo = self._jaw_items(skip=sid)
        if up is None or lo is None:
            return False
        scan = self.scans[sid]["scan"]
        return shared_share(scan, up["scan"]) >= SHARED_SHARE and shared_share(scan, lo["scan"]) >= SHARED_SHARE

    def _place_bites(self):
        """Сканы прикуса — на сканы челюстей; нижний скан — в прикус по ним, если челюсти не в прикусе."""
        # Любое изменение совмещения требует заново передать модели в окно
        # артикулятора, иначе просмотр мог бы использовать старую гипсовку.
        self._articulation_ready = False
        try:
            self._place_bite_scans()
        finally:
            self._update_bite_motion()

    def _update_bite_motion(self):
        """Как переезжает нижняя челюсть с КТ в прикус сканов: по сканам прикуса или прикус со сканера."""
        up, lo = self._jaw_items()
        motion = None
        # Без КТ: сканы челюстей не в прикусе и сканов прикуса нет — нижний стоит как в файле (подсказка у скана).
        self.jaws_apart = bool(self.case is None and up is not None and lo is not None and lo.get("lower_bite") is None
                               and not (in_occlusion(up["scan"], lo["scan"]) or in_occlusion(lo["scan"], up["scan"])))
        if up is not None and lo is not None and self.case is not None:
            if lo.get("lower_bite") is not None:
                motion = lo["lower_bite"] @ np.linalg.inv(lo["reg"].transform)
            elif in_occlusion(up["scan"], lo["scan"]) or in_occlusion(lo["scan"], up["scan"]):
                motion = up["reg"].transform @ np.linalg.inv(lo["reg"].transform)  # общие координаты сканера
        self.bite_motion, self._bite_meshes = motion, {}
        if motion is None:
            self.bite_view = False

    def _lower_moves(self, item) -> bool:
        """Скан нижней челюсти, который в прикусе сканов стоит не как на КТ."""
        return bool(self.bite_view and self.bite_motion is not None and item["role"] != BITE
                    and item["reg"] is not None and item["reg"].jaw == "lower")

    def shown_transform(self, item):
        """Где скан показан: как на КТ или, с включённым прикусом, нижние — в прикусе сканов."""
        T = item["transform"]
        if self.case is None and T is not None and item.get("lower_bite") is not None:
            return item["lower_bite"]  # без КТ прикус один — по сканам прикуса
        return self.bite_motion @ T if T is not None and self._lower_moves(item) else T

    def _shown_structure(self, key: str):
        """Сетка структуры, как показана: с включённым прикусом структуры нижней челюсти — за ней,
        у смешанных (импланты, коронки, мосты) — их нижняя часть."""
        mesh = self.structures[key]
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        jaw = jaw_of(key)
        if not self.bite_view or self.bite_motion is None or jaw == "upper":
            return v, f, id(mesh)
        cached = self._bite_meshes.get(key)
        if cached is None:
            from ...segment import Mesh

            if jaw == "lower":
                v = apply(self.bite_motion, v)
            else:
                vs, fs, offset = [], [], 0
                for part, m in split_by_jaw(Mesh(v, f), self.case).items():
                    vs.append(apply(self.bite_motion, m.vertices) if part == "lower" else m.vertices)
                    fs.append(m.faces + offset)
                    offset += len(m.vertices)
                v, f = np.vstack(vs), np.vstack(fs)
            cached = self._bite_meshes[key] = (v, f, ("bite", id(mesh), self.bite_motion.tobytes()))
        return cached

    def set_bite_view(self, on: bool) -> dict:
        if on and self.bite_motion is None:
            raise ValueError("прикуса сканов нет: сканы челюстей выгружены не в прикусе, а сканов прикуса нет")
        self.bite_view, self._bite_meshes = bool(on), {}
        return self.bite_info()

    def bite_info(self) -> dict:
        info = {"on": self.bite_view, "available": self.bite_motion is not None, "shift_mm": None}
        if self.bite_motion is not None:
            _up, lo = self._jaw_items()
            at = apply(lo["reg"].transform, lo["scan"].vertices[lo["scan"].crowns])
            info["shift_mm"] = round(float(np.linalg.norm(apply(self.bite_motion, at) - at, axis=1).mean()), 2)
        return info

    def _place_bite_scans(self):
        up, lo = self._jaw_items()
        for item in self.scans.values():
            item["lower_bite"] = None
        bites = [item for item in self.scans.values() if item["role"] == BITE]
        if not bites:
            return
        if up is None and lo is None:
            for item in bites:
                item.update(reg=None, auto=None, transform=None, note=None)
            return
        result = place_bites([item["scan"] for item in bites], up and up["reg"], lo and lo["reg"], self.case,
                             on_ct=self.case is not None)
        with self.lock:
            for k, item in enumerate(bites):
                reg = result.regs[k]
                item.update(reg=reg, auto=reg, transform=None if reg is None else reg.transform, guided=True,
                            note=result.failed.get(k))
            if lo is not None:
                lo["lower_bite"] = result.lower

    def scan_info(self, sid: str) -> dict:
        item = self.scans[sid]
        reg: Registration | None = item["reg"]
        return {"id": sid, "name": item["scan"].name, "path": item["path"], "color": item["color"],
                "vertices": len(item["scan"].vertices),
                "jaw": BITE if item["role"] == BITE else (reg.jaw if reg else item["jaw"]), "role": item["role"],
                "bite_from_scans": item.get("lower_bite") is not None,
                "transform": None if item["transform"] is None else self.shown_transform(item).tolist(),
                "auto_transform": None if item["auto"] is None else item["auto"].transform.tolist(),
                "registered": reg is not None, "accepted": item["accepted"],
                "stats": reg.stats if reg else None, "edge_shift_mm": round(reg.edge_shift, 3) if reg else None,
                "warnings": ((([] if item["guided"] or item["role"] == BITE else [UNGUIDED]) + reg.warnings
                              + ([JAWS_APART] if self._apart(item) else [])) if reg
                             else [item["note"]] if item.get("note") else []),
                "segments": [vars(s) for s in reg.segments] if reg else [],
                "corrected_mm": self._corrected(sid)}

    def _apart(self, item) -> bool:
        return bool(getattr(self, "jaws_apart", False) and item["reg"] is not None and item["role"] != BITE
                    and item["reg"].jaw == "lower")

    def _corrected(self, sid: str) -> float | None:
        item = self.scans[sid]
        if item["auto"] is None or item["transform"] is None:
            return None
        v = item["scan"].vertices
        return round(float(np.linalg.norm(apply(item["transform"], v) - apply(item["auto"].transform, v), axis=1).max()), 3)

    def scan_mesh(self, sid: str) -> bytes:
        s = self.scans[sid]["scan"]
        return mesh_bytes(s.vertices, s.faces)

    def scan_colors(self, sid: str) -> bytes:
        reg = self.scans[sid]["reg"]
        if reg is None:
            return b""
        return np.ascontiguousarray(deviation_colors(reg.deviation)[:, :3], np.uint8).tobytes()

    def _require_ct(self):
        if self.case is None:
            raise ValueError("сначала откройте КТ")

    def register(self, sid: str, jaw: str | None = None, pairs=None, start=None, progress=None) -> dict:
        """Совместить скан с КТ; без КТ — поставить в координатах сканера (как в файле)."""
        if self.case is None and (start is not None or pairs is not None):
            raise ValueError("без КТ скан стоит как в файле: корректировать не по чему")
        item = self.scans[sid]
        if jaw is not None:  # челюсть указал пользователь (значок у скана)
            self.set_jaw(sid, jaw)
            jaw = None if jaw == BITE else jaw
        if item["role"] != BITE and jaw is None and start is None and pairs is None and self._on_both_jaws(sid):
            item["role"] = BITE  # лежит и на верхнем, и на нижнем скане — это скан прикуса
        if self._lower_moves(item) and (start is not None or pairs is not None):
            raise ValueError("выключите «Прикус», чтобы корректировать нижний скан: коррекция — по КТ")
        if item["role"] == BITE:  # с КТ не совмещается: стоит на сканах челюстей
            self._place_bites()
            if item["reg"] is None and not item.get("note"):
                raise ValueError("скан прикуса ставится на сканы челюстей: сначала добавьте и совместите их")
            return self.scan_info(sid)
        jaw = jaw or item["jaw"]
        if self.case is None:
            return self._place_without_ct(sid, jaw)
        reg = self.case.register(item["scan"], jaw=jaw, pairs=pairs,
                                 start=None if start is None else np.asarray(start, float))
        with self.lock:
            item["reg"], item["transform"], item["accepted"] = reg, reg.transform, False
            item["guided"] = reg.jaw in self.case.guided
            if item["auto"] is None or (start is None and pairs is None):
                item["auto"] = reg
        self._place_bites()
        return self.scan_info(sid)

    def _place_without_ct(self, sid: str, jaw: str | None) -> dict:
        """Без КТ: скан челюсти — в координатах своего файла (у выгрузки сканера они общие и в прикусе);
        челюсть — по имени файла, иначе противоположная уже поставленной."""
        item = self.scans[sid]
        if jaw is None:
            jaw = jaw_by_name(item["scan"].name)
        if jaw is None:
            taken = {it["reg"].jaw for s, it in self.scans.items()
                     if s != sid and it["reg"] is not None and it["role"] != BITE}
            jaw = {frozenset({"upper"}): "lower", frozenset({"lower"}): "upper"}.get(frozenset(taken))
        if jaw is None:
            raise ValueError(f"{item['scan'].name}: без КТ челюсть по имени файла не понять — укажите её "
                             "(щёлкните по значку челюсти у скана)")
        reg = scanner_registration(item["scan"], jaw)
        with self.lock:
            item.update(reg=reg, auto=reg, transform=reg.transform, accepted=False, guided=True, jaw=jaw)
        self._place_bites()
        return self.scan_info(sid)

    def evaluate(self, sid: str, transform) -> dict:
        self._require_ct()
        item = self.scans[sid]
        if item["role"] == BITE:  # скан прикуса двигается вместе со сканами челюстей
            self._place_bites()
            return self.scan_info(sid)
        if self._lower_moves(item):
            raise ValueError("выключите «Прикус», чтобы корректировать нижний скан: коррекция — по КТ")
        reg = self.case.evaluate(item["scan"], np.asarray(transform, float), jaw=item["reg"].jaw if item["reg"] else None)
        with self.lock:
            item["reg"], item["transform"], item["accepted"] = reg, reg.transform, False
            item["guided"] = reg.jaw in self.case.guided
        self._place_bites()
        return self.scan_info(sid)

    def set_jaw(self, sid: str, jaw: str | None):
        """Челюсть скана (upper/lower) или «скан прикуса» (bite)."""
        item = self.scans[sid]
        item["role"] = BITE if jaw == BITE else "jaw"
        item["jaw"] = None if jaw == BITE else (jaw or None)

    def accept(self, sid: str) -> dict:
        item = self.scans[sid]
        if item["reg"] is None:
            raise ValueError("скан ещё не совмещён")
        if item["role"] == BITE:
            raise ValueError("скан прикуса стоит на сканах челюстей — принимать нужно их")
        self._require_ct()
        from ...learning import MIN_CASES

        rec = self.memory.record(self.vol.device, item["auto"] or item["reg"], item["reg"])
        item["accepted"] = True
        # выученное — сразу в работу: следующие сканы этого кейса совмещаются уже с новой поправкой
        self.case.edge_prior, self.case.prior_weight = self.memory.prior(self.vol.device)
        return {"scan": self.scan_info(sid),
                "record": {k: rec[k] for k in ("corrected_mm", "edge_shift_mm")} | {"learned": self.memory.usable(rec)},
                "memory": self.memory.summary().get(self.vol.device or "unknown"), "min_cases": MIN_CASES}
