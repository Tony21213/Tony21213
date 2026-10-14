"""Экспорт: в папку, в проект exocad, для exocad с артикулятором — методы Session (разделено из session.py; контракты — там же, где были)."""

from ..common import (  # noqa: F401
    BITE,
    EXOCAD_SUBFOLDER,
    LOWER_BONES,
    UPPER_BONES,
    dataclasses,
    errorlog,
    exocad_project,
    export_case,
    np,
    open_folder,
    os,
    trimesh,
)


class ExportSteps:
    # --- экспорт ------------------------------------------------------------
    def export(self, out_dir: str, bite: str = "scan", frame: str = "exocad", include: list[str] | None = None,
               reference: str | None = None) -> dict:
        errorlog.private(out_dir, "папка экспорта")
        if self.case is None and not any(item["reg"] is not None for item in self.scans.values()):
            raise ValueError("нечего экспортировать: откройте КТ или добавьте сканы")
        if self.case is None:  # только сканы: координаты сканера, прикус сканов
            frame, bite, reference = "exocad", "scan", None
        regs = []
        for item in self.scans.values():
            reg = item["reg"]
            if reg is not None and item.get("lower_bite") is not None:  # нижний — в прикусе по сканам прикуса
                reg = dataclasses.replace(reg, bite=item["lower_bite"])
            if reg is not None:
                regs.append(reg)
        meshes = {}
        from ...segment import Mesh

        for key, mesh in self.structures.items():
            if include is None or key in include:
                meshes[key] = Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        project = exocad_project.find(out_dir)
        if not regs:  # без сканов — только КТ: структуры в координатах КТ (DICOM)
            if project is not None:
                raise ValueError("для проекта exocad нужны совмещённые сканы: структуры КТ ставятся к ним")
            if not meshes:
                raise ValueError("нечего экспортировать: сегментируйте КТ или совместите сканы")
            return self._export_ct_only(out_dir, meshes)
        if not self.structures and self.case is not None:  # без сегментации — зубы из КТ по плотности (как в 3D)
            meshes["ct_teeth"] = self.case.surfaces(step=1)["ct_teeth"]
        if project is not None:
            return self._export_to_exocad(project, regs, meshes, frame, bite)
        matrix, name = self.reference(reference) if frame == "reference" else (None, "")
        self.last_export = out_dir
        report = export_case(out_dir, regs, meshes, bite=bite, frame=frame, ct=self.case, reference=matrix,
                             reference_name=name, without_ct=self.case is None)
        return {"out_dir": out_dir, "files": sorted(report["files"]), "notes": report["notes"],
                "bite": report["ct_bite_vs_scans"], "frame": report["frame"]}

    def validate_exocad_articulator(self, project_dir: str, out_dir: str | None = None) -> tuple[str, str]:
        """Проверить исходный проект, путь назначения, монтаж по готовым мыщелкам и обе челюсти.

        Возвращает целевую папку и имя копии; вызывается до постановки фоновой задачи, чтобы ошибки
        пользовательского ввода и готовности пришли HTTP 400, а не скрылись внутри ответа job.
        """
        from ... import exocad_articulator_export as xa

        errorlog.private(project_dir, "проект exocad")
        errorlog.private(out_dir, "папка выгрузки exocad")
        if not isinstance(project_dir, str) or not project_dir.strip():
            raise ValueError("выберите папку проекта exocad")
        if out_dir is not None and (not isinstance(out_dir, str) or not out_dir.strip()):
            raise ValueError("папка выгрузки должна быть путём или null")

        condyles = self.articulation_condyles()
        if not condyles["available"]:
            raise ValueError("для выгрузки с артикулятором откройте КТ")
        if not condyles["ready"]:
            problems = [c["text"] for c in condyles["checks"] if c["level"] == "error"]
            missing = [s for s, p in condyles["points"].items() if p["ct"] is None]
            detail = "; ".join(([f"нет точки мыщелка: {', '.join(missing)}"] if missing else []) + problems)
            raise ValueError("точки мыщелков не готовы" + (f": {detail}" if detail else ""))
        if not condyles["applied"]:
            raise ValueError("выполните монтаж моделей по точкам мыщелков")
        if self.jaw.anatomy is None:
            raise ValueError("сначала смонтируйте модели в шаге «Артикулятор»")
        upper, lower = self._jaw_items()
        if upper is None or lower is None or upper.get("reg") is None or lower.get("reg") is None:
            raise ValueError("нужны совмещённые сканы обеих челюстей")

        folder, stem = xa.target(project_dir, out_dir)
        if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
            raise FileExistsError("папка для копии проекта уже существует и не пуста — выберите другую")
        return folder, stem

    def export_exocad_articulator(self, project_dir: str, out_dir: str | None = None, progress=None) -> dict:
        """Выгрузить копию проекта exocad после проверки готовых и применённых точек мыщелков."""
        from ... import exocad_articulator_export as xa

        self.validate_exocad_articulator(project_dir, out_dir)
        if progress:
            progress(0.1, "Собираю сканы, кости и движения")
        case = xa.case_from_session(self)
        if progress:
            progress(0.35, "Создаю копию проекта exocad")
        result = xa.export(case, project_dir, out_dir)
        if progress:
            progress(1.0, "Выгрузка готова")
        return result

    def open_export_folder(self):
        """Показать папку последнего экспорта в проводнике (только её — путь не приходит снаружи)."""
        folder = getattr(self, "last_export", None)
        if not folder or not os.path.isdir(folder):
            raise ValueError("экспорта ещё не было")
        open_folder(folder)

    @staticmethod
    def open_log_folder():
        """Показать папку журнала ошибок."""
        folder = errorlog.folder()
        if not folder or not os.path.isdir(folder):
            raise ValueError("журнал ошибок не ведётся")
        open_folder(folder)

    def _export_to_exocad(self, project, regs, meshes: dict, frame: str, bite: str = "scan") -> dict:
        """Папка проекта exocad: всё — в подпапку проекта, в координатах его сцены (прикус — bite).

        Файлы exocad не меняются. Копии сканов там же — по ним видно в exocad, что всё встало на место.
        """
        jaws = [r for r in regs if r.jaw != BITE]
        if not jaws:
            raise ValueError("нет совмещённых сканов челюстей: скан прикуса ставится на них")
        ref = next((r for r in jaws if r.jaw == "upper"), jaws[0])  # опорный скан — как в export_case
        matched = project.match(ref.scan.vertices)
        scene = project.to_scene(matched) if matched else project.default
        out_dir = os.path.join(project.folder, EXOCAD_SUBFOLDER)
        self.last_export = out_dir
        report = export_case(out_dir, regs, meshes, bite=bite, frame="exocad", ct=self.case, scene=scene,
                             without_ct=self.case is None)
        notes = [f"Проект exocad: файлы в подпапке {EXOCAD_SUBFOLDER}, в координатах сцены проекта (матрица — "
                 f"{project.source}). В exocad: «Load mesh as …»; копии сканов там же должны совпасть со сканами проекта."]
        bone_files, bone_notes = self._bones_as_frameworks(project, jaws)
        if bone_files:
            notes.append("Кости — bone_upper.stl (кость верхней челюсти: череп и верхняя челюсть) и bone_lower.stl "
                         "(кость нижней челюсти); exocad подгрузит их сам и будет двигать с челюстями в артикуляторе.")
        notes += bone_notes
        if frame == "dicom" and self.case is not None:
            notes.append("Для проекта exocad выгрузка всегда в координатах его сцены, а не КТ.")
        if matched is None:
            notes.append(f"Скан {ref.scan.name} не найден среди сканов проекта — положение в exocad не гарантировано "
                         "(взята общая матрица проекта). Загрузите в программу сканы из папки этого проекта.")
        return {"out_dir": out_dir, "files": sorted(report["files"]), "notes": notes + report["notes"],
                "bite": report["ct_bite_vs_scans"], "frame": report["frame"], "frameworks": bone_files}

    def _bones_as_frameworks(self, project, jaws) -> tuple[list, list]:
        """Кости КТ — в координаты файла скана своей челюсти и в папку проекта (bone_upper.stl, bone_lower.stl)."""
        bones, notes = {}, []
        for jaw, keys in (("upper", UPPER_BONES), ("lower", LOWER_BONES)):
            reg = next((r for r in jaws if r.jaw == jaw), None)
            parts = [self.structures[k] for k in keys if k in self.structures]
            if reg is None or not parts:
                continue
            if project.match(reg.scan.vertices) is None:
                notes.append(f"{'Верхний' if jaw == 'upper' else 'Нижний'} скан не из этого проекта — кость "
                             "каркасом не записана.")
                continue
            mesh = trimesh.util.concatenate([trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.faces), process=False)
                                             for m in parts])
            mesh.apply_transform(np.linalg.inv(reg.transform))  # КТ → координаты файла скана этой челюсти
            bones[jaw] = mesh
        files, more = exocad_project.write_bone_frameworks(project, bones)
        return files, notes + more

    def _export_ct_only(self, out_dir: str, meshes: dict) -> dict:
        import json

        self.last_export = out_dir
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for key, m in meshes.items():
            path = os.path.join(out_dir, *key.split("/")) + ".stl"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            trimesh.Trimesh(m.vertices, m.faces, process=False).export(path)
            files.append(key + ".stl")
        with open(os.path.join(out_dir, "case.json"), "w", encoding="utf-8") as f:
            json.dump({"frame": "dicom", "note": "координаты пациента из DICOM (мм)", "files": sorted(files)}, f,
                      ensure_ascii=False, indent=1)
        return {"out_dir": out_dir, "files": sorted(files + ["case.json"]),
                "notes": ["сканов нет — структуры КТ в координатах КТ (DICOM)"], "bite": None, "frame": "dicom"}


def route(h, s, p, method, q):
    """HTTP-маршруты шага (вызывает server.Handler.api): h — обработчик запроса, s — сессия; True — запрос обработан."""
    if p == ["export", "open"] and method == "POST":
        s.open_export_folder()
        return h.json({"ok": True})
    if p == ["export"] and method == "POST":
        b = h.body()
        return h.job("Экспорт", lambda progress: s.export(b["out_dir"], b.get("bite", "scan"),
                                                              b.get("frame", "exocad"), b.get("include"),
                                                              b.get("reference")))
    if p == ["export", "exocad-articulator"] and method == "POST":
        b = h.body()
        project, out_dir = b.get("project"), b.get("out")
        s.validate_exocad_articulator(project, out_dir)
        return h.job("Экспорт для exocad с артикулятором",
                        lambda progress: s.export_exocad_articulator(project, out_dir, progress))
