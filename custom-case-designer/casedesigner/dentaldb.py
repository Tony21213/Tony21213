"""Контекст заказа DentalDB и публикация комплекта в его папку.

DentalDB передаёт точный .dentalProject; сведения пациента не используются как
идентификатор. Несколько файлов публикуются с журналом/резервными копиями,
а не как якобы атомарная замена папки. Закрытые сцены CAD не редактируются.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import uuid
import xml.etree.ElementTree as ET

import numpy as np

SUB = "KStomCaseDesigner"
MANIFEST = f"{SUB}/dentaldb-manifest.json"
JOURNAL = f"{SUB}/dentaldb-journal.json"
MESH_EXT = (".ply", ".stl", ".obj")


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("ожидался обычный файл заказа")
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def relative(root: Path, name: str) -> Path:
    """Только файл внутри заказа; XML и манифест не дают доступа к другим папкам."""
    p = Path(name.replace("\\", "/"))
    if p.is_absolute() or ":" in str(p) or ".." in p.parts or not p.parts:
        raise ValueError("недопустимая ссылка на файл заказа")
    out = root.joinpath(p)
    if not out.resolve().is_relative_to(root.resolve()) or any(x.is_symlink() for x in [out, *out.parents] if x != root.parent):
        raise ValueError("ссылка выходит за папку заказа")
    return out


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _xml(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError("некорректный XML заказа exocad") from e


def _alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # нет доступа — считаем процесс живым
        try:
            code = wintypes.DWORD()
            return not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclass
class Project:
    path: Path
    case_id: str
    baseline: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    matrices: dict = field(default_factory=dict)
    owned: dict = field(default_factory=dict)
    mutex: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def folder(self):
        return self.path.parent

    @property
    def stem(self):
        return self.path.stem

    @classmethod
    def open(cls, path: str) -> Project:
        p = Path(path).resolve()
        if p.suffix.lower() != ".dentalproject" or not p.is_file():
            raise ValueError("нужен существующий файл .dentalProject из DentalDB")
        if _xml(p).tag.rsplit("}", 1)[-1] != "Treatment":
            raise ValueError("файл не является заказом DentalDB")
        obj = cls(p, uuid.uuid4().hex[:8].upper())
        manifest = relative(obj.folder, MANIFEST)
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("schema") != 1 or not isinstance(data.get("files"), dict):
                    raise ValueError("неподдерживаемый манифест интеграции DentalDB")
                obj.owned = data["files"]
                obj.case_id = data["case_id"]
                for name, value in obj.owned.items():
                    relative(obj.folder, name)
                    if not isinstance(value, str) or len(value) != 64:
                        raise ValueError("некорректная контрольная сумма в манифесте")
            except (KeyError, json.JSONDecodeError) as e:
                raise ValueError("повреждён манифест интеграции DentalDB") from e
        obj._sources()
        obj.baseline = obj.inventory()
        return obj

    def _sources(self):
        from .exocad_project import read_matrix

        info = self.path.with_suffix(".scanInfo")
        default = np.eye(4)
        per_file = {}
        preferred = set()
        if info.exists():
            root = _xml(info)
            matrix = root.find("MatrixToScanDataFiles")
            if matrix is not None:
                default = np.linalg.inv(read_matrix(matrix))
            for item in root.iter("ScanFile"):
                name = item.findtext("FileName")
                if name:
                    file = relative(self.folder, name)
                    preferred.add(file.name.lower())
                    matrix = item.find("TransformationMatrix")
                    if matrix is not None:
                        per_file[file.name.lower()] = np.linalg.inv(read_matrix(matrix))
        elif self.path.with_suffix(".matrix4").exists():
            default = read_matrix(_xml(self.path.with_suffix(".matrix4")))
        files = {p.name.lower(): p for p in self.folder.iterdir() if p.is_file()}

        def pick(role, endings):
            found = [files[(self.stem + ending + ext).lower()] for ending in endings for ext in MESH_EXT
                     if (self.stem + ending + ext).lower() in files]
            active = [p for p in found if p.name.lower() in preferred]
            candidates = active or found
            if len(candidates) > 1:
                raise ValueError(f"неоднозначные файлы роли «{role}»: оставьте один активный скан")
            if candidates:
                p = candidates[0]
                self.sources[role] = p
                self.matrices[role] = per_file.get(p.name.lower(), default).copy()

        pick("upper", ("-upperjaw",))
        pick("lower", ("-lowerjaw",))
        if len(self.sources) == 1:
            pick("lower" if "upper" in self.sources else "upper", ("-antagonist",))
        import re
        bites = sorted({m.group(1) for name in files
                        if (m := re.match(re.escape(self.stem.lower()) + r"(-totaljaw\d+)\.(ply|stl|obj)$", name))})
        for i, ending in enumerate(bites):
            pick(f"bite{i}", (ending,))
        for matrix in self.matrices.values():
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("некорректная матрица скана заказа")

    def inventory(self):
        files = [p for p in self.folder.iterdir() if p.is_file() and not p.name.startswith(".")]
        managed = relative(self.folder, SUB)
        if managed.exists():
            files += [relative(self.folder, name) for name in self.owned]
            if relative(self.folder, MANIFEST).exists():
                files.append(relative(self.folder, MANIFEST))
        return {p.relative_to(self.folder).as_posix(): digest(p) for p in files if p.exists()}

    def verify(self):
        if self.inventory() != self.baseline:
            raise ValueError("файлы заказа изменены другой программой; откройте заказ заново")
        if relative(self.folder, JOURNAL).exists():
            raise ValueError("предыдущая сборка прервана; сначала восстановите файлы заказа")
        if len(list(self.folder.glob("*.dentalProject"))) != 1:
            raise ValueError("для публикации в папке должен быть один заказ DentalDB")
        if any(p.suffix.lower() in (".dentalcad", ".constructioninfo", ".partialcad")
               or p.stem.lower().endswith("_cad") for p in self.folder.iterdir() if p.is_file()):
            raise ValueError("в заказе уже есть моделировка: создайте Duplicate в DentalDB")
        # Такие данные нельзя оставить в прежних координатах при новом монтаже.
        if any(p.name.startswith(self.stem + "-") and
               any(word in p.name.lower() for word in ("-situ", "-waxup", "-scanabutment", ".implantposition"))
               for p in self.folder.iterdir() if p.is_file()):
            raise ValueError("в заказе есть дополнительные данные моделирования; подготовьте отдельный вариант")
        for name, checksum in self.owned.items():
            if digest(relative(self.folder, name)) != checksum:
                raise ValueError("подготовленные файлы изменены вне KStom; требуется новый вариант заказа")

    def publish(self, staged: Path, delete: set[str] | None = None):
        """Готовый набор файлов → заказ; контроль версий, журнал, откат при исключении.

        staged содержит только проверенные результаты сборки, никогда файлы программы.
        Ошибки ОС не включают пути в сообщениях интерфейса.
        """
        if not self.mutex.acquire(blocking=False):
            raise ValueError("этот заказ уже собирается")
        lock = relative(self.folder, f"{SUB}/dentaldb-publish.lock")
        acquired = False
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            try:
                with lock.open("x", encoding="ascii") as f:
                    f.write(str(os.getpid()))
                acquired = True
            except FileExistsError as e:
                raise ValueError("сборка заказа уже выполняется или была прервана") from e
            self.verify()
            generated = {p.relative_to(staged).as_posix(): p for p in staged.rglob("*") if p.is_file()}
            if not generated or self.path.name not in generated:
                raise ValueError("в комплекте нет файла заказа")
            permitted = {p.name for p in self.sources.values()} | {self.path.name,
                         self.stem + ".scanInfo", self.stem + ".matrix4"} | set(self.owned)
            removals = set(delete or ()) | (set(self.owned) - set(generated))
            for name in generated.keys() | removals:
                target = relative(self.folder, name)
                if target.exists() and name not in permitted and name != MANIFEST:
                    raise ValueError("комплект пересекается с чужими файлами заказа; создайте Duplicate")
            files = {name: digest(path) for name, path in generated.items()}
            write_json(relative(staged, MANIFEST), {"schema": 1, "case_id": self.case_id, "files": files})
            generated[MANIFEST] = relative(staged, MANIFEST)
            backup_rel = f"{SUB}/backups/{uuid.uuid4().hex}"
            backup = relative(self.folder, backup_rel)
            entries = {}
            for name in generated.keys() | removals:
                dst = relative(self.folder, name)
                old = digest(dst)
                if old is not None:
                    copy = relative(backup, name)
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dst, copy)
                entries[name] = {"old": old, "new": digest(generated[name]) if name in generated else None}
            record = {"schema": 1, "backup": backup_rel, "entries": entries}
            write_json(relative(self.folder, JOURNAL), record)
            try:
                for name, entry in entries.items():
                    dst = relative(self.folder, name)
                    if digest(dst) != entry["old"]:
                        raise ValueError("во время сборки файл заказа был изменён другой программой")
                    if name in generated:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(generated[name], dst)
                    elif dst.exists():
                        dst.unlink()
            except BaseException:
                self._restore(record)
                raise
            relative(self.folder, JOURNAL).unlink()
            self.owned = files
            self.baseline = self.inventory()
            return {"files": len(files), "case_id": self.case_id, "backup": backup_rel}
        except OSError as e:
            raise ValueError("не удалось записать заказ; проверьте доступ и свободное место, затем повторите") from e
        finally:
            if acquired:
                lock.unlink(missing_ok=True)
            self.mutex.release()

    def _restore(self, record):
        backup = relative(self.folder, record["backup"])
        for name, entry in record["entries"].items():
            target = relative(self.folder, name)
            current = digest(target)
            if current not in (entry["old"], entry["new"]):
                raise ValueError("файл изменён после прерывания сборки; автоматический откат остановлен")
            if entry["old"] is not None:
                source = relative(backup, name)
                if digest(source) != entry["old"]:
                    raise ValueError("резервная копия заказа повреждена")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif target.exists():
                target.unlink()
        relative(self.folder, JOURNAL).unlink(missing_ok=True)

    def recover(self):
        """Восстановление после аварии; не заменяет файлы, изменённые после неё."""
        lock = relative(self.folder, f"{SUB}/dentaldb-publish.lock")
        if lock.exists():
            try:
                pid = int(lock.read_text(encoding="ascii"))
            except ValueError as e:
                raise ValueError("неизвестный владелец блокировки заказа") from e
            if _alive(pid):
                raise ValueError("сборка заказа ещё выполняется; восстановление недоступно")
        journal = relative(self.folder, JOURNAL)
        if not journal.exists():
            lock.unlink(missing_ok=True)
            return False
        try:
            record = json.loads(journal.read_text(encoding="utf-8"))
            self._restore(record)
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError("повреждён журнал восстановления заказа") from e
        self.baseline = self.inventory()
        lock.unlink(missing_ok=True)
        return True
