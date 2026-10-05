"""Журнал ошибок программы — без имён пациентов.

Имена пациентов встречаются в путях и именах файлов (папка КТ, сканы из
exocad, файл кейса), а из имён файлов — в названиях объектов. Поэтому в
журнал не попадает ни один путь пользователя:
  * строки, которые сессия знает как личные (пути КТ, сканов, кейса, папки
    экспорта, их части и названия сканов), заменяются метками <КТ>, <скан 1>…;
  * любой оставшийся путь — на <путь>.расширение;
  * домашняя папка — на ~.
Пишется то, что нужно для разбора: время, версия, система, операция, тип
ошибки, текст и стек вызовов программы. Журнал — %LOCALAPPDATA%\\CustomCaseDesigner\\logs
(на других системах ~/.casedesigner/logs), не больше LOG_FILES файлов по LOG_BYTES.
"""

import logging
import logging.handlers
import os
import platform
import re
import sys
import threading
import traceback

LOG_BYTES = 1 << 20
LOG_FILES = 3
_logger = logging.getLogger("casedesigner.errors")
_private: dict[str, str] = {}  # личная строка → метка
_lock = threading.Lock()
_folder: str | None = None

# Путь Windows (C:\…, \\сервер\…) или Unix (/…, ~/…): до пробела-кавычки-скобки, с хотя бы одним разделителем.
# Путь Windows (C:\…, \\сервер\…) — до кавычки, точки с запятой, запятой или конца строки: в именах
# папок бывают пробелы («Иванов И.И»). Путь Unix или ~/… — до пробела.
_WIN_PATH = re.compile(r"""(?:[A-Za-z]:[\\/]|\\\\)[^;,"'<>|\r\n\t*?]*""")
_UNIX_PATH = re.compile(r"""(?:~[\\/]|(?<![\w.~])/)[^\s"'<>|*?:,;()\[\]{}]*[\\/][^\s"'<>|*?:,;()\[\]{}]*""")
_FILE_EXT = re.compile(r"[^\s\\/]*\.(\w{1,8})(?=\s|$)")


def default_folder() -> str:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return os.path.join(base, "CustomCaseDesigner", "logs")
    return os.path.join(os.path.expanduser("~"), ".casedesigner", "logs")


def setup(folder: str | None = None, version: str = "") -> str:
    """Включить журнал (один раз); вернуть его папку."""
    global _folder
    folder = folder or default_folder()
    with _lock:
        if _folder == folder:
            return folder
        for h in list(_logger.handlers):
            _logger.removeHandler(h)
            h.close()
        os.makedirs(folder, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(os.path.join(folder, "errors.log"), maxBytes=LOG_BYTES,
                                                       backupCount=LOG_FILES - 1, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _folder = folder
    _logger.info(scrub(f"запуск: версия {version or '—'}, {platform.system()} {platform.release()}, "
                       f"Python {platform.python_version()}"))
    return folder


def folder() -> str | None:
    return _folder


def private(value: str | None, label: str):
    """Запомнить личную строку (путь, название) и её метку для журнала."""
    if not value:
        return
    value = str(value)
    with _lock:
        _private[value] = label
        norm = os.path.normpath(value)
        _private.setdefault(norm, label)
        for part in re.split(r"[\\/]", norm):  # папки и имя файла по отдельности: «Иванов И.И», «Иванов-upperjaw.stl»
            stem = os.path.splitext(part)[0]
            if len(stem) >= 3 and not re.fullmatch(r"[A-Za-z]:|\.+", part):
                _private.setdefault(part, f"{label}:часть")
                _private.setdefault(stem, f"{label}:часть")


def scrub(text: str) -> str:
    """Текст без личных строк и путей пользователя."""
    if not text:
        return text
    with _lock:
        items = sorted(_private.items(), key=lambda kv: -len(kv[0]))
    for value, label in items:
        text = text.replace(value, f"<{label}>")
    home = os.path.expanduser("~")
    program = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def own(p):  # файлы программы и библиотек — нужны для разбора
        return p.startswith(program) or "site-packages" in p or "dist-packages" in p

    def win(m):
        p = m.group(0)
        if own(p):
            return p.replace(home, "~") if len(home) > 2 else p
        tail = p[max(p.rfind("\\"), p.rfind("/")) + 1:]
        f = _FILE_EXT.match(tail)
        if f:  # имя файла с расширением, дальше — обычный текст сообщения
            return f"<путь>.{f.group(1)}{tail[f.end():]}"
        return "<путь>"  # без расширения не понять, где кончается путь: лучше потерять текст, чем имя

    def unix(m):
        p = m.group(0)
        if own(p):
            return p.replace(home, "~") if len(home) > 2 else p
        return f"<путь>{os.path.splitext(p.rstrip(chr(92) + '/'))[1]}"
    text = _WIN_PATH.sub(win, text)
    text = _UNIX_PATH.sub(unix, text)
    return text.replace(home, "~") if len(home) > 2 else text


def error(operation: str, exc: BaseException | None = None, details: str = ""):
    """Записать ошибку операции: тип, текст и стек — без личных данных."""
    if not _logger.handlers:
        return
    lines = [f"ОШИБКА: {operation}"]
    if details:
        lines.append(details)
    if exc is not None:
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
    _logger.error(scrub("\n".join(lines)))


def message(text: str):
    if _logger.handlers:
        _logger.info(scrub(text))


def install_hooks():
    """Неперехваченные ошибки главного потока и фоновых потоков — тоже в журнал."""
    previous = sys.excepthook

    def hook(t, e, tb):
        error("неперехваченная ошибка", e)
        previous(t, e, tb)

    sys.excepthook = hook
    previous_thread = threading.excepthook

    def thread_hook(args):
        error(f"неперехваченная ошибка в потоке {args.thread.name if args.thread else ''}", args.exc_value)
        previous_thread(args)

    threading.excepthook = thread_hook
