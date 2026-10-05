"""Запуск Custom Case Designer: локальный сервер и окно приложения.

    python -m casedesigner.app [--models ПАПКА] [--memory ФАЙЛ] [--browser]

Окно — системный веб-движок (на Windows — Edge WebView2) через pywebview;
без pywebview или с --browser интерфейс открывается в браузере.
"""

import argparse
import os
import sys
import time
import webbrowser

from .server import serve
from .session import Session

FILE_TYPES = {
    "ct": ("КТ (*.nii;*.nii.gz;*.mha;*.mhd;*.nrrd;*.dcm;*.zip)", "Все файлы (*.*)"),
    "scan": ("Сканы (*.stl;*.ply;*.obj)", "Все файлы (*.*)"),
}


class WindowApi:
    """Функции, которые интерфейс вызывает через window.pywebview.api: системные диалоги."""

    def __init__(self):
        self.window = None

    def choose(self, kind: str):
        import webview

        if kind in ("ctdir", "models", "out"):
            result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        else:
            result = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=kind == "scan",
                                                    file_types=FILE_TYPES.get(kind, ()))
        return list(result) if result else []


def main(argv=None):
    parser = argparse.ArgumentParser(prog="casedesigner.app", description="Custom Case Designer")
    parser.add_argument("--models", help="папка моделей сегментации")
    parser.add_argument("--memory", help="файл памяти совмещений (по умолчанию ~/.casedesigner/memory.jsonl)")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--browser", action="store_true", help="открыть в браузере, а не в окне приложения")
    args = parser.parse_args(argv)

    models = args.models
    if models is None:  # рядом с exe, в корне проекта (запуск из исходников) или в текущей папке
        if getattr(sys, "frozen", False):
            places = [os.path.dirname(os.path.abspath(sys.executable))]
        else:
            places = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), os.getcwd()]
        found = [os.path.join(p, "models") for p in places if os.path.isdir(os.path.join(p, "models"))]
        models = found[0] if found else None
    server = serve(Session(args.memory, models), args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    if not args.browser:
        try:
            import webview

            api = WindowApi()
            api.window = webview.create_window("Custom Case Designer", url, js_api=api, width=1480, height=920,
                                               min_size=(1100, 700), background_color="#0d0f13")
            webview.start()
            return 0
        except Exception as e:  # noqa: BLE001 — нет оконного движка: открыть в браузере
            print(f"Окно приложения недоступно ({e}); открываю в браузере.", file=sys.stderr)
    print(f"Custom Case Designer: {url}")
    webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
