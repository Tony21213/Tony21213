"""Native dialogs and the actual pywebview API registration, without patient data."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from casedesigner.app.main import FILE_TYPES, WindowApi


def test_bridge_exposes_choose_without_traversing_window(monkeypatch):
    webview = pytest.importorskip("webview")
    api = WindowApi()
    scripts = []
    window = SimpleNamespace(
        _js_api=api, _functions={}, _expose_lock=threading.Lock(),
        events=SimpleNamespace(**{name: threading.Event() for name in
                                  ("before_load", "loaded", "_pywebviewready")}),
        run_js=scripts.append,
        destroy=lambda: None,
    )
    api._window = window
    monkeypatch.setattr(webview.util, "load_js_files", lambda *args: ("init", "%(functions)s"))
    webview.util.inject_pywebview("edgechromium", window)
    assert window.events.loaded.wait(3), "API registration did not finish"
    assert window.events._pywebviewready.is_set()
    assert json.loads(scripts[-1]) == [{"func": "choose", "params": ["kind"]}]


@pytest.mark.parametrize("kind", ["ct", "scan", "ctdir", "case", "save", "out", "models"])
def test_choose_dialog_and_cancel(kind):
    webview = pytest.importorskip("webview")
    api = WindowApi()
    dialog = Mock(return_value=("C:/case-001/file",))
    api._window = SimpleNamespace(create_file_dialog=dialog)
    assert api.choose(kind) == ["C:/case-001/file"]
    args, kwargs = dialog.call_args
    expected = webview.FOLDER_DIALOG if kind in ("ctdir", "out", "models") else (
        webview.SAVE_DIALOG if kind == "save" else webview.OPEN_DIALOG)
    assert args == (expected,)
    if kind in ("ct", "scan", "case"):
        assert kwargs["file_types"] == FILE_TYPES[kind]
        assert kwargs["allow_multiple"] == (kind == "scan")
    dialog.return_value = None
    assert api.choose(kind) == []
    if kind == "save":
        dialog.return_value = "C:/case-001/case.ccdcase"
        assert api.choose(kind) == ["C:/case-001/case.ccdcase"]
