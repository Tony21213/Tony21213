"""Панель объектов окна 3D (как в exocad): видимость и прозрачность на любом шаге — проверка устройства кода.

Поведение проверено в браузере на фантоме; здесь — чтобы шаги не обходили слой видимости окна 3D."""

import re
from pathlib import Path

WEB = Path(__file__).parents[1] / "casedesigner" / "app" / "web"


def read(name):
    return (WEB / name).read_text(encoding="utf-8")


def test_viewer_keeps_user_choice_over_program_visibility():
    v = read("viewer3d.js")
    assert "this.userHidden = new Set()" in v and "this.userOpacity = new Map()" in v
    # итог на экране — видимость шага и не скрыт ли объект в панели
    assert "mesh.visible = on && !this.userHidden.has(key);" in v
    assert "this.userOpacity.get(key) ?? opacity" in v  # пересозданная сетка — с прозрачностью из панели
    for method in ("setTransform", "setVisible", "setScan"):
        body = re.search(rf"\n  {method}\(.*?\n  }}\n", v, re.S).group(0)
        assert "mesh.visible =" not in body, method


def test_steps_go_through_viewer_visibility():
    """Шаги не присваивают .visible сеткам напрямую — иначе скрытое в панели объектов появится снова."""
    for path in [WEB / "core.js", WEB / "app.js", *(WEB / "steps").glob("*.js")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(?!state\b)\w+\.visible\s*=(?!=)", text), path.name  # state.visible — флажки структур


def test_bar_wired_on_every_step():
    bar, core, app = read("objectbar.js"), read("core.js"), read("app.js")
    assert "data-ob-eye" in bar and "data-ob-op" in bar and "setUserHidden" in bar and "toggleStructures" in bar
    assert "RENDER.objectBar = renderObjectBar" in app
    assert "RENDER.objectBar?.()" in core
    assert "!viewer.userHidden.has(s.id)" in core  # скрытый скан — и без контура на срезах
    assert ".objbar" in read("style.css")
