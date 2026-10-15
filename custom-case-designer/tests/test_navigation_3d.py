"""Мышь в окне 3D как в exocad (решение владельца 15.10): правая — вращение, левая + правая — сдвиг, щелчок колесом —
центр вращения в точку. Поведение проверено в браузере событиями мыши; здесь — устройство кода и подсказка."""

from pathlib import Path

WEB = Path(__file__).parents[1] / "casedesigner" / "app" / "web"


def test_buttons_like_exocad():
    v = (WEB / "viewer3d.js").read_text(encoding="utf-8")
    assert "this.controls.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: THREE.MOUSE.ROTATE };" in v
    assert "(e.buttons & 3) === 3" in v  # левая и правая вместе — сдвиг
    assert "this.centerAt(e.clientX, e.clientY)" in v  # щелчок колесом
    assert v.count("}, true);") >= 3  # слушатели — в фазе захвата, раньше OrbitControls


def test_help_lists_3d_mouse():
    case = (WEB / "steps" / "case.js").read_text(encoding="utf-8")
    assert "['Окно 3D', [['Правая кнопка', 'вращать'], ['Левая + правая', 'сдвинуть" in case
    assert "['Щелчок колесом', 'центр вращения — в точку']" in case
