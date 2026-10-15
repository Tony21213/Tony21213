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


def test_slices_like_3d_right_button_scrolls():
    """Срезы КТ (решение владельца 15.10): как окно 3D, но правая кнопка листает срезы; яркость и контраст —
    Ctrl + правая; вторая кнопка аккорда — первым обработчиком движения, раньше обработчиков нажатия."""
    s = (WEB / "slices.js").read_text(encoding="utf-8")
    assert "const windowing = e.button === 2 && (e.ctrlKey || e.metaKey);" in s
    assert "this.step(n * (ev.shiftKey ? 5 : 1));" in s and "PX_PER_SLICE" in s
    assert s.index("this.onChord(e)") < s.index("this.onHover(e)")
    assert "(e.buttons & 3) === 3" in s and "e.stopImmediatePropagation()" in s
    assert "this.centerAt(ev.offsetX, ev.offsetY)" in s
    assert "!mode && !g.chord" in s  # после сдвига двумя кнопками перекрестие не ставится
    case = (WEB / "steps" / "case.js").read_text(encoding="utf-8")
    assert "['Правая кнопка', 'листать срезы: вверх-вниз (Shift — по 5)']" in case
    assert "['Ctrl + правая', 'яркость и контраст КТ']" in case
