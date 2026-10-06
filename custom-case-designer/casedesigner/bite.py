"""Исправление прикуса сканов, как в Bite-Finder: нижняя челюсть садится в
положение максимального фиссурно-бугоркового контакта.

Прикус со сканера часто неточен: «провален» (зубы друг в друге), не сомкнут,
перекошен на сторону — щёчный скан прикуса снят на нескольких зубах одной
стороны. Здесь нижняя челюсть «оседает» в верхнюю под силой смыкания, как
настоящая: по всем шести степеням свободы (три сдвига, три поворота), пока
бугры не встанут в фиссуры и дальше сомкнуться нельзя.

Как считается (квазистатика с контактами):
  * зубы — точки нижнего скана у верхних зубов и поверхность верхнего скана
    (ближайшая вершина и её нормаль: расстояние со знаком, минус — внутри);
  * провал сначала убирается раскрытием вокруг шарнирной оси;
  * на каждом шаге решается линейная задача: какой малый сдвиг и поворот
    (не больше STEP_MM на зубах, затем FINE_STEP_MM) смыкает челюсти сильнее
    всего, если ни одна точка не уходит в верхние зубы глубже TOL_MM. Сила
    смыкания — в центр жевательной поверхности, вдоль нормали окклюзионной
    плоскости;
  * мыщелки остаются у своих мест: вверх, в суставную ямку, — не больше
    CONDYLE_UP_MM, в остальные стороны — CONDYLE_MM (по КТ — точно, без КТ —
    по треугольнику Бонвилля). Поэтому челюсть смыкается по дуге вокруг
    шарнирной оси, а не «подъезжает» вверх целиком и не проваливается туда,
    где верхнего зуба нет (препарированный зуб, край скана);
  * от прикуса сканера — не дальше MAX_SHIFT_MM в плоскости и MAX_TURN_DEG:
    исправляются ошибки скана, а не ищется другой прикус. Все допуски
    отсчитываются от прикуса сканера;
  * челюсть, вставшая бугром на бугор, «постукивается»: раскрывается на
    TAP_OPEN_MM, сдвигается на TAP_SHIFT_MM в четыре стороны и оседает
    снова; берётся самое сомкнутое положение.

Проверено на четырёх выгрузках exocad (в пятой сканы не в прикусе): прикус,
сбитый поворотом вокруг шарнирной оси (провал или раскрытие на 0,2–0,6 мм по
резцам) и перекосом на сторону до 0,3°, возвращается в то же положение
точнее 0,12 мм, если контакты есть на обеих сторонах. Если на одной стороне
контактов нет (нет антагонистов, препарированные зубы), наклон в эту сторону
определён плохо — ошибка до 0,5 мм; отчёт об этом предупреждает.

Поиск прикуса без скана прикуса (сканы в разных системах координат) — впереди:
первая попытка (постановка по осям дуг и оседание) ошибалась на 1,5–20 мм.

Прикус сканера в приоритете: результат — предложение. Отчёт говорит, на
сколько он сдвинул челюсть (по резцам и молярам, в градусах) и как
изменились контакты по участкам.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import motion as mo
from .register import apply

TOL_MM = 0.02  # допустимое проникновение
CONTACT_MM = 0.1  # точка в контакте, если ближе к верхним зубам
REACH_MM = 3.0  # точки нижних зубов дальше этого от верхних в прикусе не участвуют
STEP_MM = 0.05  # наибольший сдвиг точек зубов за шаг
FINE_STEP_MM = 0.01  # второй проход — точнее
NEAR_MM = 0.8  # в ограничения шага — точки ближе этого (за шаг дальше не дотянуться)
SIGNED_REACH_MM = 2.0
MAX_POINTS = 12000
MAX_STEPS = 300  # крупных шагов на одно оседание
FINE_STEPS = 80  # мелких
STOP_MM = 2e-4  # шаг смыкания меньше — челюсть стоит
STOP_STEPS = 4
MAX_SHIFT_MM = 0.3  # от прикуса сканера — не дальше (в плоскости)
MAX_OPEN_MM = 3.0  # раскрыть, чтобы убрать проникновение, — не больше
MAX_TURN_DEG = 1.0
CONDYLE_MM = 0.3  # мыщелки — не дальше от исходного места (вниз, вперёд-назад, в стороны)
CONDYLE_FREE_MM = 0.3  # без КТ (мыщелки по Бонвиллю) — так же
CONDYLE_UP_MM = 0.1  # вверх, в суставную ямку, мыщелок почти не уходит
CONDYLE_UP_FREE_MM = 0.3
TAP_ROUNDS = 3
TURN_COST = 0.02  # лёгкий штраф за сдвиги и повороты: из равных положений — ближайшее
TAP_OPEN_MM = 0.3  # «постукивание»: раскрыть на столько и сдвинуть
TAP_SHIFT_MM = 0.2


@dataclass
class BiteResult:
    transform: np.ndarray  # 4×4: нижний скан (как дан) → исправленное положение
    report: dict = field(default_factory=dict)


def _orient_normals(points, normals, toward):
    """Нормали верхних зубов — наружу, к нижним (по среднему знаку у точек toward)."""
    tree = cKDTree(points)
    _d, i = tree.query(toward)
    if np.mean(np.sum((toward - points[i]) * normals[i], axis=1)) < 0:
        normals = -normals
    return tree, normals


class _Upper:
    def __init__(self, vertices, faces):
        import trimesh

        mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces), process=False)
        self.points = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.tree = cKDTree(self.points)

    def orient(self, toward):
        self.tree, self.normals = _orient_normals(self.points, self.normals, toward)

    def signed(self, p):
        d, i = self.tree.query(p, distance_upper_bound=SIGNED_REACH_MM, workers=-1)
        out = np.full(len(p), np.inf)
        ok = np.isfinite(d)
        out[ok] = np.sum((p[ok] - self.points[i[ok]]) * self.normals[i[ok]], axis=1)
        n = np.zeros((len(p), 3))
        n[ok] = self.normals[i[ok]]
        return out, n


def _twist(omega, v, centre):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    T[:3, 3] = centre + v - T[:3, :3] @ centre
    return T


def _open(T, points, condyles, up, mm):
    """Раскрыть челюсть на mm (по центру зубов) поворотом вокруг шарнирной оси; без мыщелков — сдвигом."""
    if len(condyles) == 2:
        cr, cl = (apply(T, c[None])[0] for c in condyles)
        axis = (cr - cl) / np.linalg.norm(cr - cl)
        hinge = (cr + cl) / 2
        centre = apply(T, points).mean(0)
        lever = np.linalg.norm(np.cross(centre - hinge, axis))
        for sign in (1.0, -1.0):
            R = Rotation.from_rotvec(sign * mm / max(lever, 1.0) * axis).as_matrix()
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R, hinge - R @ hinge
            if (apply(M, centre[None])[0] - centre) @ up < 0:  # центр зубов — от верхних
                return M @ T
    return _twist(np.zeros(3), -up * mm, np.zeros(3)) @ T


def settle(upper: "_Upper", points, start, up, condyles, condyle_mm, max_shift, max_turn_deg, max_open=MAX_OPEN_MM,
           condyle_up_mm=CONDYLE_UP_FREE_MM, anchor=None):
    """Опустить нижнюю челюсть в контакт: 4×4 и число шагов.

    points и condyles — точки зубов и мыщелков нижней челюсти в координатах
    нижнего скана (как дан), start — его положение; up — направление
    смыкания (к верхним зубам) в координатах кейса. anchor — положение, от
    которого отсчитываются допуски мыщелков, сдвига и поворота (по умолчанию
    start; при повторных оседаниях — исходный прикус, чтобы допуски не
    складывались).
    """
    T = start.copy()
    up = up / np.linalg.norm(up)
    # 1. Убрать проникновение: раскрыть вдоль up (−up), пока все точки не снаружи.
    opened = 0.0
    while True:
        d, _n = upper.signed(apply(T, points))
        worst = -np.min(d)
        if worst <= TOL_MM or opened > max_open:
            break
        step = worst + 0.05
        T = _open(T, points, condyles, up, step)
        opened += step
    anchor = start.copy() if anchor is None else anchor
    base = apply(anchor, points).mean(0)  # центр жевательной поверхности при исходном положении
    radius = max(float(np.percentile(np.linalg.norm(apply(T, points) - apply(T, points).mean(0), axis=1), 95)), 10.0)
    co0 = [apply(anchor, c[None])[0] for c in condyles]
    side = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    side /= np.linalg.norm(side)
    basis = np.array([side, np.cross(up, side), up])
    if len(condyles) == 2:  # оси, вокруг которых поворот ограничен: вертикаль и «вперёд» (перпендикуляр к шарниру)
        hinge = co0[0] - co0[1]
        fwd = np.cross(up, hinge)
        tilt = np.array([up, fwd / np.linalg.norm(fwd)])
    else:
        tilt = np.array([up, side])
    still, steps, step_mm, fine_from = 0, 0, STEP_MM, None
    w_lim = step_mm / radius
    for steps in range(1, MAX_STEPS + FINE_STEPS + 1):
        if fine_from is None and steps > MAX_STEPS:
            break
        if fine_from is not None and steps - fine_from > FINE_STEPS:
            break
        p = apply(T, points)
        rel = T @ np.linalg.inv(anchor)
        centre = apply(rel, base[None])[0]  # центр жевательной поверхности — с челюстью
        shift = centre - base  # от исходного положения
        turn = Rotation.from_matrix(rel[:3, :3]).as_rotvec()
        d, n = upper.signed(p)
        near = d < NEAR_MM
        r = p[near] - centre
        J = np.c_[np.cross(r, n[near]), n[near]]  # δd = (r×n)·ω + n·v
        # переменные: ξ = (ω, v) = x⁺ − x⁻ ≥ 0 — для штрафа |ξ|
        A_ub = [np.c_[-J, J]]
        b_ub = [d[near] + TOL_MM]
        for c0, c in zip(co0, (apply(T, cc[None])[0] for cc in condyles)):  # мыщелки — у своих мест
            rc = c - centre
            Jc = basis @ np.c_[np.array([[0, rc[2], -rc[1]], [-rc[2], 0, rc[0]], [rc[1], -rc[0], 0]]), np.eye(3)]
            moved = basis @ (c - c0)  # по осям: в плоскости, в плоскости, вверх
            top = np.array([condyle_mm, condyle_mm, condyle_up_mm])  # вверх — в ямку — почти нельзя
            A_ub += [np.c_[Jc, -Jc], np.c_[-Jc, Jc]]
            b_ub += [top - moved, condyle_mm + moved]
        # от исходного положения: без мыщелков — сдвиг в плоскости не больше max_shift (с мыщелками его
        # держат они); поворот набок и вокруг вертикали — не больше max_turn_deg. Вращение вокруг
        # шарнирной оси (открывание и закрывание) свободно: по дуге центр зубов сдвигается и вперёд-назад.
        if len(condyles) != 2:
            inplane = np.eye(3) - np.outer(up, up)
            A_ub += [np.c_[np.zeros((3, 3)), inplane, np.zeros((3, 3)), -inplane],
                     np.c_[np.zeros((3, 3)), -inplane, np.zeros((3, 3)), inplane]]
            b_ub += [max_shift - inplane @ shift, max_shift + inplane @ shift]
        lim = np.radians(max_turn_deg)
        A_ub += [np.c_[tilt, np.zeros((2, 3)), -tilt, np.zeros((2, 3))],
                 np.c_[-tilt, np.zeros((2, 3)), tilt, np.zeros((2, 3))]]
        b_ub += [lim - tilt @ turn, lim + tilt @ turn]
        A_ub, b_ub = np.vstack(A_ub), np.concatenate(b_ub)
        cost = TURN_COST * np.r_[np.full(3, radius), np.ones(3)]
        g = np.r_[np.zeros(3), -up]  # цель — сомкнуть: центр жевательной поверхности — к верхним зубам
        c_obj = np.r_[g + cost, -g + cost]
        bounds = [(0, w_lim)] * 3 + [(0, step_mm)] * 3 + [(0, w_lim)] * 3 + [(0, step_mm)] * 3
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status != 0:
            break
        x = res.x[:6] - res.x[6:]
        omega, v = x[:3], x[3:]
        T = _twist(omega, v, centre) @ T
        gain = -float(g @ x)
        still = still + 1 if gain < STOP_MM * step_mm / STEP_MM else 0
        if still >= STOP_STEPS:
            if step_mm == FINE_STEP_MM:
                break
            step_mm, still, fine_from = FINE_STEP_MM, 0, steps  # стоит на крупном шаге — досадить мелким
            w_lim = step_mm / radius
    return T, steps


def _contacts(upper, points, T, frame):
    d, _n = upper.signed(apply(T, points))
    hit = d <= CONTACT_MM
    local = apply(frame, apply(T, points[hit]))
    sectors = {"right": int((local[:, 0] > 8).sum()), "front": int((np.abs(local[:, 0]) <= 8).sum()),
               "left": int((local[:, 0] < -8).sum())}
    return {"points": int(hit.sum()), "sectors": sectors,
            "penetration_mm": round(float(max(0.0, -np.min(d))), 3)}


def _anatomy(lower_v, upper_v, condyles):
    """Монтаж: окклюзионная плоскость, вперёд, резцовая точка; мыщелки — по КТ или Бонвиллю."""
    anat = mo.anatomy_average(lower_v, upper_v)
    if condyles is not None:
        anat.points["condyle_right"], anat.points["condyle_left"] = (np.asarray(condyles[0], float),
                                                                      np.asarray(condyles[1], float))
    return anat


def _sample(points, n, seed=0):
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def _report(upper, pts, T, anat, before_T, mode):
    frame = anat.frame
    inc = anat.points["incisal"]
    right, left = anat.points["condyle_right"], anat.points["condyle_left"]
    # точки для отчёта: резцы и первые моляры (≈ 25 мм кзади от резцов, по 22 мм в стороны)
    R = frame[:3, :3]
    molar = lambda side: inc + R.T @ np.array([22.0 * side, -25.0, 0.0])  # noqa: E731
    delta = T @ np.linalg.inv(before_T)
    move = lambda p: float(np.linalg.norm(apply(delta, p[None])[0] - p))  # noqa: E731
    shift = apply(frame, apply(delta, inc[None]))[0] - apply(frame, inc[None])[0]
    angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
    return {
        "mode": mode,
        "incisal_mm": round(move(inc), 3),
        "incisal_shift_mm": {"right": round(float(shift[0]), 3), "forward": round(float(shift[1]), 3),
                             "up": round(float(shift[2]), 3)},
        "molar_right_mm": round(move(molar(1)), 3), "molar_left_mm": round(move(molar(-1)), 3),
        "condyle_right_mm": round(move(right), 3), "condyle_left_mm": round(move(left), 3),
        "turn_deg": round(angle, 3),
        "before": _contacts(upper, pts, before_T, frame), "after": _contacts(upper, pts, T, frame),
    }


def correct_bite(upper_vertices, upper_faces, lower_vertices, condyles=None, max_shift_mm=MAX_SHIFT_MM,
                 max_turn_deg=MAX_TURN_DEG) -> BiteResult:
    """Исправить прикус сканов (нижний скан уже стоит в прикусе сканера).

    condyles — (правый, левый) мыщелки в координатах сканов, если известны по
    КТ; иначе — треугольник Бонвилля. Возвращает матрицу «нижний скан →
    исправленный» и отчёт.
    """
    lower = np.asarray(lower_vertices, float)
    upper = _Upper(upper_vertices, upper_faces)
    d, _ = upper.tree.query(lower, distance_upper_bound=REACH_MM, workers=-1)
    near = lower[np.isfinite(d)]
    if len(near) < 50:
        raise ValueError(f"сканы не в прикусе: нижние зубы дальше {REACH_MM:g} мм от верхних — "
                         "используйте поиск прикуса без скана прикуса")
    upper.orient(near)
    anat = _anatomy(lower, upper.points, condyles)
    up = anat.frame[:3, :3][2]  # к верхним зубам
    pts = _sample(near, MAX_POINTS)
    co = [anat.points["condyle_right"], anat.points["condyle_left"]]
    cmm = CONDYLE_MM if condyles is not None else CONDYLE_FREE_MM
    cup = CONDYLE_UP_MM if condyles is not None else CONDYLE_UP_FREE_MM
    T, steps = settle(upper, pts, np.eye(4), up, co, cmm, max_shift_mm, max_turn_deg, condyle_up_mm=cup)
    for _ in range(TAP_ROUNDS):  # пока «постукивание» находит положение сомкнутее
        T2, taps = _tap(upper, pts, T, up, co, cmm, max_shift_mm, max_turn_deg, anat.frame, cup, np.eye(4))
        steps += taps
        if T2 is T:
            break
        T = T2
    T = _refine(upper, near, T, up, co, condyles, max_shift_mm, max_turn_deg)
    rep = _report(upper, near, T, anat, np.eye(4), "прикус сканера")
    rep["steps"] = steps
    rep["condyles"] = "КТ" if condyles is not None else "треугольник Бонвилля"
    rep["notes"] = _notes(rep)
    return BiteResult(T, rep)


def _notes(rep: dict) -> list[str]:
    """Что сказать пользователю об исправлении."""
    notes = []
    moved = max(rep["incisal_mm"], rep["molar_right_mm"], rep["molar_left_mm"])
    if rep["before"]["penetration_mm"] > 0.05:
        notes.append(f"прикус сканера провален на {rep['before']['penetration_mm']:.2f} мм")
    if moved < 0.05:
        notes.append("прикус сканера верный: исправлять нечего")
    elif moved > 0.25:
        notes.append(f"прикус сканера, похоже, неточен: челюсть сдвинута до {moved:.2f} мм — проверьте")
    after = rep["after"]["sectors"]
    for side, name in (("right", "справа"), ("left", "слева")):
        if after[side] == 0:
            notes.append(f"{name} нет контактов — наклон челюсти в эту сторону определён плохо, проверьте")
    if rep["condyles"] != "КТ":
        notes.append("мыщелки — по треугольнику Бонвилля; с КТ точнее")
    return notes


def _closure(upper, pts, T, up):
    """Насколько сомкнута челюсть (центр зубов вдоль up) и сколько точек в контакте."""
    p = apply(T, pts)
    d, _ = upper.signed(p)
    return float(p.mean(0) @ up), int((d <= CONTACT_MM).sum())


def _tap(upper, pts, T, up, co, cmm, max_shift, max_turn, frame, cup=CONDYLE_UP_FREE_MM, anchor=None):
    """«Постучать»: из найденного положения раскрыть, сдвинуть в четыре стороны и снова опустить.

    Челюсть, вставшая бугром на бугор, так соскальзывает в фиссуры: из всех
    устойчивых положений выбирается самое сомкнутое (потом — с большим
    контактом). Сдвиги — в осях монтажа (вправо, вперёд).
    """
    R = frame[:3, :3]
    best = (_closure(upper, pts, T, up), T)
    steps = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        move = R.T @ np.array([dx * TAP_SHIFT_MM, dy * TAP_SHIFT_MM, 0.0])
        start = _twist(np.zeros(3), move, np.zeros(3)) @ _open(T, pts, co, up, TAP_OPEN_MM)
        Ts, n = settle(upper, pts, start, up, co, cmm, max_shift, max_turn, condyle_up_mm=cup, anchor=anchor)
        steps += n
        score = _closure(upper, pts, Ts, up)
        if score[0] > best[0][0] + 0.01 or (abs(score[0] - best[0][0]) <= 0.01 and score[1] > best[0][1]):
            best = (score, Ts)
    return best[1], steps


def _refine(upper, all_points, T, up, co, condyles, max_shift, max_turn):
    """Проверить по всем точкам: если какие-то ушли глубже допуска — досадить с ними."""
    d, _ = upper.signed(apply(T, all_points))
    bad = all_points[d < -TOL_MM - 0.01]
    if len(bad) == 0:
        return T
    pts = np.vstack([_sample(all_points, MAX_POINTS, 1), bad])
    T2, _ = settle(upper, pts, T, up, co, CONDYLE_MM if condyles is not None else CONDYLE_FREE_MM,
                   max_shift, max_turn, condyle_up_mm=CONDYLE_UP_MM if condyles is not None else CONDYLE_UP_FREE_MM,
                   anchor=np.eye(4))
    return T2
