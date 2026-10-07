"""Сверка лицевой дуги с образцом exocad (Zebris, проект 012 из CAD-Data exocad).

1. Вилка библиотеки (zebris_type_sd) ищется на скане маркера образца: ICP из 24 ориентаций; из зеркальных
   вариантов (вилка почти симметрична) берётся тот, где верхние зубы — со стороны +y вилки (ось y вилки —
   вверх: так по записи движений образца 011, где при открывании метки уходят по −y).
2. По меткам вилки на скане и в .jawmotion восстанавливается «сканы → регистратор» (как у exocad).
3. Та же система подаётся в нашу цепочку (exocad_facebow): наша вилка, наши метки → «сканы → регистратор»
   по нашим файлам; положения моделей в артикуляторе (MovementregisterToArticulatorTransformation SAM 2P)
   сравниваются с положениями по файлам образца.

Запуск: python tools/verify_facebow_sample.py [папка DentalCADApp exocad]
"""

import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from casedesigner import exocad_facebow as ef  # noqa: E402
from casedesigner.register import apply, rigid  # noqa: E402


def kabsch(A, B):
    """Жёсткое T: B ≈ T · A."""
    ca, cb = A.mean(0), B.mean(0)
    U, _s, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return rigid(R, cb - R @ ca)


def find_fork(fork: trimesh.Trimesh, marker: np.ndarray, upper: np.ndarray):
    """Положение вилки (вилка → координаты сканов) на скане маркера: ICP, зеркальные варианты — по стороне зубов."""
    extra = marker[cKDTree(upper).query(marker)[0] > 1.0]  # не верхний скан: вилка и материал
    target = cKDTree(extra)
    pts = np.asarray(fork.sample(20000))

    def icp(T, iters):
        for _ in range(iters):
            p = apply(T, pts)
            d, j = target.query(p)
            k = d < np.quantile(d, 0.8)
            T = kabsch(p[k], extra[j[k]]) @ T
        d = target.query(apply(T, pts))[0]
        return T, float((d < 0.3).mean())

    found = []
    for R in Rotation.create_group("O").as_matrix():
        T, share = icp(rigid(R, extra.mean(0) - R @ pts.mean(0)), 40)
        teeth_up = float(np.median((upper - T[:3, 3]) @ T[:3, 1]))  # верхние зубы — по +y вилки?
        found.append((share, teeth_up, T))
    found.sort(key=lambda f: -f[0])
    good = [f for f in found if f[1] > 0]
    T, share = icp(good[0][2], 80)
    return T, share, found[:6]


def main(app=None):
    app = app or ef.find_exocad()
    data = os.path.join(os.path.dirname(app), "CAD-Data", "2025-03-25_99999-012")
    stem = os.path.join(data, "2025-03-25_99999-012-")
    reg = ef.load_register(app)
    upper = trimesh.load_mesh(stem + "upperjaw.stl")
    lower = trimesh.load_mesh(stem + "lowerjaw.stl")
    marker = trimesh.load_mesh(stem + "movementmarker.stl")
    jm = ef.read_jawmotion(os.path.join(data, "Facebow_articulator_settings.jawmotion"))
    uv, lv = np.asarray(upper.vertices), np.asarray(lower.vertices)

    B, share, cands = find_fork(reg.fork, np.asarray(marker.vertices), uv)
    print("вилка на скане маркера: варианты (доля точек ближе 0.3 мм, верхние зубы по +y вилки):")
    for s, up, _T in cands:
        print(f"   {s:.0%}  {'да' if up > 0 else 'нет'}")
    print(f"взят: {share:.0%} точек вилки ближе 0.3 мм к скану маркера")
    A = kabsch(apply(B, reg.marks), jm["marks"])  # сканы → регистратор (exocad)
    print("метки: расхождение треугольников, мм:", np.round(np.linalg.norm(apply(A @ B, reg.marks) - jm["marks"], axis=1), 3))
    U, L = apply(A, uv), apply(A, lv)
    print(f"анатомия в системе регистратора (x влево, y вверх, z вперёд): верхний скан y {U[:, 1].mean():+.1f}, "
          f"нижний y {L[:, 1].mean():+.1f}; резцы z {U[:, 2].max():.1f} мм; середина дуги x {np.median(U[:, 0]):+.1f} мм")
    up_axis = A[:3, :3].T @ [0, 1, 0]
    fork_vs_head = float(np.degrees(Rotation.from_matrix((A @ B)[:3, :3]).magnitude()))
    print(f"настоящая вилка относительно осей регистратора (головы): поворот {fork_vs_head:.1f}°")

    # Наша цепочка с той же системой: мыщелки на шарнирной оси (±50 мм), резцовая точка — передняя точка верхнего.
    frame = np.linalg.inv(ef.TO_REGISTER) @ A  # сканы → система монтажа (x вправо, y вперёд, z вверх)
    inv = np.linalg.inv(A)
    inc = uv[np.argmax(U[:, 2])]
    fb = ef.facebow(frame, apply(inv, [[-50.0, 0, 0]])[0], apply(inv, [[50.0, 0, 0]])[0], inc, reg)
    ours = kabsch(apply(fb.fork_pose, reg.marks), fb.marks)  # как exocad поставит модели по нашим файлам
    M = ef.REGISTER_TO_ARTICULATOR

    def to_art(T, p):
        q = apply(T, p)
        return q @ M[:3, :3] + M[3, :3]

    d = np.linalg.norm(to_art(ours, uv) - to_art(A, uv), axis=1)
    ang = np.degrees(Rotation.from_matrix(ours[:3, :3] @ A[:3, :3].T).magnitude())
    print(f"модели в артикуляторе: наша цепочка против образца — до {d.max():.4f} мм, поворот {ang:.4f}°")
    ours_vs_real_fork = float(np.degrees(Rotation.from_matrix(fb.fork_pose[:3, :3] @ B[:3, :3].T).magnitude()))
    print(f"наша вилка относительно настоящей: поворот {ours_vs_real_fork:.1f}°")
    return {"share": share, "max_mm": float(d.max()), "deg": float(ang), "upper_y": float(U[:, 1].mean()),
            "lower_y": float(L[:, 1].mean()), "incisal_z": float(U[:, 2].max()), "A": A, "B": B,
            "fork_vs_head_deg": fork_vs_head, "ours_vs_real_fork_deg": ours_vs_real_fork}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
