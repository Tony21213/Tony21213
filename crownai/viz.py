"""Preview images: 3D view plus mesiodistal / buccolingual cross-sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .design import CrownResult
from .mesh import Mesh


def slice_mesh(mesh: Mesh, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Intersection segments of ``mesh`` with a plane, shape (S, 2, 3)."""
    tri = mesh.triangles
    d = (tri - origin) @ normal
    segs = []
    sign = d > 0
    cross = sign.any(1) & ~sign.all(1)
    tri, d = tri[cross], d[cross]
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = d[:, i], d[:, j]
        hit = (di > 0) != (dj > 0)
        t = np.where(hit, di / np.where(hit, di - dj, 1.0), np.nan)
        pts.append(tri[:, i] + t[:, None] * (tri[:, j] - tri[:, i]))
    pts = np.stack(pts, 1)  # (T, 3 edges, 3)
    for k in range(len(pts)):
        good = pts[k][~np.isnan(pts[k, :, 0])]
        if len(good) >= 2:
            segs.append(good[:2])
    return np.array(segs).reshape(-1, 2, 3)


def render_preview(result: CrownResult, prep: Mesh, path: str | Path,
                   antagonist: Mesh | None = None, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    f = result.frame
    fig = plt.figure(figsize=(15, 5.2))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    crown_loc = Mesh(f.to_local(result.crown.vertices), result.crown.faces)
    prep_loc = Mesh(f.to_local(prep.vertices), prep.faces)
    light = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
    for mesh, color, alpha in ((prep_loc, (0.85, 0.55, 0.5), 0.35), (crown_loc, (0.95, 0.93, 0.86), 1.0)):
        shade = 0.45 + 0.55 * np.clip(mesh.face_normals() @ light, 0, 1)
        fc = np.clip(np.array(color)[None] * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=np.c_[fc, np.full(len(fc), alpha)],
                                             linewidths=0))
    m = f.to_local(result.margin)
    ax.plot(*np.vstack([m, m[:1]]).T, color="tab:blue", lw=1.5)
    lim = 7
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-4, 10), xlabel="MD", ylabel="BL", zlabel="occlusal")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-60)
    ax.set_title("Crown on preparation")

    for k, (normal, axis_idx, name) in enumerate(((f.x, 1, "Buccolingual section"),
                                                   (f.y, 0, "Mesiodistal section"))):
        ax2 = fig.add_subplot(1, 3, 2 + k)
        items = [(prep, "tab:red", "die"), (result.crown, "black", "crown")]
        if antagonist is not None:
            items.append((antagonist, "tab:green", "antagonist"))
        for mesh, color, label in items:
            segs = slice_mesh(mesh, f.origin, normal)
            if len(segs):
                loc = f.to_local(segs.reshape(-1, 3)).reshape(-1, 2, 3)
                ax2.add_collection(LineCollection(loc[:, :, [axis_idx, 2]], colors=color, linewidths=1.2,
                                                  label=label))
        ax2.set(xlim=(-8, 8), ylim=(-4, 10), aspect="equal", xlabel="mm", ylabel="mm", title=name)
        ax2.grid(alpha=0.3)
        ax2.legend(loc="lower right", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
