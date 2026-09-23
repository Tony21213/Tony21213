"""Compare a designed crown with a reference (e.g. the technician's wax-up)."""

from __future__ import annotations

import numpy as np

from .mesh import Mesh, nearest_distance


def surface_samples(mesh: Mesh, per_triangle: int = 6, seed: int = 0) -> np.ndarray:
    """Vertices plus random points on every triangle (dense, uniform-ish sampling)."""
    tri = mesh.triangles
    w = np.random.default_rng(seed).dirichlet([1.0, 1.0, 1.0], size=(len(tri), per_triangle))
    return np.vstack([mesh.vertices, np.einsum("tnk,tkd->tnd", w, tri).reshape(-1, 3)])


def compare_to_reference(outer_points: np.ndarray, crown: Mesh, reference: Mesh,
                         frame, margin_top: float) -> dict:
    """Two-way surface deviation between the crown's anatomy and a reference crown (mm).

    Only the reference above the margin is compared (its fitting surface and
    anything below the finish line are not anatomy).
    """
    ref = surface_samples(reference)
    ref_z = frame.to_local(ref)[:, 2]
    ref_top = ref[ref_z > margin_top + 0.5]
    d_ours, _ = nearest_distance(outer_points, ref)
    d_ref, _ = nearest_distance(ref_top, surface_samples(crown, 3))
    return {
        "crown_to_reference_mean_mm": round(float(d_ours.mean()), 3),
        "crown_to_reference_p90_mm": round(float(np.percentile(d_ours, 90)), 3),
        "reference_to_crown_mean_mm": round(float(d_ref.mean()), 3),
        "reference_to_crown_p90_mm": round(float(np.percentile(d_ref, 90)), 3),
    }


def outer_skin(mesh: Mesh, depth: float = 1.2) -> Mesh:
    """Drop inner surfaces (a screw channel, a ti-base or die cavity) from a crown mesh.

    A face is inner when the ray along its outward normal runs into the crown
    again farther than ``depth`` mm away (fissure walls see each other closer).
    """
    from .mesh import raycast

    n = mesh.face_normals()
    t = raycast(mesh.triangles.mean(axis=1) + 0.02 * n, n, mesh)
    keep = ~(np.isfinite(t) & (t > depth))
    return Mesh(mesh.vertices, mesh.faces[keep])
