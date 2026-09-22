"""Continual learning from finished cases.

Every crown a technician approves (or corrects in exocad and exports) becomes a
training example:

* **features** - measurements of the preparation / margin (what the designer
  knows before designing),
* **target**  - the approved crown's outer anatomy as a normalised radial
  signature plus its real dimensions.

Per tooth class the learner keeps

* a statistical shape model (PCA) of all approved crowns - the space of
  plausible anatomies, re-fitted whenever a case is added, and
* a regressor ``features -> anatomy``: closed-form ridge regression while the
  library is small, switching to a small neural network (numpy MLP) once there
  are enough cases.  The network is warm-started from its previous weights on
  every retrain, so it keeps improving as cases are added instead of starting
  over.

The predicted anatomy replaces the generic library tooth in
:func:`crownai.design.design_crown` (``learner=`` argument).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .anatomy import ShapeModel, sphere_directions, tooth_type_for_fdi
from .margin import detect_margin, make_frame, order_margin
from .mesh import Mesh, raycast

N_THETA, N_PHI, CENTER_Z = 64, 32, 0.45
MLP_MIN_EXAMPLES = 20  # below this, ridge regression generalises better
FEATURE_NAMES = ("half_md", "half_bl", "margin_z_range", "prep_height", "margin_perimeter")


def case_features(margin_local: np.ndarray, prep_top: float) -> np.ndarray:
    """Pre-design measurements of a preparation in its tooth frame (mm)."""
    m = margin_local
    perim = np.linalg.norm(np.diff(np.vstack([m, m[:1]]), axis=0), axis=1).sum()
    return np.array([np.abs(m[:, 0]).max(), np.abs(m[:, 1]).max(), np.ptp(m[:, 2]),
                     prep_top - m[:, 2].mean(), perim])


def crown_signature(crown: Mesh, frame, margin_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised outer-surface signature and (A, B, H) of a finished crown.

    Uses the same normalisation as :class:`crownai.anatomy.ModelTooth` so the
    learned signature can be fed straight back into the designer.
    """
    loc = frame.to_local(crown.vertices)
    z0 = margin_local[:, 2].min() - 1.0
    A, B, H = np.abs(loc[:, 0]).max(), np.abs(loc[:, 1]).max(), loc[:, 2].max()
    norm = Mesh(np.column_stack([loc[:, 0] / A, loc[:, 1] / B, (loc[:, 2] - z0) / (H - z0)]), crown.faces)
    dirs = sphere_directions(N_THETA, N_PHI)
    # Farthest hit = outer anatomy (nearest would be the intaglio).
    sig = raycast(np.array([[0.0, 0.0, CENTER_Z]]), dirs, norm, farthest=True).reshape(N_PHI, N_THETA)
    for j in range(N_THETA):  # rays below the margin miss the crown: extend from above
        col = sig[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            raise ValueError("crown does not surround the preparation centre")
        sig[:, j] = np.interp(np.arange(N_PHI), np.flatnonzero(ok), col[ok])
    return sig.reshape(-1), np.array([A, B, H])


# --------------------------------------------------------------------------
# Regressors
# --------------------------------------------------------------------------

def _ridge_fit(X, Y, alpha=1.0):
    Xb = np.column_stack([X, np.ones(len(X))])
    reg = alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0  # do not shrink the intercept
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def _ridge_predict(W, X):
    return np.column_stack([X, np.ones(len(X))]) @ W


def _mlp_init(n_in, n_hidden, n_out, rng):
    return {"W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out)), "b2": np.zeros(n_out)}


def _mlp_predict(w, X):
    return np.tanh(X @ w["W1"] + w["b1"]) @ w["W2"] + w["b2"]


def _mlp_train(w, X, Y, epochs=600, lr=3e-3, weight_decay=1e-3):
    """Full-batch Adam on MSE; ``w`` is updated in place (warm start)."""
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(val) for k, val in w.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(X)
    for t in range(1, epochs + 1):
        h = np.tanh(X @ w["W1"] + w["b1"])
        err = (h @ w["W2"] + w["b2"]) - Y
        g2 = 2 * err / (n * Y.shape[1])
        grads = {"W2": h.T @ g2 + weight_decay * w["W2"], "b2": g2.sum(0)}
        gh = (g2 @ w["W2"].T) * (1 - h ** 2)
        grads["W1"] = X.T @ gh + weight_decay * w["W1"]
        grads["b1"] = gh.sum(0)
        for k in w:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            w[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
    return float(np.mean(err ** 2))


# --------------------------------------------------------------------------
# Learner
# --------------------------------------------------------------------------

@dataclass
class Prediction:
    shape_model: ShapeModel
    signature: np.ndarray
    dims: np.ndarray  # A, B, H
    n_examples: int
    kind: str


class CrownLearner:
    """A folder-backed library of approved crowns with per-class models."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}

    # ---- storage --------------------------------------------------------
    def _examples_path(self, cls):
        return self.root / f"{cls}_examples.npz"

    def _model_path(self, cls):
        return self.root / f"{cls}_model.npz"

    def examples(self, cls: str) -> dict:
        path = self._examples_path(cls)
        if not path.exists():
            return {"ids": np.array([], dtype=str), "features": np.zeros((0, len(FEATURE_NAMES))),
                    "signatures": np.zeros((0, N_THETA * N_PHI)), "dims": np.zeros((0, 3))}
        d = np.load(path)
        return {k: d[k] for k in ("ids", "features", "signatures", "dims")}

    def add_example(self, cls: str, case_id: str, features, signature, dims) -> int:
        ex = self.examples(cls)
        keep = ex["ids"] != case_id  # re-learning a case replaces it
        ex = {k: v[keep] for k, v in ex.items()}
        ex["ids"] = np.append(ex["ids"], case_id)
        ex["features"] = np.vstack([ex["features"], features])
        ex["signatures"] = np.vstack([ex["signatures"], signature])
        ex["dims"] = np.vstack([ex["dims"], dims])
        np.savez_compressed(self._examples_path(cls), **ex)
        self._log({"event": "add", "class": cls, "case": case_id, "n": int(len(ex["ids"]))})
        return len(ex["ids"])

    def _log(self, entry: dict) -> None:
        entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.root / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- training -------------------------------------------------------
    def train(self, cls: str, seed: int = 0) -> dict:
        ex = self.examples(cls)
        n = len(ex["ids"])
        if n == 0:
            raise ValueError(f"no examples for tooth class '{cls}'")
        pca = ShapeModel.fit(ex["signatures"], N_THETA, N_PHI, CENTER_Z, variance=0.98,
                             max_components=min(12, max(n - 1, 0)))
        X, Y = self._design_matrices(ex, pca)
        if n >= MLP_MIN_EXAMPLES:
            # The network predicts the full (fixed-size) signature rather than
            # PCA coefficients, so its weights stay valid as the library - and
            # with it the PCA basis - grows: every retrain continues from them.
            Y = np.column_stack([ex["signatures"], Y[:, len(pca.stddev):]])
        x_mu, x_sd = X.mean(0), X.std(0) + 1e-6
        y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-6
        Xs, Ys = (X - x_mu) / x_sd, (Y - y_mu) / y_sd

        model = {"n": n, "x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd,
                 "pca_mean": pca.mean, "pca_components": pca.components, "pca_stddev": pca.stddev}
        prev = self._load_model(cls)
        if n >= MLP_MIN_EXAMPLES:
            n_hidden = 16
            rng = np.random.default_rng(seed)
            w = _mlp_init(Xs.shape[1], n_hidden, Ys.shape[1], rng)
            warm = prev is not None and prev.get("kind") == "mlp" and prev["W2"].shape == w["W2"].shape
            if warm:  # continual learning: continue from the previous network
                w = {k: prev[k].copy() for k in ("W1", "b1", "W2", "b2")}
            loss = _mlp_train(w, Xs, Ys, epochs=300 if warm else 1500)
            model.update(kind="mlp", warm_start=warm, train_mse=loss, **w)
        else:
            model.update(kind="ridge", W=_ridge_fit(Xs, Ys, alpha=max(1.0, 10.0 / n)))
        np.savez_compressed(self._model_path(cls), **{k: np.asarray(v) for k, v in model.items()})
        self._cache.pop(cls, None)
        info = {"class": cls, "examples": n, "regressor": model["kind"], "shape_components": len(pca.stddev)}
        self._log({"event": "train", **info})
        return info

    @staticmethod
    def _design_matrices(ex, pca: ShapeModel):
        X = ex["features"]
        if len(pca.stddev):
            coeffs = ((ex["signatures"] - pca.mean) @ pca.components.T) / pca.stddev
        else:
            coeffs = np.zeros((len(X), 0))
        # Predict dimensions relative to the margin so the model transfers across sizes.
        rel = np.log(ex["dims"] / np.column_stack([X[:, 0], X[:, 1], X[:, 3]]))
        return X, np.column_stack([coeffs, rel])

    def _load_model(self, cls: str) -> dict | None:
        if cls in self._cache:
            return self._cache[cls]
        path = self._model_path(cls)
        if not path.exists():
            return None
        d = np.load(path)
        model = {k: d[k] for k in d.files}
        model["kind"] = str(model["kind"])
        self._cache[cls] = model
        return model

    # ---- inference ------------------------------------------------------
    def predict(self, cls: str, features: np.ndarray) -> Prediction | None:
        m = self._load_model(cls)
        if m is None:
            return None
        xs = ((np.asarray(features) - m["x_mu"]) / m["x_sd"])[None]
        if m["kind"] == "mlp":
            ys = _mlp_predict({k: m[k] for k in ("W1", "b1", "W2", "b2")}, xs)
        else:
            ys = _ridge_predict(m["W"], xs)
        y = (ys * m["y_sd"] + m["y_mu"])[0]
        pca = ShapeModel(N_THETA, N_PHI, CENTER_Z, m["pca_mean"], m["pca_components"], m["pca_stddev"])
        k = len(pca.stddev)
        if m["kind"] == "mlp":  # project the predicted signature onto the shape space
            sig, rel = y[:-3], y[-3:]
            coeffs = ((sig - pca.mean) @ pca.components.T) / pca.stddev if k else np.zeros(0)
        else:
            coeffs, rel = y[:k], y[k:k + 3]
        coeffs = np.clip(coeffs, -3.0, 3.0)  # stay within the plausible shape space
        f = np.asarray(features)
        dims = np.exp(rel) * np.array([f[0], f[1], f[3]])
        return Prediction(pca, pca.reconstruct(coeffs), dims, int(m["n"]), m["kind"])

    def evaluate(self, cls: str, max_folds: int = 10) -> dict:
        """Cross-validated error of the learned predictor on this library (mm)."""
        ex = self.examples(cls)
        n = len(ex["ids"])
        if n < 3:
            return {"class": cls, "examples": n, "note": "need at least 3 cases to evaluate"}
        folds = np.array_split(np.random.default_rng(0).permutation(n), min(n, max_folds))
        errs, base = [], []
        tmp = CrownLearner(self.root / ".cv")
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            for cls_ex in (tmp._examples_path(cls), tmp._model_path(cls)):
                cls_ex.unlink(missing_ok=True)
            tmp._cache.clear()
            for i in train:
                tmp.add_example(cls, str(ex["ids"][i]), ex["features"][i], ex["signatures"][i], ex["dims"][i])
            tmp.train(cls)
            mean_sig = ex["signatures"][train].mean(0)
            for i in test:
                pred = tmp.predict(cls, ex["features"][i])
                scale = ex["dims"][i].mean()
                errs.append(np.abs(pred.signature - ex["signatures"][i]).mean() * scale)
                base.append(np.abs(mean_sig - ex["signatures"][i]).mean() * scale)
        import shutil

        shutil.rmtree(tmp.root, ignore_errors=True)
        return {"class": cls, "examples": n, "shape_error_mm": round(float(np.mean(errs)), 3),
                "mean_shape_error_mm": round(float(np.mean(base)), 3)}

    def status(self) -> dict:
        out = {}
        for p in sorted(self.root.glob("*_examples.npz")):
            cls = p.name[: -len("_examples.npz")]
            m = self._load_model(cls)
            out[cls] = {"examples": int(len(self.examples(cls)["ids"])),
                        "trained_on": int(m["n"]) if m else 0,
                        "regressor": m["kind"] if m else None}
        return out


def learn_from_crown(learner: CrownLearner, crown: Mesh, *, prep: Mesh | None = None,
                     margin: np.ndarray | None = None, tooth: int | None = None,
                     axis=(0.0, 0.0, 1.0), md_direction=(1.0, 0.0, 0.0),
                     case_id: str | None = None, retrain: bool = True) -> dict:
    """Add an approved crown (with its preparation and/or margin) to the library."""
    if prep is None and margin is None:
        raise ValueError("need the preparation scan or the margin line to place the crown")
    if margin is None:
        margin = detect_margin(prep, axis=axis)
    margin = order_margin(np.asarray(margin, dtype=np.float64), axis)
    frame = make_frame(margin.mean(axis=0), axis, md_direction)
    m_loc = frame.to_local(margin)
    if prep is not None:
        prep_top = frame.to_local(prep.vertices)[:, 2].max()
    else:  # highest point of the crown's fitting surface is a good proxy
        prep_top = frame.to_local(crown.vertices)[:, 2].max() - 1.5
    feats = case_features(m_loc, prep_top)
    sig, dims = crown_signature(crown, frame, m_loc)
    cls = tooth_type_for_fdi(tooth).name
    case_id = case_id or f"case_{int(time.time() * 1000)}"
    n = learner.add_example(cls, case_id, feats, sig, dims)
    info = {"class": cls, "case": case_id, "examples": n}
    if retrain:
        info.update(learner.train(cls))
    return info
