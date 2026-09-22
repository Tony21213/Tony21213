import numpy as np
import pytest

from crownai import learning
from crownai.anatomy import TOOTH_TYPES, ShapeModel, ToothType, synthetic_library
from crownai.design import design_crown
from crownai.learning import CrownLearner, learn_from_crown
from crownai.synthetic import make_prepared_molar


def test_shape_model_fit_and_partial(tmp_path):
    sigs = synthetic_library(16, np.random.default_rng(0), n_theta=32, n_phi=16)
    model = ShapeModel.fit(sigs[1:], 32, 16, 0.45)  # sigs[0] is held out
    assert 0 < len(model.stddev) <= 14
    target = sigs[0]
    mask = np.zeros((16, 32), bool)
    mask[:, :16] = True  # only half of the circumference observed
    mask = mask.reshape(-1)
    rec = model.reconstruct(model.fit_partial(mask, target[mask], regularization=0.1))
    err = np.abs(rec[~mask] - target[~mask]).mean()
    assert err < 0.8 * np.abs(model.mean[~mask] - target[~mask]).mean()
    model.save(tmp_path / "m.npz")
    back = ShapeModel.load(tmp_path / "m.npz")
    ones = np.ones(len(model.stddev))
    assert np.allclose(back.reconstruct(ones), model.reconstruct(ones))


def _technician_case(scale, cusp_boost, params):
    """A prep of a given size and a 'technician-approved' crown for it."""
    prep = make_prepared_molar(n_theta=48)
    prep.vertices[:, :2] *= scale
    base = TOOTH_TYPES["molar"]
    style = ToothType("molar", base.mesiodistal * scale, base.buccolingual * scale, base.height,
                      base.exponent, base.top_taper_md, base.top_taper_bl,
                      tuple((x, y, h * cusp_boost) for x, y, h in base.cusps), base.fossa_depth)
    return prep, design_crown(prep, tooth=style, params=params).crown


@pytest.fixture(scope="module")
def library(tmp_path_factory, fast_params):
    lib = CrownLearner(tmp_path_factory.mktemp("lib"))
    for k, scale in enumerate([0.9, 0.95, 1.0, 1.05, 1.1]):
        prep, crown = _technician_case(scale, 1.3, fast_params)
        learn_from_crown(lib, crown, prep=prep, tooth=36, case_id=f"c{k}")
    return lib


def test_library_grows_and_trains(library):
    st = library.status()["molar"]
    assert st["examples"] == 5 and st["trained_on"] == 5 and st["regressor"] == "ridge"
    assert (library.root / "history.jsonl").exists()


def test_relearning_a_case_replaces_it(library, fast_params):
    prep, crown = _technician_case(1.0, 1.3, fast_params)
    info = learn_from_crown(library, crown, prep=prep, tooth=36, case_id="c2")
    assert info["examples"] == 5


def test_prediction_follows_preparation_size(library):
    small = library.predict("molar", np.array([4.0, 4.2, 0.8, 4.6, 36.0]))
    large = library.predict("molar", np.array([5.2, 5.5, 0.8, 4.6, 46.0]))
    assert large.dims[0] > small.dims[0] and large.dims[1] > small.dims[1]


def test_design_uses_learned_anatomy(library, fast_params):
    res = design_crown(make_prepared_molar(n_theta=48), tooth=36, learner=library, params=fast_params)
    assert res.report["anatomy_source"].startswith("learned")
    assert res.crown.is_watertight()


def test_evaluate(library):
    ev = library.evaluate("molar", max_folds=3)
    assert ev["shape_error_mm"] >= 0


def test_neural_regressor_warm_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "MLP_MIN_EXAMPLES", 4)
    lib = CrownLearner(tmp_path)
    rng = np.random.default_rng(0)
    sigs = synthetic_library(6, rng, n_theta=learning.N_THETA, n_phi=learning.N_PHI)
    for k in range(6):
        s = 0.9 + 0.04 * k
        feats = np.array([4.8 * s, 5.0 * s, 0.8, 4.6, 40 * s])
        lib.add_example("molar", f"c{k}", feats, sigs[k], np.array([5.2 * s, 5.6 * s, 7.4]))
        if k >= 3:
            info = lib.train("molar")
    assert info["regressor"] == "mlp"
    model = lib._load_model("molar")
    assert bool(model["warm_start"])  # second+ MLP training continued from saved weights
    pred = lib.predict("molar", np.array([4.8, 5.0, 0.8, 4.6, 40.0]))
    assert np.isfinite(pred.signature).all() and (pred.dims > 0).all()
