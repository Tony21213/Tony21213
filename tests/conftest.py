import pytest

from crownai.design import CrownParameters
from crownai.synthetic import make_antagonist, make_prepared_molar


@pytest.fixture(scope="session")
def prep():
    return make_prepared_molar(n_theta=64)


@pytest.fixture(scope="session")
def antagonist():
    return make_antagonist(n=30)


@pytest.fixture(scope="session")
def fast_params():
    # Coarser sampling keeps the suite quick; the defaults are 128 x 48.
    return CrownParameters(n_theta=64, n_v=28)
