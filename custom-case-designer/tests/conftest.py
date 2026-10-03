import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phantom  # noqa: E402


@pytest.fixture(scope="session")
def jaw_ct():
    from casedesigner.fusion import CaseCT

    return CaseCT(phantom.make_volume())
