import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _deterministic():
    np.random.seed(0)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture
def model():
    from superfluid import TinyTransformer
    return TinyTransformer(32, d=32, n_layers=2, n_ctx=32, seed=0)
