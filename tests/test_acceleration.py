"""Feature 3 -- hardware acceleration: unit tests (numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluidlayer.acceleration import (  # noqa: E402
    benchmark_vectorization,
    detect_acceleration,
    loop_scores,
    vectorized_scores,
)


def test_detection_report_shape():
    info = detect_acceleration()
    assert isinstance(info, dict)
    for key in ("torch", "cuda", "cuda_devices", "mps", "numba", "cupy"):
        assert key in info
    assert info["vectorized_numpy"] is True


def test_vectorized_matches_loop():
    rng = np.random.default_rng(0)
    content = rng.normal(size=(64, 8))
    Wq = rng.normal(size=(8, 8))
    Wk = rng.normal(size=(8, 8))
    protos = rng.normal(size=(16, 8))
    a = loop_scores(content, Wq, Wk, protos)
    b = vectorized_scores(content, Wq, Wk, protos)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_speedup_positive():
    res = benchmark_vectorization(n=128, d=8, m=16, reps=10)
    assert res["outputs_match"] is True
    assert res["speedup"] > 1.0
