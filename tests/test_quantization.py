"""Feature 1 -- quantization: unit tests (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluidlayer.quantization import (  # noqa: E402
    dequantize_int8,
    quantization_report,
    quantize_int8,
)


def test_roundtrip_error_bounded():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 8)).astype(np.float32)
    q, scale = quantize_int8(x)
    assert q.dtype == np.int8
    back = dequantize_int8(q, scale)
    assert np.allclose(x, back, atol=scale)


def test_memory_reduction_4x():
    x = np.zeros((1000, 100), dtype=np.float32)
    q, _scale = quantize_int8(x)
    assert q.nbytes * 4 == x.nbytes


def test_zero_array_stable():
    q, scale = quantize_int8(np.zeros((4, 4), dtype=np.float32))
    assert scale == 1.0
    assert float(np.max(np.abs(dequantize_int8(q, scale)))) == 0.0


def test_report_reasonable():
    rng = np.random.default_rng(1)
    rep = quantization_report(rng.normal(size=(500, 64)))
    assert rep["reduction_x"] == 4.0
    assert rep["relative_error_pct"] < 1.0
