#!/usr/bin/env python3
"""INT8 quantization for fluidlayer states and embeddings.

Feature 1 of the "Coming soon" list: quantize float32 tensors
(fluid state, embeddings, fused hidden states) to int8 -- 4x
smaller -- with a BOUNDED round-trip error:
max |x - dequant(quant(x))| <= scale = max|x| / 127,
i.e. under 0.4% relative error (measured in e21 + unit tests).
"""
from __future__ import annotations

import numpy as np

__all__ = ["quantize_int8", "dequantize_int8", "quantization_report"]


def quantize_int8(arr):
    """Symmetric int8 quantization. Returns (int8 array, scale)."""
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return np.zeros(x.shape, dtype=np.int8), 1.0
    peak = float(np.max(np.abs(x)))
    scale = peak / 127.0 if peak > 0.0 else 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    """Dequantize int8 back to float32."""
    return (np.asarray(q, dtype=np.float32) * float(scale)).astype(np.float32)


def quantization_report(arr):
    """Measured summary for one tensor: bytes + round-trip error."""
    x = np.asarray(arr, dtype=np.float32)
    q, scale = quantize_int8(x)
    back = dequantize_int8(q, scale)
    max_err = float(np.max(np.abs(x - back))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return {
        "float32_bytes": int(x.nbytes),
        "int8_bytes": int(q.nbytes),
        "reduction_x": round(x.nbytes / max(q.nbytes, 1), 2),
        "scale": scale,
        "max_abs_error": max_err,
        "relative_error_pct": (
            round(100.0 * max_err / peak, 4) if peak > 0 else 0.0
        ),
    }
