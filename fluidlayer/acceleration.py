#!/usr/bin/env python3
"""Hardware acceleration: framework detection + vectorized fast path.

Feature 3 of the "Coming soon" list. Two parts:
1. detect_acceleration(): what is available on THIS machine
   (torch, CUDA + device count, MPS, numba, cupy) -- pick a backend
   accordingly;
2. a VECTORIZED fast path for the fluid routing scores, benchmarked
   against the naive per-row path (measured in e21). The vectorized
   path is the CPU-accelerated default; the same function shape is
   what a CUDA/cupy backend accelerates on GPU machines.
"""
from __future__ import annotations

import time

import numpy as np

__all__ = [
    "detect_acceleration",
    "loop_scores",
    "vectorized_scores",
    "benchmark_vectorization",
]


def detect_acceleration():
    """Available acceleration frameworks on this machine (honest)."""
    info = {
        "torch": None,
        "cuda": False,
        "cuda_devices": 0,
        "mps": False,
        "numba": False,
        "cupy": False,
        "vectorized_numpy": True,
    }
    try:
        import torch

        info["torch"] = str(torch.__version__)
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["cuda_devices"] = int(torch.cuda.device_count())
        try:
            info["mps"] = bool(torch.backends.mps.is_available())
        except Exception:
            info["mps"] = False
    except Exception:
        pass
    for mod in ("numba", "cupy"):
        try:
            __import__(mod)
            info[mod] = True
        except Exception:
            info[mod] = False
    return info


def loop_scores(content, Wq, Wk, protos):
    """Naive per-row routing scores (the pre-acceleration path)."""
    out = np.zeros((content.shape[0], protos.shape[0]), dtype=np.float64)
    inv = content.shape[1] ** 0.5
    kk = protos @ Wk
    for t in range(content.shape[0]):
        q = content[t] @ Wq
        out[t] = (q @ kk.T) / inv
    return out


def vectorized_scores(content, Wq, Wk, protos):
    """Vectorized routing scores -- the accelerated fast path."""
    inv = content.shape[1] ** 0.5
    q = content @ Wq
    kk = protos @ Wk
    return (q @ kk.T) / inv


def _time_best(fn, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def benchmark_vectorization(n=256, d=16, m=32, reps=20, seed=0):
    """Measured loop-vs-vectorized benchmark on THIS machine."""
    rng = np.random.default_rng(seed)
    content = rng.normal(size=(n, d))
    Wq = rng.normal(size=(d, d))
    Wk = rng.normal(size=(d, d))
    protos = rng.normal(size=(m, d))
    loop_s = _time_best(
        lambda: loop_scores(content, Wq, Wk, protos), reps)
    vec_s = _time_best(
        lambda: vectorized_scores(content, Wq, Wk, protos), reps)
    match = bool(np.allclose(
        loop_scores(content, Wq, Wk, protos),
        vectorized_scores(content, Wq, Wk, protos),
        rtol=1e-9, atol=1e-9,
    ))
    return {
        "rows": n,
        "dim": d,
        "protos": m,
        "reps": reps,
        "loop_s": round(loop_s, 6),
        "vectorized_s": round(vec_s, 6),
        "speedup": round(loop_s / max(vec_s, 1e-12), 2),
        "outputs_match": match,
    }
