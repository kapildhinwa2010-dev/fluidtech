"""
fluidlayer.losses — training objective for the fluid stack.

  L_total = CE(next-token)
          + w_E  * sheaf energy        (smoothness / consistency on edges)
          + w_D  * band decorrelation  (keep the filter bank from collapsing)
          + w_G  * spectral gap bonus  (anti-oversmoothing, diagnostic only)

The "topological sufficiency" term of the framework spec is implemented as a
*probe* (see `probe.py`) rather than as a gradient, because the routing
decision is a hard top-k selection and carries no gradient path.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cross_entropy", "sheaf_energy_loss", "band_decorrelation", "gap_bonus"]


def cross_entropy(logits, targets) -> float:
    """Mean token cross-entropy from logits (n, V) and targets (n,)."""
    z = logits.data
    z = z - z.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(z).sum(axis=1))
    ll = z[np.arange(len(targets)), targets] - logZ
    return float(-ll.mean())


def cross_entropy_grad(logits, targets) -> np.ndarray:
    """d(mean CE)/d(logits) — (n, V)."""
    z = logits.data
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    g = p.copy()
    g[np.arange(len(targets)), targets] -= 1.0
    return g / len(targets)


def cross_entropy_tensor(logits, targets):
    """Differentiable mean token cross-entropy (returns a Tensor)."""
    n = len(targets)
    m = logits.max(axis=1, keepdims=True)
    z = logits - m
    logZ = z.exp().sum(axis=1, keepdims=True).log() + m
    picked = logits[np.arange(n), np.asarray(targets, dtype=np.int64)]
    return (-(picked - logZ.reshape(-1))).mean()


def sheaf_energy_loss(block) -> float:
    """Total variation of the field over the routed edges (diagnostic scalar)."""
    d = block.diag
    if not d:
        return 0.0
    return float(d.get("sheaf_energy", 0.0))


def band_decorrelation(bands) -> float:
    """1 - mean |cos| between band responses.

    If two bands collapse onto the same spectral mode the filter bank stops
    being a frame and information is lost (the rank-c trap in a different
    disguise).  This penalises that.
    """
    if not bands or len(bands) < 2:
        return 0.0
    M = []
    for b in bands:
        v = b.data.reshape(-1)
        nrm = np.linalg.norm(v)
        M.append(v / nrm if nrm > 1e-12 else v)
    M = np.stack(M)
    C = M @ M.T
    off = C - np.eye(len(M)) * np.diag(C)
    iu = np.triu_indices(len(M), k=1)
    return float(np.mean(np.abs(off[iu])))


def gap_bonus(L) -> float:
    """lambda_2 of a (numpy) Laplacian -- anti-oversmoothing diagnostic."""
    from .diffusion import spectral_gap
    try:
        return float(spectral_gap(L))
    except Exception:
        return 0.0
