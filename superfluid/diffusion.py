"""
superfluid.diffusion — the "freezing" operator.

Everything needed to run the fluid in a forward pass, in closed form:

  * power iteration for lambda_max
  * Chebyshev expansion of exp(-t L)   (DCT-I interpolation, numerically stable)
  * a multi-scale BAND filter bank      (the fix for the rank-c trap)
  * spectral gap lambda_2               (the "critical temperature")
  * predicted jamming time t_jam        (= log(1/eps) / lambda_2)

Cost model (per layer, per token):
  * m Laplacian matvecs for ALL bands combined (the Chebyshev basis is shared)
  * B x (m+1) coefficient combinations afterwards (free)
  * so the fluid costs m matvecs, NOT m*B matvecs.
"""

from __future__ import annotations
import math
import numpy as np

__all__ = [
    "power_iteration_lmax", "power_iteration_lmax_autodiff", "cheb_coeffs_exp",
    "cheb_basis", "apply_bands", "band_energies", "spectral_gap", "jam_time",
    "cheb_degree", "exact_exp_action", "spectral_summary", "band_times",
                        "cheb_coeffs_exp_autodiff", "spectral_bound_autodiff",
]


def power_iteration_lmax_autodiff(L, iters: int = 15, seed: int = 0):
    """Differentiable lambda_max estimate (power iteration, no stop-gradient).

    The spectral scale enters the Chebyshev recurrence as 1/lambda_max, so
    treating it as a constant silently drops a large part of the gradient
    w.r.t. the restriction maps.  Keeping it in the graph costs a handful of
    extra mat-vecs and makes the gradient exact.
    """
    from .autodiff import Tensor
    n = L.shape[0]
    rng = np.random.default_rng(seed)
    v = Tensor(rng.standard_normal(n))
    for _ in range(iters):
        w = L @ v
        nrm = (w * w).sum().sqrt()
        if float(nrm.data) < 1e-12:
            break
        v = w / nrm
    lam = (v * (L @ v)).sum()          # Rayleigh quotient v^T L v (1-D safe)
    return lam


# --------------------------------------------------------------------- norms
def power_iteration_lmax(L: np.ndarray, iters: int = 60, tol: float = 1e-10) -> float:
    """Largest eigenvalue of a symmetric PSD matrix (power iteration)."""
    n = L.shape[0]
    if n == 0:
        return 0.0
    v = np.random.default_rng(0).standard_normal(n)
    v /= np.linalg.norm(v) + 1e-16
    lam = 0.0
    for _ in range(iters):
        w = L @ v
        nw = np.linalg.norm(w)
        if nw < 1e-16:
            return 0.0
        v_new = w / nw
        lam_new = float(v_new @ (L @ v_new))
        if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
            lam = lam_new
            v = v_new
            break
        lam, v = lam_new, v_new
    return max(lam, 0.0)


def spectral_summary(L: np.ndarray) -> dict:
    """Exact spectrum for small matrices; the key numbers of the fluid."""
    n = L.shape[0]
    if n <= 512:
        w = np.linalg.eigvalsh(L)
        w = np.clip(w, 0.0, None)
        nz = w[w > 1e-10]
        lam2 = float(nz[0]) if nz.size else 0.0
        return {
            "lam_max": float(w.max()) if w.size else 0.0,
            "lam_min": float(w.min()) if w.size else 0.0,
            "lam2": lam2,
            "kernel_dim": int(n - nz.size),
            "eigs": w,
        }
    lam_max = power_iteration_lmax(L)
    return {"lam_max": lam_max, "lam_min": 0.0, "lam2": float("nan"),
            "kernel_dim": -1, "eigs": None}


# --------------------------------------------------------------- chebyshev
def cheb_coeffs_exp(t: float, lam_max: float, m: int) -> np.ndarray:
    """Coefficients a_k with   exp(-t L) ~= sum_k a_k T_k(Ltilde),
    where Ltilde = 2L/lam_max - I  (spectrum in [-1, 1]).

    Method: Chebyshev interpolation of g(x) = exp(-alpha x) on [-1,1],
    alpha = t*lam_max/2, at the Chebyshev-Gauss nodes, then scaled by exp(-alpha).
    """
    if m < 0:
        raise ValueError("m must be >= 0")
    if lam_max <= 0.0:
        a = np.zeros(m + 1)
        a[0] = 1.0
        return a
    alpha = 0.5 * t * lam_max
    if alpha > 500.0:
        # exp(+alpha) would overflow; beyond this the truncated expansion is
        # meaningless anyway -- see band_times() for the safe regime.
        raise ValueError(
            f"Chebyshev expansion of exp(-tL) unstable for alpha=t*lam_max/2={alpha:.1f} "
            f"(>500). Use band_times() to place bands in spectral time units.")
    j = np.arange(m + 1)
    x = np.cos(np.pi * (j + 0.5) / (m + 1))
    g = np.exp(-alpha * x)
    k = np.arange(m + 1)
    Cm = np.cos(np.pi * np.outer(j + 0.5, k) / (m + 1))       # (m+1, m+1)
    c = (2.0 / (m + 1)) * (g @ Cm)                            # DCT-I coeffs
    a = np.exp(-alpha) * c
    a[0] *= 0.5                                                # c_0 / 2 convention
    return a


def spectral_bound_autodiff(L, tau: float = 0.5):
    """Differentiable UPPER BOUND on lambda_max (Gershgorin + smooth max).

    The Chebyshev recurrence T_k(2L/l - I) is only stable when l >= lambda_max.
    A power-iteration ESTIMATE can undershoot, and then the recurrence
    explodes geometrically (measured: band energies of 1e15).  So we normalise
    with a guaranteed upper bound instead:

        lambda_max <= max_i ( L_ii + sum_{j != i} |L_ij| ) = max_i rowsum|L|

    The max is replaced by a log-sum-exp (smooth, differentiable, and still an
    upper bound), so the whole thing stays in the autodiff graph.
    """
    row_abs = L.abs().sum(axis=1)                 # (n,) differentiable
    e = (row_abs / tau).exp()
    lse = tau * e.sum().log()
    return 2.0 * lse


def cheb_coeffs_exp_autodiff(t: float, lam_max_t, m: int):
    """Differentiable version of cheb_coeffs_exp.

    The Chebyshev coefficients depend on the spectral scale alpha = t*lam_max/2,
    so treating lam_max as a constant inside the coefficient computation drops
    a real part of the gradient w.r.t. the restriction maps.  The whole
    expansion is a matrix multiply, so it can be kept in the autodiff graph.
    """
    from .autodiff import Tensor
    j = np.arange(m + 1)
    k = np.arange(m + 1)
    x = np.cos(np.pi * (j + 0.5) / (m + 1))
    Cm = np.cos(np.pi * np.outer(j + 0.5, k) / (m + 1))
    Cm[0, :] *= 0.5                       # the c_0 / 2 convention
    alpha = 0.5 * t * lam_max_t           # Tensor scalar
    g = (-alpha * Tensor(x)).exp()        # g(x) = exp(-alpha x), differentiable
    a = (-alpha).exp() * ((2.0 / (m + 1)) * (g @ Tensor(Cm)))
    return a


def cheb_basis(L: np.ndarray, lam_max: float, X: np.ndarray, m: int):
    """Y[k] = T_k(Ltilde) @ X for k = 0..m.  (m matvecs total, shared by all bands)

    Ltilde = 2L/lam_max - I  so that sigma(Ltilde) subset [-1, 1].
    Chebyshev recurrence:  T_0 = I,  T_1 = Ltilde,  T_k = 2*Ltilde*T_{k-1} - T_{k-2}.
    """
    if lam_max <= 0.0:
        return [X.copy()]
    Lt = (2.0 / lam_max) * L - np.eye(L.shape[0])
    Y = [X.copy()]
    if m >= 1:
        Y.append(Lt @ X)
    for k in range(2, m + 1):
        Y.append(2.0 * (Lt @ Y[k - 1]) - Y[k - 2])
    return Y


def band_times(lam_max: float, m: int, n_bands: int, alpha_min: float = 0.05,
               alpha_max: float | None = None):
    """Spectrum-adaptive band times.

    The Chebyshev expansion of exp(-tL) needs alpha = t*lam_max/2 <= m, otherwise
    the truncation error blows up (this is the "conditioning cliff").  So the
    bands are placed in *spectral* time units alpha_j, and converted to t at run
    time.  The band range then always spans the whole spectrum of L, whatever
    its scale, and the Chebyshev degree stays sufficient by construction.

    t_j = 2*alpha_j / lam_max
    """
    if alpha_max is None:
        alpha_max = float(m) * 0.75
    a = np.geomspace(alpha_min, max(alpha_max, alpha_min * 2), n_bands + 1)
    if lam_max <= 0:
        return a
    return 2.0 * a / lam_max


def apply_bands(L: np.ndarray, lam_max: float, X: np.ndarray,
                ts: np.ndarray, m: int) -> np.ndarray:
    """Multi-scale band filter bank.

    band_j = exp(-t_j L) - exp(-t_{j+1} L),  t = [t_0, ..., t_B] (B+1 times)

    Returns array of shape (B, n, d).  Cost: m Laplacian matvecs total
    (the Chebyshev basis T_k(Ltilde) X is computed once and reused by every band).
    """
    ts = np.asarray(ts, dtype=np.float64)
    B = ts.shape[0] - 1
    if B <= 0:
        raise ValueError("need at least two time constants")
    Y = cheb_basis(L, lam_max, X, m)
    out = np.zeros((B,) + X.shape)
    for j in range(B):
        a = cheb_coeffs_exp(ts[j], lam_max, m)
        b = cheb_coeffs_exp(ts[j + 1], lam_max, m)
        coef = a - b
        acc = np.zeros_like(X)
        for k in range(m + 1):
            if coef[k] != 0.0:
                acc += coef[k] * Y[k]
        out[j] = acc
    return out


def band_energies(bands: np.ndarray) -> np.ndarray:
    """Relative energy carried by each band: ||band_j||^2 / sum_j ||band_j||^2."""
    e = (bands ** 2).sum(axis=(1, 2))
    s = e.sum()
    return e / s if s > 0 else e


# ------------------------------------------------------------------- theory
def spectral_gap(L: np.ndarray) -> float:
    """Smallest strictly positive eigenvalue (the spectral gap lambda_2)."""
    s = spectral_summary(L)
    return s["lam2"]


def jam_time(lam2: float, eps: float = 1e-6) -> float:
    """Predicted freezing time:  t_jam = log(1/eps) / lambda_2."""
    if lam2 <= 0:
        return math.inf
    return math.log(1.0 / eps) / lam2


def cheb_degree(kappa: float, eps: float = 1e-6) -> int:
    """Chebyshev degree needed for relative accuracy eps at condition number kappa."""
    return int(math.ceil(math.sqrt(max(kappa, 1.0)) * math.log(2.0 / eps)))


def exact_exp_action(L: np.ndarray, t: float, X: np.ndarray) -> np.ndarray:
    """Ground truth exp(-tL) X via eigendecomposition (for verification only)."""
    w, V = np.linalg.eigh(L)
    return V @ (np.exp(-t * w)[:, None] * (V.T @ X))
