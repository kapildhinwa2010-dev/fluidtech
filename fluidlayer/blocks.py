"""
fluidlayer.blocks — the FluidLayer block.

One block, five stages.  Every stage exists to fix a named weakness of the
naive "replace attention with diffusion" idea:

  ROUTER      top-k content routing  -> semantic edges
  TOPOLOGY    semantic edges + temporal chain  -> the complex
  SHEAF       learned signed restriction maps -> content-adaptive geometry
  FREEZE      multi-scale band filter bank (exp(-t_j L) - exp(-t_{j+1} L))
  READ        channel mixing + ONE nonlinearity + residual + sharp route head

Stage notes
-----------
* The signed restriction maps make the G-balance phase transition learnable:
  a balanced component freezes onto a non-trivial sign pattern, an unbalanced
  one decays to zero.  `tanh` keeps |alpha| <= 1 so L stays bounded and
  gradients cannot explode.
* The band filter bank (not a single exponential) is what prevents the
  "frozen state has rank c" information loss.
* The route head is a hard top-k gather (winner-take-all).  It is the sharp
  retrieval path the smooth fluid cannot provide -- this is what keeps the
  model from being a pure smoother.
"""

from __future__ import annotations

import numpy as np

from .autodiff import Tensor, concat, softmax_rows, softplus
from .diffusion import (
    band_times,
    cheb_coeffs_exp,
    cheb_coeffs_exp_autodiff,
    power_iteration_lmax,
    spectral_bound_autodiff,
)
from .topology import (
    chain_edges,
    connected_components,
    edges_from_topk,
    topo_descriptor,
)

__all__ = ["FluidLayerBlock", "RMSNorm", "softplus"]


class RMSNorm:
    def __init__(self, d: int, seed: int = 0):
        self.g = Tensor(np.ones(d))
        self.eps = 1e-6

    def __call__(self, x: Tensor) -> Tensor:
        ms = (x * x).mean()
        inv = (ms + self.eps).pow(-0.5)
        n = x * inv
        return n * self.g


class FluidLayerBlock:
    def __init__(self, d: int, k_neighbors: int = 8, bands: int = 6, cheb_m: int = 12,
                 t_min: float = 1e-3, t_max: float = 1e3, edge_hidden: int = 16,
                 use_route_head: bool = True, seed: int = 0,
                 alpha_min: float = 0.05, alpha_max: float | None = None):
        rng = np.random.default_rng(seed)
        self.d = d
        self.k = int(k_neighbors)
        self.B = int(bands)
        self.m = int(cheb_m)
        # Bands live in SPECTRAL time units alpha = t*lam_max/2 so that the
        # Chebyshev degree m stays sufficient for any graph scale.
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max if alpha_max is not None else 0.75 * self.m
        self.ts = np.geomspace(t_min, t_max, self.B + 1)   # fallback / reference
        self.use_route_head = use_route_head

        s = d ** -0.5
        self.Wq = Tensor(rng.normal(0, s, (d, d)))
        self.Wk = Tensor(rng.normal(0, s, (d, d)))
        self.Wv = Tensor(rng.normal(0, s, (d, d)))
        self.Wo = Tensor(rng.normal(0, s, (d, d)))
        self.Wmix = Tensor(rng.normal(0, s, (d, d)))
        self.Wr = Tensor(rng.normal(0, s, (d, d))) if use_route_head else None
        self.edge1 = Tensor(rng.normal(0, (2 * d) ** -0.5, (2 * d, edge_hidden)))
        self.edge2 = Tensor(rng.normal(0, edge_hidden ** -0.5, (edge_hidden, 1)))
        self.beta_chain = Tensor(np.array([[0.3]]))       # order / temporal edges
        # Band gates start NEAR ZERO (softplus(-3) ~ 0.05): the transient is
        # much larger in magnitude than the harmonic component, so starting
        # with the bands "open" would swamp the frozen state.
        self.band_g = Tensor(np.full((self.B, 1), -3.0))    # band gates (softplus)
        self.harm_g = Tensor(np.array([[1.0]]))            # gate on the HARMONIC branch
        self.norm = RMSNorm(d)
        self.diag = {}

    # ------------------------------------------------------------- helpers
    def _laplacian(self, n: int, edges: np.ndarray, alpha: Tensor) -> Tensor:
        """Differentiable signed sheaf Laplacian  L = D_abs - A."""
        m = edges.shape[0]
        if m == 0:
            return Tensor(np.zeros((n, n)))
        idx_uv = (edges[:, 0].astype(np.int64) * n + edges[:, 1].astype(np.int64))
        idx_vu = (edges[:, 1].astype(np.int64) * n + edges[:, 0].astype(np.int64))
        Z = Tensor(np.zeros(n * n))
        # two SEPARATE scatter bases, then added: index_add starts from zeros,
        # so chaining it would silently discard the first contribution.
        A = (Z.index_add(idx_uv, alpha) + Z.index_add(idx_vu, alpha)).reshape(n, n)
        deg = A.abs().sum(axis=1)                          # (n,)
        diag_idx = np.arange(n) * n + np.arange(n)
        D = Tensor(np.zeros(n * n)).index_add(diag_idx, deg).reshape(n, n)
        return D - A

    def _harmonic(self, x: Tensor, edges: np.ndarray):
        """Exact Hodge projection of the graph Laplacian: the PER-COMPONENT MEAN.

        ker(L) for a graph Laplacian is the space of per-component constants,
        so the frozen state of the diffusion is the component mean.  Cost
        O(n + m): union-find for the components, then a scatter-add and a
        gather.  This is the branch that lets the block represent a GLOBAL
        aggregate -- the band bank alone only carries the transient.
        """
        n, d = x.shape[0], x.shape[1]
        labels, ncomp = connected_components(n, edges)
        if ncomp <= 0:
            return Tensor(np.zeros((n, d)))
        sums = Tensor(np.zeros((ncomp, d))).index_add(labels, x)
        ones = Tensor(np.ones(n))
        counts = Tensor(np.zeros(ncomp)).index_add(labels, ones).reshape(ncomp, 1)
        comp_mean = sums / counts
        return comp_mean.gather(labels)

    def _bands(self, L: Tensor, lam_max: float, X: Tensor, lam_max_t=None):
        """Chebyshev band filter bank; the basis T_k(Ltilde)X is computed ONCE."""
        n = L.shape[0]
        inv = (2.0 / lam_max) if lam_max_t is None else (2.0 / lam_max_t)
        Y = [X]
        if self.m >= 1:
            Lt = inv * L - Tensor(np.eye(n))
            Y.append(Lt @ X)
            for k in range(2, self.m + 1):
                Y.append(2.0 * (Lt @ Y[k - 1]) - Y[k - 2])
        ts = band_times(lam_max, self.m, self.B, self.alpha_min, self.alpha_max)
        out = []
        for j in range(self.B):
            if lam_max_t is not None:
                a = cheb_coeffs_exp_autodiff(ts[j], lam_max_t, self.m)
                b = cheb_coeffs_exp_autodiff(ts[j + 1], lam_max_t, self.m)
                coef = a - b
                acc = None
                for k in range(self.m + 1):
                    if abs(float(coef.data[k])) > 0.0:
                        term = Y[k] * coef[k]
                        acc = term if acc is None else acc + term
            else:
                a = cheb_coeffs_exp(ts[j], lam_max, self.m)
                b = cheb_coeffs_exp(ts[j + 1], lam_max, self.m)
                coef = a - b
                acc = None
                for k in range(self.m + 1):
                    if coef[k] != 0.0:
                        term = Y[k] * float(coef[k])
                        acc = term if acc is None else acc + term
            out.append(acc)
        self.last_ts = ts
        return out

    # -------------------------------------------------------------- forward
    def __call__(self, x: Tensor):
        n = x.shape[0]
        d = self.d
        xn = self.norm(x)

        # ---- ROUTER --------------------------------------------------------
        q = xn @ self.Wq
        k = xn @ self.Wk
        scores = (q @ k.T) * (d ** -0.5)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        sc_masked = scores.data - 1e9 * causal
        idx = np.argsort(-sc_masked, axis=1)[:, : self.k]   # (n, k) neighbours
        row = np.arange(n)[:, None]
        sel = scores[row, idx]                                # Tensor (n, k)

        # ---- TOPOLOGY: semantic edges + temporal chain ----------------------
        sem = edges_from_topk(idx)
        chain = chain_edges(n)

        # ---- SHEAF: content-adaptive signed restriction maps -----------------
        if sem.shape[0] > 0:
            eu = xn[sem[:, 0]]
            ev = xn[sem[:, 1]]
            feat = concat([eu, ev], axis=1)
            h = (feat @ self.edge1).tanh()
            alpha = (h @ self.edge2).reshape(-1)              # (m,) in (-1, 1)
        else:
            alpha = Tensor(np.zeros(0))
        L_sem = self._laplacian(n, sem, alpha)

        if chain.shape[0] > 0:
            beta = softplus(self.beta_chain)                   # (1,1) > 0
            a_ch = beta.reshape(1) * Tensor(np.ones(chain.shape[0]))
            L_chain = self._laplacian(n, chain, a_ch)
        else:
            L_chain = Tensor(np.zeros((n, n)))
        L = L_sem + L_chain

        # Spectral scale.  We use a GUARANTEED UPPER BOUND (Gershgorin +
        # smooth max) for the Chebyshev normalisation: an estimate that
        # undershoots lambda_max makes the recurrence explode.  The bound is
        # differentiable, so no gradient is lost.
        lam_bound_t = spectral_bound_autodiff(L)
        lam_est = power_iteration_lmax(L.data, iters=40)
        lam_max_t = lam_bound_t
        lam_max = float(lam_max_t.data)
        if lam_max < 1e-8:
            lam_max = 1e-8
            lam_max_t = Tensor(np.array([[lam_max]]))

        # ---- FREEZE: multi-scale band filter bank --------------------------
        bands = self._bands(L, lam_max, xn, lam_max_t)
        gates = softplus(self.band_g).reshape(self.B)
        z = None
        for j in range(self.B):
            term = bands[j] * gates[j]
            z = term if z is None else z + term

        # ---- HARMONIC BRANCH: the frozen state (exact Hodge projection) -----
        # For a graph Laplacian, ker(L) = per-component constants, so the
        # Hodge limit of the diffusion is the PER-COMPONENT MEAN.  The band
        # bank only carries the transient, so without this branch the block
        # cannot represent a global aggregate at all (measured: sign acc =
        # chance on the global-mean task).  Cost: O(n + m) via union-find.
        all_edges = (
            np.concatenate([sem, chain], axis=0)
            if sem.shape[0] or chain.shape[0]
            else np.zeros((0, 2), dtype=np.int64)
        )
        harmonic = self._harmonic(xn, all_edges)
        _, diag_ncomp = connected_components(n, all_edges)
        z = z + self.harm_g * harmonic

        # ---- READ: mix channels, one nonlinearity, residual -----------------
        z = z @ self.Wmix
        z = z.gelu()
        out = x + z @ self.Wo

        # ---- sharp retrieval path (winner-take-all over the routed edges) ---
        if self.use_route_head:
            v = xn @ self.Wv
            vg = v.gather_rows(idx)                            # (n, k, d)
            w = softmax_rows(sel)                              # (n, k)
            sr = (vg * w[:, :, None]).sum(axis=1)              # (n, d)
            out = out + sr @ self.Wr

        # ---- diagnostics ---------------------------------------------------
        try:
            desc = topo_descriptor(n, sem, np.abs(alpha.data) + 1e-9)
        except Exception:
            desc = np.zeros(9)
        self.diag = {
            "n_edges": int(sem.shape[0]),
            "n_components": int(diag_ncomp),
            "lam_max": float(lam_max),
            "lam_est": float(lam_est),
            "alpha_abs_mean": (
                float(np.abs(alpha.data).mean()) if alpha.data.size else 0.0
            ),
            "alpha_neg_frac": (
                float((alpha.data < 0).mean()) if alpha.data.size else 0.0
            ),
            "band_energy": np.array(
                [float((band.data**2).sum()) for band in bands]
            ),
            "topo": desc,
            "balance_res": float(np.max(np.abs(L.data.sum(axis=1)))),
        }
        return out
