"""
superfluid.sheaf — Cellular sheaf Laplacians on graphs.

The mathematical object that makes "Superfluid" real.

A cellular sheaf F on a graph G assigns:
  * a stalk (vector space) F(v) ~ R^d  to every vertex v
  * a restriction map  F_{v<-e} : F(v) -> F(e)  to every incidence (v, e)

Its coboundary  delta : C^0(G;F) -> C^1(G;F)  induces the **sheaf Laplacian**

    L_F = delta* delta = delta^T delta      (PSD by construction)

and the diffusion (the "fluid") is the heat flow on that operator:

    dX/dt = -L_F X      =>      X(t) = exp(-t L_F) X(0)

Two facts we rely on throughout (Hansen & Ghrist 2019; Bodnar et al. 2022):

1.  lim_{t->inf} exp(-t L_F) = P_{ker(L_F)}      (the Hodge limit: the fluid
    freezes onto the space of *global sections* -- consistent assignments).
2.  On ker(L_F)^perp the flow contracts at rate exp(-lam_2 t).

Implementation note: we use the **signed** sheaf Laplacian with absolute degree

    L = D_abs - A ,   D_abs[i,i] = sum_j |A[i,j]| ,   A[i,j] = alpha_ij

which is PSD for any sign pattern, reduces to the ordinary graph Laplacian
when all signs are positive, and reproduces the G-balance phase transition:
a *balanced* signed component freezes onto a NON-TRIVIAL sign pattern,
an *unbalanced* one decays to zero (it "evaporates" instead of freezing).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_laplacian", "sheaf_energy", "balance_residual", "is_balanced",
    "laplacian_from_dense_A", "signed_laplacian_from_edges",
]


def build_laplacian(n: int, edges: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Signed sheaf Laplacian L = D_abs - A.

    edges : (m, 2) int array of undirected edges (u, v), u != v
    alpha : (m,) float array of signed edge weights (the restriction maps)
    Returns L : (n, n) symmetric PSD.
    """
    L = np.zeros((n, n), dtype=np.float64)
    if edges.shape[0] == 0:
        return L
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    w = np.asarray(alpha, dtype=np.float64)

    # A contribution (symmetric)
    np.add.at(L, (u, v), w)
    np.add.at(L, (v, u), w)
    # D_abs: sum of |A[i,j]| over j
    absA = np.zeros_like(L)
    np.add.at(absA, (u, v), np.abs(w))
    np.add.at(absA, (v, u), np.abs(w))
    diag = absA.sum(axis=1)
    L = np.diag(diag) - L          # L = D_abs - A   (PSD)
    return L


def laplacian_from_dense_A(A: np.ndarray) -> np.ndarray:
    """L = D_abs - A for a dense symmetric A."""
    A = np.asarray(A, dtype=np.float64)
    A = 0.5 * (A + A.T)
    D = np.diag(np.abs(A).sum(axis=1))
    return D - A


def signed_laplacian_from_edges(
    n: int, edges: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Alias kept for readability at call sites."""
    return build_laplacian(n, edges, alpha)


def sheaf_energy(x: np.ndarray, edges: np.ndarray, alpha: np.ndarray) -> float:
    """Total variation of the field across the sheaf edges.

    E = sum_e  alpha_e^2 * (x_u - x_v)^2      (scalar stalk; x is (n,) or (n,d))

    This is the quantity the "smoothness" regularizer anneals toward, and it
    is exactly x^T L x / 2 for the signed Laplacian up to the sign convention.
    """
    if edges.shape[0] == 0:
        return 0.0
    u = edges[:, 0]
    v = edges[:, 1]
    w = np.asarray(alpha, dtype=np.float64)
    diff = x[u] - x[v]
    if diff.ndim == 1:
        diff = diff[:, None]
    return float(np.sum((w[:, None] ** 2) * (diff ** 2)))


def balance_residual(L: np.ndarray) -> float:
    """||L 1||_inf for the constant vector 1.

    Zero  <=>  the sheaf is (G-)balanced: every node's incident restriction
    maps sum to zero, so the constant field is a global section and the fluid
    has a non-trivial frozen state.
    """
    n = L.shape[0]
    if n == 0:
        return 0.0
    return float(np.max(np.abs(L @ np.ones(n))))


def is_balanced(L: np.ndarray, tol: float = 1e-8) -> bool:
    return balance_residual(L) < tol


def component_balance(L: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Per-component balance residual.

    For each connected component (of the underlying unsigned graph) returns
    ||L 1_C||_inf.  Zero means that component is balanced (freezes onto a
    sign pattern); non-zero means it is unbalanced (decays to zero).
    """
    n = L.shape[0]
    seen = np.zeros(n, dtype=bool)
    out = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            nb = np.nonzero(adj[u])[0]
            for w in nb:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        idx = np.array(comp)
        res = L[np.ix_(idx, idx)].sum(axis=1)
        out.append(float(np.max(np.abs(res))))
    return np.array(out)
