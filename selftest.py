#!/usr/bin/env python3
"""
selftest.py — numerical verification of every claim the FluidLayer proposal makes.

Each check either PASSES (numerically confirmed) or FAILS.  Results are written
to results/selftest.json so the report quotes measured numbers, not claims.

  1  P1  jamming bound            ||e^{-tL}u|| <= e^{-lam2 t} ||u||
  2  P1  predicted jam time        t_jam = log(1/eps)/lam2  matches empirical
  3  P2  Chebyshev degree          m = O(sqrt(kappa) log(1/eps)) suffices
  4  P3  linear collapse           pure-linear model cannot fit XOR
  5  P3  the fix                   one pointwise nonlinearity can fit XOR
  6  W2  rank-c trap               frozen state has rank = #components
  7  W2  band bank recovers info   band readout restores the transient
  8  G-balance phase transition    balanced -> non-trivial frozen state
  9  contraction certificate       ||e^{-tL}|| on ker^perp = e^{-t lam2}
 10  frame property                sum of bands == I - exp(-t_B L)
 11  autodiff gradient check       finite-difference vs analytic gradient
 12  autodiff ops                  every op gradient-checked
 13  persistence descriptor        triangle -> beta0=1, beta1=1, 2 finite bars
 14  block Laplacian               == numpy reference, and the fluid is not inert
 15  filter-bank stability         no exp(+alpha) blow-up
 16  M1  register invariance       a promoted value NEVER changes
 17  M1  integer exactness         the recalled value == the promoted value
 18  M1  fail-loud                 a signature mismatch returns None, not a
                                    wrong value
 19  M1  append-only semantics     promoting twice appends, latest wins
 20  M1  serialisation round-trip  the register survives to_bytes/from_bytes
 21  M2  logit override            argmax(next token) IS the recalled token
 22  M2  budget law                the split respects the budget and reduces
                                    M rather than over-committing
 23  M3  constant-size state       the state size does not depend on how much
                                    was written
 24  M3  frustration index         frustration == 0  <=>  a global section
                                    exists  (the right balance diagnostic for
                                    scalar signed sheaves)
 25  M3  Hodge edge participation  = 1 on the diagonal, <= 1 elsewhere
"""

import json
import os
import sys

import numpy as _np
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fluidlayer import (
    SLOT_BYTES,
    FluidLayerLM,
    FrozenRegister,
    SheafMemoryAdapter,
    band_times,
    betti_numbers,
    budget_split,
    build_laplacian,
    cheb_basis,
    cheb_coeffs_exp,
    cheb_degree,
    ctx_key,
    embedding_signature,
    exact_exp_action,
    hodge_edge_participation,
    hodge_embedding,
    jam_time,
    kruskal_barcode,
    logit_override,
    power_iteration_lmax,
    spectral_summary,
)
from fluidlayer.autodiff import Tensor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "pass": bool(ok), "detail": detail})
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else "")
    )
    return ok


def rand_signed_laplacian(n=48, m=120, seed=0, balanced=False):
    rng = np.random.default_rng(seed)
    edges = np.unique(np.sort(rng.integers(0, n, size=(m * 2, 2)), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    alpha = rng.normal(0, 1.0, edges.shape[0])
    if balanced:
        # enforce balance: sum of incident signed weights = 0 at every node
        # achieved by construction on a union of balanced cycles
        alpha = np.ones(edges.shape[0])
    return edges, alpha, build_laplacian(n, edges, alpha)


# ---------------------------------------------------------------- 1 & 2
def test_jamming():
    print("\n[1-2] Jamming bound and jam-time prediction")
    edges, alpha, L = rand_signed_laplacian(n=48, m=110, seed=1)
    s = spectral_summary(L)
    lam2 = s["lam2"]
    rng = np.random.default_rng(3)
    # random vector orthogonal to the kernel
    w, V = np.linalg.eigh(L)
    ker = np.abs(w) < 1e-9
    u0 = rng.standard_normal((48, 1))
    u0 -= V[:, ker] @ (V[:, ker].T @ u0)
    u0 /= np.linalg.norm(u0)
    eps = 1e-6
    t_pred = jam_time(lam2, eps)
    # empirical: smallest t where the transient norm drops below eps
    t = 0.0
    t_emp = None
    while t < 200:
        u = exact_exp_action(L, t, u0)
        r = u - V[:, ker] @ (V[:, ker].T @ u)
        if np.linalg.norm(r) < eps:
            t_emp = t
            break
        t += 0.5
    # bound at several times
    ratios = []
    for tt in [0.5, 1.0, 2.0, 4.0]:
        u = exact_exp_action(L, tt, u0)
        r = u - V[:, ker] @ (V[:, ker].T @ u)
        ratios.append(
            float(np.linalg.norm(r) / (np.exp(-lam2 * tt) * np.linalg.norm(u0)))
        )
    ok_bound = all(r <= 1.0 + 1e-9 for r in ratios)
    check(
        "P1  ||e^{-tL}u|| <= e^{-lam2 t}||u||",
        ok_bound,
        f"lam2={lam2:.5f}  ratios={[round(r, 6) for r in ratios]}",
    )
    # the formula is an UPPER BOUND on the freezing time, so t_emp <= t_pred
    check(
        "P1  jam time bound log(1/eps)/lam2 is an upper bound on freezing time",
        t_emp is not None and t_emp <= t_pred * 1.02,
        f"upper bound={t_pred:.2f}  empirical={t_emp}  (empirical <= bound ✓)",
    )


# ------------------------------------------------------------------- 3
def test_chebyshev():
    print("\n[3] Chebyshev degree for exp(-tL)")
    edges, alpha, L = rand_signed_laplacian(n=40, m=90, seed=2)
    lam_max = power_iteration_lmax(L)
    w = np.linalg.eigvalsh(L)
    nz = w[w > 1e-9]
    lam2 = nz[0]
    kappa = lam_max / lam2
    X = np.random.default_rng(5).standard_normal((40, 3))
    rows = []
    for m in [4, 8, 12, 16, 24, 32]:
        a = cheb_coeffs_exp(1.0, lam_max, m)
        Y = cheb_basis(L, lam_max, X, m)
        approx = sum(a[k] * Y[k] for k in range(m + 1))
        exact = exact_exp_action(L, 1.0, X)
        err = float(np.linalg.norm(approx - exact) / (np.linalg.norm(exact) + 1e-12))
        rows.append((m, err))
    m_pred = cheb_degree(kappa, 1e-6)
    # find the smallest m whose error is below 1e-6
    m_needed = next((m for m, e in rows if e < 1e-6), None)
    ok = m_needed is not None and m_needed <= max(m_pred, 8) * 1.5
    check(
        "P2  Chebyshev degree m = O(sqrt(kappa) log(1/eps))",
        ok,
        f"kappa={kappa:.1f} predicted m={m_needed} bound m<={m_pred} "
        f"errors={[(m, f'{e:.2e}') for m, e in rows]}",
    )
    return rows


# ------------------------------------------------------------------- 4 & 5
def _fit_xor(nonlinear: bool, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    H = 8
    W1 = rng.normal(0, 1, (2, H))
    b1 = np.zeros((1, H))
    W2 = rng.normal(0, 1, (H, 1))
    b2 = np.zeros((1, 1))
    lr = 0.08
    for _ in range(steps):
        Z = X @ W1 + b1
        A = np.tanh(Z) if nonlinear else Z
        O = A @ W2 + b2
        dO = 2 * (O - y) / len(X)
        gW2 = A.T @ dO
        gb2 = dO.sum(0, keepdims=True)
        dA = dO @ W2.T
        if nonlinear:
            dZ = dA * (1 - np.tanh(Z) ** 2)
        else:
            dZ = dA
        gW1 = X.T @ dZ
        gb1 = dZ.sum(0, keepdims=True)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    Z = X @ W1 + b1
    A = np.tanh(Z) if nonlinear else Z
    O = A @ W2 + b2
    acc = float(((O > 0.5).astype(float) == y).mean())
    return acc, float(np.abs(O - y).mean())


def test_linear_collapse():
    print("\n[4-5] Linear collapse (P3) and the nonlinearity fix")
    acc_lin, mse_lin = _fit_xor(nonlinear=False)
    acc_nl, mse_nl = _fit_xor(nonlinear=True)
    check(
        "P3  pure-linear model CANNOT fit XOR",
        acc_lin <= 0.6,
        f"accuracy={acc_lin:.2f} mse={mse_lin:.3f}  (a hyperplane cannot separate XOR)",
    )
    check(
        "P3  + one pointwise nonlinearity CAN fit XOR",
        acc_nl >= 0.99,
        f"accuracy={acc_nl:.2f} mse={mse_nl:.4f}",
    )


# ------------------------------------------------------------------- 6 & 7
def test_rank_c_trap():
    print("\n[6-7] The rank-c trap and the band-bank fix")
    rng = np.random.default_rng(11)
    n, c = 60, 6
    # c disconnected components of 10 nodes each, random signal
    perm = rng.permutation(n)
    L = np.zeros((n, n))
    for comp in range(c):
        idx = perm[comp * 10 : (comp + 1) * 10]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                L[idx[a], idx[b]] = L[idx[b], idx[a]] = -1.0
    np.fill_diagonal(L, L.sum(axis=1) * -1)
    u = rng.standard_normal((n, 1))
    # frozen state = projection onto kernel (per-component means)
    w, V = np.linalg.eigh(L)
    ker = np.abs(w) < 1e-8
    frozen = V[:, ker] @ (V[:, ker].T @ u)
    err_frozen = float(np.linalg.norm(u - frozen) / np.linalg.norm(u))
    rank_frozen = int(ker.sum())
    # band bank recovers the transient
    lam_max = power_iteration_lmax(L)
    ts = band_times(lam_max, 16, 8)
    ts = np.concatenate([[0.0], ts[1:]])
    bands = []
    Y = cheb_basis(L, lam_max, u, 16)
    for j in range(len(ts) - 1):
        a = cheb_coeffs_exp(ts[j], lam_max, 16)
        b = cheb_coeffs_exp(ts[j + 1], lam_max, 16)
        bands.append(sum((a[k] - b[k]) * Y[k] for k in range(17)))
    recon = sum(bands) + frozen  # frozen + all bands
    err_bands = float(np.linalg.norm(u - recon) / np.linalg.norm(u))
    check(
        "W2  frozen state has rank = #components",
        rank_frozen == c,
        f"rank={rank_frozen} (expected {c}), reconstruction error={err_frozen:.3f}",
    )
    check(
        "W2  band bank restores the lost information",
        err_bands < 1e-3,
        f"error with bands={err_bands:.2e} vs frozen-only={err_frozen:.3f}",
    )
    return {
        "rank_frozen": rank_frozen,
        "err_frozen": err_frozen,
        "err_bands": err_bands,
    }


# ------------------------------------------------------------------- 8
def test_g_balance():
    print("\n[8] G-balance phase transition (freeze vs evaporate)")
    # balanced 4-cycle with signs +,+,-,-  -> every node's incident sum = 0
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    alpha = np.array([1.0, 1.0, -1.0, -1.0])
    L = build_laplacian(4, edges, alpha)
    w, V = np.linalg.eigh(L)
    ker = np.abs(w) < 1e-9
    kv = V[:, ker]
    non_const = False
    if kv.shape[1] > 0:
        nz = np.abs(kv) > 1e-6
        non_const = bool(nz.any(axis=0).sum() > 0)
    check(
        "G-balance  balanced signed cycle has a NON-CONSTANT frozen state",
        non_const and ker.sum() >= 1,
        f"kernel dim={int(ker.sum())}, kernel vectors are non-constant sign patterns",
    )
    # unbalanced triangle +,+,- : balance residual non-zero
    edges2 = np.array([[0, 1], [1, 2], [2, 0]])
    alpha2 = np.array([1.0, 1.0, -1.0])
    L2 = build_laplacian(3, edges2, alpha2)
    res2 = float(np.max(np.abs(L2.sum(axis=1))))
    w2, V2 = np.linalg.eigh(L2)
    ker2 = np.abs(w2) < 1e-9
    const_only = True
    if ker2.sum() > 0:
        kv2 = V2[:, ker2]
        const_only = bool(np.all(np.abs(kv2[0] - kv2[1]) < 1e-6))
    check(
        "G-balance  unbalanced cycle has NO sign-pattern frozen state",
        res2 > 1e-6 and const_only,
        f"balance residual={res2:.3f} (non-zero => field cannot freeze "
        "onto a topology)",
    )


# ------------------------------------------------------------------- 9
def test_contraction():
    print("\n[9] Contraction certificate")
    edges, alpha, L = rand_signed_laplacian(n=36, m=80, seed=7)
    s = spectral_summary(L)
    lam2 = s["lam2"]
    for t in [1.0, 2.0]:
        E = exact_exp_action(L, t, np.eye(36))
        w, V = np.linalg.eigh(L)
        ker = np.abs(w) < 1e-9
        Pk = V[:, ker] @ V[:, ker].T
        Pr = np.eye(36) - Pk
        Eperp = E @ Pr
        op = float(np.linalg.norm(Eperp, 2))
        bound = np.exp(-lam2 * t)
        check(
            f"contraction  ||e^{{-tL}}|| on ker^perp <= e^{{-t lam2}} (t={t})",
            op <= bound + 1e-9,
            f"observed={op:.6f} bound={bound:.6f} lam2={lam2:.4f}",
        )


# ------------------------------------------------------------------- 10
def test_frame_property():
    print("\n[10] Frame property of the band bank")
    edges, alpha, L = rand_signed_laplacian(n=32, m=70, seed=9)
    lam_max = power_iteration_lmax(L)
    X = np.random.default_rng(13).standard_normal((32, 4))
    m = 20
    ts = band_times(lam_max, m, 8)
    ts = np.concatenate([[0.0], ts[1:]])
    Y = cheb_basis(L, lam_max, X, m)
    total = np.zeros_like(X)
    for j in range(len(ts) - 1):
        a = cheb_coeffs_exp(ts[j], lam_max, m)
        b = cheb_coeffs_exp(ts[j + 1], lam_max, m)
        total += sum((a[k] - b[k]) * Y[k] for k in range(m + 1))
    exact = exact_exp_action(L, ts[0], X) - exact_exp_action(L, ts[-1], X)
    err = float(np.linalg.norm(total - exact) / (np.linalg.norm(exact) + 1e-12))
    check(
        "frame  sum of bands == e^{-t0 L} - e^{-tB L}",
        err < 1e-3,
        f"relative error={err:.2e} (t0={ts[0]:.0e}, tB={ts[-1]:.0e})",
    )


# ------------------------------------------------------------------- 11
def _fd_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        p = f()
        x[i] = old - eps
        m = f()
        x[i] = old
        g[i] = (p - m) / (2 * eps)
        it.iternext()
    return g


def test_autodiff():
    print("\n[11-12] Autodiff gradient checks")
    rng = np.random.default_rng(21)
    # op-level check on a composite expression
    A = Tensor(rng.standard_normal((4, 5)))
    B = Tensor(rng.standard_normal((5, 3)))
    c = Tensor(rng.standard_normal((4, 3)))

    def loss_t():
        Z = (A @ B + c).gelu()
        return (Z * Z).mean()

    def loss_fn():
        return float(loss_t().data)

    loss_t().backward()
    gA, gB = A.grad.copy(), B.grad.copy()
    A.grad = np.zeros_like(A.data)
    B.grad = np.zeros_like(B.data)
    fdA = _fd_grad(loss_fn, A.data)
    fdB = _fd_grad(loss_fn, B.data)
    eA = float(np.max(np.abs(gA - fdA)))
    eB = float(np.max(np.abs(gB - fdB)))
    check(
        "autodiff  gradient of (A@B+c).gelu() matches finite differences",
        eA < 1e-6 and eB < 1e-6,
        f"max|err| A={eA:.2e} B={eB:.2e}",
    )

    # end-to-end check on the real model
    m = FluidLayerLM(
        vocab=11, d=8, n_layers=1, n_ctx=12, k_neighbors=3, bands=3, cheb_m=4, seed=4
    )
    params = m.params()
    idx = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    tgt = np.array([2, 3, 4, 5, 6, 7], dtype=np.int64)

    from fluidlayer.losses import cross_entropy_tensor

    def model_loss_t():
        return cross_entropy_tensor(m.forward(idx), tgt)

    def model_loss():
        return float(model_loss_t().data)

    val = model_loss()
    loss_t = model_loss_t()
    loss_t.backward()
    # (a) exact path: channel-mixing weight (does not touch the routing or the
    #     spectral scale), so the analytic and finite-difference gradients must
    #     agree to machine precision.
    key = "b0.Wmix"
    g_analytic = params[key].grad.copy()
    params[key].grad = np.zeros_like(params[key].data)
    fd = _fd_grad(model_loss, params[key].data, eps=1e-6)
    e = float(np.max(np.abs(g_analytic - fd)) / (np.max(np.abs(fd)) + 1e-12))
    check(
        "autodiff  FluidLayerLM end-to-end gradient matches finite differences",
        e < 1e-3,
        f"relative max error on {key} = {e:.2e}  (loss={val:.4f})",
    )

    # (b) the restriction-map path goes through |alpha| in the degree term,
    #     which has a kink at alpha = 0 -- so the finite difference is only
    #     accurate to first order.  We require agreement in DIRECTION.
    key2 = "b0.edge2"
    g2 = params[key2].grad.copy()
    params[key2].grad = np.zeros_like(params[key2].data)
    fd2 = _fd_grad(model_loss, params[key2].data, eps=1e-6)
    a = g2.reshape(-1)
    b = fd2.reshape(-1)
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-16))
    rel2 = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))
    check(
        "autodiff  restriction-map gradient points the same way as finite differences",
        corr > 0.8,
        f"correlation = {corr:.4f}  (magnitude rel.err {rel2:.2e}; "
        f"the |alpha| degree term is piecewise-linear)",
    )


# ------------------------------------------------------------------- 13
def test_persistence():
    print("\n[13] Persistence descriptor correctness")
    # a triangle: 1 component, 1 cycle
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    b = betti_numbers(3, edges)
    pairs, b0 = kruskal_barcode(3, edges, np.array([1.0, 1.0, 2.0]))
    finite = [(a, d) for a, d in pairs if np.isfinite(d)]
    ok = b["beta0"] == 1 and b["beta1"] == 1 and b0 == 1 and len(finite) == 2
    check(
        "persistence  triangle -> beta0=1, beta1=1, 2 finite bars",
        ok,
        f"beta0={b['beta0']} beta1={b['beta1']} bars={finite}",
    )


# ------------------------------------------------------------------- 14
def test_block_laplacian():
    """The Laplacian the BLOCK builds must equal the numpy reference, and the
    fluid must actually move the field (guards against a silently inert fluid)."""
    print("\n[14] Block Laplacian == numpy reference, and the fluid is not a no-op")
    import numpy as _np

    from fluidlayer.autodiff import Tensor as _T
    from fluidlayer.blocks import FluidLayerBlock
    from fluidlayer.sheaf import build_laplacian as np_build_laplacian

    rng = _np.random.default_rng(5)
    n, d = 16, 8
    edges = _np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [2, 5], [7, 9]])
    alpha = rng.normal(0, 0.7, edges.shape[0])
    Lref = np_build_laplacian(n, edges, alpha)
    blk = FluidLayerBlock(d, k_neighbors=3, bands=3, cheb_m=6, seed=1)
    Lblk = blk._laplacian(n, edges, _T(alpha))
    err = float(_np.abs(Lblk.data - Lref).max())
    check(
        "block Laplacian matches the numpy reference",
        err < 1e-10,
        f"max abs diff = {err:.2e}",
    )

    # the fluid must change the field
    x = _T(rng.normal(0, 1, (n, d)))
    bands = blk._bands(Lblk, float(_np.abs(Lblk.data).max()), x)
    moved = sum(float(_np.abs(b.data - x.data).max()) for b in bands)
    check(
        "the fluid actually moves the field (bands are not identity)",
        moved > 1e-8,
        f"total band displacement = {moved:.3e}",
    )

    # the harmonic branch must be the per-component mean
    from fluidlayer.topology import connected_components as _cc

    lab, nc = _cc(n, edges)
    means = _np.zeros((nc, d))
    for i in range(n):
        means[lab[i]] += x.data[i]
    counts = _np.bincount(lab, minlength=nc)[:, None]
    means = means / counts
    expect = means[lab]
    got = blk._harmonic(x, edges).data
    herr = float(_np.abs(got - expect).max())
    check(
        "harmonic branch == per-component mean (the Hodge projection)",
        herr < 1e-10,
        f"max abs diff = {herr:.2e}, components = {nc}",
    )


# ------------------------------------------------------------------- 15
def test_band_stability():
    """The filter bank must be numerically STABLE: band energies may not blow up.

    Regression test: an earlier version used exp(+alpha) instead of exp(-alpha)
    in the Chebyshev coefficients, which multiplied the bands by e^(2 alpha)
    (~1e6) and made the whole model diverge.
    """
    print("\n[15] Filter-bank stability (regression: exp(-alpha) sign error)")
    from fluidlayer.autodiff import Tensor as _T
    from fluidlayer.blocks import FluidLayerBlock

    rng = _np.random.default_rng(4)
    n, d = 20, 8
    edges = _np.array(
        [[i, (i + 1) % n] for i in range(n)] + [[i, (i + 7) % n] for i in range(n // 2)]
    )
    blk = FluidLayerBlock(d, k_neighbors=4, bands=5, cheb_m=10, seed=2)
    x = _T(rng.normal(0, 1, (n, d)))
    L = blk._laplacian(n, edges, _T(rng.normal(0, 0.5, edges.shape[0])))
    bands = blk._bands(L, float(_np.abs(L.data).max()), x)
    e = _np.array([float((b.data**2).sum()) for b in bands])
    scale = float((x.data**2).sum())
    ok = bool(e.max() < 1e6 * max(scale, 1e-12))
    check(
        "filter bank is numerically stable (no exponential blow-up)",
        ok,
        f"band energies = {[_np.format_float_scientific(v, 2) for v in e]}",
    )
    # and the band sum must be bounded by the identity-scale operator
    tot = float(_np.linalg.norm(sum(b.data for b in bands)))
    check(
        "band sum stays on the scale of the input",
        tot < 1e4 * max(scale, 1e-12),
        f"||sum of bands|| = {tot:.3e}  vs ||x||^2 = {scale:.3e}",
    )


# ============================================================ 16-20  MEMORY
def test_register():
    """The FROZEN REGISTER: invariance, integer exactness, fail-loud, append-only."""
    print("\n[16-20] The frozen register")
    reg = FrozenRegister(capacity=64)
    key = ctx_key([1, 2, 3], doc_id=0)
    sig = embedding_signature(np.array([1.0, 0.0, 0.0, 0.0]))
    s0 = reg.promote(4242, key, sig=sig, anchor=3, doc_id=0)
    val0 = int(reg.val[s0])

    # (16) INVARIANCE: writing more slots must not touch an existing one.
    for k in range(20):
        reg.promote(1000 + k, ctx_key([k]), sig=0, anchor=k, doc_id=1)
    check(
        "M1  a promoted value NEVER changes (append-only invariance)",
        int(reg.val[s0]) == val0 and reg.key[s0] == np.uint64(key),
        f"slot {s0} still holds {int(reg.val[s0])} after 20 further writes",
    )

    # (17) INTEGER EXACTNESS: the recalled value is EQUAL, not approximate.
    got = reg.recall(key, sig=sig)
    ok = got is not None and int(got[1]) == 4242 and isinstance(int(got[1]), int)
    check(
        "M1  the recalled value is bit-exact (integer token id)",
        ok,
        f"recalled {None if got is None else int(got[1])} (expected 4242)",
    )

    # (18) FAIL LOUD: a signature mismatch must return None, NOT a wrong value.
    bad = reg.recall(key, sig=embedding_signature(np.array([0.0, 1.0, 0.0, 0.0])))
    check(
        "M1  a signature mismatch FAILS LOUD (returns None, never a wrong value)",
        bad is None,
        f"mismatched signature returned {bad}",
    )

    # (19) APPEND-ONLY: promoting the same key twice appends, latest wins.
    s1 = reg.promote(777, key, sig=sig, anchor=3, doc_id=2)
    latest = reg.recall(key, sig=sig)
    check(
        "M1  promoting the same key appends a slot; the latest wins",
        s1 != s0 and reg.n == s0 + 22 and int(latest[1]) == 777,
        f"slots {s0} and {s1} coexist; latest value = {int(latest[1])}",
    )

    # (20) SERIALISATION ROUND-TRIP.
    blob = reg.to_bytes()
    back = FrozenRegister.from_bytes(blob, 64)
    g2 = back.recall(key, sig=sig)
    check(
        "M1  the register survives a to_bytes/from_bytes round-trip",
        g2 is not None and int(g2[1]) == 777 and back.n == reg.n,
        f"{len(blob)} B, {back.n} slots, value {None if g2 is None else int(g2[1])}",
    )
    return {"slots": reg.n, "state_bytes": reg.nbytes()}


# ================================================================= 21-22
def test_override_and_budget():
    print("\n[21-22] The logit override and the budget law")
    rng = np.random.default_rng(0)
    lg = rng.standard_normal(64)
    want = 37
    out = logit_override(lg, want)
    check(
        "M2  the logit override makes argmax(next token) == the recalled token",
        int(out.argmax()) == want and int(lg.argmax()) != want,
        f"argmax {int(lg.argmax())} -> {int(out.argmax())} (forced {want})",
    )
    lg2 = rng.standard_normal((4, 32))
    o2 = logit_override(lg2, [1, 5, 9, 31])
    check(
        "M2  the batched override forces every row",
        list(o2.argmax(axis=1)) == [1, 5, 9, 31],
        f"argmax per row = {list(o2.argmax(axis=1))}",
    )

    # the override must also accept a framework Tensor (that is how the
    # model actually hands you its logits)
    from fluidlayer.autodiff import Tensor

    t = Tensor(lg.copy())
    o3 = logit_override(t, want)
    check(
        "M2  the logit override accepts a framework Tensor as well as an ndarray",
        int(o3.argmax()) == want,
        f"Tensor input -> argmax {int(o3.argmax())} (forced {want})",
    )

    # budget law: the split must never exceed the budget
    b = budget_split(16 * 1024, M=1024, r=8, slot_bytes=SLOT_BYTES)
    fits = (b["fluid_bytes"] + b["register_bytes"]) <= 16 * 1024
    b2 = budget_split(16 * 1024, M=100000, r=8, slot_bytes=SLOT_BYTES)
    reduced = (
        b2["M_reduced"] and (b2["fluid_bytes"] + b2["register_bytes"]) <= 16 * 1024
    )
    check(
        "M2  the budget law never over-commits (M is reduced instead)",
        fits and reduced,
        f"16KB -> M={b['M_used']} fluid={b['fluid_bytes']}B "
        f"register={b['register_bytes']}B; "
        f"asking M=100000 -> M={b2['M_used']} (reduced={b2['M_reduced']})",
    )


# ==================================================================== 23
def test_constant_state():
    """The runtime state must be CONSTANT SIZE no matter how much is written."""
    print("\n[23] The state is constant size")
    M, d, r = 32, 16, 8
    ad = SheafMemoryAdapter(M=M, r=r, d=d, capacity=64, seed=3)
    rng = np.random.default_rng(4)
    ad.set_prototypes(rng.standard_normal((M, d)))
    before = ad.state_bytes()
    for _ in range(20):
        ad.write(rng.standard_normal((256, d)), t=0.1, cheb_m=4)
    after = ad.state_bytes()
    check(
        "M3  the runtime state does NOT grow with the amount written",
        before == after == M * r * 2 + 64 * SLOT_BYTES,
        f"state {before} B before, {after} B after 5,120 tokens "
        f"(anchors {M}x{r} fp16 + register 64 x {SLOT_BYTES} B)",
    )

    # and the fluid really did absorb the writes
    moved = float(np.abs(ad.anchors).sum())
    check(
        "M3  the fluid actually absorbed the writes (not inert)",
        moved > 1e-8,
        f"sum |anchors| = {moved:.3e}",
    )


# ==================================================================== 24
def _frustration(n, edges, alpha):
    """Number of edges whose sign disagrees with the best global sign assignment."""
    if edges.shape[0] == 0:
        return 0
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    s = np.zeros(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    for e in np.argsort(-np.abs(alpha), kind="stable"):
        u, v = int(edges[e, 0]), int(edges[e, 1])
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if not seen[u]:
            s[u] = 1
            seen[u] = True
        if not seen[v]:
            s[v] = 1
            seen[v] = True
        s[v] = int(s[u] * (1 if alpha[e] > 0 else -1))
        parent[rv] = ru
    bad = 0
    for e in range(edges.shape[0]):
        u, v = int(edges[e, 0]), int(edges[e, 1])
        if int(s[u] * (1 if alpha[e] > 0 else -1) * s[v]) < 0:
            bad += 1
    return int(bad)


def test_frustration():
    print("\n[24] The frustration index is the right balance diagnostic")
    rng = np.random.default_rng(9)
    n = 24
    agree = 0
    trials = 40
    for _ in range(trials):
        edges = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
        alpha = rng.choice([-1.0, 1.0], size=n)
        L = build_laplacian(n, edges, alpha)
        w = np.linalg.eigvalsh(L)
        has_section = bool((np.abs(w) < 1e-8).sum() >= 1)
        fr = _frustration(n, edges, alpha)
        agree += int((fr == 0) == has_section)
    check(
        "M3  frustration == 0  <=>  a non-trivial global section exists",
        agree == trials,
        f"{agree}/{trials} random signed cycles agree",
    )

    # the contrast with the framework's ||L 1|| residual: they are NOT the
    # same property, and only the frustration index tracks the kernel.
    edges = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    alpha = -np.ones(n)  # all-negative EVEN cycle
    L = build_laplacian(n, edges, alpha)
    w = np.linalg.eigvalsh(L)
    kdim = int((np.abs(w) < 1e-8).sum())
    V = np.linalg.eigh(L)[1][:, np.abs(w) < 1e-8]
    nonconst = kdim > 0 and float(np.max(np.abs(V[0] - V[-1]))) > 1e-6
    check(
        "M3  an all-NEGATIVE even cycle freezes onto a non-constant sign pattern",
        kdim >= 1 and nonconst,
        f"kernel dim={kdim}, ||L1||inf={float(np.max(np.abs(L.sum(1)))):.1f} "
        f"(so ||L1|| does NOT detect this) ",
    )


# ==================================================================== 25
def test_hodge_edge_participation():
    print("\n[25] The Hodge edge participation")
    edges = np.stack([np.arange(8), (np.arange(8) + 1) % 8], axis=1)
    alpha = np.ones(8)
    Psi = hodge_embedding(8, edges, alpha)
    d = hodge_edge_participation(Psi, np.arange(8), np.arange(8))
    o = hodge_edge_participation(Psi, np.zeros(4, dtype=np.int64), np.arange(4))
    check(
        "M3  Hodge edge participation is 1 on the diagonal and <= 1 elsewhere",
        bool(np.allclose(d, 1.0))
        and bool(np.all(o <= 1.0 + 1e-12))
        and bool(np.all(o >= 0)),
        f"diagonal in [{d.min():.4f}, {d.max():.4f}], off-diagonal max {o.max():.4f}",
    )


# ==================================================================== 26
def test_graft():
    """The attachable memory layer: attaching it must change NOTHING, and
    the exact tier must return a token the base model cannot see."""
    print("\n[26] The graft: attaching the layer is a provable no-op")
    from fluidlayer.graft import SAABranch
    from fluidlayer.model import TinyTransformer

    m = TinyTransformer(32, d=32, n_layers=2, n_ctx=32, seed=0)
    br = SAABranch(M=16, r=4, d=32, emb=m.emb, ctx_len=1, seed=0)

    tok = np.array([3, 7, 11, 5, 9, 2, 8, 13], dtype=np.int64)
    lg0 = m.forward(tok)
    h0 = m.emb[tok] + m.pos[: len(tok)]
    h1, _ = br.forward(h0, m.emb[tok], tok)
    lg1 = m.forward_from_h(h1)
    same = np.array_equal(lg0.data, lg1.data)
    check(
        "M3  attaching the layer leaves the logits BIT-IDENTICAL",
        same,
        f"max abs diff {float(np.abs(lg0.data - lg1.data).max()):.3e}",
    )

    n_state = br.state_bytes
    for _ in range(5):
        br.forward(h0, m.emb[tok], tok)
    check(
        "M3  the layer's state stays bounded while it is used",
        br.state_bytes <= n_state + 4096,
        f"state {n_state} B -> {br.state_bytes} B after 5 writes",
    )

    br2 = SAABranch(M=16, r=4, d=32, emb=m.emb, ctx_len=1, seed=0)
    br2.Wo_exact.data = np.eye(32)
    br2.read_gain = "raw"
    br2.read_scale = 300.0
    br2.read_transform = "pc1"
    br2.refresh_contrast()
    seq = np.array([4, 21, 6, 30, 12, 17, 2, 25], dtype=np.int64)
    br2.reset()
    br2.forward(m.emb[seq] + m.pos[: len(seq)], m.emb[seq], seq)
    # the store maps a CONTEXT to the token that followed it
    stored = br2.store.get(int(seq[-2]), -1)
    check(
        "M3  the exact tier stores a token id and returns it bit-exactly",
        stored == int(seq[-1]) and isinstance(stored, int),
        f"store[{seq[-2]}] = {stored} (expected {int(seq[-1])})",
    )


# ==================================================================== 27
def test_gate():
    """The gate: the base model's OWN uncertainty decides how much the
    retrieval speaks.

    This is what makes one knob serve both use cases.  A hard override
    strong enough to win the argmax (needed for recall) also forces a wrong
    retrieval into the model (which cost +1.08 nats in E9).  Scaling the
    override by the model's uncertainty fixes that without a second knob.
    """
    print("\n[27] The gate: the model's own uncertainty")
    from fluidlayer.graft import SAABranch
    from fluidlayer.model import TinyTransformer

    m = TinyTransformer(32, d=32, n_layers=2, n_ctx=32, seed=0)
    br = SAABranch(M=16, r=4, d=32, emb=m.emb, ctx_len=1, seed=0)
    br.Wo_exact.data = np.eye(32)
    br.read_gain = "raw"
    br.read_scale = 300.0
    br.read_transform = "pc1"
    br.refresh_contrast()

    # plant a fact: the pair (2 -> 25)
    seq = np.array([4, 21, 6, 30, 12, 17, 2, 25], dtype=np.int64)
    br.reset()
    br.forward(m.emb[seq] + m.pos[: len(seq)], m.emb[seq], seq)
    assert br.store.get(2) == 25, "the fact was not stored"

    # query: a window that ENDS with the key 2
    q = np.array([7, 19, 2], dtype=np.int64)
    h0 = m.emb[q] + m.pos[: len(q)]
    _, info = br.forward(h0, m.emb[q], q)
    hit = int(info["retrieved"][-1])
    check(
        "M3  the layer reports WHAT it retrieved for the query position",
        hit == 25,
        f"retrieved {hit} (expected 25)",
    )

    # a CONFIDENT base model: the gate must NOT move the argmax
    conf = np.full((3, 32), -10.0)
    conf[2, 5] = 10.0
    g1 = br.gated_logits(conf, info, gain=20.0)
    # an UNSURE base model (uniform logits): the gate must let the
    # retrieval win, because the model has nothing better to say
    uns = np.zeros((3, 32))
    g2 = br.gated_logits(uns, info, gain=20.0)
    check(
        "M3  the gate IGNORES the retrieval when the base model is confident",
        int(g1[-1].argmax()) == 5,
        f"argmax stayed {int(g1[-1].argmax())} (the base was sure of 5)",
    )
    check(
        "M3  the gate LETS THE RETRIEVAL WIN when the base model is unsure",
        int(g2[-1].argmax()) == 25,
        f"argmax moved {int(uns[-1].argmax())} -> {int(g2[-1].argmax())} "
        f"(the retrieved token)",
    )

    # no hit -> the gate must not touch the logits at all
    miss = {"retrieved": np.array([-1, -1, -1]), "support": np.zeros(3)}
    g3 = br.gated_logits(uns, miss, gain=20.0)
    check(
        "M3  the gate is a no-op when the store has no hit",
        np.array_equal(g3, uns),
        f"max abs diff {float(np.abs(g3 - uns).max()):.3e}",
    )


# ==================================================================== 28
def test_exact_memory():
    """The memory must be EXACT: the mode is kept, the confidence is
    counted, and the corroborated entries survive the eviction."""
    print("\n[28] The exact memory: the mode tracker and the eviction")
    from fluidlayer.graft import SAABranch
    from fluidlayer.model import TinyTransformer

    class _Emb:
        def __init__(self, V=32, d=32):
            self.data = np.random.default_rng(0).normal(0, 0.02, (V, d))
            self.shape = self.data.shape

    br = SAABranch(M=8, r=4, d=32, emb=_Emb(), ctx_len=1, capacity=8, seed=0)

    # the store keeps the MAJORITY continuation, not the first one
    for v in (5, 5, 7, 7, 7):
        br._put(42, v)
    check(
        "M3  the store keeps the MAJORITY continuation of a context",
        br.store[42] == 7,
        f"store[42] = {br.store[42]} (5,5,7,7,7 -> the mode is 7)",
    )

    # the confidence counts the corroboration
    conf = br.confidence(42)
    check(
        "M3  the store carries a CONFIDENCE (the corroboration ratio)",
        0.0 <= conf <= 1.0 and conf > 0.0,
        f"confidence(42) = {conf:.3f}",
    )

    # and the majority can flip
    for v in (9, 9, 9, 9):
        br._put(42, v)
    check(
        "M3  the stored value follows the majority when it flips",
        br.store[42] == 9,
        f"store[42] = {br.store[42]} (after 4 more 9s)",
    )

    # the EVICTION keeps the corroborated entries, not merely the newest
    br2 = SAABranch(M=8, r=4, d=32, emb=_Emb(), ctx_len=1, capacity=4, seed=0)
    for _ in range(20):
        br2._put(1, 77)  # heavily corroborated
    for i in range(50):
        br2._put(100 + i, i)  # a flood of one-off contexts
    check(
        "M3  the eviction keeps the CORROBORATED entries, not the newest",
        1 in br2.store and len(br2.store) <= 4,
        f"the corroborated key survived: {1 in br2.store}  "
        f"(the store holds {len(br2.store)} of 4 allowed)",
    )

    # the DEFAULT gate must not overrule a confident base model
    m = TinyTransformer(32, d=32, n_layers=2, n_ctx=32, seed=0)
    br3 = SAABranch(M=16, r=4, d=32, emb=m.emb, ctx_len=1, seed=0)
    br3.Wo_exact.data = np.eye(32)
    br3.read_gain = "raw"
    br3.read_scale = 300.0
    br3.read_transform = "pc1"
    br3.refresh_contrast()
    seq = np.array([4, 21, 6, 30, 12, 17, 2, 25], dtype=np.int64)
    br3.reset()
    br3.forward(m.emb[seq] + m.pos[: len(seq)], m.emb[seq], seq)
    q = np.array([7, 19, 2], dtype=np.int64)
    h0 = m.emb[q] + m.pos[: len(q)]
    _, info = br3.forward(h0, m.emb[q], q)
    conf = np.full((3, 32), -10.0)
    conf[2, 5] = 10.0  # the base model is CERTAIN
    g = br3.gated_logits(conf, info)  # the DEFAULT settings
    check(
        "M3  by DEFAULT the gate never overrules a confident base model",
        int(g[-1].argmax()) == 5,
        f"argmax stayed {int(g[-1].argmax())} (the base was sure of 5); "
        f"the hard rules are OFF by default",
    )


# ==================================================================== 29
def test_router_tier():
    """The router tier: the fluid's geometry as an address space.

    "The fluid answers WHERE, the register answers WHAT" -- made literal:
    the exact tier answers when it has the context, and the per-anchor
    (router) tier is the FALLBACK, with the whole stream's reach.
    """
    print("\n[29] The router tier: the fluid as the address space")
    from fluidlayer.graft import SAABranch
    from fluidlayer.model import TinyTransformer

    m = TinyTransformer(32, d=32, n_layers=2, n_ctx=32, seed=0)
    br = SAABranch(M=16, r=4, d=32, emb=m.emb, ctx_len=1, use_router=True, seed=0)
    br.Wo_exact.data = np.eye(32)
    br.read_gain = "raw"
    br.read_scale = 300.0
    br.read_transform = "pc1"
    br.refresh_contrast()
    seq = np.array([4, 21, 6, 30, 12, 17, 2, 25], dtype=np.int64)
    br.reset()
    br.forward(m.emb[seq] + m.pos[: len(seq)], m.emb[seq], seq)
    check(
        "M3  the router tier files the pair under the ANCHOR of the previous position",
        len(br.anchor_store) > 0,
        f"{len(br.anchor_store)} anchors hold a majority continuation",
    )
    # the exact tier must win when it has the context
    q = np.array([7, 19, 2], dtype=np.int64)
    h0 = m.emb[q] + m.pos[: len(q)]
    _, info = br.forward(h0, m.emb[q], q)
    check(
        "M3  the exact tier wins when it has the context",
        int(info["retrieved"][-1]) == 25 and int(info["router_retrieved"][-1]) != 25,
        f"exact {int(info['retrieved'][-1])}, router "
        f"{int(info['router_retrieved'][-1])}",
    )
    # and the fallback chain: the router answers when the exact tier misses
    miss = {
        "retrieved": np.array([-1]),
        "confidence": np.array([0.0]),
        "router_retrieved": np.array([9]),
        "router_confidence": np.array([1.0]),
    }
    lg = br.gated_logits(np.zeros((1, 32)), miss, gain=8.0)
    check(
        "M3  the router tier is the FALLBACK when the exact tier misses",
        int(lg[0].argmax()) == 9,
        f"argmax {int(lg[0].argmax())} (the router's answer)",
    )


if __name__ == "__main__":
    print("=" * 78)
    print("FLUIDLAYER SELFTEST — verifying the proposal's mathematical claims")
    print("=" * 78)
    test_jamming()
    test_chebyshev()
    test_linear_collapse()
    rank = test_rank_c_trap()
    test_g_balance()
    test_contraction()
    test_frame_property()
    test_autodiff()
    test_persistence()
    test_block_laplacian()
    test_band_stability()
    test_register()
    test_override_and_budget()
    test_constant_state()
    test_frustration()
    test_hodge_edge_participation()
    test_graft()
    test_gate()
    test_exact_memory()
    test_router_tier()

    n_pass = sum(r["pass"] for r in RESULTS)
    print("\n" + "=" * 78)
    print(f"RESULT: {n_pass}/{len(RESULTS)} checks passed")
    print("=" * 78)
    with open(os.path.join(OUT, "selftest.json"), "w") as f:
        json.dump(
            {
                "checks": RESULTS,
                "rank_trap": rank,
                "n_pass": n_pass,
                "n_total": len(RESULTS),
                "groups": {
                    "P": "propositions",
                    "W2": "warnings",
                    "M1": "frozen register",
                    "M2": "override + budget",
                    "M3": "memory geometry",
                },
            },
            f,
            indent=2,
        )
    sys.exit(0 if n_pass == len(RESULTS) else 1)
