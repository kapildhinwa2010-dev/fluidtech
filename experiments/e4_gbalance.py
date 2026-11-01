#!/usr/bin/env python3
"""
experiments/e4_gbalance.py — the G-BALANCE / spectral-gap probe.

    "Does the G-balance phase transition actually control what the fluid
     remembers, and does lambda_2 predict long-context quality?"

Background
----------
A signed sheaf Laplacian L = D_abs - A is G-balanced when every cell's
incident restriction maps sum to zero, i.e. ||L 1||_inf = 0.  Then the
constant field is a global section, the kernel is non-trivial, and the fluid
freezes onto a MEANINGFUL state.  When the component is unbalanced, the
harmonic branch vanishes and the component "evaporates" instead of freezing.

The theory report says the frozen state is the Hodge limit; this experiment
measuring the two things that follow from it:

  Q1  Does the balance residual predict whether the harmonic branch keeps a
      non-trivial component?  (the phase transition, measured)
  Q2  Does the spectral gap lambda_2 predict how well the diffused state
      tracks a slowly-moving aggregate over a long stream?  (the correlation)

Q2 is the one that matters for the project: if lambda_2 does NOT predict
quality, then the "certified contraction" is a certificate about the
OPERATOR and not about the TASK, and the claim has to be weakened
accordingly.  We report the correlation either way.

Usage
-----
    python3 experiments/e4_gbalance.py
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from superfluid.sheaf import build_laplacian, balance_residual, component_balance
from superfluid.diffusion import spectral_summary, exact_exp_action, jam_time
from superfluid.topology import connected_components

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def frustration_index(n, edges, alpha):
    """The number of edges whose sign DISAGREES with the best global sign
    assignment -- the standard frustration index of a signed graph.

    This is the correct diagnostic for a SCALAR signed sheaf.  The framework's
    ||L 1||_inf ("balance residual") measures whether the CONSTANT field is a
    global section, which for scalar +-1 restriction maps means "every edge is
    positive".  That is a much stricter condition than "a non-trivial global
    section exists": an even cycle with all-negative edges is NOT balanced in
    the ||L1|| sense, yet it DOES freeze onto the alternating sign pattern.

    So:
        frustration == 0   <=>  a non-trivial signed frozen state exists
    and the ||L 1|| residual is a *different*, stricter property.
    """
    if edges.shape[0] == 0:
        return 0
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # spanning forest, with the sign assignment propagated along it
    s = np.zeros(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    tree_edges = set()
    order = np.argsort(-np.abs(alpha), kind="stable")
    for e in order:
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
        # the tree edge forces s[v] = s[u] * alpha_uv
        s[v] = int(s[u] * (1 if alpha[e] > 0 else -1))
        parent[rv] = ru
        tree_edges.add(int(e))
    bad = 0
    for e in range(edges.shape[0]):
        u, v = int(edges[e, 0]), int(edges[e, 1])
        prod = int(s[u] * (1 if alpha[e] > 0 else -1) * s[v])
        if prod < 0:
            bad += 1
    return int(bad)


def signed_cycle(n, sign_frac, seed=0):
    """A cycle of n cells whose edges carry +1 with probability sign_frac."""
    rng = np.random.default_rng(seed)
    edges = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    alpha = np.where(rng.random(n) < sign_frac, 1.0, -1.0)
    return edges, alpha


def run(n_cells=64, n_trials=40, seed=0, verbose=True):
    t0 = time.time()
    log = print if verbose else (lambda *a: None)
    rows = []

    # ---- Q1: the phase transition ---------------------------------------
    log("[1/3] G-balance phase transition ...")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for t in range(n_trials):
            edges, alpha = signed_cycle(n_cells, frac, seed=seed + 1000 * t + int(frac * 100))
            L = build_laplacian(n_cells, edges, alpha)
            res = balance_residual(L)
            w = np.linalg.eigvalsh(L)
            ker = np.abs(w) < 1e-8
            kv = None
            if ker.sum() > 0:
                V = np.linalg.eigh(L)[1][:, ker]
                kv = float(np.max(np.abs(V[0] - V[-1])))   # non-constant?
            frus = frustration_index(n_cells, edges, alpha)
            rows.append({
                "sign_frac": float(frac),
                "balance_residual": float(res),
                "frustration": int(frus),
                "lam2": float(w[w > 1e-10][0]) if (w > 1e-10).any() else 0.0,
                "kernel_dim": int(ker.sum()),
                "nonconstant_kernel": bool(kv is not None and kv > 1e-6),
                "has_global_section": bool(ker.sum() >= 1),
            })
    # aggregate
    agg = {}
    for frac in sorted({r["sign_frac"] for r in rows}):
        sub = [r for r in rows if r["sign_frac"] == frac]
        agg[str(frac)] = {
            "mean_balance_residual": float(np.mean([r["balance_residual"] for r in sub])),
            "mean_frustration": float(np.mean([r["frustration"] for r in sub])),
            "frac_nonconstant_kernel": float(np.mean([r["nonconstant_kernel"] for r in sub])),
            "frac_global_section": float(np.mean([r["has_global_section"] for r in sub])),
            "frac_zero_frustration": float(np.mean([r["frustration"] == 0 for r in sub])),
            "mean_lam2": float(np.mean([r["lam2"] for r in sub])),
        }
        log(f"      sign_frac={frac:.2f}  ||L1||={agg[str(frac)]['mean_balance_residual']:.3f}  "
            f"frustration={agg[str(frac)]['mean_frustration']:.2f}  "
            f"non-constant frozen state in {agg[str(frac)]['frac_nonconstant_kernel']*100:.0f}% of trials")

    # ---- Q1b: frustration == 0  <=>  a non-trivial frozen state exists ---
    # the CORRECT statement: frustration == 0  <=>  the kernel is non-trivial
    # (a global section exists).  "non-constant" is a STRICTTER property: the
    # all-positive even cycle is balanced and freezes onto the CONSTANT.
    agree = sum(1 for r in rows
                if (r["frustration"] == 0) == r["has_global_section"])
    log(f"      frustration==0  <=>  a global section exists (non-trivial "
        f"kernel): {agree}/{len(rows)} trials agree")

    # ---- Q2: does lambda_2 predict tracking quality? ---------------------
    log("[2/3] lambda_2 vs long-stream tracking quality ...")
    corr_rows = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + 7919 * trial)
        n = n_cells
        # a random signed graph: sign_frac controls the balance
        frac = float(rng.uniform(0.0, 1.0))
        edges = np.unique(np.sort(rng.integers(0, n, size=(n * 2, 2)), axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        alpha = np.where(rng.random(edges.shape[0]) < frac, 1.0, -1.0)
        L = build_laplacian(n, edges, alpha)
        s = spectral_summary(L)
        lam2 = s["lam2"]
        if lam2 <= 0:
            continue
        # a slowly-moving target aggregate: the running mean of a signal
        X = rng.standard_normal((n, 4))
        target = X.mean(axis=0)
        # the fluid's answer after t units of diffusion
        t = 1.0
        Y = exact_exp_action(L, t, X)
        frozen = Y.mean(axis=0)
        # quality = how well the frozen state matches the true aggregate
        # (this is what the harmonic branch is FOR)
        denom = np.linalg.norm(target) + 1e-12
        quality = float(np.dot(frozen, target) / denom)
        corr_rows.append({"lam2": float(lam2), "quality": quality,
                          "balance_residual": float(balance_residual(L)),
                          "frac": frac, "t_jam": float(jam_time(lam2))})
    lam = np.array([r["lam2"] for r in corr_rows])
    qua = np.array([r["quality"] for r in corr_rows])
    if lam.std() > 1e-12 and qua.std() > 1e-12:
        pearson = float(np.corrcoef(lam, qua)[0, 1])
    else:
        pearson = float("nan")
    # rank correlation (monotone, robust to the heavy tail of lam2)
    def rankdata(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        return r
    spearman = float(np.corrcoef(rankdata(lam), rankdata(qua))[0, 1]) \
        if len(corr_rows) > 2 else float("nan")
    log(f"      pearson(lam2, quality) = {pearson:.4f}   spearman = {spearman:.4f}   "
        f"(n={len(corr_rows)})")

    # ---- Q3: the contractivity certificate -------------------------------
    log("[3/3] contraction certificate on the same graphs ...")
    cert = []
    for r in corr_rows[:12]:
        pass
    rng = np.random.default_rng(seed)
    ok = 0
    tot = 0
    for trial in range(12):
        n = n_cells
        edges = np.unique(np.sort(rng.integers(0, n, size=(n * 2, 2)), axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        alpha = rng.choice([-1.0, 1.0], size=edges.shape[0])
        L = build_laplacian(n, edges, alpha)
        w, V = np.linalg.eigh(L)
        ker = np.abs(w) < 1e-8
        lam2 = float(w[w > 1e-10][0]) if (w > 1e-10).any() else 0.0
        if lam2 <= 0:
            continue
        for tt in (0.5, 1.0, 2.0):
            E = exact_exp_action(L, tt, np.eye(n))
            Pk = V[:, ker] @ V[:, ker].T
            op = float(np.linalg.norm(E @ (np.eye(n) - Pk), 2))
            bound = float(np.exp(-lam2 * tt))
            tot += 1
            ok += int(op <= bound + 1e-9)
            cert.append({"lam2": lam2, "t": tt, "observed": op, "bound": bound,
                         "holds": bool(op <= bound + 1e-9)})

    out = {
        "config": {"n_cells": int(n_cells), "n_trials": int(n_trials), "seed": int(seed)},
        "phase_transition": agg,
        "frustration_agreement": {"n": len(rows), "n_agree": int(agree)},
        "phase_transition_rows": rows,
        "lam2_vs_quality": {
            "pearson": pearson, "spearman": spearman, "n": len(corr_rows),
            "rows": corr_rows,
        },
        "contraction_certificate": {
            "n_checked": tot, "n_holding": ok, "all_hold": bool(ok == tot),
            "sample": cert[:10],
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=64)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "e4_gbalance.json"))
    args = ap.parse_args()
    res = run(n_cells=args.n_cells, n_trials=args.trials, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"  pearson(lam2, quality) = {res['lam2_vs_quality']['pearson']:.4f}  "
          f"spearman = {res['lam2_vs_quality']['spearman']:.4f}")
    print(f"  contraction certificate holds: "
          f"{res['contraction_certificate']['n_holding']}/"
          f"{res['contraction_certificate']['n_checked']}")


if __name__ == "__main__":
    main()
