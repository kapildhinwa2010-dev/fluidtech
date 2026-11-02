#!/usr/bin/env python3
"""
experiments/e1_prior_perplexity.py — DOES A 16 KB DOMAIN PRIOR HELP A FROZEN LM?

This is the experiment the whole "download a domain prior" story rests on, and
it is also the project's largest credibility gap: at the scale we can train
here, a plain transformer still beats the fluid on raw language modelling.

The question is therefore NOT "is the fluid better than a transformer".  It is:

    "Can a 16 KB state, built with NO gradient steps, carry enough domain
     information to improve a FROZEN model's next-token distribution -- and
     does it carry more than a frequency table of the same size?"

Arms (all on the SAME frozen model, all evaluated on the same held-out split)
-----------------------------------------------------------------------------
  A  base            the frozen FluidLayerLM, no prior
  B0 + static prior  the 16 KB fluid state, read at the anchor of the CURRENT
                     token.  The feature is a function of the token identity,
                     so this is (structurally) a COMPRESSED per-token logit bias.
  B1 + live prior    the same state, but it EVOLVES with the stream, so the
                     feature depends on the PREFIX.  This is the arm that can
                     carry information a frequency table cannot.
  C  + unigram bias  the trivial baseline: a per-token logit bias fitted in
                     closed form.  V parameters.
  D  + bigram bias   a strong reference: a per-(prev,cur) residual table.
                     V^2 parameters -- i.e. ~25 KB, MORE than the 16 KB fluid.

Every correction is fitted in CLOSED FORM (ridge regression / residual means).
No gradient step is ever taken on the model.  That is the G0 protocol.

Usage
-----
    python3 experiments/e1_prior_perplexity.py
    python3 experiments/e1_prior_perplexity.py --steps 200 --quick
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

from fluidlayer import FluidLayerLM
from fluidlayer.data import CharTokenizer, get_corpus
from fluidlayer.losses import cross_entropy_tensor
from fluidlayer.memory import SheafMemoryAdapter
from fluidlayer.optim import Adam

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------ helpers
def zero_grads(params):
    for p in params.values():
        p.grad = np.zeros_like(p.data)


def sample_batch(tokens, batch, ctx, rng):
    ix = rng.integers(0, len(tokens) - ctx - 1, size=batch)
    x = np.stack([tokens[i:i + ctx] for i in ix])
    y = np.stack([tokens[i + 1:i + 1 + ctx] for i in ix])
    return x, y


def forward_logits(model, X, chunk=8):
    """Logits without keeping the autodiff graph alive.

    `model.forward_batch` concatenates every sub-batch, so the graphs of ALL
    sequences stay alive at once -- 2,000 x 64 tokens x 6 bands blew the
    sandbox out of memory.  Chunking and dropping the graph as we go fixes it.
    """
    outs = []
    for s in range(0, len(X), chunk):
        lg = model.forward_batch(X[s:s + chunk])
        outs.append(lg.data.copy())
        del lg
    return np.concatenate(outs, axis=0)


def nll_np(logits, targets):
    """Mean cross-entropy (nats) from numpy arrays -- no Tensor needed."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    picked = z[np.arange(len(targets)), targets]
    return float(np.mean(lse - picked))


def count_params(model):
    return int(sum(p.data.size for p in model.params().values()))


def kmeans(X, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = int(min(k, n))
    C = np.empty((k, X.shape[1]), dtype=X.dtype)
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for i in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        C[i] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[i]) ** 2).sum(1))
    for _ in range(iters):
        lab = np.empty(n, dtype=np.int64)
        for s in range(0, n, 4096):
            blk = X[s:s + 4096]
            sc = blk @ C.T
            sc -= (blk ** 2).sum(1)[:, None]
            sc -= (C ** 2).sum(1)[None, :]
            lab[s:s + 4096] = np.argmin(sc, axis=1)
        counts = np.bincount(lab, minlength=k)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
        empty = np.nonzero(counts == 0)[0]
        if empty.size:
            d = np.linalg.norm(X - C[lab], axis=1)
            far = np.argsort(-d)
            for t, j in enumerate(empty):
                C[j] = X[far[t % far.size]]
    return C


def ridge_solve(Phi, R, lam=1e-3):
    """A = argmin ||Phi A - R||^2 + lam||A||^2   (closed form, no gradients)."""
    p = Phi.shape[1]
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(p), Phi.T @ R)


# ------------------------------------------------------------------- the run
def run(steps=300, d=64, layers=2, ctx=64, batch=16, lr=3e-3, seed=0,
        M=1024, r=8, live_chunk=16, n_stream=40000, n_windows=2000,
        verbose=True):
    t0 = time.time()
    log = print if verbose else (lambda *a: None)

    # ---- 1. corpus + model ----------------------------------------------
    log("[1/7] corpus and frozen model ...")
    text = get_corpus(max_chars=120000)
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    # THREE-WAY split.  The model is trained on 0-90%; the PRIOR is fitted on
    # 90-95% (positions the model has NEVER seen, so its residuals are
    # representative); the final evaluation is on 95-100%.
    s1 = int(len(ids) * 0.90)
    s2 = int(len(ids) * 0.95)
    train_ids, fit_ids, val_ids = ids[:s1], ids[s1:s2], ids[s2:]
    V = tok.vocab
    log(f"      corpus {len(text)} chars, vocab {V}, train {len(train_ids)} "
        f"/ fit {len(fit_ids)} / val {len(val_ids)} tokens")

    model = FluidLayerLM(vocab=V, d=d, n_layers=layers, n_ctx=ctx,
                         k_neighbors=8, bands=6, cheb_m=12, seed=seed)
    params = model.params()
    opt = Adam(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, clip=1.0)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        x, y = sample_batch(train_ids, batch, ctx, rng)
        zero_grads(params)
        logits = model.forward_batch(x)
        lt = cross_entropy_tensor(logits, y.reshape(-1))
        lt.backward()
        opt.step(opt.lr_at(step, steps))
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            vx, vy = sample_batch(val_ids, 8, ctx, rng)
            vl = nll_np(forward_logits(model, vx), vy.reshape(-1))
            log(f"      step {step:>4}/{steps}  val {vl:.4f}  "
                f"({time.time()-t0:.0f}s)")
    log(f"      model trained in {time.time()-t0:.0f}s  "
        f"({count_params(model)} params)  -- FROZEN from here on")

    # ---- 2. base logits on sampled windows ------------------------------
    log("[2/7] base logits on sampled windows ...")
    r2 = np.random.default_rng(seed + 1)
    fstarts = r2.integers(0, len(fit_ids) - ctx - 1, size=n_windows)
    Xtr = np.stack([fit_ids[s:s + ctx] for s in fstarts])
    Ytr = np.stack([fit_ids[s + 1:s + 1 + ctx] for s in fstarts])
    Ltr = forward_logits(model, Xtr).reshape(-1, V)
    Ttr = Ytr.reshape(-1)
    Itr = Xtr.reshape(-1)
    Pprev = np.concatenate([Itr[0:1], Itr[:-1]])     # previous token (causal)
    vstarts = r2.integers(0, len(val_ids) - ctx - 1, size=max(n_windows // 2, 200))
    Xva = np.stack([val_ids[s:s + ctx] for s in vstarts])
    Yva = np.stack([val_ids[s + 1:s + 1 + ctx] for s in vstarts])
    Lva = forward_logits(model, Xva).reshape(-1, V)
    Tva = Yva.reshape(-1)
    Iva = Xva.reshape(-1)
    Pva = np.concatenate([Iva[0:1], Iva[:-1]])
    log(f"      fit positions {Ltr.shape[0]:,}, val positions {Lva.shape[0]:,}")

    def nll(logits, targets):
        return nll_np(logits, targets)

    base_tr = nll(Ltr, Ttr)
    base_va = nll(Lva, Tva)
    log(f"      base  fit {base_tr:.4f}  val {base_va:.4f} nats")

    # ---- 3. the 16 KB domain prior --------------------------------------
    log("[3/7] building the 16 KB domain prior ...")
    emb = model.emb.data                     # (V, d) the model's own embeddings
    protos = kmeans(emb, M, iters=20, seed=seed)
    M_use = int(min(M, protos.shape[0]))
    if M_use < M:
        log(f"      note: the corpus has only {V} distinct symbols, so the "
            f"codebook has {M_use} cells (not {M})")
    ad = SheafMemoryAdapter(M=M_use, r=r, d=d, capacity=64, seed=seed)
    ad.set_prototypes(protos)
    rng3 = np.random.default_rng(seed + 2)
    ad.W_in = rng3.normal(0, d ** -0.5, (d, r))

    # stream the training corpus through the fluid, recording the feature of
    # every token position (the LIVE prior: the state evolves)
    stream_len = min(len(ids), n_stream)
    log(f"[4/7] streaming {stream_len:,} corpus tokens (live prior, "
        f"chunk={live_chunk}) ...")
    feats = np.zeros((stream_len, r), dtype=np.float64)
    s = 0
    n_diff = 0
    while s < stream_len:
        e = min(s + live_chunk, stream_len)
        blk = emb[ids[s:e]]
        # read BEFORE absorbing the chunk (causal)
        idx, _ = ad.route(blk, k=1)
        feats[s:e] = ad.anchors[idx]
        ad.write(blk, t=0.05, cheb_m=10, diffuse=True)
        n_diff += 1
        s = e
    log(f"      {n_diff} diffusion steps in {time.time()-t0:.0f}s")

    # the STATIC prior: the same state, but frozen, read at the anchor of the
    # current token (a function of the token identity only)
    static_state = ad.anchors.copy()

    def static_feature(tok_ids):
        idx, _ = ad.route(emb[tok_ids], k=1)
        return static_state[idx]

    def live_feature(global_idx):
        """feature of a token at a global stream index (the live state)"""
        g = np.clip(global_idx, 0, stream_len - 1)
        return feats[g]

    # map the sampled window positions to their global stream indices
    off_fit = s1
    off_val = s2
    gidx_tr = np.clip(off_fit + fstarts[:, None] + np.arange(ctx)[None, :],
                      0, stream_len - 1).reshape(-1)
    gidx_va = np.clip(off_val + vstarts[:, None] + np.arange(ctx)[None, :],
                      0, stream_len - 1).reshape(-1)

    # ---- 5. fit + evaluate ----------------------------------------------
    log("[5/7] fitting the corrections in closed form ...")
    results = {"base": {"fit_nll": base_tr, "val_nll": base_va,
                        "val_ppl": float(np.exp(base_va))}}

    # The gradient of the cross-entropy w.r.t. the logits is (p - onehot),
    # which lives in PROBABILITY space.  Fitting the correction to the
    # LOGIT-space residual (onehot - logits) over-corrects by a factor
    # (1 - p) and made every arm WORSE (measured).  This is the fix.
    def prob_residual(logits, targets):
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        R = -p
        R[np.arange(len(targets)), targets] += 1.0
        return R

    R_tr = prob_residual(Ltr, Ttr)          # the residual we want to predict

    # --- B0: static fluid prior
    P0_tr = static_feature(Itr)
    A0 = ridge_solve(P0_tr, R_tr)
    l0_va = Lva + static_feature(Iva) @ A0
    l0_tr = Ltr + P0_tr @ A0
    results["fluid_prior_static"] = {
        "fit_nll": nll(l0_tr, Ttr), "val_nll": nll(l0_va, Tva),
        "fit_params": int(A0.size), "state_bytes": int(ad.state_bytes()),
    }

    # --- B1: live fluid prior
    P1_tr = live_feature(gidx_tr)
    A1 = ridge_solve(P1_tr, R_tr)
    l1_va = Lva + live_feature(gidx_va) @ A1
    l1_tr = Ltr + P1_tr @ A1
    results["fluid_prior_live"] = {
        "fit_nll": nll(l1_tr, Ttr), "val_nll": nll(l1_va, Tva),
        "fit_params": int(A1.size), "state_bytes": int(ad.state_bytes()),
    }

    # --- C: unigram bias (the trivial baseline)
    bias = np.zeros((V, V), dtype=np.float64)
    cnt = np.zeros(V, dtype=np.float64)
    for c in range(V):
        m = Itr == c
        if m.any():
            bias[c] = R_tr[m].mean(axis=0)
            cnt[c] = m.sum()
    l2_va = Lva + bias[Iva]
    l2_tr = Ltr + bias[Itr]
    results["unigram_bias"] = {
        "fit_nll": nll(l2_tr, Ttr), "val_nll": nll(l2_va, Tva),
        "fit_params": int(V * V), "state_bytes": int(V * V * 4),
    }

    # --- C2: the BEST rank-8 unigram correction (a fair control for B0).
    # The static fluid prior is, structurally, a rank-<=r map from the token
    # identity to a logit residual -- so it must be compared against the best
    # possible rank-r unigram map, not against the full-rank one.
    U, S, Vt = np.linalg.svd(bias, full_matrices=False)
    B8 = (U[:, :r] * S[:r]) @ Vt[:r, :]
    f8 = B8[Itr]                            # (N, V) the rank-8 features
    alpha = float(np.sum(f8 * R_tr) / (np.sum(f8 * f8) + 1e-12))
    l2b_va = Lva + alpha * B8[Iva]
    l2b_tr = Ltr + alpha * f8
    results["unigram_lowrank"] = {
        "fit_nll": nll(l2b_tr, Ttr), "val_nll": nll(l2b_va, Tva),
        "fit_params": int(r * V), "state_bytes": int(r * V * 4),
        "rank": int(r), "alpha": alpha,
    }

    # --- D: bigram residual table (a strong reference, V^2 params)
    big = {}
    for i in range(len(Ttr)):
        key = (int(Itr[i]), int(Pprev[i]))
        big.setdefault(key, []).append(i)
    bias2 = np.zeros((V, V, V), dtype=np.float64)   # (prev, cur) -> V-vector
    for (p, c), idxs in big.items():
        bias2[p, c] = R_tr[idxs].mean(axis=0)
    # back off to the unigram where the pair was unseen
    seen = np.zeros((V, V), dtype=bool)
    for (p, c) in big:
        seen[p, c] = True
    # unseen (prev,cur) pairs fall back to the CURRENT token's unigram bias
    bias2 = np.where(seen[:, :, None], bias2, bias[None, :, :])
    l3_va = Lva + bias2[Pva, Iva, :]
    l3_tr = Ltr + bias2[Pprev, Itr, :]
    results["bigram_bias"] = {
        "fit_nll": nll(l3_tr, Ttr), "val_nll": nll(l3_va, Tva),
        "fit_params": int(V * V * V), "state_bytes": int(V * V * V * 4),
    }

    # ---- 6. report ------------------------------------------------------
    log("[6/7] results (val nats; lower is better) ...")
    log(f"      base                 {base_va:.4f}")
    for k in ("fluid_prior_static", "fluid_prior_live", "unigram_bias",
              "unigram_lowrank", "bigram_bias"):
        v = results[k]["val_nll"]
        delta = v - base_va
        log(f"      {k:<20} {v:.4f}   delta {delta:+.4f} nats "
            f"({int(results[k]['fit_params'])} fitted params, "
            f"{results[k]['state_bytes']:,} B)")
    # byte efficiency: nats saved per KB of STATE
    base_v = base_va
    for k in list(results.keys()):
        if k in ("base", "config", "runtime_sec"):
            continue
        gain = base_v - results[k]["val_nll"]
        kb = max(results[k].get("state_bytes", 0), 1) / 1024.0
        results[k]["nats_gained"] = float(gain)
        results[k]["nats_per_kb"] = float(gain / kb)
    results["config"] = {
        "steps": steps, "d": d, "layers": layers, "ctx": ctx, "batch": batch,
        "seed": seed, "M": M, "r": r, "live_chunk": live_chunk,
        "n_stream": n_stream, "stream_len": int(stream_len), "n_windows": n_windows, "vocab": V,
        "model_params": count_params(model),
    }
    results["runtime_sec"] = round(time.time() - t0, 2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="run several training lengths and report how the "
                         "prior's benefit depends on the base model's quality")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "e1_prior.json"))
    args = ap.parse_args()
    if args.quick:
        args.steps = min(args.steps, 60)
    if args.sweep:
        rows = []
        for st in (30, 100, 250, 500):
            print(f"\n########## training length = {st} steps ##########")
            r = run(steps=st, d=args.d, layers=args.layers, ctx=args.ctx,
                    batch=args.batch, lr=args.lr, seed=args.seed, M=args.M,
                    r=args.r, n_windows=args.windows)
            rows.append({
                "steps": st,
                "base_val_nll": r["base"]["val_nll"],
                "base_val_ppl": r["base"]["val_ppl"],
                "gains": {k: r[k]["nats_gained"] for k in
                          ("fluid_prior_static", "fluid_prior_live",
                           "unigram_bias", "unigram_lowrank", "bigram_bias")},
                "nats_per_kb": {k: r[k]["nats_per_kb"] for k in
                                ("fluid_prior_static", "fluid_prior_live",
                                 "unigram_bias", "unigram_lowrank", "bigram_bias")},
            })
        out = {"sweep": rows, "config": {"d": args.d, "ctx": args.ctx,
                                         "M": args.M, "r": args.r,
                                         "windows": args.windows,
                                         "seed": args.seed}}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")
        print(f"  {'steps':>6} {'base nll':>10} {'fluid':>10} {'unigram':>10} "
              f"{'uni-r8':>10}")
        for r in rows:
            print(f"  {r['steps']:>6} {r['base_val_nll']:>10.4f} "
                  f"{r['gains']['fluid_prior_static']:>+10.4f} "
                  f"{r['gains']['unigram_bias']:>+10.4f} "
                  f"{r['gains']['unigram_lowrank']:>+10.4f}")
        return 0

    res = run(steps=args.steps, d=args.d, layers=args.layers, ctx=args.ctx,
              batch=args.batch, lr=args.lr, seed=args.seed, M=args.M, r=args.r,
              n_windows=args.windows)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")
    b = res["base"]["val_nll"]
    print(f"  base val nats {b:.4f}")
    for k in ("fluid_prior_static", "fluid_prior_live", "unigram_bias",
              "unigram_lowrank", "bigram_bias"):
        v = res[k]["val_nll"]
        print(f"  {k:20s} {v:.4f}   delta {v-b:+.4f}   "
              f"{res[k]['nats_per_kb']:.2f} nats/KB  "
              f"({res[k]['state_bytes']:,} B)")


if __name__ == "__main__":
    main()
