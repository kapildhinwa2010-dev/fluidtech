#!/usr/bin/env python3
"""
experiments/e2_scaling.py — the SCALING CURVE.

    "How does quality depend on the size of the state?"

Nobody knows the answer, and that is precisely why the curve is worth
publishing: EITHER outcome is a result.

  * "quality saturates at 256 KB"  -> a famous NEGATIVE result, and it
    retroactively justifies the 16 KB headline.
  * "quality keeps climbing"        -> a famous POSITIVE result, and it says
    16 KB was too small.

What is swept
-------------
The RUNTIME budget S in {16 KB, 64 KB, 256 KB, 1 MB, 4 MB, 16 MB}.  For each
budget the budget law fixes the split:

    fluid  = M * r * 2 bytes          (M = S / 2 / (r*2),  r = 8)
    register = the rest                (20 B per exact slot)

so a bigger budget buys MORE ANCHORS *and* MORE SLOTS at the same time, and
we can see which one actually pays.

Measured per budget
-------------------
  gist_cosine   how well the diffused state carries the local aggregate
                (the fluid's job: aggregate recall)
  precision     of the selection criterion: of the K promoted slots, how many
                are real fact contexts (the criterion's job: choosing what
                deserves exactness)
  fact_recall   how many test facts come back bit-exactly
  flops/token   the per-token cost of the fluid (linear in M)

Usage
-----
    python3 experiments/e2_scaling.py
    python3 experiments/e2_scaling.py --tokens 200000
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

from superfluid.memory import (SheafMemoryAdapter, FrozenRegister, SLOT_BYTES,
                               embedding_signature, ctx_key, budget_split)
from superfluid.topology import hodge_relevance

sys.path.insert(0, HERE)
from e5_graft import Corpus, kmeans, route_all, cos_mean, CTX

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BUDGETS = [16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024,
           4 * 1024 * 1024, 16 * 1024 * 1024]


def one_budget(corp, protos, anchor_of_symbol, budget, r=8, d=64, budget_slots=64,
               n_test=150, seed=0, chunk=8192, cheb_m=12, M_override=None,
               verbose=True):
    """Run the whole pipeline for ONE runtime budget and return the metrics."""
    log = print if verbose else (lambda *a: None)
    M_req = 1024 if M_override is None else int(M_override)
    split = budget_split(budget, M=M_req, r=r, slot_bytes=SLOT_BYTES)
    M = split["M_used"] if M_override is None else M_req
    n_slots = max(split["pins"], 1)

    ad = SheafMemoryAdapter(M=M, r=r, d=d, capacity=max(n_slots, 16), seed=seed)
    ad.set_prototypes(protos[:M])
    P = protos[:M]

    # route with the (possibly smaller) codebook
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    tok_anchor = np.empty(corp.n_tokens, dtype=np.int64)
    hist = np.zeros(M, dtype=np.int64)
    sum_proj = np.zeros((M, r), dtype=np.float64)
    for s in range(0, corp.n_tokens, chunk):
        blk = corp.emb[corp.tokens[s:s + chunk]]
        bn = blk / (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-12)
        a = np.argmax(bn @ Pn.T, axis=1)
        tok_anchor[s:s + chunk] = a
        np.add.at(hist, a, 1)
        np.add.at(sum_proj, a, blk @ ad.W_in)

    # stream
    n_chunks = (corp.n_tokens + chunk - 1) // chunk
    t_chunk = 1.0 / max(n_chunks, 1)
    for s in range(0, corp.n_tokens, chunk):
        ad.write(corp.emb[corp.tokens[s:s + chunk]], t=t_chunk, cheb_m=cheb_m)

    # gist quality
    counts = np.maximum(hist, 1)[:, None]
    true_local = sum_proj / counts
    live = hist > 0
    gist = cos_mean(ad.anchors[live], true_local[live]) if live.any() else 0.0

    # selection + exact recall
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(corp.n_facts)
    te = perm[corp.n_facts // 3:][:n_test]
    q_all = np.arange(CTX, corp.n_tokens - 1, dtype=np.int64)
    valid = corp.tokens[q_all] != corp.tokens[np.minimum(q_all + 1, corp.n_tokens - 1)]
    q_all = q_all[valid]
    pc = corp.tokens[:-1] * corp.n_symbols + corp.tokens[1:]
    srt = np.argsort(pc, kind="stable")
    p = pc[srt]
    newp = np.ones(p.size, dtype=bool)
    newp[1:] = p[1:] != p[:-1]
    pf = np.zeros(pc.size, dtype=bool)
    pf[srt[newp]] = True
    pair_novel = pf[q_all]
    e1 = corp.emb[corp.tokens[q_all]]
    e2 = corp.emb[corp.tokens[np.minimum(q_all + 1, corp.n_tokens - 1)]]
    cs = np.sum(e1 * e2, axis=1) / (np.linalg.norm(e1, axis=1) *
                                     np.linalg.norm(e2, axis=1) + 1e-12)
    score = pair_novel * cs
    k = int(min(budget_slots, score.shape[0]))
    idx = np.argpartition(-score, kth=k - 1)[:k]
    positions = set(int(x) for x in q_all[idx])

    fact_positions = set(int(x) for x in corp.fact_pos)
    reg = FrozenRegister(capacity=len(positions) + 8)
    for q in sorted(positions):
        w = [int(x) for x in corp.tokens[q - CTX + 1: q + 1]]
        reg.promote(int(corp.tokens[q + 1]), ctx_key(w, doc_id=0),
                    sig=embedding_signature(corp.emb[corp.tokens[q]]),
                    anchor=int(tok_anchor[q]), doc_id=0)
    exact = 0
    n_promoted_test = 0
    for i in te:
        q = int(corp.fact_pos[i])
        if q not in positions:
            continue
        n_promoted_test += 1
        w = [int(x) for x in corp.tokens[q - CTX + 1: q + 1]]
        got = reg.recall(ctx_key(w, doc_id=0),
                         sig=embedding_signature(corp.emb[corp.tokens[q]]))
        if got is not None and int(got[1]) == int(corp.fact_val_sym[i]):
            exact += 1
    precision = sum(1 for q in positions if q in fact_positions) / max(len(positions), 1)

    # per-token cost: the fluid is linear in M (m matvecs of the anchor graph)
    flops = 12 * 2 * M * M * 2 + 4 * M * r
    return {
        "budget_bytes": int(budget),
        "M_override": M_override,
        "budget_slots": int(budget_slots),
        "M": int(M),
        "r": int(r),
        "slots": int(n_slots),
        "fluid_bytes": int(M * r * 2),
        "register_bytes": int(budget_slots * SLOT_BYTES),
        "gist_cosine": float(gist),
        "pin_precision": float(precision),
        "fact_recall": float(exact / max(len(te), 1)),
        "exact_given_promoted": float(exact / max(n_promoted_test, 1)),
        "flops_per_token": int(flops),
        "n_test": int(len(te)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "e2_scaling.json"))
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/4] corpus ({args.tokens:,} tokens) ...")
    corp = Corpus(n_tokens=args.tokens, d=64, seed=args.seed)
    print("[2/4] codebook (M=4096) ...")
    sample = np.random.default_rng(args.seed).choice(corp.n_symbols, 4096, replace=False)
    protos = kmeans(corp.emb[sample].astype(np.float64),
                     min(4096, corp.n_symbols), iters=20, seed=args.seed)

    # ---- AXIS A: the ANCHOR count (M), pins fixed -----------------------
    print("[3/4] axis A -- anchor count M (exact budget fixed) ...")
    axis_m = []
    for M in (256, 1024, 4096):
        row = one_budget(corp, protos, None, budget=1 << 30, M_override=M,
                         budget_slots=args.slots, seed=args.seed)
        axis_m.append(row)
        print(f"   M={M:>6}  gist={row['gist_cosine']:.4f}  "
              f"precision={row['pin_precision']:.3f}  recall={row['fact_recall']:.3f}  "
              f"flops/tok={row['flops_per_token']:,}")

    # ---- AXIS B: the EXACT budget (K), anchors fixed ---------------------
    print("[4/4] axis B -- exact budget K (anchors fixed at 1024) ...")
    axis_k = []
    for K in (16, 64, 256, 1024):
        row = one_budget(corp, protos, None, budget=1 << 30, M_override=1024,
                         budget_slots=K, seed=args.seed)
        axis_k.append(row)
        print(f"   K={K:>6}  gist={row['gist_cosine']:.4f}  "
              f"precision={row['pin_precision']:.3f}  recall={row['fact_recall']:.3f}")

    out = {
        "config": {"n_tokens": int(args.tokens), "seed": int(args.seed),
                   "slot_bytes": SLOT_BYTES},
        "axis_anchors_M": axis_m,
        "axis_exact_budget_K": axis_k,
        "runtime_sec": round(time.time() - t0, 2),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
