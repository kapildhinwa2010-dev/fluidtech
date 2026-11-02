"""
E12 -- giving the diffusion tier a real job: THE ROUTER.

THE FAILURE THIS FIXES
----------------------
N7: the diffusion tier has **no upside**.  Its aggregate is a worse
next-token predictor than the base model (3.81 vs 2.59 nats), so the
optimal injection gain for it is **zero**.  The layer's thesis says "the
fluid answers WHERE, the register answers WHAT" -- but the fluid was only
ever a lossy summary that answered NEITHER.

THE ROUTER
----------
The fluid's geometry (its M anchors) is used as an ADDRESS SPACE.  Each
anchor holds the MAJORITY continuation of the contexts routed to it.  The
fluid therefore answers "WHERE does this context belong?", and the
per-anchor store answers "what usually follows contexts like this?".

The consequences, all measured below:

  * the router tier has the **WHOLE stream's reach** (M entries, never
    evicted) where the exact tier's reach is the last ~capacity contexts;
  * the router tier's answer is a **cluster-level regularity**, not a
    fact -- so it must earn its place on the hard contexts, where the base
    model has no idea;
  * the exact tier always wins when it has the context; the router tier is
    the FALLBACK.

WHAT IS MEASURED
----------------
E12a  the reach:  a fact planted deep in a long stream -- does the hybrid
      still recall it after the exact tier has evicted it?
E12b  the precision as a function of M:  more anchors = tighter clusters
      = a more useful cluster-level answer.
E12c  the real test:  on the HIGH-ENTROPY positions (where the base model
      is unsure and the router fires), does the layer beat the base model?
E12d  the cost:  the routing is O(M*d) per token -- what does that buy
      against the KV cache?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluidlayer import SAABranch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--capacity", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e12_router.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"config": vars(args)}

    # ---------------------------------------------------------- E12a/b
    print("[1/4] E12a/E12b -- the reach and the precision vs M")
    from experiments.e11_exact import make_doc, train_base
    rng0 = np.random.default_rng(1)
    train_docs = [make_doc(rng0) for _ in range(120)]
    rng1 = np.random.default_rng(2)
    test_docs = [make_doc(rng1) for _ in range(40)]
    model = train_base(train_docs, d=args.d, ctx=args.ctx, steps=300)

    def new_branch(M, ctx_len=1, capacity=None, use_router=True):
        br = SAABranch(M=M, r=8, d=args.d, emb=model.emb, ctx_len=ctx_len,
                       capacity=capacity or args.capacity, use_exact=True,
                       use_lossy=False, use_router=use_router, seed=0)
        br.Wo_exact.data = np.eye(args.d)
        br.read_gain = "raw"
        br.read_scale = 300.0
        br.read_transform = "pc1"
        br.refresh_contrast()
        return br

    # a long stream with facts planted at increasing depth
    print(f"    building a {args.tokens:,}-token stream ...")
    ids = np.random.default_rng(5).integers(0, 100, size=args.tokens).astype(np.int64)
    depths = [int(args.tokens * f) for f in (0.05, 0.3, 0.6, 0.85)]
    facts = []
    for i, d in enumerate(depths):
        key, val = 90 + i, 95 + i
        ids[d] = key
        ids[d + 1] = val
        facts.append({"depth": d, "key": key, "val": val})

    def stream_tokens(br, arr):
        """Fill both tiers WITHOUT a model forward (the exact tier is
        model-free; the router tier needs the route head, which needs the
        content -- so it does need the embeddings, hence the model)."""
        for s in range(0, len(arr) - 1, args.ctx):
            w = arr[s:s + args.ctx + 1]
            br.forward(model.emb[w[:-1]] + model.pos[:len(w) - 1],
                       model.emb[w[:-1]], w[:-1])

    results = {}
    for M in (16, 64, 256, 1024):
        br = new_branch(M)
        br.reset()
        stream_tokens(br, ids)
        hit_exact = hit_router = hit_hybrid = 0
        for f in facts:
            ctx = ids[f["depth"] - 3:f["depth"] + 1].tolist()
            key = br._key(ctx)
            exact = br.store.get(key, -1)
            # the VALUE at depth+1 was filed under the ANCHOR of the KEY's
            # position (depth), which is row 3 of this 4-token window
            _, am = br._route(model.emb[ids[f["depth"] - 3:
                                                f["depth"] + 1]].data)
            anchor = int(am[3])
            router = br.anchor_store.get(anchor, -1)
            hit_exact += (exact == f["val"])
            hit_router += (router == f["val"])
            hit_hybrid += (exact == f["val"] or router == f["val"])
        results[str(M)] = {
            "exact_tier_recall": hit_exact / len(facts),
            "router_tier_recall": hit_router / len(facts),
            "hybrid_recall": hit_hybrid / len(facts),
            "exact_store_entries": len(br.store),
            "router_entries": len(br.anchor_store),
        }
        print(f"    M={M:>5}: the exact tier {hit_exact}/{len(facts)}   "
              f"the router tier {hit_router}/{len(facts)}   "
              f"the hybrid {hit_hybrid}/{len(facts)}   "
              f"(the exact store holds {len(br.store)} contexts)")
    res["e12ab_reach_and_precision"] = results

    # ---------------------------------------------------------- E12c
    print("\n[2/4] E12c -- the real test: the HARD contexts")
    from experiments.e6_real_corpus import load_chars
    from experiments.e6_real_corpus import train_base as tb_real
    ids_c, Vr = load_chars()
    n_tr = int(len(ids_c) * 0.9)
    tr, va = ids_c[:n_tr], ids_c[n_tr:]
    rm = tb_real(tr, Vr, d=args.d, ctx=args.ctx, steps=300, seed=0)

    def eval_real(use_router, gain=8.0):
        brr = SAABranch(M=256, r=8, d=args.d, emb=rm.emb, ctx_len=12,
                        capacity=4096, use_exact=True, use_lossy=False,
                        use_router=use_router, seed=0)
        brr.Wo_exact.data = np.eye(args.d)
        brr.read_gain = "raw"
        brr.read_scale = 300.0
        brr.read_transform = "pc1"
        brr.refresh_contrast()
        brr.reset()
        for s in range(0, len(tr) - 1, args.ctx):
            w = tr[s:s + args.ctx + 1]
            brr.forward(rm.emb[w[:-1]] + rm.pos[:len(w) - 1],
                        rm.emb[w[:-1]], w[:-1])
        tot = cnt = 0.0
        hard_tot = hard_cnt = 0.0
        tot_base = hard_base = 0.0
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            base = rm.forward(w[:-1])
            z = np.asarray(base.data)
            e = np.exp(z - z.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            u = 1.0 - p.max(axis=1)
            hard = u > 0.9            # the base model has NO IDEA here
            h0 = rm.emb[w[:-1]] + rm.pos[:len(w) - 1]
            _, info = brr.forward(h0, rm.emb[w[:-1]], w[:-1])
            lg = np.asarray(brr.gated_logits(base.data, info, gain=gain))
            z0 = z - z.max(axis=1, keepdims=True)
            ll = -(z0[np.arange(len(z0)), w[1:]] - np.log(np.exp(z0).sum(axis=1)))
            z1 = lg - lg.max(axis=1, keepdims=True)
            ll1 = -(z1[np.arange(len(z1)), w[1:]] - np.log(np.exp(z1).sum(axis=1)))
            tot += ll1.sum()
            tot_base += ll.sum()
            cnt += len(ll)
            hard_tot += ll1[hard].sum()
            hard_base += ll[hard].sum()
            hard_cnt += int(hard.sum())
        return (tot / cnt, tot_base / cnt, hard_tot / max(hard_cnt, 1),
                hard_base / max(hard_cnt, 1), hard_cnt)

    for use_router in (False, True):
        l, lb, hl, hlb, hn = eval_real(use_router)
        tag = "with the router tier   " if use_router else "without (exact only)"
        print(f"    {tag}: overall {l:.4f} (base {lb:.4f}, delta {l - lb:+.4f})")
        print(f"    {'':22s}  on the {hn} HARD positions: {hl:.4f} "
              f"(base {hlb:.4f}, delta {hl - hlb:+.4f})")
        res[f"e12c_{'with' if use_router else 'without'}_router"] = {
            "loss": float(l), "base_loss": float(lb), "delta": float(l - lb),
            "hard_loss": float(hl), "hard_base_loss": float(hlb),
            "hard_delta": float(hl - hlb), "hard_positions": int(hn),
        }

    # ---------------------------------------------------------- E12d
    print("\n[3/4] E12d -- the cost of the routing")
    for M in (64, 256, 1024):
        flops = M * args.d * 2          # the route head: one dot product per anchor
        print(f"    M={M:>5}: the routing costs {flops:>9,} flops/token  "
              f"({flops / 1.74e9 * 100:.4f} % of one transformer layer)")
    res["e12d_cost"] = {str(M): {"flops_per_token": M * args.d * 2,
                                 "percent_of_a_layer": M * args.d * 2 / 1.74e9 * 100}
                        for M in (64, 256, 1024)}

    # ---------------------------------------------------------- E12e
    print("\n[4/4] E12e -- the end-to-end demo on the fact task")
    out = {}
    for use_router in (False, True):
        br = new_branch(256, capacity=64, use_router=use_router)
        hit = 0
        hit_base = 0
        for doc in test_docs:
            t = doc["tokens"]
            br.reset()
            for s in range(0, len(t) - 1, args.ctx):
                w = t[s:s + args.ctx + 1]
                br.forward(model.emb[w[:-1]] + model.pos[:len(w) - 1],
                           model.emb[w[:-1]], w[:-1])
            q = t[-args.ctx:]
            base = model.forward(q)
            h0 = model.emb[q] + model.pos[:len(q)]
            _, info = br.forward(h0, model.emb[q], q)
            lg = br.gated_logits(base.data, info, gain=8.0)
            hit += (int(lg[-1].argmax()) == doc["query_val"])
            hit_base += (int(base.data[-1].argmax()) == doc["query_val"])
        tag = "with the router tier   " if use_router else "without (exact only)"
        print(f"    {tag}: the fact-task recall {hit}/{len(test_docs)} "
              f"= {hit / len(test_docs):.3f}   (the base model alone: "
              f"{hit_base / len(test_docs):.3f})")
        out["with_router" if use_router else "exact_only"] = {
            "recall": hit / len(test_docs), "base_recall": hit_base / len(test_docs)}
    res["e12e_fact_task"] = out

    res["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n    wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
