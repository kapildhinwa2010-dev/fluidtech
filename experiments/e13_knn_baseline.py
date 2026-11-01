"""
E13 -- the missing baseline: kNN-LM.

THE GAP
-------
This repository never compared itself against the closest prior art.  A
**kNN-LM** (Khandelwal et al., 2020, "Retrieval Augmented Language Model")
is: a datastore of (context -> next token) plus an LM, combined by
INTERPOLATING their next-token distributions:

    p(v) = lam * p_LM(v)  +  (1 - lam) * p_knn(v)

That is, functionally, what this project's layer is -- with three
differences:

  * the datastore is CONSTANT-SIZE here (16 KB), not unbounded;
  * the combination is a GATED OVERRIDE (the model's own uncertainty
    scales the retrieval), not a flat interpolation;
  * the store is an append-only audit trail, first-write/majority tracked.

Without this comparison the project has no evidence that the *gate* -- the
actual contribution -- is worth anything.

WHAT IS MEASURED
----------------
E13a  the base model alone;
E13b  the kNN-LM baseline (the probability interpolation), lam swept;
E13c  the layer (the gated override), the gain swept;
E13d  the RECALL of each -- because the interpolation is bounded (it can
      never outvote the LM) while the gate can, and that is the whole
      argument;
E13e  the datastore size effect: the same code with a 16 KB store and with
      a 1 MB store.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superfluid import SAABranch, Tensor, cross_entropy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e13_knn.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"config": vars(args)}

    from experiments.e6_real_corpus import load_chars, train_base
    ids, V = load_chars()
    n_tr = int(len(ids) * 0.9)
    tr, va = ids[:n_tr], ids[n_tr:]
    print(f"[1/4] the corpus + the base model ({len(ids):,} chars, vocab {V})")
    model = train_base(tr, V, d=args.d, ctx=args.ctx, steps=args.steps)

    def build(capacity=819, ctx_len=12, M=256):
        br = SAABranch(M=M, r=8, d=args.d, emb=model.emb, ctx_len=ctx_len,
                       capacity=capacity, use_exact=True, use_lossy=False,
                       use_router=True, seed=0)
        br.Wo_exact.data = np.eye(args.d)
        br.read_gain = "raw"
        br.read_scale = 300.0
        br.read_transform = "pc1"
        br.refresh_contrast()
        br.reset()
        for s in range(0, len(tr) - 1, args.ctx):
            w = tr[s:s + args.ctx + 1]
            br.forward(model.emb[w[:-1]] + model.pos[:len(w) - 1],
                       model.emb[w[:-1]], w[:-1])
        return br

    def losses(br, mode, param):
        """mode: 'knn' (the interpolation) or 'gate' (the override)."""
        tot = cnt = 0.0
        rec = 0
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            base = model.forward(w[:-1])
            z = np.asarray(base.data)
            h0 = model.emb[w[:-1]] + model.pos[:len(w) - 1]
            _, info = br.forward(h0, model.emb[w[:-1]], w[:-1])
            r = np.asarray(info["retrieved"])
            rr = np.asarray(info["router_retrieved"])
            fallback = (r < 0) & (rr >= 0)
            ret = np.where(fallback, rr, r)
            if mode == "knn":
                # p = lam * p_LM + (1 - lam) * p_knn
                e = np.exp(z - z.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
                lam = float(param)
                pknn = np.zeros_like(p)
                rows = np.nonzero(ret >= 0)[0]
                pknn[rows, ret[rows]] = 1.0
                pm = lam * p + (1.0 - lam) * pknn
                lg = np.log(np.clip(pm, 1e-30, None))
            else:
                lg = np.asarray(br.gated_logits(base.data, info, gain=float(param)))
            tot += cross_entropy(Tensor(lg), w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
            pred = lg.argmax(axis=1)
            tgt = w[1:]
            rec += int((pred == tgt).sum())
        return tot / cnt, rec / cnt

    print("\n[2/4] E13a/E13b -- the base model and the kNN-LM baseline")
    br = build(capacity=819)
    base_l, base_acc = losses(br, "knn", 1.0)   # lam=1 -> the plain LM
    print(f"    the base model alone        : {base_l:.4f} "
          f"(ppl {math.exp(base_l):.2f})  token accuracy {base_acc:.4f}")
    res["base"] = {"loss": float(base_l), "ppl": float(math.exp(base_l)),
                   "token_accuracy": float(base_acc)}
    knn = {}
    for lam in (0.95, 0.9, 0.8, 0.5, 0.2):
        l, a = losses(br, "knn", lam)
        knn[str(lam)] = {"loss": float(l), "delta": float(l - base_l),
                         "token_accuracy": float(a)}
        print(f"    kNN-LM, lam={lam:<5}          : {l:.4f} "
              f"(ppl {math.exp(l):.2f})  delta {l - base_l:+.4f}  "
              f"token accuracy {a:.4f}")
    res["knn_lm"] = knn

    print("\n[3/4] E13c/E13d -- the layer (the gated override)")
    gate = {}
    for g in (2.0, 8.0, 20.0):
        l, a = losses(br, "gate", g)
        gate[str(g)] = {"loss": float(l), "delta": float(l - base_l),
                        "token_accuracy": float(a)}
        print(f"    the layer, gain={g:<5}        : {l:.4f} "
              f"(ppl {math.exp(l):.2f})  delta {l - base_l:+.4f}  "
              f"token accuracy {a:.4f}")
    res["gated_layer"] = gate

    print("\n[4/4] E13e -- the datastore size (the same code, a bigger store)")
    sizes = {}
    for cap, label in ((819, "16 KB"), (65536, "1 MB")):
        b2 = build(capacity=cap)
        l, a = losses(b2, "gate", 8.0)
        lk, ak = losses(b2, "knn", 0.8)
        sizes[str(cap)] = {
            "label": label, "layer_loss": float(l),
            "layer_delta": float(l - base_l), "layer_token_accuracy": float(a),
            "knn_loss": float(lk), "knn_delta": float(lk - base_l),
        }
        print(f"    a {label:>6} store: the layer {l:.4f} "
              f"(delta {l - base_l:+.4f})   kNN-LM {lk:.4f} "
              f"(delta {lk - base_l:+.4f})")
    res["datastore_size"] = sizes

    res["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n    wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
