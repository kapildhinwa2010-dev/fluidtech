"""
E10 -- the gate: making the retrieval safe at the gain that works.

THE FAILURE THIS FIXES
-----------------------
E9 measured the tension:

    the injection gain that maximises RECALL   : 300   (the recalled token
                                                    must win the argmax)
    the injection gain that maximises GENERATION: 0.3   (a wrong retrieval
                                                    must not overrule the
                                                    model)
    -> 1000x apart.  At the recall gain, the base model's perplexity
       goes 13.61 -> 40.20 (+1.08 nats).  The layer is unusable.

THE GATE
--------
The base model's OWN uncertainty at that position:

    u_t = 1 - max_v p(v)

and the retrieved token's logit is boosted by  gain * u_t.

One knob serves both use cases, because the two tasks differ exactly in the
base model's confidence:

    RECALL task    : the model has no idea what comes next (u ~ 1)
                    -> the retrieval decides.
    GENERATION task: the model is usually confident (u ~ 0)
                    -> the retrieval is ignored.

The gate is not a second hyper-parameter; it is the model's own confidence.

WHAT IS MEASURED
----------------
E10a  the RECALL task: does the gate let the retrieval through when the
      model is clueless?  (the recall must not drop)
E10b  the GENERATION task: does the gate keep the base model's perplexity
      intact at a LARGE gain?  (the harm must not grow with the gain)
E10c  the frontier: (recall, harm) for the ungated and the gated layer.

Everything is CPU-only and pure NumPy.
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

from superfluid import SAABranch, Tensor, TinyTransformer, cross_entropy    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
V = 128
N_KEYS, N_VALS = 8, 8
KEY_LO, KEY_HI = 0, N_KEYS
VAL_LO, VAL_HI = N_KEYS, N_KEYS + N_VALS


# ---------------------------------------------------------------- the corpus
def make_doc(rng, length=512, n_facts=4):
    filler_pool = np.arange(N_KEYS + N_VALS, V)
    beta = 0.30
    doc = np.empty(length, dtype=np.int64)
    doc[0] = int(rng.choice(filler_pool))
    for i in range(1, length):
        prev = doc[i - 1]
        doc[i] = (int(rng.choice(filler_pool)) if rng.random() < beta else prev)
    doc = np.where((doc >= KEY_LO) & (doc < KEY_LO + N_VALS),
                   filler_pool[rng.integers(0, len(filler_pool))], doc)
    keys = rng.choice(np.arange(KEY_LO, KEY_HI), size=n_facts, replace=False)
    facts = []
    for k in keys:
        p = int(rng.integers(16, length - 160))
        v = int(rng.integers(VAL_LO, VAL_HI))
        doc[p] = k
        doc[p + 1] = v
        facts.append((int(k), v))
    qk, qv = facts[0]
    doc[-1] = qk
    return {"tokens": doc, "facts": facts, "query_key": qk, "query_val": qv}


def train_base(docs, d=64, layers=2, ctx=128, steps=400, batch=16, lr=3e-3,
               seed=0):
    rng = np.random.default_rng(seed)
    model = TinyTransformer(V, d=d, n_layers=layers, n_ctx=ctx, seed=0)
    from superfluid.optim import Adam
    from superfluid.losses import cross_entropy_tensor
    opt = Adam(model.params(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    toks = np.concatenate([doc["tokens"] for doc in docs])
    for step in range(steps):
        starts = rng.integers(0, len(toks) - ctx - 1, size=batch)
        x = np.stack([toks[s:s + ctx] for s in starts])
        y = np.stack([toks[s + 1:s + ctx + 1] for s in starts])
        for p in model.params().values():
            p.grad = np.zeros_like(p.data)
        loss = cross_entropy_tensor(model.forward_batch(x), y.reshape(-1))
        loss.backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--train-docs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e10_gate.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"config": vars(args)}

    print("[1/5] the corpus + the base model ...")
    train_docs = [make_doc(np.random.default_rng(args.seed if 'seed' in vars(args) else 0 + i))
                  for i in range(0)]
    rng0 = np.random.default_rng(1)
    train_docs = [make_doc(rng0) for _ in range(args.train_docs)]
    rng1 = np.random.default_rng(2)
    test_docs = [make_doc(rng1) for _ in range(args.docs)]
    model = train_base(train_docs, d=args.d, ctx=args.ctx, steps=args.steps)
    print(f"    {args.train_docs} train / {args.docs} test documents")

    def new_branch(ctx_len=1):
        br = SAABranch(M=64, r=8, d=args.d, emb=model.emb, ctx_len=ctx_len,
                       capacity=4096, use_exact=True, use_lossy=False, seed=0)
        br.Wo_exact.data = np.eye(args.d)
        br.read_gain = "raw"
        br.read_scale = 300.0
        br.read_transform = "pc1"
        br.refresh_contrast()
        return br

    # ------------------------------------------------------------- E10a
    print("\n[2/5] E10a -- the RECALL task: does the gate let the "
          "retrieval through?")
    br = new_branch(ctx_len=1)
    res_a = {}
    for gated in (False, True):
        hit = 0
        for doc in test_docs:
            t = doc["tokens"]
            br.reset()
            for s in range(0, len(t) - 1, args.ctx):
                w = t[s:s + args.ctx + 1]
                br.forward(model.emb[w[:-1]] + model.pos[:len(w) - 1],
                           model.emb[w[:-1]], w[:-1])
            q = t[-args.ctx:]
            base = model.forward(q)
            if gated:
                # the gated override: the boost is the model's own uncertainty
                h0 = model.emb[q] + model.pos[:len(q)]
                _, info = br.forward(h0, model.emb[q], q)
                lg = br.gated_logits(base.data, info, gain=20.0)
            else:
                # the ungated override: a flat boost
                h0 = model.emb[q] + model.pos[:len(q)]
                h1, info = br.forward(h0, model.emb[q], q)
                lg = base.data.copy()
                r = info["retrieved"]
                rows = np.nonzero(r >= 0)[0]
                if len(rows):
                    lg[rows, r[rows]] += 20.0
            hit += (int(lg[-1].argmax()) == doc["query_val"])
        res_a["gated" if gated else "ungated"] = hit / len(test_docs)
        print(f"    {'GATED  ' if gated else 'ungated'} exact recall: "
              f"{hit}/{len(test_docs)} = {hit / len(test_docs):.3f}")
    res["e10a_recall"] = res_a

    # ------------------------------------------------------------- E10b
    print("\n[3/5] E10b -- the GENERATION task: does the gate keep the "
          "base model's loss intact?")
    from experiments.e6_real_corpus import load_chars, train_base as tb_real
    ids_c, Vr = load_chars()
    n_tr = int(len(ids_c) * 0.9)
    tr, va = ids_c[:n_tr], ids_c[n_tr:]
    rm = tb_real(tr, Vr, d=args.d, ctx=args.ctx, steps=300, seed=0)

    def real_eval(gain, gated, support=False, support_scale=2.0):
        brr = SAABranch(M=64, r=8, d=args.d, emb=rm.emb, ctx_len=12,
                        capacity=4096, use_exact=True, use_lossy=False, seed=0)
        brr.Wo_exact.data = np.eye(args.d)
        brr.read_gain = "raw"
        brr.read_scale = 300.0
        brr.read_transform = "pc1"
        brr.refresh_contrast()
        brr.reset()
        # fill the memory by streaming the training text through the layer
        # in the model's own window size (the position table is n_ctx long)
        for s in range(0, len(tr) - 1, args.ctx):
            w = tr[s:s + args.ctx + 1]
            brr.forward(rm.emb[w[:-1]] + rm.pos[:len(w) - 1],
                        rm.emb[w[:-1]], w[:-1])
        tot = cnt = 0.0
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            base = rm.forward(w[:-1])
            h0 = rm.emb[w[:-1]] + rm.pos[:len(w) - 1]
            _, info = brr.forward(h0, rm.emb[w[:-1]], w[:-1])
            if gated:
                lg = brr.gated_logits(base.data, info, gain=gain,
                                      use_support=support,
                                      support_scale=support_scale)
            else:
                # the UNGATED override: a FLAT boost, no uncertainty weighting
                lg = base.data.copy()
                r = info["retrieved"]
                rows = np.nonzero(r >= 0)[0]
                if len(rows):
                    lg[rows, r[rows]] += gain
            tot += cross_entropy(Tensor(np.asarray(lg)), w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
        return tot / cnt

    base_l = real_eval(0.0, True)
    print(f"    the base model alone: {base_l:.4f} (ppl {math.exp(base_l):.2f})")
    ungated = {}
    for g in (0.5, 2.0, 8.0, 20.0, 50.0):
        l = real_eval(g, False)
        ungated[str(g)] = {"loss": float(l), "delta": float(l - base_l)}
        print(f"    UNGATED gain {g:>6.1f} -> {l:.4f}  delta {l - base_l:+.4f}")
    gated = {}
    for g in (0.5, 2.0, 8.0, 20.0, 50.0, 200.0):
        l = real_eval(g, True)
        gated[str(g)] = {"loss": float(l), "delta": float(l - base_l)}
        print(f"    GATED   gain {g:>6.1f} -> {l:.4f}  "
              f"delta {l - base_l:+.4f}")
    print(f"    (the ungated numbers are in the JSON; the gated ones are "
          f"printed above)")
    res["e10b_generation"] = {"base_loss": float(base_l),
                              "gated": gated, "ungated_flat": ungated}

    # ------------------------------------------------------------- E10c
    print("\n[4/5] E10c -- the frontier")
    print(f"    {'gain':>7} {'ungated Δ':>12} {'gated Δ':>12}")
    for g in ("0.5", "2.0", "8.0", "20.0", "50.0"):
        ug = ungated.get(g, {}).get("delta")
        gt = gated.get(g, {}).get("delta")
        print(f"    {g:>7} {(f'{ug:+.4f}' if ug is not None else '-'):>12} "
              f"{(f'{gt:+.4f}' if gt is not None else '-'):>12}")

    print("\n[5/5] summary")
    res["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"    wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
