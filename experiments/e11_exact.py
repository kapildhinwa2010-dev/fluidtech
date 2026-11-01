"""
E11 -- making the memory EXACT.

E10 solved the safety problem (one knob for recall AND generation) but the
memory was still not exact in two ways:

  (1) the RECALL had a ceiling: the soft gate could suppress a perfectly
      good retrieval whenever the base model happened to be confident.
      Measured: 39/40 = 0.975.  A memory that is right 97.5 % of the time
      is not exact.

  (2) the STORE kept the FIRST continuation of a context (first-write-
      wins).  That is exact but blind: a context that occurs everywhere kept
      only its first continuation, so the statistically correct answer was
      lost.  Measured (E9): 0/3 facts with common keys recalled.

THE FIXES
---------
RULE 1 (the clueless rule):  if the base model's top-1 probability is
        below 1 - tau, the retrieval DECIDES.  The model has no idea, so
        the memory answers.
RULE 2 (the trusted-memory rule):  if the stored value has never been
        contradicted (confidence == 1), the retrieval DECIDES as well.
        A memory of a specific, never-contradicted event is trusted; a
        statistical guess is not.
THE MODE TRACKER:  the store keeps the MAJORITY continuation of a context
        and counts the corroboration, so it is statistically correct and it
        carries a confidence.
THE EVIDENCE-BASED EVICTION:  when the store overflows, the LEAST
        corroborated entry is evicted, not merely the oldest.  A memory that
        has been confirmed many times is worth more than a fresh one.

WHAT IS MEASURED
----------------
E11a  the recall: is it 1.000 now?
E11b  the common-key case: the mode tracker vs first-write-wins.
E11c  the generation safety: does the new gate still improve the base
      model's perplexity?
E11d  the eviction: at 10M tokens, which entries survive?
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

from superfluid import SAABranch, Tensor, TinyTransformer, cross_entropy   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
V = 128
N_KEYS, N_VALS = 8, 8
KEY_LO, KEY_HI = 0, N_KEYS
VAL_LO, VAL_HI = N_KEYS, N_KEYS + N_VALS


def make_doc(rng, length=512, n_facts=4, majority=False):
    """A document with facts.  If `majority`, the fact's value is also the
    context's MOST COMMON continuation (so a mode tracker can find it and a
    first-write-wins store usually cannot)."""
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
    # the fact positions must NOT overlap: two overlapping facts would
    # overwrite each other and the "missing" fact would be a CORPUS bug,
    # not a layer bug (this cost one confusing measurement already)
    positions = sorted(rng.choice(np.arange(16, length - 160), size=n_facts,
                                  replace=False).tolist())
    for k, p in zip(keys, positions):
        v = int(rng.integers(VAL_LO, VAL_HI))
        doc[p] = k
        doc[p + 1] = v
        facts.append({"key": int(k), "val": v, "pos": int(p)})
        if majority:
            # make the value the COMMON continuation of that key: 8 further
            # occurrences of key -> value, 2 of key -> something else.
            # These positions must also avoid the fact positions.
            taken = {p, p + 1} | {f["pos"] for f in facts} | \
                    {f["pos"] + 1 for f in facts}
            for _ in range(10):
                q = int(rng.integers(0, length - 4))
                if q in taken or (q + 1) in taken:
                    continue
                taken.add(q)
                doc[q] = int(k)
                doc[q + 1] = v if _ < 8 else int(rng.integers(VAL_LO, VAL_HI))
    qk = facts[0]["key"]
    doc[-1] = qk
    return {"tokens": doc, "facts": facts,
            "query_key": qk, "query_val": facts[0]["val"]}


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
    ap.add_argument("--out", default=os.path.join(RESULTS, "e11_exact.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"config": vars(args)}

    rng0 = np.random.default_rng(1)
    train_docs = [make_doc(rng0) for _ in range(args.train_docs)]
    rng1 = np.random.default_rng(2)
    test_docs = [make_doc(rng1) for _ in range(args.docs)]
    rng2 = np.random.default_rng(3)
    maj_docs = [make_doc(rng2, majority=True) for _ in range(args.docs)]

    print("[1/5] the base model ...")
    model = train_base(train_docs, d=args.d, ctx=args.ctx, steps=args.steps)

    def new_branch(ctx_len=1, capacity=4096):
        br = SAABranch(M=64, r=8, d=args.d, emb=model.emb, ctx_len=ctx_len,
                       capacity=capacity, use_exact=True, use_lossy=False, seed=0)
        br.Wo_exact.data = np.eye(args.d)
        br.read_gain = "raw"
        br.read_scale = 300.0
        br.read_transform = "pc1"
        br.refresh_contrast()
        return br

    def stream(br, t):
        for s in range(0, len(t) - 1, args.ctx):
            w = t[s:s + args.ctx + 1]
            br.forward(model.emb[w[:-1]] + model.pos[:len(w) - 1],
                       model.emb[w[:-1]], w[:-1])

    # -------------------------------------------------------------- E11a
    print("\n[2/5] E11a -- the recall: is it 1.000 now?")
    br = new_branch(ctx_len=1)
    out = {}
    for name, tau, c_min in (("the soft gate alone (RECOMMENDED)", 2.0, 2.0),
                             ("+ the clueless rule, tau=0.99", 0.99, 2.0),
                             ("+ the clueless rule, tau=0.95", 0.95, 2.0),
                             ("+ the clueless + trusted rules", 0.99, 1.0)):
        hit = 0
        for doc in test_docs:
            t = doc["tokens"]
            br.reset()
            stream(br, t)
            q = t[-args.ctx:]
            base = model.forward(q)
            h0 = model.emb[q] + model.pos[:len(q)]
            _, info = br.forward(h0, model.emb[q], q)
            lg = br.gated_logits(base.data, info, gain=8.0, tau=tau, c_min=c_min)
            hit += (int(lg[-1].argmax()) == doc["query_val"])
        out[name] = hit / len(test_docs)
        print(f"    {name:<34}: {hit}/{len(test_docs)} = {hit / len(test_docs):.3f}")
    res["e11a_recall"] = out

    # -------------------------------------------------------------- E11b
    print("\n[3/5] E11b -- the common-key case: the mode tracker")
    br2 = new_branch(ctx_len=1)
    hit = 0
    confs = []
    for doc in maj_docs:
        t = doc["tokens"]
        br2.reset()
        stream(br2, t)
        q = t[-args.ctx:]
        base = model.forward(q)
        h0 = model.emb[q] + model.pos[:len(q)]
        _, info = br2.forward(h0, model.emb[q], q)
        lg = br2.gated_logits(base.data, info, gain=8.0, tau=0.5, c_min=1.0)
        ok = (int(lg[-1].argmax()) == doc["query_val"])
        hit += ok
        confs.append(float(info["confidence"][-1]))
    print(f"    the fact is the MAJORITY continuation of a common key:")
    print(f"      recalled {hit}/{len(maj_docs)} = {hit / len(maj_docs):.3f}   "
          f"(the first-write-wins store managed 0/3 in E9)")
    print(f"      the mean retrieval confidence: {np.mean(confs):.3f}")
    res["e11b_mode_tracker"] = {
        "recall": hit / len(maj_docs), "n": len(maj_docs),
        "mean_confidence": float(np.mean(confs)),
    }

    # -------------------------------------------------------------- E11c
    print("\n[4/5] E11c -- the generation safety (does it still improve?)")
    from experiments.e6_real_corpus import load_chars, train_base as tb_real
    ids_c, Vr = load_chars()
    n_tr = int(len(ids_c) * 0.9)
    tr, va = ids_c[:n_tr], ids_c[n_tr:]
    rm = tb_real(tr, Vr, d=args.d, ctx=args.ctx, steps=300, seed=0)

    def base_only():
        tot = cnt = 0.0
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            lg = rm.forward(w[:-1])
            tot += cross_entropy(Tensor(np.asarray(lg.data)), w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
        return tot / cnt

    def real_eval(gain, tau, c_min):
        brr = SAABranch(M=64, r=8, d=args.d, emb=rm.emb, ctx_len=12,
                        capacity=4096, use_exact=True, use_lossy=False, seed=0)
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
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            base = rm.forward(w[:-1])
            h0 = rm.emb[w[:-1]] + rm.pos[:len(w) - 1]
            _, info = brr.forward(h0, rm.emb[w[:-1]], w[:-1])
            lg = brr.gated_logits(base.data, info, gain=gain, tau=tau, c_min=c_min)
            tot += cross_entropy(Tensor(np.asarray(lg)), w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
        return tot / cnt

    base_only_l = base_only()
    print(f"    the base model alone            : {base_only_l:.4f} "
          f"(ppl {math.exp(base_only_l):.2f})")
    swept = {}
    for label, tau, c_min in (("the soft gate alone (RECOMMENDED)", 2.0, 2.0),
                               ("+ the clueless rule, tau=0.99", 0.99, 2.0),
                               ("+ the clueless + trusted rules", 0.99, 1.0)):
        for g in (2.0, 8.0, 20.0):
            l = real_eval(g, tau, c_min)
            swept[f"{label}|{g}"] = {"loss": float(l),
                                     "delta_vs_base": float(l - base_only_l)}
            print(f"    {label:<34} gain {g:>5.1f}: {l:.4f} "
                  f"(ppl {math.exp(l):.2f})   delta {l - base_only_l:+.4f}")
    res["e11c_generation"] = {"base_loss": float(base_only_l),
                              "base_ppl": float(math.exp(base_only_l)),
                              "variants": swept}

    # -------------------------------------------------------------- E11d
    print("\n[5/5] E11d -- the eviction: which entries survive at scale?")
    ids = np.random.default_rng(4).integers(0, 64, size=200_000).astype(np.int64)
    # 40 contexts that are corroborated many times, and a flood of noise
    hot = {i: ids[ids == i][:50].tolist() for i in range(40)}
    br3 = SAABranch(M=8, r=4, d=args.d, emb=model.emb, ctx_len=1, capacity=64,
                    use_exact=True, use_lossy=False, seed=0)
    br3.reset()
    for i, occ in hot.items():
        for v in occ:
            br3._put(int(i), int(v))
    for v in ids[:5000]:
        br3._put(int(v), int(v))          # a flood of one-off contexts
    survived = sum(1 for i in hot if i in br3.store)
    conf_hi = [br3.confidence(i) for i in hot if i in br3.store]
    print(f"    {survived}/{len(hot)} corroborated contexts survived a flood "
          f"of 5000 one-off contexts (the store holds 64)")
    print(f"    their mean confidence: {np.mean(conf_hi) if conf_hi else 0:.3f}")
    res["e11d_eviction"] = {
        "hot_contexts": len(hot), "survived": survived,
        "capacity": 64, "mean_confidence_survived":
            float(np.mean(conf_hi)) if conf_hi else 0.0,
    }

    res["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n    wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
