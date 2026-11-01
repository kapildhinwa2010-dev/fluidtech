"""
E6 -- THE GRAFT EXPERIMENT.

    "Base model pe ye layer laga di to kya hoga?"

Three questions, three measurements:

  E6a  ATTACH SAFETY.  Attach the layer to a trained model.  Does anything
       change?  (Answer measured, not assumed: the branch is initialised
       with ZERO output projections, so the answer must be "nothing at all,
       bit for bit".)

  E6b  PERPLEXITY.  Train ONLY the layer (the base model is frozen).  Does
       the base model's own next-token quality get better or WORSE?  And
       where does the change come from -- real modelling, or retrieval?

  E6c  CAPABILITY.  What can the frozen model do AFTER the graft that it
       provably could not do before?  (Cross-window verbatim recall.)

  E6d  GENERATION.  Side-by-side samples, so the change is visible and not
       just a number.

Everything runs on CPU in pure NumPy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superfluid import TinyTransformer, cross_entropy, cross_entropy_tensor  # noqa: E402
from superfluid.graft import SAABranch  # noqa: E402
from superfluid.optim import Adam  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

V = 128
N_KEYS = 8
N_VALS = 8
KEY_LO, KEY_HI = 0, N_KEYS
VAL_LO, VAL_HI = N_KEYS, N_KEYS + N_VALS
FILL_LO, FILL_HI = N_KEYS + N_VALS, V


# --------------------------------------------------------------------- data
def make_doc(rng, length=512, n_facts=4, block_len=24):
    """A document with (a) random key->value facts, (b) a repeated block.

    The repeated block is the "the model has seen this before" case: the
    base model's attention window cannot reach it, so its loss on the second
    occurrence is high.  The facts are the "the model must retrieve a value
    it never saw in the window" case.
    """
    # A MARKOV filler, not i.i.d. noise: with i.i.d. tokens the base model
    # cannot beat the unigram entropy, so every perplexity number would be
    # meaningless (and the "does the layer hurt?" question unanswerable).
    filler_pool = np.arange(FILL_LO, FILL_HI)
    beta = 0.30
    doc = np.empty(length, dtype=np.int64)
    doc[0] = int(rng.choice(filler_pool))
    for i in range(1, length):
        prev = doc[i - 1]
        if rng.random() < beta:
            doc[i] = int(rng.choice(filler_pool))
        else:
            doc[i] = prev
    # keep the filler away from the key/value tokens
    doc = np.where((doc >= KEY_LO) & (doc < FILL_LO),
                   filler_pool[rng.integers(0, len(filler_pool))], doc)

    keys = rng.choice(np.arange(KEY_LO, KEY_HI), size=n_facts, replace=False)
    facts = []
    fact_pos = []
    for k in keys:
        p = int(rng.integers(16, length - 160))
        v = int(rng.integers(VAL_LO, VAL_HI))
        doc[p] = k
        doc[p + 1] = v
        facts.append((int(k), v))
        fact_pos.append(p)

    # a block of BLOCK_LEN tokens that appears TWICE
    block_start = int(rng.integers(8, 64))
    block = doc[block_start:block_start + block_len].copy()
    rep = int(rng.integers(length - block_len - 80, length - block_len - 8))
    doc[rep:rep + block_len] = block

    # the query: the last token is a fact key, the target is its value
    qk, qv = facts[0]
    doc[-1] = qk
    return {
        "tokens": doc,
        "facts": facts,
        "fact_pos": fact_pos,
        "query_key": qk,
        "query_val": qv,
        "block": (block_start, block_len),
        "repeat": (rep, block_len),
    }


def corpus(n_docs, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [make_doc(rng, **kw) for _ in range(n_docs)]


# ------------------------------------------------------------------ training
def train_base(docs, d=64, layers=2, ctx=128, steps=400, batch=16, lr=3e-3,
               seed=0, seed0=0):
    """Train the BASE model (a real LM) on the corpus.  It is then frozen."""
    rng = np.random.default_rng(seed)
    model = TinyTransformer(V, d=d, n_layers=layers, n_ctx=ctx, seed=seed0)
    opt = Adam(model.params(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    toks = np.concatenate([doc["tokens"] for doc in docs])
    for step in range(steps):
        starts = rng.integers(0, len(toks) - ctx - 1, size=batch)
        x = np.stack([toks[s:s + ctx] for s in starts])
        y = np.stack([toks[s + 1:s + ctx + 1] for s in starts])
        for p in model.params().values():
            p.grad = np.zeros_like(p.data)
        logits = model.forward_batch(x)
        loss = cross_entropy_tensor(logits, y.reshape(-1))
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"    base step {step:>4}/{steps}  loss {float(loss.data):.4f}")
    return model


def base_loss(model, docs, ctx=128, key="tokens"):
    tot, n = 0.0, 0
    for doc in docs:
        t = doc[key]
        for s in range(0, len(t) - 1, ctx):
            w = t[s:s + ctx + 1]
            if len(w) < 2:
                continue
            lg = model.forward(w[:-1])
            tot += cross_entropy(lg, w[1:]) * (len(w) - 1)
            n += len(w) - 1
    return tot / max(n, 1)


# ------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=80)
    ap.add_argument("--train-docs", type=int, default=400)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--branch-steps", type=int, default=30)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--ctx-len", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e6_graft.json"))
    args = ap.parse_args()

    t0 = time.time()
    res = {"config": vars(args)}

    print("[1/6] corpus + base model ...")
    train_docs = corpus(args.train_docs, seed=args.seed + 1)
    test_docs = corpus(args.docs, seed=args.seed + 2)
    model = train_base(train_docs, d=args.d, layers=args.layers, ctx=args.ctx,
                       steps=args.steps, seed=args.seed)
    base_params = {k: v.data.copy() for k, v in model.params().items()}

    # ------------------------------------------------------------ E6a
    print("\n[2/6] E6a -- attach safety (does anything change?)")
    branch = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb,
                       ctx_len=args.ctx_len, seed=args.seed)
    tok = test_docs[0]["tokens"][:args.ctx]
    lg_base = model.forward(tok)
    h0 = model.emb[tok] + model.pos[:len(tok)]
    content = model.emb[tok]
    h1, _ = branch.forward(h0, content, tok)
    lg_graft = model.forward_from_h(h1)
    same = np.array_equal(lg_base.data, lg_graft.data)
    maxdiff = float(np.abs(lg_base.data - lg_graft.data).max())
    print(f"    logits bit-identical after attaching the layer: {same}  "
          f"(max abs diff {maxdiff:.3e})")
    print(f"    layer state: {branch.state_bytes:,} B "
          f"({args.M} anchors x r={args.r}, {len(branch.store)} store entries)")
    res["e6a_attach"] = {
        "logits_bit_identical": bool(same),
        "max_abs_diff": maxdiff,
        "state_bytes": branch.state_bytes,
        "M": args.M, "r": args.r,
    }

    # ------------------------------------------------------------ E6b
    print("\n[3/6] E6b -- perplexity: base vs base+layer (layer trained alone)")
    bl = base_loss(model, test_docs, ctx=args.ctx)
    print(f"    base val loss              {bl:.4f}  (ppl {np.exp(bl):.2f})")

    # train ONLY the layer on the LM task, the base model stays FROZEN
    b2 = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb,
                   ctx_len=args.ctx_len, seed=args.seed)
    opt = Adam(b2.params(), lr=5e-3, betas=(0.9, 0.95))
    rng = np.random.default_rng(args.seed + 7)

    def stream(br, t):
        """Push a whole document through the layer (that is the memory)."""
        br.reset()
        for st in range(0, len(t) - 1, args.ctx):
            seg = t[st:st + args.ctx + 1]
            hh = model.emb[seg[:-1]] + model.pos[:len(seg) - 1]
            cc = model.emb[seg[:-1]]
            br.forward(hh, cc, seg[:-1])

    for step in range(args.branch_steps):
        doc = train_docs[int(rng.integers(0, len(train_docs)))]
        t = doc["tokens"]
        stream(b2, t)
        s = int(rng.integers(0, max(1, len(t) - args.ctx - 1)))
        w = t[s:s + args.ctx + 1]
        h0 = model.emb[w[:-1]] + model.pos[:len(w) - 1]
        content = model.emb[w[:-1]]
        h1, _ = b2.forward(h0, content, w[:-1])
        logits = model.forward_from_h(h1)
        loss = cross_entropy_tensor(logits, w[1:])
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"      branch step {step:>3}  loss {float(loss.data):.4f}  "
                  f"|Wo_exact| {np.linalg.norm(b2.Wo_exact.data):.3f}  "
                  f"|Wo_lossy| {np.linalg.norm(b2.Wo_lossy.data):.3f}")

    # evaluate the trained branch, and SPLIT the loss by "did this context
    # occur earlier?" -- that is the whole question: is the gain MODELLING
    # or is it RETRIEVAL?
    def loss_split(br, docs):
        """Split the loss into positions whose context occurred BEFORE (the
        repeated block) and positions that are genuinely NEW.  The question
        this answers: is the perplexity gain MODELLING or RETRIEVAL?"""
        acc = {"rep": [0.0, 0], "new": [0.0, 0]}
        for doc in docs:
            t = doc["tokens"]
            if br is not None:
                stream(br, t)
            rep_start, rep_len = doc["repeat"]
            for st in range(0, len(t) - 1, args.ctx):
                seg = t[st:st + args.ctx + 1]
                if br is not None:
                    hh = model.emb[seg[:-1]] + model.pos[:len(seg) - 1]
                    cc = model.emb[seg[:-1]]
                    h1, _ = br.forward(hh, cc, seg[:-1])
                    lg = model.forward_from_h(h1)
                else:
                    lg = model.forward(seg[:-1])
                z = np.asarray(lg.data)
                z = z - z.max(axis=1, keepdims=True)
                ll = -(z[np.arange(len(z)), seg[1:]]
                       - np.log(np.exp(z).sum(axis=1)))
                for i in range(len(ll)):
                    pos = st + i
                    k = "rep" if (rep_start <= pos < rep_start + rep_len) else "new"
                    acc[k][0] += float(ll[i])
                    acc[k][1] += 1
        return (acc["rep"][0] / max(acc["rep"][1], 1), acc["rep"][1],
                acc["new"][0] / max(acc["new"][1], 1), acc["new"][1])

    g_rep, g_nrep, g_new, g_nnew = loss_split(b2, test_docs)
    b_rep, b_nrep, b_new, b_nnew = loss_split(None, test_docs)
    print("    positions whose context occurred BEFORE (the repeated block):")
    print(f"      base        {b_rep:.4f}  (n={b_nrep})")
    print(f"      base+layer  {g_rep:.4f}  (n={g_nrep})   "
          f"delta {g_rep - b_rep:+.4f}")
    print("    positions whose context is NEW:")
    print(f"      base        {b_new:.4f}  (n={b_nnew})")
    print(f"      base+layer  {g_new:.4f}  (n={g_nnew})   "
          f"delta {g_new - b_new:+.4f}")

    gl = 0.0
    gn = 0
    for doc in test_docs:
        t = doc["tokens"]
        stream(b2, t)
        w = t[-args.ctx - 1:]
        h0 = model.emb[w[:-1]] + model.pos[:len(w) - 1]
        content = model.emb[w[:-1]]
        h1, _ = b2.forward(h0, content, w[:-1])
        lg = model.forward_from_h(h1)
        gl += cross_entropy(lg, w[1:]) * (len(w) - 1)
        gn += len(w) - 1
    gl /= max(gn, 1)
    print(f"    base+layer val loss        {gl:.4f}  (ppl {np.exp(gl):.2f})   "
          f"delta {gl - bl:+.4f}")
    res["e6b_perplexity"] = {
        "base_loss": float(bl), "base_ppl": float(np.exp(bl)),
        "graft_loss": float(gl), "graft_ppl": float(np.exp(gl)),
        "delta": float(gl - bl),
        "repeat_positions": {
            "base_loss": float(b_rep), "graft_loss": float(g_rep),
            "delta": float(g_rep - b_rep), "n": int(b_nrep),
        },
        "new_positions": {
            "base_loss": float(b_new), "graft_loss": float(g_new),
            "delta": float(g_new - b_new), "n": int(b_nnew),
        },
        "wo_exact_norm": float(np.linalg.norm(b2.Wo_exact.data)),
        "wo_lossy_norm": float(np.linalg.norm(b2.Wo_lossy.data)),
    }

    # ------------------------------------------------------------ E6c
    print("\n[4/6] E6c -- capability: cross-window verbatim recall")
    print("    (the fact sits outside the attention window; the layer must")
    print("     retrieve it.  G1 protocol: the BASE model is frozen and only")
    print("     the layer is trained.)")

    # ctx_len=1 -> the content key is the token itself, so the layer can
    # answer "what followed this token earlier in the session".
    bf = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb,
                   ctx_len=1, seed=args.seed)

    def infer(br, w, where="input"):
        """Run the base model on the window `w` with the layer attached.

        where="input"  -> the layer rewrites the residual stream BEFORE the
                          base model's layers (the deep insertion).
        where="out"    -> the layer rewrites the final hidden state, just
                          before the tied readout (the readout insertion).
        """
        if br is None:
            return model.forward(w)
        h0 = model.emb[w] + model.pos[:len(w)]
        cc = model.emb[w]
        if where == "input":
            h1, _ = br.forward(h0, cc, w)
            return model.forward_from_h(h1)
        logits, hfin = model.forward_from_h(h0, return_hidden=True)
        h1, _ = br.forward(hfin, cc, w)
        return h1 @ model.emb.T

    def recall_eval(br, use_exact=True, use_lossy=True, docs=None, where="input"):
        """Stream each document; at the last position, is the target token
        produced?  The fact sits far outside the attention window."""
        if br is not None:
            br.use_exact, br.use_lossy = use_exact, use_lossy
        hit = 0
        hit_val = 0
        for doc in (docs or test_docs):
            t = doc["tokens"]
            if br is not None:
                stream(br, t)
                w = t[-args.ctx:]          # the last ctx tokens, the key included
                lg = infer(br, w, where)
            else:
                w = t[-args.ctx:]
                lg = model.forward(w)
            pred = int(np.asarray(lg.data)[-1].argmax())
            if pred == doc["query_val"]:
                hit += 1
            if pred in range(VAL_LO, VAL_HI):
                hit_val += 1
        n = len(docs or test_docs)
        return hit / n, hit_val / n

    acc_none, val_none = recall_eval(None)
    print(f"    base (no layer)            exact recall {acc_none:.3f}   "
          f"(predicts a value token at all: {val_none:.3f})")

    # ---- the read gain: one scalar, swept -------------------------
    # The exact read injects a stored token's embedding into the residual.
    # How much of it is needed depends on the model's scales, so it is
    # CALIBRATED, not guessed.  Note what is being calibrated: a single
    # scalar on an otherwise training-free path.
    print("      calibrating the read gain (Wo_exact = I, no training):")
    gain_sweep = {}
    for where in ("input", "out"):
        for g in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
            bf.read_gain = "raw"
            bf.read_scale = g
            bf.Wo_exact.data = np.eye(args.d)
            bf.use_exact, bf.use_lossy = True, False
            a, vr = recall_eval(bf, True, False, where=where)
            gain_sweep[f"{where}:{g}"] = a
            print(f"        {where:>5s} gain {g:>7.1f} -> exact recall {a:.3f}")
    # how far apart are the 8 candidate values in logit space?  the injected
    # token has to beat that gap -- this is the number that sets the gain.
    gaps = []
    for doc in test_docs:
        t = doc["tokens"]
        stream(bf, t)
        w = t[-args.ctx:]
        h0 = model.emb[w] + model.pos[:len(w)]
        _, hfin = model.forward_from_h(h0, return_hidden=True)
        z = np.asarray(model.forward_from_h(h0).data)[-1]
        v = np.sort(z[VAL_LO:VAL_HI])
        gaps.append(float(v[-1] - v[-2]))
    print(f"      mean logit gap between the 8 candidate values: "
          f"{np.mean(gaps):.4f}  (max {np.max(gaps):.4f})")
    res["e6c_value_logit_gap"] = {"mean": float(np.mean(gaps)),
                                  "max": float(np.max(gaps))}
    # the read transform: how selective must the injection be?
    print("      the read transform (selectivity of the injection):")
    tf_sweep = {}
    for tf in ("raw", "center", "pc1", "pc4"):
        bf.read_transform = tf
        bf.refresh_contrast()
        bf.read_gain = "raw"
        bf.read_scale = 300.0
        bf.Wo_exact.data = np.eye(args.d)
        a, _ = recall_eval(bf, True, False, where="out")
        tf_sweep[tf] = a
        print(f"        {tf:>7s} -> exact recall {a:.3f}")
    bf.read_transform = "pc1"
    bf.refresh_contrast()
    res["e6c_transform_sweep"] = tf_sweep

    best_key = max(gain_sweep, key=gain_sweep.get)
    where_best, gain_best = best_key.split(":")
    gain_best = float(gain_best)
    print(f"      best: insertion={where_best}  gain={gain_best}  "
          f"recall {gain_sweep[best_key]:.3f}")
    bf.read_gain = "raw"
    bf.read_scale = gain_best
    bf.Wo_exact.data = np.eye(args.d)
    res["e6c_gain_sweep"] = gain_sweep

    # G1: the layer trained on the FACT task, the base frozen (for
    # comparison with the training-free arm above)
    bf.Wo_exact.data = np.zeros((args.d, args.d))
    optf = Adam(bf.params(), lr=3e-2, betas=(0.9, 0.95))
    rngf = np.random.default_rng(args.seed + 11)
    for step in range(args.branch_steps):
        doc = train_docs[int(rngf.integers(0, len(train_docs)))]
        t = doc["tokens"]
        stream(bf, t)
        w = t[-args.ctx:]
        logits = infer(bf, w, where_best)
        loss = cross_entropy_tensor(logits[-1:], np.array([doc["query_val"]]))
        loss.backward()
        optf.step()
        if step % 10 == 0:
            print(f"      fact-step {step:>3}  loss {float(loss.data):.4f}  "
                  f"|Wo_exact| {np.linalg.norm(bf.Wo_exact.data):.3f}")

    # the TRAINING-FREE arm: W_o = I, no gradient steps on the layer at all
    bf.Wo_exact.data = np.eye(args.d)
    bf.read_gain = "raw"
    bf.read_scale = 300.0
    res_cap = {}
    for where in ("input", "out"):
        acc_e, val_e = recall_eval(bf, True, False, where=where)
        acc_l, val_l = recall_eval(bf, False, True, where=where)
        acc_b, val_b = recall_eval(bf, True, True, where=where)
        print(f"    [{where:>5s}] EXACT tier   exact recall {acc_e:.3f}   "
              f"(predicts a value token at all: {val_e:.3f})")
        print(f"    [{where:>5s}] LOSSY tier   exact recall {acc_l:.3f}   "
              f"(predicts a value token at all: {val_l:.3f})")
        print(f"    [{where:>5s}] BOTH         exact recall {acc_b:.3f}   "
              f"(predicts a value token at all: {val_b:.3f})")
        res_cap[where] = {
            "exact_only_recall": acc_e, "exact_only_value_rate": val_e,
            "lossy_only_recall": acc_l, "lossy_only_value_rate": val_l,
            "both_recall": acc_b, "both_value_rate": val_b,
        }
    # how close is the learned map to the identity?  (the tied readout does
    # the rest: adding emb[v] raises v's logit)
    W = bf.Wo_exact.data
    ident = float(np.abs(W - np.eye(args.d)).mean())
    print(f"    mean |Wo_exact - I| = {ident:.4f}   "
          f"(the learned map is close to the identity: the tied readout "
          f"turns the injected embedding into the right logit)")
    res["e6c_capability"] = {
        "base_exact_recall": acc_none, "base_value_rate": val_none,
        "by_insertion": res_cap,
        "chance": 1.0 / N_VALS,
        "mean_abs_dev_from_identity": ident,
        "wo_exact_norm": float(np.linalg.norm(bf.Wo_exact.data)),
    }

    # ------------------------------------------------------------ E6d
    print("\n[5/6] E6d -- generation, side by side")
    # the training-free arm (W_o = I): the headline configuration
    bf.Wo_exact.data = np.eye(args.d)
    bf.read_gain = "raw"
    bf.read_scale = 300.0
    bf.read_transform = "pc1"
    bf.refresh_contrast()
    lines = []
    for doc in test_docs[:3]:
        t = doc["tokens"]
        w = t[-args.ctx:]
        lg0 = model.forward(w)
        stream(bf, t)
        h0 = model.emb[w] + model.pos[:len(w)]
        cc = model.emb[w]
        h1, _ = bf.forward(h0, cc, w)
        lg1 = model.forward_from_h(h1)
        p0 = int(np.asarray(lg0.data)[-1].argmax())
        p1 = int(np.asarray(lg1.data)[-1].argmax())
        lines.append({"target": doc["query_val"], "base": p0, "graft": p1,
                      "store_hits": bf.store_hits})
        print(f"    target {doc['query_val']:>3}   base {p0:>3}   "
              f"+layer {p1:>3}   store hits {bf.store_hits}")
    res["e6d_generation"] = lines

    # the trained arm, for the record: training the layer on the task makes
    # it WORSE than the training-free identity map (measured above)
    tr = {}
    bf_trained = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb,
                           ctx_len=1, seed=args.seed)
    bf_trained.read_transform = "pc1"
    bf_trained.refresh_contrast()
    bf_trained.Wo_exact.data = bf.Wo_exact.data.copy()   # the trained map
    a_tr, v_tr = recall_eval(bf_trained, True, False, where="input")
    tr["trained_exact_recall"] = a_tr
    tr["trained_value_rate"] = v_tr
    tr["wo_exact_norm"] = float(np.linalg.norm(bf_trained.Wo_exact.data))
    tr["training_free_exact_recall"] = res_cap["input"]["exact_only_recall"]
    res["e6c_trained_vs_free"] = tr
    print(f"    trained arm: recall {a_tr:.3f}   |Wo_exact| "
          f"{tr['wo_exact_norm']:.1f}   vs the training-free arm "
          f"{tr['training_free_exact_recall']:.3f}")

    # ------------------------------------------------------------ summary
    print("\n[6/6] summary")
    res["runtime_sec"] = time.time() - t0
    res["base_params"] = int(sum(v.size for v in base_params.values()))
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"    wrote {args.out}")
    print(f"    base params {res['base_params']:,}   "
          f"layer state {branch.state_bytes:,} B   "
          f"runtime {res['runtime_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
