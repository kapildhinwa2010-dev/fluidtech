"""
E7 -- fixing the diffusion tier's training signal.

THE PROBLEM (measured in E6, `results/e6_real.json`)
-----------------------------------------------------
Grafted onto a frozen base model, the lossy (diffusion) tier made the model
WORSE: +0.809 nats (ppl 13.33 -> 29.94) on real English, while the exact
tier next to it was fine (-0.030).  The layer's own training loss ROSE while
it trained (2.4306 -> 3.4387): the optimisation was diverging, not merely
failing to help.

THE DIAGNOSIS
-------------
The persistent state is a numpy accumulator OUTSIDE the autodiff graph (by
design: it is a memory, not an activation).  So the read projection is
trained against a MOVING, NON-DIFFERENTIABLE TARGET: the state changes every
step and no gradient says how the read should be shaped.  The optimiser
takes a step that is right for this state and wrong for the next one.

THE TWO FIXES, BOTH MEASURED HERE
---------------------------------
FIX 1 -- LAYER-LOCAL OBJECTIVE.  The layer gets its own next-token
prediction head (Wp) and its own loss: predict the token that followed,
from its own state.  The base model is never in the loss loop, so the
layer's parameters cannot be dragged around by the base model's gradient.

FIX 2 -- DIFFERENTIABLE STATE.  The state is REBUILT from a bounded history
INSIDE the graph on every call (truncated BPTT).  The diffusion is linear,
so the rebuilt state has the SAME VALUES as the persistent accumulator; only
the gradient path is new.  Now the write projection is trainable too.

Plus a third ingredient, needed for FIX 1 to be useful: the layer's own
prediction is injected as a SOFT PRIOR (projected through the embedding
table), not as the raw diffused read.

Everything is CPU-only and pure NumPy.
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
from superfluid.graft import SAABranch                                        # noqa: E402
from superfluid.optim import Adam                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load_chars(max_chars=120_000):
    with open(os.path.join(ROOT, "corpus.txt"), "r", encoding="utf-8",
              errors="ignore") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text[:max_chars]], dtype=np.int64)
    return ids, len(chars)


def train_base(ids, V, d=64, layers=2, ctx=128, steps=400, batch=16, lr=3e-3,
               seed=0):
    rng = np.random.default_rng(seed)
    model = TinyTransformer(V, d=d, n_layers=layers, n_ctx=ctx, seed=0)
    opt = Adam(model.params(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    n = len(ids)
    for step in range(steps):
        st = rng.integers(0, n - ctx - 1, size=batch)
        x = np.stack([ids[s:s + ctx] for s in st])
        y = np.stack([ids[s + 1:s + ctx + 1] for s in st])
        for p in model.params().values():
            p.grad = np.zeros_like(p.data)
        loss = cross_entropy_tensor(model.forward_batch(x), y.reshape(-1))
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"    base step {step:>4}/{steps}  loss {float(loss.data):.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--layer-steps", type=int, default=60)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--hist-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e7_lossy_fix.json"))
    args = ap.parse_args()

    t0 = time.time()
    print("[1/4] corpus + base model ...")
    ids, V = load_chars()
    n_tr = int(len(ids) * 0.9)
    tr, va = ids[:n_tr], ids[n_tr:]
    print(f"    {len(ids):,} chars, vocab {V}")
    model = train_base(tr, V, d=args.d, ctx=args.ctx, steps=args.steps)

    def base_loss():
        tot, cnt = 0.0, 0
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            if len(w) < 2:
                continue
            lg = model.forward(w[:-1])
            tot += cross_entropy(lg, w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
        return tot / cnt

    bl = base_loss()
    print(f"    base val loss {bl:.4f}  (ppl {np.exp(bl):.2f})")

    def attach(br, w):
        h0 = model.emb[w[:-1]] + model.pos[:len(w) - 1]
        cc = model.emb[w[:-1]]
        h1, info = br.forward(h0, cc, w[:-1])
        return model.forward_from_h(h1), info

    def eval_with(br):
        br.reset()
        tot, cnt = 0.0, 0
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            lg, _ = attach(br, w)
            tot += cross_entropy(lg, w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
        return tot / cnt

    # --------------------------------------------------------------- the runs
    out = {}

    print("\n[2/4] CONTROL -- train the layer through the BASE MODEL's loss "
          "(the broken way)")
    br = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb, ctx_len=1,
                   state_mode="persistent", seed=0)
    opt = Adam(br.params(), lr=args.lr, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)
    hist = []
    for step in range(args.layer_steps):
        s = int(rng.integers(0, len(tr) - args.ctx - 1))
        w = tr[s:s + args.ctx + 1]
        lg, _ = attach(br, w)
        loss = cross_entropy_tensor(lg, w[1:])
        loss.backward()
        opt.step()
        hist.append(float(loss.data))
        if step % 20 == 0:
            print(f"      step {step:>3}  base-loss {float(loss.data):.4f}")
    gl = eval_with(br)
    out["control_base_loss"] = {
        "base_loss": float(gl), "delta": float(gl - bl),
        "train_first": hist[0], "train_last": hist[-1],
        "converged": bool(hist[-1] < hist[0]),
    }
    print(f"    -> val loss {gl:.4f}  delta {gl - bl:+.4f}   "
          f"train {hist[0]:.4f} -> {hist[-1]:.4f}  "
          f"{'CONVERGED' if hist[-1] < hist[0] else 'DIVERGED'}")

    for mode, inject, label in (("persistent", "state", "FIX1_persistent"),
                                ("rebuild", "state", "FIX2_rebuild"),
                                ("rebuild", "prior", "FIX2_rebuild_prior")):
        print(f"\n[3/4] {label} -- layer-local objective "
              f"(state_mode={mode}, inject={inject})")
        br = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb, ctx_len=1,
                       state_mode=mode, hist_len=args.hist_len,
                       inject_mode=inject, seed=0)
        if inject == "prior":
            br.Wo_lossy.data = np.eye(args.d)     # the injection map stays FIXED
        opt = Adam(br.params(), lr=args.lr, betas=(0.9, 0.95))
        rng = np.random.default_rng(0)
        hist = []
        for step in range(args.layer_steps):
            s = int(rng.integers(0, len(tr) - args.ctx - 1))
            w = tr[s:s + args.ctx + 1]
            # --- the LAYER-LOCAL loss: the base model is not in this loop
            h0 = model.emb[w[:-1]] + model.pos[:len(w) - 1]
            cc = model.emb[w[:-1]]
            _, info = br.forward(h0, cc, w[:-1])
            z = br.layer_logits(info)
            if z is None:
                break
            loss = cross_entropy_tensor(z, w[1:])
            loss.backward()
            opt.step()
            hist.append(float(loss.data))
            if step % 20 == 0:
                print(f"      step {step:>3}  layer-loss {float(loss.data):.4f}  "
                      f"|Wp| {np.linalg.norm(br.Wp.data):.3f}")
        gl = eval_with(br)
        out[label] = {
            "base_loss": float(gl), "delta": float(gl - bl),
            "layer_train_first": hist[0], "layer_train_last": hist[-1],
            "converged": bool(hist[-1] < hist[0]),
            "wo_lossy_norm": float(np.linalg.norm(br.Wo_lossy.data)),
            "state_mode": mode, "inject_mode": inject,
        }
        print(f"    -> val loss {gl:.4f}  delta {gl - bl:+.4f}   "
              f"layer loss {hist[0]:.4f} -> {hist[-1]:.4f}  "
              f"{'CONVERGED' if hist[-1] < hist[0] else 'DIVERGED'}")

    # ---- how much of the prior can the base model afford? ---------------
    print("\n[3b/4] the prior-injection gain sweep (the layer is already "
          "trained; only the injection strength changes)")
    brp = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb, ctx_len=1,
                    state_mode="rebuild", hist_len=args.hist_len,
                    inject_mode="prior", seed=0)
    optp = Adam(brp.params(), lr=args.lr, betas=(0.9, 0.95))
    rngp = np.random.default_rng(0)
    hp = []
    for step in range(args.layer_steps):
        s2 = int(rngp.integers(0, len(tr) - args.ctx - 1))
        w2 = tr[s2:s2 + args.ctx + 1]
        h0 = model.emb[w2[:-1]] + model.pos[:len(w2) - 1]
        _, info = brp.forward(h0, model.emb[w2[:-1]], w2[:-1])
        z = brp.layer_logits(info)
        loss = cross_entropy_tensor(z, w2[1:])
        loss.backward()
        optp.step()
        hp.append(float(loss.data))
    sweep = {}
    for g in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        brp.Wo_lossy.data = g * np.eye(args.d)
        gl2 = eval_with(brp)
        sweep[str(g)] = {"loss": float(gl2), "delta": float(gl2 - bl)}
        print(f"    gain {g:>6.3f} -> val loss {gl2:.4f}  delta {gl2 - bl:+.4f}")
    out["prior_gain_sweep"] = sweep
    out["prior_layer_loss"] = {"first": hp[0], "last": hp[-1]}

    out["base"] = {"base_loss": float(bl), "ppl": float(np.exp(bl))}
    out["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"    wrote {args.out}")

    print("\n[4/4] summary")
    print(f"    {'configuration':<22} {'val loss':>9} {'ppl':>8} {'delta':>9}  "
          f"{'train loss':>22}")
    print(f"    {'base (frozen)':<22} {bl:>9.4f} {np.exp(bl):>8.2f} {'--':>9}  "
          f"{'--':>22}")
    for k, v in out.items():
        if k in ("base", "runtime_sec", "prior_gain_sweep", "prior_layer_loss"):
            continue
        t1 = v.get("train_first", v.get("layer_train_first"))
        t2 = v.get("train_last", v.get("layer_train_last"))
        print(f"    {k:<22} {v['base_loss']:>9.4f} {np.exp(v['base_loss']):>8.2f} "
              f"{v['delta']:>+9.4f}  {t1:>10.4f} -> {t2:<10.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
