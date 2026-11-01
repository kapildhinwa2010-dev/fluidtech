"""
E6-real -- does the graft's perplexity gain survive on REAL text?

The synthetic corpus in `e6_graft.py` is a Markov chain with 70 % self-loops,
which is the friendliest possible case for a content-keyed store: the token
that followed a context is very often the token itself.  So the measured
perplexity gain there is an UPPER BOUND, not a typical case.

This experiment repeats the measurement on real English (the local
`corpus.txt`, Alice in Wonderland, character level), with the same
architecture and the same protocol:

    base model  vs  base model + the memory layer (base frozen)

and it splits the loss into "the context occurred earlier in the document"
vs "the context is new", so the source of any gain is visible.
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


def load_chars(max_chars=120_000, tokenizer="char", bpe_path=None):
    """Load the corpus as token ids.

    tokenizer="char": the character level vocabulary (76 symbols) that every
    earlier experiment used.  It is convenient but MAXIMALLY REDUNDANT, which
    flatters a content-keyed store.

    tokenizer="bpe": a byte-level BPE subword vocabulary trained locally by
    `superfluid/bpe.py` (no downloads). This is the realistic granularity: the
    same text needs far fewer tokens, and the redundancy structure that the
    character level corpus has is gone.
    """
    path = os.path.join(ROOT, "corpus.txt")
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()[:max_chars]
    if tokenizer == "bpe":
        from superfluid.bpe import BPETokenizer
        bp = bpe_path or os.path.join(ROOT, "prior", "bpe_512.json")
        tok = BPETokenizer.load(bp)
        ids = np.array(tok.encode(text), dtype=np.int64)
        return ids, tok.vocab_size
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
    ap.add_argument("--branch-steps", type=int, default=40)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--ctx-len", type=int, default=12)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--max-chars", type=int, default=120_000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e6_real.json"))
    args = ap.parse_args()

    t0 = time.time()
    print("[1/4] real corpus (character level) ...")
    ids, V = load_chars(max_chars=args.max_chars, tokenizer=args.tokenizer,
                        bpe_path=args.bpe)
    n_tr = int(len(ids) * 0.9)
    tr, va = ids[:n_tr], ids[n_tr:]
    print(f"    {len(ids):,} tokens, vocab {V}, train {len(tr):,} / val {len(va):,} "
          f"({args.tokenizer} level)")

    print("[2/4] base model ...")
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

    def evaluate(br):
        if br is None:
            tot, cnt = 0.0, 0
            for s in range(0, len(va) - 1, args.ctx):
                w = va[s:s + args.ctx + 1]
                if len(w) < 2:
                    continue
                lg = model.forward(w[:-1])
                tot += cross_entropy(lg, w[1:]) * (len(w) - 1)
                cnt += len(w) - 1
            return tot / cnt
        br.reset()
        tot, cnt = 0.0, 0
        for s in range(0, len(va) - 1, args.ctx):
            w = va[s:s + args.ctx + 1]
            h0 = model.emb[w[:-1]] + model.pos[:len(w) - 1]
            cc = model.emb[w[:-1]]
            h1, _ = br.forward(h0, cc, w[:-1])
            lg = model.forward_from_h(h1)
            tot += cross_entropy(lg, w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
        return tot / cnt

    def train_layer(br, lr, steps):
        opt = Adam(br.params(), lr=lr, betas=(0.9, 0.95))
        rng = np.random.default_rng(0)
        hist = []
        for step in range(steps):
            s = int(rng.integers(0, len(tr) - args.ctx - 1))
            w = tr[s:s + args.ctx + 1]
            h0 = model.emb[w[:-1]] + model.pos[:len(w) - 1]
            cc = model.emb[w[:-1]]
            h1, _ = br.forward(h0, cc, w[:-1])
            loss = cross_entropy_tensor(model.forward_from_h(h1), w[1:])
            loss.backward()
            opt.step()
            hist.append(float(loss.data))
            if step % 10 == 0:
                print(f"      branch step {step:>3}  loss {float(loss.data):.4f}  "
                      f"|Wo_e| {np.linalg.norm(br.Wo_exact.data):.2f}  "
                      f"|Wo_l| {np.linalg.norm(br.Wo_lossy.data):.2f}")
        return hist

    print("[3/4] graft: the layer is trained ALONE, the base stays frozen ...")
    variants = {}
    for name, use_e, use_l, lr in (("lossy_only", False, True, 1e-3),
                                   ("lossy_only_lr1e-4", False, True, 1e-4),
                                   ("lossy_only_lr1e-5", False, True, 1e-5),
                                   ("exact_only", True, False, 1e-3),
                                   ("exact_only_lr1e-4", True, False, 1e-4),
                                   ("both", True, True, 1e-3),
                                   ("both_lr3e-3", True, True, 3e-3)):
        br = SAABranch(M=args.M, r=args.r, d=args.d, emb=model.emb,
                       ctx_len=args.ctx_len, seed=0)
        br.use_exact, br.use_lossy = use_e, use_l
        print(f"    -- variant {name} (lr {lr}) --")
        hist = train_layer(br, lr, args.branch_steps)
        gl = evaluate(br)
        variants[name] = {
            "loss": float(gl), "ppl": float(np.exp(gl)),
            "delta_vs_base": float(gl - bl),
            "wo_exact_norm": float(np.linalg.norm(br.Wo_exact.data)),
            "wo_lossy_norm": float(np.linalg.norm(br.Wo_lossy.data)),
            "train_loss_first": hist[0], "train_loss_last": hist[-1],
        }
        print(f"       -> val loss {gl:.4f} (ppl {np.exp(gl):.2f})  "
              f"delta {gl - bl:+.4f}   train loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    gl = variants["both_lr3e-3"]["loss"]

    res = {
        "variants": variants,
        "tokenizer": args.tokenizer,
        "corpus": f"corpus.txt (Alice in Wonderland, {args.tokenizer} level)",
        "vocab": V, "chars": int(len(ids)),
        "base_loss": float(bl), "base_ppl": float(np.exp(bl)),
        "graft_loss": float(gl), "graft_ppl": float(np.exp(gl)),
        "delta": float(gl - bl),
        "wo_exact_norm": float(np.linalg.norm(br.Wo_exact.data)),
        "wo_lossy_norm": float(np.linalg.norm(br.Wo_lossy.data)),
        "state_bytes": br.state_bytes,
        "runtime_sec": time.time() - t0,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"    wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
