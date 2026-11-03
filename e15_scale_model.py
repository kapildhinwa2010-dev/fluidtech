#!/usr/bin/env python3
"""
experiments/e15_scale_model.py -- E15: the graft on a BIGGER real model.

E14 ran the protocol on GPT-2 (124M).  This runs the SAME protocol on
Qwen2.5-0.5B (494M, a different architecture: RoPE + RMSNorm + GQA +
SwiGLU, tied embeddings), with a forward pass written from scratch in
`superfluid/qwen2_model.py` (no `transformers` at inference time).

WHY THIS MATTERS
----------------
E14's honest negative (N11) is that on a strong model the classic kNN-LM
interpolation beat the gated layer by 8x, because interpolation is
BOUNDED (a wrong retrieval can never dominate) while the gate is not.
A bigger model is the next rung of the scale ladder: if the layer's
advantage keeps shrinking with model quality, that is the honest
conclusion -- and it is what this measures.

WHAT IS MEASURED
----------------
  E15a  the attach is a provable no-op (bit-identical logits)
  E15b  the perplexity: the base model, the kNN-LM baseline, and the
        gated layer -- ONE forward per window, then every variant is
        computed from the SAME retrieval (an apples-to-apples comparison)
  E15c  the capability: a fact planted ~3,000 tokens back, far outside
        the attention window

    python experiments/e15_scale_model.py --model /workspace/models/qwen05b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

torch = None  # optional: imported lazily below


def log(*a):
    print(*a, flush=True)


def _score(acc, lg, tgt):
    """Accumulate the NLL and the token accuracy of one variant."""
    lsm = torch.log_softmax(lg.float(), dim=-1)
    nll = -lsm.gather(1, tgt.unsqueeze(1)).squeeze(1)
    acc[0] += float(nll.sum())
    acc[1] += nll.numel()
    acc[2] += int((lg.argmax(-1) == tgt).sum())


def _need_torch():
    """torch is an OPTIONAL dependency: the numpy package must work without
    it.  Import it lazily so that `--help` works on a torch-free machine
    (which is exactly what the CI does)."""
    global torch
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except ImportError:
            raise SystemExit(
                "this experiment needs torch: "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu")
    return torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/qwen05b")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus.txt"))
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--read-gain", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e15_scale_model.json"))
    ap.add_argument("--only-capability", action="store_true",
                    help="skip the perplexity sweep (fast check)")
    args = ap.parse_args()
    _need_torch()
    t0 = time.time()
    res = {"model": args.model, "config": vars(args)}

    from superfluid.qwen2_model import Qwen2Model
    from superfluid.gpt2_tokenizer import GPT2Tokenizer
    from superfluid.torch_graft import SAABranchTorch

    log("[1/4] loading the real checkpoint ...")
    model = Qwen2Model(args.model, dtype=torch.bfloat16)
    tok = GPT2Tokenizer.from_pretrained(args.model)
    n_params = model.wte.numel()
    for L in model.layers:
        for k in ("q", "k", "v", "o", "gate", "up", "down"):
            n_params += L[k].numel()
    log(f"    Qwen2: {model.n_layer} layers, d={model.n_embd}, "
        f"~{n_params / 1e6:.0f}M params, vocab {tok.vocab_size}")

    with open(args.corpus, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tok.encode(text)
    log(f"    the corpus: {len(ids):,} tokens "
        f"({len(text) / max(len(ids), 1):.2f} chars/token)")

    n_tr = int(len(ids) * 0.9)
    tr, va = ids[:n_tr], ids[n_tr:]

    def windows(seq, n, ctx):
        out = []
        for i in range(n):
            s = (i * ctx) % max(len(seq) - ctx - 1, 1)
            out.append(seq[s:s + ctx + 1])
        return out

    va_w = windows(va, args.windows, args.ctx)

    # ------------------------------------------------------------- E15a
    log("\n[2/4] E15a -- attach safety on a REAL model")
    emb = torch.nn.Parameter(model.wte, requires_grad=False)
    branch = SAABranchTorch(M=args.M, r=args.r, d=model.n_embd, emb=emb,
                            ctx_len=1, capacity=8192, read_gain=args.read_gain,
                            transform="pc1", use_exact=True, use_lossy=False)
    branch.refresh_contrast()
    w = va_w[0]
    if not args.only_capability:
        with torch.no_grad():
            x = torch.tensor(w[:-1], dtype=torch.long)
            base_logits = model.forward(x)
            # RoPE: no position embedding.  The branch runs in fp32, the
            # checkpoint in bf16 -- so the residual stream is cast at the seam.
            h = model.wte[x].float()
            h1, info = branch(h, model.wte[x].float(), x)
            graft_logits = model.forward(x, h_override=h1.to(model.dtype))
            same = bool(torch.equal(base_logits, graft_logits))
            maxdiff = float((base_logits - graft_logits).abs().max())
            del base_logits, graft_logits
        log(f"    bit-identical: {same}   max abs diff {maxdiff:.3e}")
        res["e15a_attach"] = {"bit_identical": same, "max_abs_diff": maxdiff}

    # ------------------------------------------------------------- E15b
    log("\n[3/4] E15b -- the perplexity, and the kNN-LM baseline")

    def build_branch(capacity=8192, ctx_len=1):
        br = SAABranchTorch(M=args.M, r=args.r, d=model.n_embd, emb=emb,
                            ctx_len=ctx_len, capacity=capacity,
                            read_gain=args.read_gain, transform="pc1",
                            use_exact=True, use_lossy=False)
        br.refresh_contrast()
        br.Wo_exact.data = torch.eye(model.n_embd)
        br.reset()
        return br

    br = build_branch()
    fill_tokens = 5000 if args.only_capability else 20000
    with torch.no_grad():
        for s in range(0, min(len(tr), fill_tokens), args.ctx):
            seg = tr[s:s + args.ctx + 1]
            if len(seg) < 2:
                break
            x = torch.tensor(seg[:-1], dtype=torch.long)
            h = model.wte[x].float()
            br.forward(h, model.wte[x].float(), x)
    br.readonly = True
    log(f"    the memory now holds {len(br.store)} contexts "
        f"({len(br.anchor_store)} anchors), filled from the TRAIN half")

    # ONE forward per window; EVERY variant (the base, the kNN-LM, and
    # every gate gain) is then computed from the SAME retrieval -- an
    # apples-to-apples comparison.  The variants are scored inside the
    # window loop and the logits are then dropped: the vocabulary is
    # 152k wide, so caching 40 windows of logits would need 3 GB.
    LAMBDAS = (0.95, 0.9, 0.8)
    GAINS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    if args.only_capability:
        LAMBDAS = ()
        GAINS = ()
    tot = {("knn", 1.0): [0.0, 0.0, 0],
           **{("knn", l): [0.0, 0.0, 0] for l in LAMBDAS},
           **{("gate", g): [0.0, 0.0, 0] for g in GAINS}}
    log("    running the forwards ...")
    with torch.no_grad():
        for wi, w in enumerate(va_w if not args.only_capability else []):
            x = torch.tensor(w[:-1], dtype=torch.long)
            logits = model.forward(x).float()
            h = model.wte[x].float()
            h1, info = br.forward(h, model.wte[x].float(), x)
            ret = info["retrieved"]
            tgt = torch.tensor(w[1:], dtype=torch.long)
            for lam in (1.0,) + LAMBDAS:
                p = torch.softmax(logits, dim=-1)
                pknn = torch.zeros_like(p)
                rows = torch.nonzero(ret >= 0, as_tuple=False).squeeze(-1)
                if rows.numel():
                    pknn[rows, ret[rows]] = 1.0
                lg = torch.log(torch.clamp(lam * p + (1 - lam) * pknn, min=1e-30))
                _score(tot[("knn", lam)], lg, tgt)
                del p, pknn, lg
            for g in GAINS:
                lg = br.gated_logits(logits, {"retrieved": ret}, gain=g)
                _score(tot[("gate", g)], lg, tgt)
                del lg
            del logits
            if (wi + 1) % 10 == 0:
                log(f"      {wi + 1}/{len(va_w)} windows")

    def report(key):
        t, c, a = tot[key]
        if c == 0:
            return float("nan"), float("nan")
        return t / c, a / c

    if args.only_capability:
        log("    (the perplexity sweep is skipped)")
        base_l, base_acc = report(("knn", 1.0))
        res["base"] = {"loss": float(base_l), "ppl": float(math.exp(base_l)),
                       "token_accuracy": float(base_acc)}
        res["knn_lm"] = {}
        res["gated_layer"] = {}
        base_l = float("nan")
        base_acc = float("nan")
    else:
        base_l, base_acc = report(("knn", 1.0))
    log(f"    the base model alone        : {base_l:.4f} "
        f"(ppl {math.exp(base_l):.2f})  token accuracy {base_acc:.4f}")
    res["base"] = {"loss": float(base_l), "ppl": float(math.exp(base_l)),
                   "token_accuracy": float(base_acc)}

    knn = {}
    for lam in LAMBDAS:
        l, a = report(("knn", lam))
        knn[str(lam)] = {"loss": float(l), "delta": float(l - base_l),
                         "token_accuracy": float(a)}
        log(f"    kNN-LM, lam={lam:<5}          : {l:.4f} "
            f"(ppl {math.exp(l):.2f})  delta {l - base_l:+.4f}  acc {a:.4f}")
    res["knn_lm"] = knn

    gated = {}
    for g in GAINS:
        l, a = report(("gate", g))
        gated[str(g)] = {"loss": float(l), "delta": float(l - base_l),
                         "token_accuracy": float(a)}
        log(f"    the LAYER (gated), gain={g:<4} : {l:.4f} "
            f"(ppl {math.exp(l):.2f})  delta {l - base_l:+.4f}  acc {a:.4f}")
    res["gated_layer"] = gated

    # ------------------------------------------------------------- E15c
    log("\n[4/4] E15c -- the capability on a REAL model")
    # A fact planted far outside the attention window.  Two tokens that do
    # NOT occur anywhere in the corpus: a common token's store entry holds
    # the MAJORITY continuation, and a token that occurs in the corpus has
    # already claimed its store slot.
    used = set(ids)
    key_tok = next(t for t in range(150000, 1000, -1) if t not in used)
    val_tok = next(t for t in range(1000, 150000)
                   if t not in used and t != key_tok)
    log(f"    the key token {key_tok} and the value token {val_tok} occur "
        f"nowhere in the corpus")
    filler = list(tr[:args.ctx * 12])
    mid = args.ctx // 2          # the pair lands INSIDE one window
    planted = filler[:mid] + [key_tok, val_tok] + filler[mid:] + [key_tok]
    query = planted[-args.ctx:]
    br2 = build_branch()
    with torch.no_grad():
        for s in range(0, len(planted) - 1, args.ctx):
            seg = planted[s:s + args.ctx + 1]
            x = torch.tensor(seg[:-1], dtype=torch.long)
            br2.forward(model.wte[x].float(), model.wte[x].float(), x)
        # the model and the branch BOTH see the full query (the key token
        # included): the model's prediction at the last position is the
        # token AFTER the key, and the branch's read at that position is
        # the key's stored continuation.
        x = torch.tensor(query, dtype=torch.long)
        logits = model.forward(x, keep_last=True)
        h = model.wte[x].float()
        h1, info = br2.forward(h, model.wte[x].float(), x)
        lg = br2.gated_logits(logits, {"retrieved": info["retrieved"][-1:]},
                              gain=8.0)
        p = torch.softmax(logits.float(), dim=-1)
        u = float(-torch.log(p.max(dim=-1).values[-1]))
        ret = int(info["retrieved"][-1]) if info["retrieved"].numel() else -1
    base_pred = int(logits[-1].argmax())
    graft_pred = int(lg[-1].argmax())
    dist = len(planted) - args.ctx
    log(f"    the retrieval at the query: {ret} (the fact is {val_tok}), "
        f"the base model's uncertainty u = {u:.4f} nats")
    log(f"    the fact was planted {dist} tokens before the query")
    log(f"    the base model predicts {base_pred}, the LAYER predicts "
        f"{graft_pred}  (the fact is {val_tok})")
    res["e15c_capability"] = {
        "key_token": int(key_tok),
        "value_token": int(val_tok),
        "fact_distance_tokens": int(dist),
        "retrieved": int(ret),
        "uncertainty_nats": float(u),
        "base_top": base_pred,
        "layer_top": graft_pred,
        "correct": bool(graft_pred == val_tok),
    }

    res["seconds"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    log(f"\nwrote {args.out}  ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
