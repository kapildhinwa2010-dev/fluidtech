"""
E14 -- THE REAL THING: the graft on a REAL pretrained GPT-2 (124M).

This is the experiment that was impossible in the development sandbox (no
GPU, no torch, no network).  It runs on CPU, on a genuine pretrained
checkpoint, with the genuine GPT-2 BPE tokenizer (50,257 tokens).

    python experiments/e14_real_gpt2.py --model /path/to/gpt2

THE THREE MEASUREMENTS (the same three as everywhere else)
-----------------------------------------------------------
E14a  ATTACH SAFETY -- are the logits bit-identical after attaching the
      layer to a REAL model?
E14b  PERPLEXITY -- does the layer change GPT-2's own next-token loss, and
      does it beat the kNN-LM baseline (the closest prior art)?
E14c  CAPABILITY -- can the frozen GPT-2 now quote a fact that was planted
      far outside its attention window?

Everything is measured; nothing is assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/gpt2")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus.txt"))
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--read-gain", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "e14_real_gpt2.json"))
    args = ap.parse_args()
    global torch
    try:
        import torch
    except ImportError as error:
        ap.error(f"PyTorch is required; install the optional GPU dependencies: {error}")

    from superfluid.gpt2_model import GPT2Model
    from superfluid.gpt2_tokenizer import GPT2Tokenizer
    from superfluid.torch_graft import SAABranchTorch

    t0 = time.time()
    res = {"model": args.model, "config": vars(args)}

    log("[1/4] loading the real checkpoint ...")
    model = GPT2Model(args.model, dtype=torch.float32)
    tok = GPT2Tokenizer.from_pretrained(args.model)
    n_params = sum(p.numel() for p in [model.wte, model.wpe]) + \
        sum(b["attn_w"].numel() + b["proj_w"].numel() + b["fc_w"].numel() +
            b["mlp_w"].numel() for b in model.blocks)
    log(f"    GPT-2: {model.n_layer} layers, d={model.n_embd}, "
        f"~{n_params / 1e6:.0f}M params, vocab {tok.vocab_size}")

    with open(args.corpus, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tok.encode(text)
    log(f"    the corpus: {len(ids):,} tokens "
        f"({len(text) / max(len(ids), 1):.2f} chars/token)")

    # a train/val split of the TOKEN stream
    n_tr = int(len(ids) * 0.9)
    tr, va = ids[:n_tr], ids[n_tr:]

    def windows(seq, n, ctx):
        out = []
        for i in range(n):
            s = (i * ctx) % max(len(seq) - ctx - 1, 1)
            out.append(seq[s:s + ctx + 1])
        return out

    va_w = windows(va, args.windows, args.ctx)

    # ------------------------------------------------------------- E14a
    log("\n[2/4] E14a -- attach safety on a REAL model")
    emb = torch.nn.Parameter(model.wte, requires_grad=False)
    branch = SAABranchTorch(M=args.M, r=args.r, d=model.n_embd, emb=emb,
                            ctx_len=1, capacity=8192, read_gain=args.read_gain,
                            transform="pc1", use_exact=True, use_lossy=False)
    branch.read_transform = "pc1"
    branch.refresh_contrast()
    w = va_w[0]
    with torch.no_grad():
        base_logits = model.forward(torch.tensor(w[:-1], dtype=torch.long))
        h = model.wte[torch.tensor(w[:-1], dtype=torch.long)] + \
            model.wpe[:len(w) - 1]
        h1, info = branch(h, h, torch.tensor(w[:-1], dtype=torch.long))
        graft_logits = model.forward(torch.tensor(w[:-1], dtype=torch.long),
                                    h_override=h1)
    same = bool(torch.equal(base_logits, graft_logits))
    maxdiff = float((base_logits - graft_logits).abs().max())
    log(f"    bit-identical: {same}   max abs diff {maxdiff:.3e}")
    res["e14a_attach"] = {"bit_identical": same, "max_abs_diff": maxdiff}

    # ------------------------------------------------------------- E14b
    log("\n[3/4] E14b -- the perplexity, and the kNN-LM baseline")

    def build_branch(capacity=8192, ctx_len=1):
        br = SAABranchTorch(M=args.M, r=args.r, d=model.n_embd, emb=emb,
                            ctx_len=ctx_len, capacity=capacity,
                            read_gain=args.read_gain, transform="pc1",
                            use_exact=True, use_lossy=False)
        br.refresh_contrast()
        br.Wo_exact.data = torch.eye(model.n_embd)
        br.reset()
        return br

    # fill the memory with the TRAIN half of the stream (a real session)
    br = build_branch()
    with torch.no_grad():
        for s in range(0, min(len(tr), 20000), args.ctx):
            seg = tr[s:s + args.ctx + 1]
            if len(seg) < 2:
                break
            ids = torch.tensor(seg[:-1], dtype=torch.long)
            h = model.wte[ids] + model.wpe[:len(seg) - 1]
            # the routing content is the POSITION-FREE embedding, so the same
            # context always lands on the same anchor
            br.forward(h, model.wte[ids], ids)
    br.readonly = True          # the evaluation must not write
    log(f"    the memory now holds {len(br.store)} contexts "
        f"({len(br.anchor_store)} anchors), filled from the TRAIN half")

    def evaluate(mode, param):
        """ONE forward per window; the kNN-LM interpolation and the gated
        override are then computed from the SAME retrieval -- an apples to
        apples comparison."""
        tot = cnt = 0.0
        acc = 0
        with torch.no_grad():
            for w in va_w:
                x = torch.tensor(w[:-1], dtype=torch.long)
                logits = model.forward(x)
                h = model.wte[x] + model.wpe[:x.shape[0]]
                h1, info = br.forward(h, model.wte[x], x)
                ret = info["retrieved"]
                if mode == "knn":
                    p = torch.softmax(logits.float(), dim=-1)
                    lam = float(param)
                    pknn = torch.zeros_like(p)
                    rows = torch.nonzero(ret >= 0, as_tuple=False).squeeze(-1)
                    if rows.numel():
                        pknn[rows, ret[rows]] = 1.0
                    lg = torch.log(torch.clamp(lam * p + (1 - lam) * pknn,
                                               min=1e-30))
                else:
                    lg = br.gated_logits(logits, info, gain=float(param))
                lsm = torch.log_softmax(lg.float(), dim=-1)
                tgt = torch.tensor(w[1:], dtype=torch.long)
                nll = -lsm.gather(1, tgt.unsqueeze(1)).squeeze(1)
                tot += float(nll.sum())
                cnt += nll.numel()
                acc += int((lg.argmax(-1) == tgt).sum())
        return tot / cnt, acc / cnt

    base_l, base_acc = evaluate("knn", 1.0)     # lam = 1 -> the plain model
    log(f"    the base GPT-2 alone        : {base_l:.4f} "
        f"(ppl {math.exp(base_l):.2f})  token accuracy {base_acc:.4f}")
    res["base"] = {"loss": float(base_l), "ppl": float(math.exp(base_l)),
                   "token_accuracy": float(base_acc)}

    knn = {}
    for lam in (0.95, 0.9, 0.8):
        l, a = evaluate("knn", lam)
        knn[str(lam)] = {"loss": float(l), "delta": float(l - base_l),
                         "token_accuracy": float(a)}
        log(f"    kNN-LM, lam={lam:<5}          : {l:.4f} "
            f"(ppl {math.exp(l):.2f})  delta {l - base_l:+.4f}  acc {a:.4f}")
    res["knn_lm"] = knn

    gated = {}
    for g in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        l, a = evaluate("gate", g)
        gated[str(g)] = {"loss": float(l), "delta": float(l - base_l),
                         "token_accuracy": float(a)}
        log(f"    the LAYER (gated), gain={g:<4} : {l:.4f} "
            f"(ppl {math.exp(l):.2f})  delta {l - base_l:+.4f}  acc {a:.4f}")
    res["gated_layer"] = gated

    # ------------------------------------------------------------- E14c
    log("\n[4/4] E14c -- the capability on a REAL model")
    # a fact planted far outside the attention window
    # A key token that DOES NOT OCCUR in the corpus at all.  Two reasons:
    # a common token's store entry holds the MAJORITY continuation, and any
    # token that occurs in the corpus has already claimed its store slot.
    used = set(ids)
    key_tok = next(t for t in range(50256, 1000, -1) if t not in used)
    val_tok = next(t for t in range(1000, 50000)
                   if t not in used and t != key_tok)
    log(f"    the key token {key_tok} and the value token {val_tok} do not "
        f"occur anywhere in the corpus")
    filler = list(tr[:args.ctx * 12])
    # the fact is planted early; the QUERY ends with the key, so the read
    # looks up the key's context and finds the value
    planted = filler + [key_tok, val_tok] + filler + [key_tok]
    query = planted[-args.ctx:]
    br2 = build_branch()
    with torch.no_grad():
        for s in range(0, len(planted) - 1, args.ctx):
            seg = planted[s:s + args.ctx + 1]
            ids = torch.tensor(seg[:-1], dtype=torch.long)
            h = model.wte[ids] + model.wpe[:len(seg) - 1]
            br2.forward(h, model.wte[ids], ids)
        # the model and the branch BOTH see the full query (the key token
        # included): the model's prediction at the last position is then the
        # token AFTER the key, and the branch's read at the last position is
        # the key's context.
        x = torch.tensor(query, dtype=torch.long)
        logits = model.forward(x)
        h = model.wte[x] + model.wpe[:x.shape[0]]
        h1, info = br2.forward(h, model.wte[x], x)
        lg = br2.gated_logits(logits, info, gain=5.0)
        base_pred = int(logits[-1].argmax())
        graft_pred = int(lg[-1].argmax())
        p = torch.softmax(logits.float(), dim=-1)
        u = float(1.0 - p.max(dim=-1).values[-1])
        ret = int(info["retrieved"][-1]) if info["retrieved"].numel() else -1
        log(f"    the retrieval at the query: {ret} (the fact is {val_tok}), "
            f"the base model's uncertainty u = {u:.4f}, "
            f"its top-1 probability {float(p.max(dim=-1).values[-1]):.4f}")
    log(f"    the fact ({val_tok}) was planted {len(planted) - args.ctx} tokens "
        f"before the query")
    log(f"    the base GPT-2 predicts {base_pred}, the layer predicts "
        f"{graft_pred}  (the fact is {val_tok})")
    res["e14c_capability"] = {
        "retrieved_at_query": ret, "base_uncertainty": u,
        "base_prediction": base_pred, "graft_prediction": graft_pred,
        "target": int(val_tok),
        "base_correct": bool(base_pred == val_tok),
        "graft_correct": bool(graft_pred == val_tok),
        "planted_tokens_back": len(planted) - args.ctx,
    }

    res["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"\n    wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
