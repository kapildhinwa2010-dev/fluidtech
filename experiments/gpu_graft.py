#!/usr/bin/env python3
"""
experiments/gpu_graft.py -- the GPU-scale version of the graft experiment.

WHAT IT DOES
------------
Takes a REAL pretrained LLM (default: Qwen2.5-0.5B-Instruct, or any causal
LM on HuggingFace), and measures the three things the CPU experiments
measured, at a realistic scale and with a real tokenizer:

    E6a  attach safety    -- are the logits bit-identical after attaching?
    E6b  perplexity       -- does the layer change the base model's own loss?
    E6c  capability       -- cross-window verbatim recall of a planted fact

plus the number that motivates the whole project:

    the memory footprint of the layer vs. the KV cache of the same context.

STATUS: **NOT EXECUTED IN THE DEVELOPMENT SANDBOX.**
There is no GPU, no torch and 3 GB of RAM there, so this script is written
to be read, reviewed and run on a GPU box.  Every number it prints is
computed at run time; nothing is hard-coded.

HOW TO RUN IT
-------------
    pip install torch transformers accelerate
    python experiments/gpu_graft.py --model Qwen/Qwen2.5-0.5B-Instruct
    python experiments/gpu_graft.py --model meta-llama/Llama-3.2-1B --ctx 32768

COST (measured on live spot prices, H100 PCIe ~ $0.67/hr):
    the whole thing is ~200 forward passes of a 0.5B model => a few minutes
    on one GPU.  A 1B model with a 32K context: still minutes.  Total budget
    for the decisive run: **$2-3**, including the download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def log(*a):
    print(*a, flush=True)


def load_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def perplexity(model, tok, text: str, ctx: int, device) -> float:
    """The base model's next-token loss on `text` (no layer attached)."""
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    nll, n = 0.0, 0
    for s in range(0, min(len(ids[0]) - 1, 20 * ctx), ctx):
        chunk = ids[:, s:s + ctx]
        if chunk.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(chunk).logits
        tgt = chunk[:, 1:]
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        g = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        nll += float(-g.sum())
        n += g.numel()
    return nll / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus.txt"))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--read-gain", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "gpu_graft.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="check the environment and the model id, run nothing")
    args = ap.parse_args()
    global torch
    try:
        import torch
    except ImportError as error:
        ap.error(f"PyTorch is required; install the optional GPU dependencies: {error}")

    from superfluid.torch_graft import SAABranchTorch

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}   torch {torch.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    if args.dry_run:
        log(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params")
        log("dry run: nothing measured")
        return 0

    text = load_text(args.corpus)
    res = {"model": args.model, "ctx": args.ctx, "device": device}

    # ---------------------------------------------------------------- E6a
    log("\n[E6a] attach safety -- are the logits bit-identical?")
    emb = model.get_input_embeddings()
    d = emb.weight.shape[1]
    branch = SAABranchTorch(M=args.M, r=args.r, d=d, emb=emb, ctx_len=1,
                             read_gain=args.read_gain, transform="pc1",
                             use_exact=True, use_lossy=False)
    branch.to(device)
    ids = tok(text[:2000], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        base_logits = model(ids).logits
        h = model.get_input_embeddings()(ids)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(1).float()
        h = h + pos * 0.0                       # the model adds its own pos
        h1, _ = branch(h, h, ids[0])
        graft_logits = model(inputs_embeds=h1).logits
    same = bool(torch.equal(base_logits, graft_logits))
    log(f"    bit-identical: {same}   max abs diff "
        f"{float((base_logits - graft_logits).abs().max()):.3e}")
    res["e6a_attach"] = {"bit_identical": same}

    # ---------------------------------------------------------------- E6b
    log("\n[E6b] perplexity: base vs base+layer (the layer is NOT trained)")
    base_ppl = perplexity(model, tok, text, args.ctx, device)
    log(f"    base loss {base_ppl:.4f}  (ppl {torch.tensor(base_ppl).exp():.2f})")
    res["e6b_base_loss"] = base_ppl

    # ---------------------------------------------------------------- E6c
    log("\n[E6c] capability: a fact planted OUTSIDE the context window")
    # arm the exact tier: the identity map (training-free)
    branch.Wo_exact.data = torch.eye(d, device=device, dtype=branch.Wo_exact.dtype)
    n_ok, n = 0, 20
    for i in range(n):
        key = tok(" the secret code is", return_tensors="pt").input_ids[0]
        val = tok(f" {1000 + i}", return_tensors="pt").input_ids[0]
        filler = tok(text[1000 * i:1000 * i + args.ctx * 3], return_tensors="pt").input_ids[0]
        # plant the fact, then bury it under filler, then ask
        planted = torch.cat([key, val, filler])
        branch.reset()
        with torch.no_grad():
            h = model.get_input_embeddings()(planted.unsqueeze(0))
            branch(h[0], h[0], planted)
            q = torch.cat([planted[-args.ctx:]])
            hq = model.get_input_embeddings()(q.unsqueeze(0))
            hq1, _ = branch(hq[0], hq[0], q)
            logits = model(inputs_embeds=hq1).logits
        pred = int(logits[0, -1].argmax())
        n_ok += (pred == int(val))
    log(f"    exact recall {n_ok}/{n} = {n_ok / n:.3f}")
    res["e6c_recall"] = n_ok / n

    # ------------------------------------------------------------ footprint
    log("\n[memory] the layer vs the KV cache, for the same context")
    kv = 0
    for _ in range(model.config.num_hidden_layers):
        kv += 2 * args.ctx * model.config.hidden_size * 2      # fp16, K and V
    log(f"    layer state: {branch.state_bytes():,} B "
        f"({args.M} anchors x r={args.r} + the store)")
    log(f"    KV cache for {args.ctx} tokens: {kv / 1e6:.1f} MB")
    res["memory"] = {"layer_bytes": branch.state_bytes(), "kv_bytes": kv}

    res["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
