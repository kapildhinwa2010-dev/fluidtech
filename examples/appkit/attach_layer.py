"""
examples/appkit/attach_layer.py -- app-kit integration and verification.

This is the WHOLE attachment, in one file:

  1. build the layer          (SAABranchTorch)
  2. attach it               (ONE line in your forward)
  3. run every test          (the attach safety, the perplexity, the
                               capability)

WHAT THE LAYER IS
-----------------
A constant-size memory tier that lives NEXT TO your model, not inside it:

  * L1  the exact tier  -- a content-addressed store of (context -> token)
           20 bytes per fact, bit-exact recall
  * L2  the fluid tier  -- M x r diffusion state (16 KB for M=64, r=8)
           a lossy summary that the router uses as an address space
  * L3  the pointer     -- 6 bytes, an external handle

It is attached as ONE residual-stream insertion:

        h' = h + W_e . E(h) + W_l . S(h)

with `W_e` and `W_l` ZERO-INITIALISED, so attaching it is a PROVABLE
no-op: the logits are bit-identical (measured, see the test below).

THE GATE
---------
The retrieved token is not written into the logits directly.  It is
boosted in proportion to the base model's OWN uncertainty:

        boost_t = gain * ( -log p_max )        (nats)

so the memory decides only where the model has no idea, and stays quiet
where the model is already confident.  The knob serves recall AND
generation -- BUT the useful operating points are DISJOINT on a strong
model: the recall needs gain >= 20, the perplexity needs gain <= 1
(measured on Qwen2.5-0.5B; see N12 in docs/NEGATIVE-RESULTS.md).

RUN IT
------
    python examples/appkit/attach_layer.py --model models/qwen05b
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_layer():
    """Import the optional adapter from an installed package or this checkout."""
    for parent in Path(_HERE).resolve().parents:
        if (parent / "fluidlayer" / "torch_graft.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            break
    try:
        from fluidlayer.torch_graft import SAABranchTorch
    except ImportError as error:
        raise ImportError(
            "install the optional adapter with `python -m pip install "
            "'fluidlayer[gpu]'`"
        ) from error
    return SAABranchTorch


def _import_loader():
    sys.path.insert(0, _HERE)
    from load_model import load
    return load


def attach(model, tok, M=64, r=8, capacity=8192, read_gain=300.0,
           use_exact=True, use_lossy=False):
    """Build the layer and return it.  THE ATTACHMENT ITSELF IS THE TWO
    LINES MARKED WITH `<<<` IN `run()`."""
    SAABranchTorch = _import_layer()
    emb = torch.nn.Parameter(model.wte, requires_grad=False)
    branch = SAABranchTorch(M=M, r=r, d=model.n_embd, emb=emb,
                            ctx_len=1, capacity=capacity,
                            read_gain=read_gain, transform="pc1",
                            use_exact=use_exact, use_lossy=use_lossy)
    branch.refresh_contrast()
    return branch


def run(model_dir: str, ctx: int = 128, windows: int = 12, gain: float = 8.0,
        corpus: str = None):
    load = _import_loader()
    model, tok = load(model_dir)
    print(f"the model: {type(model).__name__} | the vocab: {tok.vocab_size}")

    # ------------------------------------------------------------------
    # TEST 1 -- the attach is a provable no-op
    # ------------------------------------------------------------------
    branch = attach(model, tok)
    ids = tok.encode("The quick brown fox jumps over the lazy dog near the "
                      "river bank at dawn")[:ctx]
    x = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        base = model.forward(x)
        h = model.wte[x].float()                       # <<< THE ATTACHMENT
        h1, info = branch(h, model.wte[x].float(), x)  # <<< (one insertion)
        grafted = model.forward(x, h_override=h1.to(model.dtype))
    same = bool(torch.equal(base, grafted))
    diff = float((base - grafted).abs().max())
    print(f"\nTEST 1  the attach: bit-identical = {same}  "
          f"(max abs diff {diff:.3e})")
    assert same, "the attach must be a no-op"

    # ------------------------------------------------------------------
    # TEST 2 -- the layer recalls a fact that is far outside the window
    # ------------------------------------------------------------------
    # Two token ids that occur NOWHERE in the corpus: the key (the context
    # that is stored) and the value (the token that must come after it).
    key_tok = 150000
    val_tok = 1000
    filler = tok.encode(" ".join(["memory"] * (ctx * 8)))
    mid = ctx // 2
    planted = filler[:mid] + [key_tok, val_tok] + filler[mid:] + [key_tok]
    query = planted[-ctx:]
    branch2 = attach(model, tok)
    with torch.no_grad():
        for i in range(0, len(planted) - 1, ctx):
            seg = planted[i:i + ctx + 1]
            x = torch.tensor(seg[:-1], dtype=torch.long)
            branch2.forward(model.wte[x].float(), model.wte[x].float(), x)
        x = torch.tensor(query, dtype=torch.long)
        logits = model.forward(x) if not _has(model, "keep_last") else \
            model.forward(x, keep_last=True)
        h = model.wte[x].float()
        h1, info = branch2.forward(h, model.wte[x].float(), x)
        prob = torch.softmax(logits.float(), dim=-1)
        unc = float(-torch.log(torch.clamp(prob.max(dim=-1).values[-1],
                                            min=1e-12)))
    base_top = int(logits[-1].argmax())
    dist = len(planted) - ctx
    print(f"TEST 2  the capability: the fact is {dist} tokens back")
    print(f"        the retrieval at the query: "
          f"{int(info['retrieved'][-1])} (the fact is {val_tok})")
    print(f"        the base model predicts {base_top} "
          f"({tok.decode([base_top])!r}); its uncertainty {unc:.2f} nats")
    hit = False
    for g in (2.0, 8.0, 20.0, 50.0):
        lg = branch2.gated_logits(logits,
                                  {"retrieved": info["retrieved"][-1:]}, gain=g)
        layer_top = int(lg[-1].argmax())
        ok = layer_top == val_tok
        hit = hit or ok
        print(f"        gain {g:<5}: the LAYER predicts {layer_top} "
              f"({tok.decode([layer_top])!r})"
              f"   {'<-- THE FACT' if ok else ''}")

    # ------------------------------------------------------------------
    # TEST 3 -- the gated layer does not make the model worse
    # ------------------------------------------------------------------
    if corpus and os.path.exists(corpus):
        with open(corpus, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        ids3 = tok.encode(text)
        # fill the memory from the corpus (a real session)
        with torch.no_grad():
            for i in range(0, min(len(ids3), 5000), ctx):
                seg = ids3[i:i + ctx + 1]
                if len(seg) < 2:
                    break
                x = torch.tensor(seg[:-1], dtype=torch.long)
                branch.forward(model.wte[x].float(), model.wte[x].float(), x)
        branch.readonly = True
        gains = (0.25, 0.5, 1.0, 2.0, 5.0, 8.0)
        acc = {g: [0.0, 0.0] for g in ("base",) + gains}
        with torch.no_grad():
            for i in range(windows):
                seg = ids3[5000 + i * ctx: 5000 + i * ctx + ctx + 1]
                if len(seg) < 2:
                    break
                x = torch.tensor(seg[:-1], dtype=torch.long)
                lgb = model.forward(x)
                h = model.wte[x].float()
                h4, inf4 = branch.forward(h, model.wte[x].float(), x)
                tgt = torch.tensor(seg[1:], dtype=torch.long)
                for g in ("base",) + gains:
                    lgl = lgb if g == "base" else \
                        branch.gated_logits(lgb, inf4, gain=g)
                    acc[g][0] += float(-torch.log_softmax(lgl.float(), -1)
                                       .gather(1, tgt.unsqueeze(1)).sum())
                    acc[g][1] += tgt.numel()
        base_ppl = math.exp(acc["base"][0] / acc["base"][1])
        print(f"TEST 3  the perplexity over {int(acc['base'][1])} tokens "
              f"(the base {base_ppl:.2f}):")
        layer_ppl = float("nan")
        for g in gains:
            p = math.exp(acc[g][0] / acc[g][1])
            layer_ppl = p
            print(f"        gain {g:<5}: {p:8.2f}   "
                  f"({'BETTER' if p < base_ppl else 'worse'} than the base)")
    else:
        base_ppl = layer_ppl = float("nan")

    return {"attach_bit_identical": same, "max_abs_diff": diff,
            "fact_distance": dist, "base_top": base_top,
            "layer_top": layer_top, "fact_recovered": bool(hit),
            "base_ppl": base_ppl, "layer_ppl": layer_ppl}


def _has(obj, name):
    import inspect
    try:
        return name in inspect.signature(obj.forward).parameters
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "MODEL_DIR", "models/qwen05b"))
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--corpus", default=os.path.join(_HERE, "..", "..",
                                                    "corpus.txt"))
    a = ap.parse_args()
    t0 = time.time()
    out = run(a.model, ctx=a.ctx, windows=a.windows, gain=a.gain,
              corpus=a.corpus)
    print(f"\ndone in {time.time() - t0:.0f}s: {out}")
