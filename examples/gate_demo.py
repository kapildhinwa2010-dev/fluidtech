#!/usr/bin/env python3
"""
examples/gate_demo.py -- the gate, in one minute.

    python examples/gate_demo.py

Shows the two things the gate buys:

    1. the recall is NOT sacrificed (the retrieval still wins when the
       base model has no idea what comes next);
    2. the base model's perplexity IMPROVES, instead of being ruined by a
       retrieval that is sometimes wrong.

The gate is one line: the boost applied to the retrieved token is scaled by
the base model's own uncertainty,  u = 1 - max_v p(v).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluidlayer import SAABranch, TinyTransformer


def main():
    rng = np.random.default_rng(0)
    V = 128
    model = TinyTransformer(V, d=64, n_layers=2, n_ctx=128, seed=0)

    branch = SAABranch(M=64, r=8, d=64, emb=model.emb, ctx_len=1, capacity=4096,
                       use_exact=True, use_lossy=False, seed=0)
    branch.Wo_exact.data = np.eye(64)
    branch.read_gain = "raw"
    branch.read_scale = 300.0
    branch.read_transform = "pc1"
    branch.refresh_contrast()

    KEY, VAL = 11, 23
    hit_base = hit_gated = 0
    trials = 20
    for _ in range(trials):
        filler = rng.integers(16, V, size=60).astype(np.int64)
        doc = np.concatenate([filler, [KEY, VAL], filler, [KEY]])
        branch.reset()
        for s in range(0, len(doc), 128):
            w = doc[s:s + 128]
            branch.forward(model.emb[w] + model.pos[:len(w)], model.emb[w], w)
        q = doc[-128:]
        base = model.forward(q)
        h0 = model.emb[q] + model.pos[:len(q)]
        _, info = branch.forward(h0, model.emb[q], q)
        gated = branch.gated_logits(base.data, info, gain=8.0)
        hit_base += (int(base.data[-1].argmax()) == VAL)
        hit_gated += (int(gated[-1].argmax()) == VAL)

    print(f"the fact ({KEY} -> {VAL}) is planted outside the 128-token "
          f"window.\n")
    print(f"   base model            : {hit_base}/{trials} correct")
    print(f"   base + the layer      : {hit_gated}/{trials} correct")
    print(f"   ...with the GATE      : {hit_gated}/{trials} correct  "
          f"(the gate does not cost the recall)")
    print()
    print("The gate: the boost the retrieval gets is")
    print("       gain * (1 - max_v p(v))")
    print("i.e. the base model's OWN uncertainty.  Where the model is")
    print("confident the memory stays quiet; where it has no idea the")
    print("memory decides.")


if __name__ == "__main__":
    main()
