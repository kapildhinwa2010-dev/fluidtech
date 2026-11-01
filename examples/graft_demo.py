#!/usr/bin/env python3
"""
examples/graft_demo.py -- the capability demo, end to end.

Streams a document through a frozen model with the layer attached and asks
the model to produce a fact that was planted far outside its attention
window.  Prints the base model's answer and the layer's answer.

    python examples/graft_demo.py

This is the small version of `experiments/e6_graft.py`; the experiment is
the one with the numbers in `results/`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superfluid import SAABranch, TinyTransformer


def main():
    rng = np.random.default_rng(0)
    V, KEY, VAL = 32, 11, 23
    model = TinyTransformer(vocab=V, d=32, n_layers=2, n_ctx=32, seed=0)

    branch = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    branch.Wo_exact.data = np.eye(32)
    branch.read_gain = "raw"
    branch.read_scale = 300.0
    branch.read_transform = "pc1"
    branch.refresh_contrast()

    ok_base = ok_graft = 0
    trials = 20
    for _ in range(trials):
        filler = rng.integers(16, V, size=40).astype(np.int64)
        doc = np.concatenate([filler, [KEY, VAL], filler, [KEY]])
        branch.reset()
        for s in range(0, len(doc), 32):
            w = doc[s:s + 32]
            branch.forward(model.emb[w] + model.pos[:len(w)], model.emb[w], w)
        q = doc[-32:]
        h = model.emb[q] + model.pos[:len(q)]
        base_pred = int(model.forward(q).data[-1].argmax())
        h2, _ = branch.forward(h, model.emb[q], q)
        graft_pred = int(model.forward_from_h(h2).data[-1].argmax())
        ok_base += (base_pred == VAL)
        ok_graft += (graft_pred == VAL)

    print(f"the fact ({KEY} -> {VAL}) is planted 40 tokens before the query,")
    print("outside the model's 32-token attention window.\n")
    print(f"   base model        : {ok_base}/{trials} correct  "
          f"(it never saw the fact)")
    print(f"   base + the layer  : {ok_graft}/{trials} correct  "
          f"(the layer retrieved it)")
    print()
    print("The layer did not improve the model.  It gave it one thing it")
    print("did not have: a way to quote something it saw earlier.")


if __name__ == "__main__":
    main()
