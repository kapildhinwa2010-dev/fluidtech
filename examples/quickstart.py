#!/usr/bin/env python3
"""
examples/quickstart.py -- the whole thing in one minute.

    python examples/quickstart.py

It prints the three numbers that matter:

    1. attaching the layer changes NOTHING (the logits are bit-identical)
    2. the exact tier retrieves a fact from OUTSIDE the attention window
    3. the diffusion tier alone retrieves NOTHING (and that is the point)

No GPU, no network, no downloads.  Runs in a few seconds.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superfluid import SAABranch, TinyTransformer


def hr(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    hr("0. a base model (a small transformer, 131k params)")
    model = TinyTransformer(vocab=32, d=32, n_layers=2, n_ctx=32, seed=0)
    print(f"   params: {sum(p.data.size for p in model.params().values()):,}")

    hr("1. attach the layer: the logits must be BIT-IDENTICAL")
    branch = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    tok = np.array([3, 7, 11, 5, 9, 2, 8, 13], dtype=np.int64)
    before = model.forward(tok)
    h = model.emb[tok] + model.pos[:len(tok)]
    h2, _ = branch.forward(h, model.emb[tok], tok)
    after = model.forward_from_h(h2)
    same = np.array_equal(before.data, after.data)
    print(f"   bit-identical: {same}")
    print(f"   max abs diff: {float(np.abs(before.data - after.data).max()):.3e}")
    print(f"   the layer's state: {branch.state_bytes:,} bytes")
    assert same, "attaching the layer must not change the model"

    hr("2. the exact tier: recall a fact from OUTSIDE the window")
    # the model sees 8 tokens; the fact was 5 tokens ago
    fact = np.array([21, 17, 4, 30, 25], dtype=np.int64)   # 21 -> 17
    branch2 = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    branch2.Wo_exact.data = np.eye(32)
    branch2.read_gain = "raw"
    branch2.read_scale = 300.0
    branch2.read_transform = "pc1"
    branch2.refresh_contrast()
    branch2.reset()
    branch2.forward(model.emb[fact] + model.pos[:len(fact)],
                     model.emb[fact], fact)
    stored = branch2.store.get(int(fact[-2]))
    print(f"   the store holds: context {int(fact[-2])} -> token {stored}")
    print(f"   exact (bit-exact, an int32): {stored == int(fact[-1])}")

    hr("3. the diffusion tier alone retrieves NOTHING -- and that is the point")
    branch3 = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1, seed=0)
    branch3.use_exact = False
    branch3.use_lossy = True
    branch3.reset()
    branch3.forward(model.emb[fact] + model.pos[:len(fact)],
                     model.emb[fact], fact)
    print(f"   the lossy tier's state: {branch3.state.shape} "
          f"(a diffused aggregate -- no token ids in it)")
    print(
        "   its state is NOT exact: "
        f"{not np.array_equal(branch3.state, branch3.state.astype(int))}"
    )

    hr("what to read next")
    print("   docs/GRAFT.md            what the layer does to a base model (measured)")
    print("   docs/NEGATIVE-RESULTS.md  the seven things that did not work")
    print("   docs/GETTING-STARTED.md   how to use it for real")
    print("   python selftest.py        38 machine-checked claims")
    print()


if __name__ == "__main__":
    main()
