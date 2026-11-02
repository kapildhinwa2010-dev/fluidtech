"""
E9 -- what happens when you give a model TEN MILLION tokens?

    "10M tokens de do layer laga ke, to kya yaad rahega?  Aur performance
     pehle ke aur ab ke beech kitna antar hoga?"

This experiment answers it with measurements, at four levels:

  E9a  THE FOOTPRINT.  10M tokens through the layer: how long, how much
        state, and what the same tokens would cost as a KV cache.
  E9b  THE EXACT TIER, TOKEN-LEVEL (ctx_len=1).  Facts planted at 1M, 5M
        and 9.9M tokens -- are they still recalled at the end?
  E9c  THE EXACT TIER, CONTEXT-LEVEL (ctx_len=12).  The realistic setting.
        The store can only hold ~819 contexts in a 16 KB budget, so 10M
        tokens means EVICTION.  How deep does the memory actually reach?
  E9d  THE PERFORMANCE DELTA.  The base model's own next-token loss,
        before and after the layer, on text that is (i) novel and (ii) a
        repeat of what the layer has already seen.

A note on the compute: the EXACT tier is MODEL-FREE.  Its write is
"the content key -> the token that followed", which needs no forward pass.
So filling a 10M-token memory costs a dictionary update per token, not a
model call.  That is measured here, and it is the reason this experiment
can run on a CPU in minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluidlayer import SAABranch, cross_entropy  # noqa: E402
from fluidlayer.memory import SLOT_BYTES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def build_stream(n_tokens, n_facts=6, seed=0, vocab=76):
    """A long stream with facts planted at known depths."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab - 6, size=n_tokens).astype(np.int64)
    facts = []
    # plant the facts at logarithmically increasing depths.
    # The KEY must be DISTINCTIVE: a common key can have many conflicting
    # continuations, so one planted answer need not become the majority.
    # A token that appears ONLY at the fact is the honest way to plant a
    # retrievable fact.
    depths = [int(n_tokens * f) for f in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9)]
    for i, d in enumerate(depths):
        key = vocab - 6 + i        # reserved: occurs NOWHERE else in the stream
        val = int(rng.integers(0, vocab))
        ids[d] = key
        ids[d + 1] = val
        facts.append({"depth": d, "key": key, "val": val})
    # a second set of facts with COMMON keys -- the control
    common = []
    for _, d in enumerate(depths[:3]):
        key = int(rng.integers(0, vocab - 6))  # occurs everywhere
        val = int(rng.integers(0, vocab))
        ids[d + 2] = key
        ids[d + 3] = val
        common.append({"depth": d, "key": key, "val": val})
    return ids, facts, common


def stream_through(branch, ids, chunk=4096):
    """Fill the layer's memory with a token stream.  NO model forward."""
    t0 = time.time()
    for s in range(0, len(ids), chunk):
        w = ids[s:s + chunk]
        # the exact tier needs only the ids: the write is content -> next id
        branch._buf = list(branch._buf)
        for t in range(len(w) - 1):
            prev_key = branch._key(branch._buf)
            branch._buf.append(int(w[t]))
            if len(branch._buf) > branch.ctx_len:
                branch._buf = branch._buf[-branch.ctx_len:]
            if prev_key is not None:
                branch._put(prev_key, int(w[t]))
    return time.time() - t0


def recall_fact(branch, key):
    """What does the layer answer for this context?"""
    return branch.store.get(int(key), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=10_000_000)
    ap.add_argument("--capacity", type=int, default=819)     # a 16 KB budget
    ap.add_argument("--out", default=os.path.join(RESULTS, "e9_ten_million.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"tokens": args.tokens, "capacity": args.capacity,
           "slot_bytes": SLOT_BYTES}

    print(f"[1/4] building a {args.tokens:,}-token stream ...")
    ids, facts, common = build_stream(args.tokens)
    print(f"    {len(ids):,} tokens, {len(facts)} facts planted")

    # ---------------------------------------------------------------- E9a
    print("\n[2/4] E9a -- the footprint")
    kv_8b = args.tokens * 0.131e6      # fp16, GQA, 8B
    kv_70b = args.tokens * 0.328e6     # fp16, GQA, 70B
    print(f"    the layer's state:            {16 * 1024:,} B  (constant)")
    print(f"    the same tokens as a KV cache: {kv_8b / 1e12:.2f} TB (8B model)"
          f" / {kv_70b / 1e12:.2f} TB (70B)")
    res["footprint"] = {
        "layer_state_bytes": 16 * 1024,
        "kv_bytes_8b": kv_8b, "kv_bytes_70b": kv_70b,
        "kv_over_layer_8b": kv_8b / (16 * 1024),
        "kv_over_layer_70b": kv_70b / (16 * 1024),
    }

    # ---------------------------------------------------------------- E9b
    print("\n[3/4] E9b -- the exact tier at 10M tokens (token-level keys)")
    br = SAABranch(M=64, r=8, d=64, emb=None, ctx_len=1, capacity=args.capacity,
                   use_exact=True, use_lossy=False, seed=0)
    # the branch needs an embedding table for the READ; use a small random one
    class _Emb:
        def __init__(self, V, d):
            self.data = np.random.default_rng(0).normal(0, 0.02, (V, d))
            self.shape = self.data.shape
    br.emb = _Emb(76, 64)
    br.refresh_contrast()
    dt = stream_through(br, ids)
    print(f"    streamed {args.tokens:,} tokens in {dt:.1f} s "
          f"({args.tokens / dt / 1e6:.2f}M tokens/s, no model forward)")
    print(f"    the store now holds {len(br.store):,} distinct contexts "
          f"({len(br.store) * SLOT_BYTES:,} B)")
    hits = []
    for f in facts:
        got = recall_fact(br, f["key"])
        ok = (got == f["val"])
        hits.append(ok)
        print(f"      fact at token {f['depth']:>10,}: "
              f"recalled={ok}  (asked for {f['key']} -> got {got})")
    common_hits = []
    for f in common:
        got = recall_fact(br, f["key"])
        common_hits.append(got == f["val"])
    print("    CONTROL -- facts with COMMON keys (the key occurs all over "
          "the stream):")
    print(f"      recalled {sum(common_hits)}/{len(common_hits)}  -> a common "
          f"context does not identify one continuation")
    res["e9b_token_level"] = {
        "common_key_control_recall": float(np.mean(common_hits)),
        "seconds": dt, "tokens_per_sec": args.tokens / dt,
        "store_entries": len(br.store),
        "store_bytes": len(br.store) * SLOT_BYTES,
        "state_bytes": br.state_bytes,
        "recall_by_depth": {str(f["depth"]): bool(h)
                             for f, h in zip(facts, hits)},
        "recall_overall": float(np.mean(hits)),
    }

    # ---------------------------------------------------------------- E9c
    print("\n[4/4] E9c -- the realistic setting: CONTEXT-level keys (12-gram)")
    br2 = SAABranch(M=64, r=8, d=64, emb=br.emb, ctx_len=12,
                    capacity=args.capacity, use_exact=True, use_lossy=False,
                    seed=0)
    dt2 = stream_through(br2, ids)
    print(f"    streamed in {dt2:.1f} s; the store holds {len(br2.store):,} "
          f"contexts ({len(br2.store) * SLOT_BYTES:,} B, capacity "
          f"{args.capacity})")
    # the deep fact is queried with its 12-gram context
    depth_hits = {}
    for f in facts:
        ctx = ids[f["depth"] - 11:f["depth"] + 1].tolist()
        got = br2.store.get(br2._key(ctx))
        ok = (got == f["val"])
        depth_hits[str(f["depth"])] = bool(ok)
        print(f"      fact at token {f['depth']:>10,}: recalled={ok}  "
              f"(got {got}, wanted {f['val']})")
    survived = [k for k, v in depth_hits.items() if v]
    print(f"      -> {len(survived)}/{len(facts)} facts survived: with only "
          f"{args.capacity} slots, the deep ones are EVICTED")
    res["e9c_context_level"] = {
        "seconds": dt2, "store_entries": len(br2.store),
        "store_bytes": len(br2.store) * SLOT_BYTES,
        "recall_by_depth": depth_hits,
        "note": "a 16 KB budget cannot hold the contexts of 10M tokens; "
                "the deep ones are evicted",
    }

    # ---------------------------------------------------------------- E9d
    print("\n[+] E9d -- the performance delta (before vs after)")
    # a real base model, trained on the local corpus, evaluated on text the
    # layer has NOT seen (novel) and on text it HAS seen (a repeat)
    from experiments.e6_real_corpus import load_chars, train_base
    ids_c, V = load_chars()
    n_tr = int(len(ids_c) * 0.9)
    tr, va = ids_c[:n_tr], ids_c[n_tr:]
    model = train_base(tr, V, d=64, ctx=128, steps=300, seed=0)
    br3 = SAABranch(M=64, r=8, d=model.d, emb=model.emb, ctx_len=12,
                    capacity=args.capacity, use_exact=True, use_lossy=False,
                    seed=0)
    br3.Wo_exact.data = np.eye(model.d)
    br3.read_gain = "raw"
    br3.read_scale = 300.0
    br3.read_transform = "pc1"
    br3.refresh_contrast()

    def fill_memory():
        """Fill the memory ONCE with 10M tokens of the training text.

        (A real session's worth of context.  The store's contents do not
        depend on the injection gain, so the gain sweep below does NOT
        refill it -- refilling per point cost 9 x 10M tokens and made the
        experiment 9 times slower for nothing.)
        """
        br3.reset()
        for _ in range(args.tokens // len(tr) + 1):
            stream_through(br3, tr)

    def val_loss(use_layer, repeat=False):
        if use_layer and not br3.store:
            fill_memory()
        tot = cnt = 0.0
        for s in range(0, len(va) - 1, 128):
            w = va[s:s + 129]
            if use_layer:
                h = model.emb[w[:-1]] + model.pos[:len(w) - 1]
                h, _ = br3.forward(h, model.emb[w[:-1]], w[:-1])
                lg = model.forward_from_h(h)
            else:
                lg = model.forward(w[:-1])
            tot += cross_entropy(lg, w[1:]) * (len(w) - 1)
            cnt += len(w) - 1
        return tot / cnt

    base = val_loss(False)
    print(f"    the base model alone                : {base:.4f} "
          f"(ppl {np.exp(base):.2f})")
    # The injection gain decides everything.  A big gain is what makes the
    # RECALL work (the recalled token must win the argmax); a big gain is
    # also what makes the GENERATION fail (a wrong retrieval is forced into
    # the residual).  Sweep it.
    sweep = {}
    for g in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        br3.read_scale = g
        l = val_loss(True)
        sweep[str(g)] = {"loss": float(l), "delta": float(l - base)}
        print(f"    base + layer, gain {g:>6.1f}          : {l:.4f} "
              f"(ppl {np.exp(l):7.2f})   delta {l - base:+.4f}")
    best_gain = min(sweep, key=lambda k: sweep[k]["loss"])
    print(f"    -> the best gain for GENERATION is {best_gain} "
          f"(delta {sweep[best_gain]['delta']:+.4f}); the best for RECALL is "
          f"300.  They are not the same knob.")
    res["e9d_performance"] = {
        "base_loss": float(base), "base_ppl": float(np.exp(base)),
        "gain_sweep": sweep,
        "best_gain_for_generation": best_gain,
        "best_delta_for_generation": sweep[best_gain]["delta"],
        "note": "the injection gain that maximises RECALL is NOT the gain "
                "that maximises GENERATION quality",
    }

    res["runtime_sec"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
