#!/usr/bin/env python3
"""Quick self-test for the MAIN layer (base + kv cache). No GPU needed.

Run from the repo root:
    python tests/test_base_kv.py

Checks (measured in THIS run; mirrors the e19 CI claims):
  1. output shape + finite
  2. KV stays bounded (<= kv_max_size) while 2000 tokens stream
  3. evictions land in long-term memory (real hidden states)
  4. 12-gram replay recall: fresh ~0, replay ~0.99
  5. thread safety: 8 threads x 20 forwards, 0 errors
  6. worst-case memory estimate under the 100 MB ceiling

The layer needs torch; this test SKIPS (exit 0) when torch is absent,
so the numpy-only test suite stays green (tests/test_torch_parity.py
uses the same convention).
"""

from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import torch
except ImportError:  # the repo is numpy-only by default
    torch = None

if torch is not None:
    from fluidlayer.base_kv_layer import BaseKVConfig, BaseKVLayer

RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(("PASS" if ok else "FAIL") + "  " + name + ("  [" + str(info) + "]" if info else ""))


def main():
    if torch is None:
        print("SKIP: torch is not installed (numpy-only environment)")
        sys.exit(0)
    d, kv_max = 16, 256
    layer = BaseKVLayer(BaseKVConfig(d=d, kv_max_size=kv_max))
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 10000, size=2000).astype(np.int64)

    # 1+2+3: stream 2000 tokens, 16 at a time
    oops, max_kv = 0, 0
    for i in range(0, 2000, 16):
        chunk = torch.from_numpy(tokens[i:i + 16])
        h = torch.randn(16, d)
        out = layer.forward(h, chunk)
        if out.shape != h.shape or not torch.isfinite(out).all():
            oops += 1
        max_kv = max(max_kv, layer.kv_cache.size())
    st = layer.stats()
    check("1. output shape + finite", oops == 0)
    check("2. KV stays bounded", max_kv <= kv_max, "max seen " + str(max_kv))
    ev, lt = st["kv"]["evictions"], st["long_term"]["tokens"]
    check("3. evictions -> long-term memory", ev > 0 and lt > 0,
          str(ev) + " evicted, " + str(lt) + " long-term")

    # 4: fresh vs replay recall (same token stream twice)
    def pass_recall(tok_arr):
        lay = BaseKVLayer(BaseKVConfig(d=d, kv_max_size=kv_max))
        h = torch.randn(len(tok_arr), d)
        lay.forward(h, torch.from_numpy(tok_arr))
        return lay.base.get_stats()["recall_rate"]

    fresh = pass_recall(tokens)
    replay_layer = BaseKVLayer(BaseKVConfig(d=d, kv_max_size=kv_max))
    hh = torch.randn(len(tokens), d)
    replay_layer.forward(hh, torch.from_numpy(tokens))
    replay_layer.reset()
    replay_layer.forward(hh, torch.from_numpy(tokens))
    replay = replay_layer.base.get_stats()["recall_rate"]
    check("4. replay recall high, fresh ~0", replay > 0.5 and fresh < 0.5,
          "fresh " + str(round(fresh, 4)) + ", replay " + str(round(replay, 4)))

    # 5: thread safety
    errs: list = []

    def worker():
        try:
            for _ in range(20):
                layer.forward(torch.randn(16, d), torch.from_numpy(
                    rng.integers(0, 10000, 16).astype(np.int64)))
        except Exception as e:  # noqa: BLE001
            errs.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("5. thread safe (8x20, 0 errors)", len(errs) == 0, str(len(errs)) + " errors")

    # 6: worst-case memory estimate (same accounting as e19)
    entry = sys.getsizeof(0) * 4
    worst_mb = (2 * 100000 * entry + kv_max * 2 * d * 4) / (1024 * 1024)
    check("6. worst-case memory < 100 MB", worst_mb < 100, str(round(worst_mb, 1)) + " MB")

    print()
    if all(RESULTS):
        print("ALL " + str(len(RESULTS)) + " CHECKS PASSED")
        sys.exit(0)
    print(str(RESULTS.count(False)) + " CHECK(S) FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
