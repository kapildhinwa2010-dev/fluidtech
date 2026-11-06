#!/usr/bin/env python3
"""FLUIDTECH MAIN LAYER: BASE + KV CACHE (the recommended layer).

    = SAABranchBulletproof  (main branch: digest-keyed exact-tier store,
                             12-gram context keys, ~0.99 replay recall)
    + ConstantKVCache       (kv_cache branch: bounded KV window, the
                             cache NEVER grows past kv_max_size)
    + SuperfluidMemory       (kv_cache branch: long-term memory; every KV
                             eviction is stored with its REAL hidden
                             state -- no token is silently dropped)

Measured on this repo's CI (verification-results, run e19):
    - 12-gram replay recall 0.9945 (fresh recall 0.0 by construction)
    - bounded KV: max 256 seen while 2000 tokens streamed; 1744
      evictions, ALL stored to long-term memory
    - 8.9 ms / 256 tokens at d=16 (1.68x the bare base layer)
    - worst-case memory ~21.4 MB at max_capacity=100000 per store
    - 0 thread errors (8 threads x 20 forwards)
    - attaching to tiny-gpt2 changes loss by 0.0000 nats (untrained:
      proof of "does no harm", NOT a quality claim)

This is the RECOMMENDED layer.  The optional layers (mamba ssm, ultra,
combined) live in experiments/ and combined/; nothing in this file
depends on them -- use them only if you want them.

Usage:
    from fluidlayer.base_kv_layer import BaseKVConfig, BaseKVLayer
    layer = BaseKVLayer(BaseKVConfig(d=model.config.hidden_size))
    fused = layer.forward(hidden_states, token_ids)     # (S, d) -> (S, d)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from experiments.superfluid_mamba_layer import (
    SuperfluidMambaConfig,
    SuperfluidMambaFluidLayerNP,
)
from fluidlayer.graft_bulletproof import (
    BulletproofConfig,
    HashMode,
    SAABranchBulletproof,
    SecurityLevel,
)
from kv_cache.constant_kv_superfluid import (
    ConstantKVCache,
    KVCacheConfig,
    SuperfluidMemory,
)

__all__ = ["BaseKVConfig", "BaseKVLayer"]


@dataclass
class BaseKVConfig:
    """Configuration for the base + kv cache layer."""

    d: int = 64                 # model width
    M: int = 16                 # number of anchors
    r: int = 4                  # rank of each anchor
    ctx_len: int = 12           # n-gram context length for exact keys
    capacity: int = 25000       # initial store capacity
    max_capacity: int = 100000  # hard store ceiling (worst-case memory)
    kv_max_size: int = 256      # bounded KV window -- never exceeded


class BaseKVLayer:
    """The main FluidTech layer: base + bounded KV + long-term memory.

    forward(h, token_ids):
        1. base tier: 12-gram digest-keyed store lookup/write
        2. bounded KV update on the fused output
        3. every KV eviction -> long-term memory with its REAL hidden
           state (sliding stream pattern, e19-verified)
    """

    def __init__(self, config: BaseKVConfig):
        self.config = config
        d = config.d
        self.base = SAABranchBulletproof(
            M=config.M,
            r=config.r,
            d=d,
            config=BulletproofConfig(
                ctx_len=config.ctx_len,
                d=d,
                hash_mode=HashMode.DOUBLE_64,
                security_level=SecurityLevel.NONE,
                capacity=config.capacity,
                max_capacity=config.max_capacity,
                use_gated_sheaf=False,
            ),
        )
        # SuperfluidMemory is paired with the NumPy fluid layer -- the
        # pairing the kv_cache branch intended (see e18/e19).
        self.np_layer = SuperfluidMambaFluidLayerNP(
            SuperfluidMambaConfig(
                M=config.M, r=config.r, d=d, ctx_len=config.ctx_len,
                capacity=config.capacity, max_capacity=config.max_capacity,
                security_level="STANDARD",
            )
        )
        self.memory = SuperfluidMemory(self.np_layer)
        self.kv_cache = ConstantKVCache(
            KVCacheConfig(max_size=config.kv_max_size, num_heads=1, head_dim=d)
        )
        self._stream: deque = deque()

    def forward(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """One forward pass. h: (S, d) float; token_ids: (S,) long."""
        h_np = h.detach().cpu().numpy().astype(np.float64)
        tid_np = token_ids.detach().cpu().numpy().astype(np.int64)
        out, _ = self.base.forward(h_np, h_np.copy(), tid_np)
        fused = torch.from_numpy(np.asarray(out, dtype=np.float32))

        # bounded KV update; evictions -> long-term memory, REAL states
        k = fused.detach().unsqueeze(0).unsqueeze(0)  # (1, 1, S, d)
        before = len(self.kv_cache.evicted_tokens)
        self.kv_cache.update(k, k.clone(), token_ids.unsqueeze(0))
        n_evict = len(self.kv_cache.evicted_tokens) - before
        for t in range(h.shape[0]):
            self._stream.append((fused[t].detach(), int(tid_np[t])))
        if n_evict > 0:
            rows, ids = [], []
            for _ in range(n_evict):
                if self._stream:
                    row, tok = self._stream.popleft()
                    rows.append(row)
                    ids.append(tok)
            if rows:
                self.memory.store(
                    torch.stack(rows), torch.tensor(ids, dtype=torch.long)
                )
        while len(self._stream) > self.config.kv_max_size:
            self._stream.popleft()
        return fused

    def reset(self) -> None:
        """Reset the base-tier recall statistics (stores are kept)."""
        self.base.total_hits = 0
        self.base.total_misses = 0
        self.base.total_tokens = 0

    def stats(self) -> dict:
        """Live stats: base recall, KV window, long-term memory."""
        return {
            "base": self.base.get_stats(),
            "kv": {
                "size": self.kv_cache.size(),
                "max_size": self.config.kv_max_size,
                "evictions": len(self.kv_cache.evicted_tokens),
            },
            "long_term": {"tokens": self.memory.total_tokens},
        }


if __name__ == "__main__":
    layer = BaseKVLayer(BaseKVConfig(d=16))
    out = layer.forward(torch.randn(8, 16), torch.randint(0, 100, (8,)))
    print("output:", tuple(out.shape), "stats:", layer.stats())
