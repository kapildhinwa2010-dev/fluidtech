"""COMBINED FLUID LAYER - all four branches fused into ONE layer.

Branches combined:
- main:         SAABranchBulletproof -- the verified exact-tier memory
                layer (100% replay recall, security levels, threads).
- experiments:  SuperfluidMambaFluidLayer (Mamba-SSM + superfluid
                diffusion) and UltraFluidStack (multi-layer stacking,
                residuals, cross-attention fusion, adaptive routing).
- kv_cache:     ConstantKVCache (bounded, evicting KV cache) and
                SuperfluidMemory (long-term memory fed by evictions).

Honest status:
- The fusion gates are untrained constants: an untrained layer cannot
  IMPROVE a model's loss, it can only attach and run without
  breaking it.
- The KV cache stores PROXY K/V tensors projected from the fused
  hidden states (the kv_cache branch stored random tensors); the
  bounded-size behaviour is real, the K/V VALUES are placeholders.
- Bugs fixed here (found while combining the branches):
  1. the kv_cache branch passed 3-D hidden states and 2-D token ids
     straight into the fluid layer (shape mismatch); this module
     passes correctly shaped tensors.
  2. the kv_cache branch stored torch.zeros() as the "hidden states"
     of evicted tokens; this module stores the REAL evicted states.
  3. ConstantKVCache.update() tracks evicted ids from the NEW chunk
     instead of the evicted window (kept upstream); this module keeps
     its own correct eviction bookkeeping.
- Memory: the four stores share a budget (max_capacity=100000 each);
  100 MB is a hard CEILING, not a target.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.superfluid_mamba_layer import (
    SuperfluidMambaConfig,
    SuperfluidMambaFluidLayer,
)
from experiments.ultra_fluid_layer import UltraFluidConfig, UltraFluidStack
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


@dataclass
class CombinedFluidConfig:
    """Configuration for the Combined Fluid Layer (four branches)."""

    d: int = 64
    ctx_len: int = 12
    M: int = 16
    r: int = 4
    capacity: int = 25000
    max_capacity: int = 100000
    kv_max_size: int = 256
    num_heads: int = 4
    ultra_layers: int = 2
    gate_base: float = 0.5
    gate_ultra: float = 0.5

    def validate(self) -> None:
        assert self.d > 0, "d must be positive"
        assert self.ctx_len > 0, "ctx_len must be positive"
        assert self.d % self.num_heads == 0, "d must be divisible by num_heads"
        assert self.kv_max_size > 0, "kv_max_size must be positive"
        assert self.ultra_layers >= 1, "ultra_layers must be >= 1"


class CombinedFluidLayer(nn.Module):
    """ONE layer that runs all four branches' components together."""

    def __init__(self, config: CombinedFluidConfig):
        super().__init__()
        self.config = config
        config.validate()

        # --- main branch: the verified exact-tier memory layer ---
        self.base = SAABranchBulletproof(
            M=config.M,
            r=config.r,
            d=config.d,
            config=BulletproofConfig(
                ctx_len=config.ctx_len,
                d=config.d,
                hash_mode=HashMode.DOUBLE_64,
                security_level=SecurityLevel.NONE,
                capacity=config.capacity,
                max_capacity=config.max_capacity,
                use_gated_sheaf=False,
            ),
        )

        # --- experiments branch: superfluid-mamba (also feeds the
        #     kv_cache branch's SuperfluidMemory long-term store) ---
        self.mamba_memory = SuperfluidMambaFluidLayer(
            SuperfluidMambaConfig(
                M=config.M,
                r=config.r,
                d=config.d,
                ctx_len=config.ctx_len,
                capacity=config.capacity,
                max_capacity=config.max_capacity,
                security_level="STANDARD",
            )
        )

        # --- experiments branch: the ultra stack ---
        self.ultra = UltraFluidStack(
            UltraFluidConfig(
                M=config.M,
                r=config.r,
                d=config.d,
                ctx_len=config.ctx_len,
                num_layers=config.ultra_layers,
                capacity=config.capacity,
                max_capacity=config.max_capacity,
                security_level="STANDARD",
            )
        )

        # --- kv_cache branch: bounded KV cache + long-term memory ---
        head_dim = config.d // config.num_heads
        self.kv_cache = ConstantKVCache(
            KVCacheConfig(
                max_size=config.kv_max_size,
                num_heads=config.num_heads,
                head_dim=head_dim,
            )
        )
        self.superfluid_memory = SuperfluidMemory(self.mamba_memory)

        # sliding window of (hidden_state, token_id) for honest eviction
        self._stream: deque = deque()
        self.total_tokens = 0
        self.start_time = time.time()

    def forward(
        self,
        h: torch.Tensor,
        content: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through the combined layer.

        Args:
            h: hidden states of shape (seq_len, d)
            content: content embeddings of shape (seq_len, d)
            token_ids: token ids of shape (seq_len,)

        Returns:
            (fused hidden states, info dict)
        """
        seq_len, d = h.shape
        h_np = h.detach().cpu().numpy().astype(np.float64)
        c_np = content.detach().cpu().numpy().astype(np.float64)
        tid_np = token_ids.detach().cpu().numpy().astype(np.int64)

        # 1) main branch: exact-tier memory
        base_out, base_info = self.base.forward(h_np, c_np, tid_np)
        base_t = torch.from_numpy(np.asarray(base_out, dtype=np.float32))

        # 2) experiments branch: ultra stack (mamba + superfluid inside)
        ultra_out, ultra_infos = self.ultra(h, content, token_ids)
        ultra_out = ultra_out.detach().float()

        # 3) fuse (untrained constant gates)
        fused = (
            h
            + self.config.gate_base * (base_t - h)
            + self.config.gate_ultra * (ultra_out - h)
        )

        # 4) kv_cache branch: bounded KV cache; evictions go to the
        #    long-term superfluid memory with their REAL hidden states
        head_dim = d // self.config.num_heads
        k = fused.detach().reshape(1, seq_len, self.config.num_heads, head_dim)
        k = k.permute(0, 2, 1, 3).contiguous()
        v = k.clone()
        evicted_before = len(self.kv_cache.evicted_tokens)
        self.kv_cache.update(k, v, token_ids.unsqueeze(0))
        n_evict = len(self.kv_cache.evicted_tokens) - evicted_before
        if n_evict > 0:
            rows, ids = [], []
            for _ in range(n_evict):
                if self._stream:
                    row, tok = self._stream.popleft()
                    rows.append(row)
                    ids.append(tok)
            if rows:
                self.superfluid_memory.store(
                    torch.stack(rows),
                    torch.tensor(ids, dtype=torch.long),
                )
        for t in range(seq_len):
            self._stream.append((fused[t].detach(), int(tid_np[t])))
        while len(self._stream) > self.config.kv_max_size:
            self._stream.popleft()

        self.total_tokens += seq_len
        info = {
            "base": base_info,
            "ultra": ultra_infos,
            "kv_size": self.kv_cache.size(),
            "kv_evictions_total": len(self.kv_cache.evicted_tokens),
            "long_term_tokens": self.superfluid_memory.total_tokens,
            "total_tokens": self.total_tokens,
        }
        return fused, info

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "components": [
                "base (main)",
                "superfluid-mamba (experiments)",
                "ultra x" + str(self.config.ultra_layers) + " (experiments)",
                "constant-kv-cache (kv_cache)",
                "superfluid long-term memory (kv_cache)",
            ],
            "base": self.base.get_stats(),
            "mamba_long_term": self.mamba_memory.get_stats(),
            "kv_cache_size": self.kv_cache.size(),
            "kv_max_size": self.kv_cache.get_max_size(),
            "kv_evicted_total": len(self.kv_cache.evicted_tokens),
            "long_term_tokens": self.superfluid_memory.total_tokens,
        }

    def reset(self) -> None:
        self.base.total_hits = 0
        self.base.total_misses = 0
        self.base.total_tokens = 0
        self.mamba_memory.reset()
        for layer in self.ultra.layers:
            layer._context_buffer = []
            layer.total_hits = 0
            layer.total_misses = 0
            layer.total_tokens = 0
        self.kv_cache.reset()
        self._stream = deque()
        self.total_tokens = 0
        self.start_time = time.time()


def create_combined_layer(
    d: int = 64,
    ctx_len: int = 12,
    kv_max_size: int = 256,
    ultra_layers: int = 2,
) -> CombinedFluidLayer:
    """Factory: one layer with all four branches inside."""
    return CombinedFluidLayer(
        CombinedFluidConfig(
            d=d,
            ctx_len=ctx_len,
            kv_max_size=kv_max_size,
            ultra_layers=ultra_layers,
        )
    )
