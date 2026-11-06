#!/usr/bin/env python3
"""Multi-GPU support: device detection + sharded KV cache.

Feature 2 of the "Coming soon" list. ShardedKVCache shards cached
(k, v) tokens round-robin across the detected devices:
- multiple CUDA GPUs -> shards live on separate GPUs;
- MPS -> the Apple backend;
- CPU-only (CI) -> the SAME sharding logic runs on CPU "virtual
  devices", so correctness is testable everywhere; only placement
  changes on real GPU machines (honest: speedup itself needs GPUs).
"""
from __future__ import annotations

from collections import deque

import torch

__all__ = ["detect_devices", "ShardedKVCache"]


def detect_devices():
    """All usable compute devices: CUDA first, MPS next, CPU fallback."""
    devices = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device("cuda:" + str(i)))
    except Exception:
        pass
    try:
        if not devices and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
    except Exception:
        pass
    if not devices:
        devices = [torch.device("cpu")]
    return devices


class ShardedKVCache:
    """Bounded (k, v) cache sharded round-robin across devices.

    Eviction takes the oldest entry of the fullest shard, so shards
    stay balanced. gather() rebuilds global insertion order.
    """

    def __init__(self, max_size, head_dim, devices=None):
        self.devices = list(devices) if devices else detect_devices()
        self.max_size = int(max_size)
        self.head_dim = int(head_dim)
        self._shards = [deque() for _ in self.devices]
        self._rr = 0
        self._seq = 0
        self.evictions = 0

    def update(self, k, v):
        """Push (k, v) rows; shape (S, head_dim) or (head_dim,)."""
        k = torch.as_tensor(k, dtype=torch.float32)
        v = torch.as_tensor(v, dtype=torch.float32)
        if k.dim() == 1:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        rows = min(k.shape[0], v.shape[0])
        for i in range(rows):
            idx = self._rr % len(self.devices)
            self._rr += 1
            dev = self.devices[idx]
            self._shards[idx].append((
                self._seq,
                k[i].detach().to(dev).clone(),
                v[i].detach().to(dev).clone(),
            ))
            self._seq += 1
            while self.size() > self.max_size:
                self._evict_one()

    def size(self):
        return sum(len(s) for s in self._shards)

    def _evict_one(self):
        idx = max(range(len(self._shards)),
                   key=lambda i: len(self._shards[i]))
        if self._shards[idx]:
            self._shards[idx].popleft()
            self.evictions += 1

    def shard_sizes(self):
        return [len(s) for s in self._shards]

    def gather(self, device=None):
        """All cached v vectors in insertion order, on one device."""
        dev = device if device is not None else self.devices[0]
        entries = []
        for shard in self._shards:
            entries.extend(shard)
        entries.sort(key=lambda e: e[0])
        if not entries:
            return torch.zeros(0, self.head_dim, device=dev)
        return torch.stack([e[2].to(dev) for e in entries])

    def bounded(self):
        return self.size() <= self.max_size
