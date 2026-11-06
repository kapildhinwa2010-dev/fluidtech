"""Feature 2 -- multi-GPU sharded KV cache: unit tests (torch)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")  # noqa: E402

from fluidlayer.multi_gpu import ShardedKVCache, detect_devices  # noqa: E402


def test_detect_devices_fallback():
    devices = detect_devices()
    assert len(devices) >= 1
    assert all(isinstance(d, torch.device) for d in devices)


def test_sharding_round_robin_bounded():
    cpu = torch.device("cpu")
    cache = ShardedKVCache(max_size=64, head_dim=8, devices=[cpu, cpu, cpu])
    for i in range(100):
        t = torch.full((1, 8), float(i))
        cache.update(t, t.clone())
    assert cache.size() == 64
    assert cache.evictions == 36
    assert cache.bounded()
    sizes = cache.shard_sizes()
    assert max(sizes) - min(sizes) <= 1


def test_gather_preserves_insertion_order():
    cpu = torch.device("cpu")
    cache = ShardedKVCache(max_size=256, head_dim=4, devices=[cpu] * 4)
    for i in range(300):
        t = torch.full((4,), float(i))
        cache.update(t, t.clone())
    got = cache.gather()
    assert got.shape == (256, 4)
    vals = [round(float(v), 6) for v in got[:, 0]]
    assert vals == [float(i) for i in range(44, 300)]


def test_default_constructor_single_device():
    cache = ShardedKVCache(max_size=16, head_dim=4)
    t = torch.ones((2, 4))
    cache.update(t, t.clone())
    assert cache.size() == 2
    assert cache.gather().shape == (2, 4)
