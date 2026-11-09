"""Regression tests for bounded KV cache eviction bookkeeping."""

import pytest

torch = pytest.importorskip("torch")
_cache_module = pytest.importorskip("kv_cache.constant_kv_superfluid")
ConstantKVCache = _cache_module.ConstantKVCache
KVCacheConfig = _cache_module.KVCacheConfig


def _update(cache, ids):
    token_ids = torch.tensor([ids], dtype=torch.long)
    states = torch.tensor(ids, dtype=torch.float32).view(1, 1, len(ids), 1)
    cache.update(states, states, token_ids)


def test_first_update_larger_than_capacity_is_trimmed_and_tracked():
    cache = ConstantKVCache(KVCacheConfig(max_size=3, num_heads=1, head_dim=1))

    _update(cache, [10, 11, 12, 13, 14])

    assert cache.size() == 3
    assert cache.get_evicted_tokens() == [10, 11]
    assert cache.get()[0].flatten().tolist() == [12.0, 13.0, 14.0]


def test_evictions_refer_to_oldest_cached_tokens_across_updates():
    cache = ConstantKVCache(KVCacheConfig(max_size=3, num_heads=1, head_dim=1))

    _update(cache, [10, 11])
    _update(cache, [12, 13])
    _update(cache, [14])

    assert cache.size() == 3
    assert cache.get_evicted_tokens() == [10, 11]
    assert cache.get()[0].flatten().tolist() == [12.0, 13.0, 14.0]
