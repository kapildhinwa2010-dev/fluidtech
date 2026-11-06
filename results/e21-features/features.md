# E21: quantization + multi-GPU + acceleration (measured)

- commit: e539e4ad49c346ec423836cf045e9cdfe59019c6

## 1. Quantization (int8)

- memory: float32 -> int8 = 4.0x smaller (measured)
- round-trip relative error: 0.3936 % (bound: < 0.4 %)

## 2. Multi-GPU sharded KV

- detected devices: ['cpu']
- 4 virtual shards: 1000 inserts -> size 256, evictions 744, order preserved, shards balanced (measured)

## 3. Hardware acceleration

- detected: {'torch': '2.15.0+cpu', 'cuda': False, 'cuda_devices': 0, 'mps': False, 'numba': False, 'cupy': False, 'vectorized_numpy': True}
- benchmark: loop 0.000668s vs vectorized 2e-05s -> 34.24x speedup (measured)

## Claims (measured in THIS run)

- **quantization_memory_4x**: True
- **quantization_error_under_1pct**: True
- **multi_gpu_detection_works**: True
- **multi_gpu_shard_correct**: True
- **multi_gpu_kv_bounded**: True
- **acceleration_detection_ok**: True
- **acceleration_outputs_match**: True
- **acceleration_speedup**: 34.24
- **acceleration_speedup_gt_1**: True
- **all_features_positive**: True

## Full log

```
======================================================================
E21: FEATURES (quantization + multi-gpu + acceleration)
======================================================================

--- Feature 1: quantization (int8) ---
layer output: 8192 B -> 2048 B
round-trip error: 0.3936 %

--- Feature 2: multi-gpu sharded KV ---
devices: ['cpu']
kv: size=256 evictions=744 shards=[64, 64, 64, 64]

--- Feature 3: hardware acceleration ---
backend: torch=2.15.0+cpu cuda=False mps=False
benchmark: loop=0.000668s vector=2e-05s speedup=34.24x
```