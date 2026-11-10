# Base + KV wrapper (e19)

- commit: 998c924bb9fbd6e7e79654c31a32bf1d08068a0f
- branch: combined-layer
- date: 2026-11-10 00:56:38 UTC
- same harness as e18: d=16, ctx_len=12, capacity 25k/100k, KV_MAX=256
- question: does base+kv give the combined layer's bounded-KV
  + long-term features at close-to-base speed and memory?

## Measured (base vs base+kv)

| Layer | 12-gram replay recall | fresh recall | ms / 256 tok | worst-case MB | threads | KV bounded | GPT-2 delta (nats) |
|---|---|---|---|---|---|---|---|
| base (main) | 0.9945 / 0.0 | 5.3 | 10.7 | 0 | - | 0.0 |
| base+kv (e19) | 0.9945 / 0.0 | 8.9 | 21.4 | 0 | yes | 0.0 |

## Winners

- **fastest**: base (main)
- **lowest_memory**: base (main)
- **best_replay_recall**: base (main)
- **smallest_attach_perturbation**: base (main)
- **thread_safe**: ['base (main)', 'base+kv (e19)']
- **bounded_kv**: ['base+kv (e19)']

## Claims (measured in THIS run)

- **recall_parity_with_base**: True
- **kv_stays_bounded**: True
- **evictions_go_to_long_term**: True
- **memory_under_100mb_ceiling**: True
- **thread_safe**: True
- **gpt2_attach_finite**: True
- **gpt2_delta_small**: True
- **speed_overhead_vs_base**: 1.68

Reference (e18, same harness): combined (all 4 branches) measured
26.4 ms/256tok and 42.9 MB worst-case with the same bounded-KV
behaviour; base was 2.9 ms and 10.7 MB.

## Full log

```
======================================================================
BASE + KV WRAPPER RUN (e19)  sha=998c924bb9fbd6e7e79654c31a32bf1d08068a0f
======================================================================
corpus tokens: 2000
comparing: base (main), base+kv (e19)

--- base (main) ---

--- base+kv (e19) ---

--- real GPT-2 A/B (base vs base+kv) ---
```