# Layer comparison (all branches)

- commit: 4c43a3b89f7bc3e10832dacdef0facf1620b5be1
- branch: combined-layer
- date: 2026-11-06 10:25:48 UTC
- identical harness for every layer: d=16, ctx_len=12, capacity 25k/100k, same 2000-token corpus slice, same GPT-2 chunks

## Measured comparison

| Layer | 12-gram replay recall | fresh recall | ms / 256 tok | worst-case MB | threads | KV bounded | GPT-2 delta (nats) |
|---|---|---|---|---|---|---|---|
| base (main) | 0.9945 / 0.0 | 3.9 | 10.7 | 0 | - | 0.0 |
| mamba (experiments) | 0.994 / 0.0 | 8.8 | 10.7 | 0 | - | 0.0017 |
| ultra (experiments) | 0.994 / 0.0 | 24.7 | 21.4 | 0 | - | 0.0001 |
| kv (kv_cache) | n/a (wrapper) | 5.0 | 10.7 | 0 | yes | 0.0 |
| combined (all 4) | 0.9945 / 0.0 | 36.4 | 42.9 | 0 | yes | 0.0001 |

## Winners

- **fastest**: base (main)
- **lowest_memory**: base (main)
- **best_replay_recall**: base (main)
- **smallest_attach_perturbation**: base (main)
- **thread_safe**: ['base (main)', 'mamba (experiments)', 'ultra (experiments)', 'kv (kv_cache)', 'combined (all 4)']
- **bounded_kv**: ['kv (kv_cache)', 'combined (all 4)']

## Full log

```
======================================================================
LAYER COMPARISON RUN  sha=4c43a3b89f7bc3e10832dacdef0facf1620b5be1
======================================================================
corpus tokens: 2000
comparing: base (main), mamba (experiments), ultra (experiments), kv (kv_cache), combined (all 4)

--- base (main) ---

--- mamba (experiments) ---

--- ultra (experiments) ---

--- kv (kv_cache) ---

--- combined (all 4) ---

--- real GPT-2 A/B (all layers) ---
```