# Layer comparison (all branches)

- commit: f6ef8fc44c76f65a91940f7dae67d11e3bde4e46
- branch: combined-layer
- date: 2026-11-10 01:02:11 UTC
- identical harness for every layer: d=16, ctx_len=12, capacity 25k/100k, same 2000-token corpus slice, same GPT-2 chunks

## Measured comparison

| Layer | 12-gram replay recall | fresh recall | ms / 256 tok | worst-case MB | threads | KV bounded | GPT-2 delta (nats) |
|---|---|---|---|---|---|---|---|
| base (main) | 0.9945 / 0.0 | 3.9 | 10.7 | 0 | - | 0.0 |
| mamba (experiments) | 0.994 / 0.0 | 9.0 | 10.7 | 0 | - | 0.0017 |
| ultra (experiments) | 0.994 / 0.0 | 22.2 | 21.4 | 0 | - | 0.0001 |
| kv (kv_cache) | n/a (wrapper) | 5.1 | 10.7 | 0 | yes | 0.0 |
| combined (all 4) | 0.9945 / 0.0 | 37.9 | 42.9 | 0 | yes | 0.0001 |

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
LAYER COMPARISON RUN  sha=f6ef8fc44c76f65a91940f7dae67d11e3bde4e46
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