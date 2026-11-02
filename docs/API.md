# API reference

This page covers the public NumPy API and the optional PyTorch graft. For
installation and a first-use walkthrough, see
[Getting started](GETTING_STARTED.md).

## `SAABranch`

`SAABranch` is the NumPy reference implementation of the attachable memory
layer:

```python
from superfluid import SAABranch

branch = SAABranch(
    M=64,
    r=8,
    d=hidden_width,
    emb=embedding_table,
    use_exact=True,
    use_lossy=False,
    capacity=4096,
    ctx_len=1,
)
```

| Argument | Meaning |
|---|---|
| `M`, `r` | Anchor count and rank. The fluid state contains `M × r` values. |
| `d` | Residual/embedding width. Must match `emb` when supplied. |
| `emb` | Embedding table used to look up stored token IDs. |
| `ctx_len` | Number of preceding token IDs used to form a context key. |
| `capacity` | Maximum number of exact-store entries. The least-corroborated entry is evicted when full. |
| `use_exact`, `use_lossy` | Enable the exact and diffusion paths independently. The diffusion path has no demonstrated benefit on the tested real-text task. |
| `state_mode` | `"persistent"` or `"rebuild"` for the diffusion state. |
| `read_transform` | `"raw"`, `"center"`, or a principal-component projection such as `"pc1"`. |
| `read_gain` | `"norm"` or `"raw"`; calibrate retrieval behavior for the target model and task. |

The NumPy branch expects `Tensor` values for `h` and `content`, and a
one-dimensional array of consumed token IDs:

```python
h_out, info = branch.forward(h, content, token_ids)
```

### Methods and properties

| API | Description |
|---|---|
| `forward(h, content, token_ids)` | Returns `(h_out, info)`. `content` is position-free routing input; `token_ids` are the tokens consumed at each position. |
| `reset()` | Clears memory state for a new independent session. |
| `refresh_contrast()` | Invalidates the cached transformed embedding table after embeddings change. |
| `gated_logits(logits, info, ...)` | Applies uncertainty-gated retrieval to logits. Tune and evaluate the gate for your task. |
| `layer_logits(info)` | Returns the layer-local prediction when the corresponding head is enabled. |
| `state_bytes` | Logical payload estimate; excludes Python and allocator overhead. |
| `params()` | Returns trainable NumPy parameters. |

The exact and lossy output projections start at zero, so attaching a new
branch initially leaves the residual unchanged. This initialization is not
a guarantee that a trained or armed branch is behavior-preserving.

The exact store tracks majority continuations and evicts the
least-corroborated entry. A retrieved token ID is bit-exact as stored; its
truth and provenance are not verified.

## `FrozenRegister`

`FrozenRegister` is a lower-level fixed-capacity slot table. It is not the
same implementation as the branch's majority-tracking store:

```python
from superfluid import FrozenRegister

register = FrozenRegister(capacity=1024)
slot = register.promote(value_token_id, key, sig=0)
result = register.recall(key, sig=0)
```

`promote` appends a slot and raises `MemoryError` when the register is full.
`recall` returns `(slot_index, value)` or `None` when the key is not found or
the supplied signature does not match. `to_bytes()` and `from_bytes(...)`
provide explicit serialization.

Each logical slot is 20 bytes:

```text
key uint64 | value int32 | signature uint32 | anchor uint16 | metadata uint16
```

This describes the serialized slot payload, not the Python memory cost of
`SAABranch`'s dictionaries.

## Other public helpers

| Helper | Description |
|---|---|
| `ctx_key(tokens)` / `ctx_hash(tokens)` | Deterministic 64-bit context hash. A hash is an identifier, not cryptographic authentication. |
| `logit_override(logits, token_ids)` | Sets recalled token logits so the selected IDs win the argmax. Use only with a deliberate trust policy. |
| `budget_split(total_bytes, M, r, slot_bytes)` | Estimates a split between fluid state and serialized register slots. |
| `TinyTransformer.forward_from_h(h)` | Continues the included reference model from a supplied residual stream. |

## Optional PyTorch API

Install with `python -m pip install ".[gpu]"`, then construct
`SAABranchTorch` from `superfluid.torch_graft`. Its `forward` signature is
`forward(h, content, token_ids)`; see the executable-shaped example in
[Getting started](GETTING_STARTED.md). The Torch adapter has parity tests,
but that does not constitute validation on production-scale models.
