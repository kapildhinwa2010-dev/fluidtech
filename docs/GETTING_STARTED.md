# Getting started

This guide covers installation and the first steps for the NumPy reference
implementation. The optional PyTorch adapter is described separately below.

## Install

From a repository checkout:

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

The core package requires NumPy. PyTorch is optional and is needed only for
the Torch adapter and GPU-oriented experiments:

```bash
python -m pip install -e ".[gpu]"
```

## Run the examples

From the repository root:

```bash
python selftest.py
python examples/quickstart.py
python examples/graft_demo.py
```

The self-test checks project claims; the examples demonstrate the NumPy
implementation and the graft behavior. To run the unit tests, install the
development extra and use `make test`.

## Use the NumPy reference

`SAABranch` operates on residual vectors, position-free content vectors,
and the token IDs consumed at those positions:

```python
import numpy as np

from superfluid import SAABranch, TinyTransformer

model = TinyTransformer(vocab=32, d=32, n_layers=2, n_ctx=32, seed=0)
branch = SAABranch(M=16, r=4, d=32, emb=model.emb, ctx_len=1)
token_ids = np.array([3, 7, 11, 5], dtype=np.int64)

residual = model.emb[token_ids] + model.pos[:len(token_ids)]
content = model.emb[token_ids]
residual_with_memory, info = branch.forward(
    residual, content, token_ids
)
logits = model.forward_from_h(residual_with_memory)
```

The output projections start at zero, so a newly attached branch initially
leaves the model output unchanged. Retrieval behavior must be enabled and
calibrated deliberately; see the [graft measurements](GRAFT.md).

## Attach the optional PyTorch adapter

Install the `gpu` extra before importing `SAABranchTorch`. The adapter's
`forward` method takes all three inputs:

```python
import torch

from superfluid.torch_graft import SAABranchTorch

embedding = model.get_input_embeddings()
branch = SAABranchTorch(
    M=64,
    r=8,
    d=embedding.weight.shape[1],
    emb=embedding,
    ctx_len=1,
    use_exact=True,
    use_lossy=False,
)

input_ids = next_window
content = embedding(input_ids[0])
residual = content
memory_residual, info = branch(
    residual, content, input_ids[0]
)
logits = model(inputs_embeds=memory_residual.unsqueeze(0)).logits[0]
gated_logits = branch.gated_logits(logits, info)
```

This is an API example, not a complete serving integration. A real
integration must handle the model's positional embeddings, attention masks,
device placement, session boundaries, and task-specific gain calibration.
Call `branch.reset()` when starting a new independent session.

## Memory behavior and safety

The attachable store is bounded. Its reported 20-byte slot cost is a logical
payload estimate; Python dictionaries and objects use additional memory.
The store tracks majority continuations and evicts the least-corroborated
entry when full. Majority support is not proof that retrieved content is
true.

`FrozenRegister` is a separate, lower-level append-only helper. It refuses
overflow and supports explicit serialization, but is not tamper-evident.
Neither component encrypts data or provides tenant isolation. Use a separate
branch per session, and do not treat retrieved content as trusted. See the
[security policy](../SECURITY.md) and [limitations](LIMITATION.md).

## Next steps

- [API reference](API.md)
- [Measured graft behavior](GRAFT.md)
- [Reproducing tests and results](REPRODUCING.md)
