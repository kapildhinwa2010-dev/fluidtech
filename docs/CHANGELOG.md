Changelog
All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]
Repository maintenance
Correct project links and documentation paths, synchronize package metadata,
remove duplicate and obsolete single-file examples, and restore reproducible
generation of the standalone bundle.

[1.2.0] — 2026-10-29
Added
`superfluid/graft.py` — the attachable memory layer.  One residual-stream
insertion, two tiers (a lossy diffusion state and an exact, content-keyed
store of token ids).  Attaching it is a provable no-op: the output
projections start at zero, so the logits are bit-identical.
`superfluid/torch_graft.py` — the same layer in PyTorch, for attaching
to a real pretrained LLM.  Not executed in the development sandbox.
`superfluid/toplevel`-level API: `SAABranch`, `GRAFT_STATE_BYTES`,
`ctx_hash` are exported from the package root.
`experiments/e6_graft.py` — the graft benchmark: attach safety, the
perplexity effect, and the cross-window verbatim-recall capability.
`experiments/e6_real_corpus.py` — the same measurement on real English,
with a `--tokenizer {char,bpe}` switch and a per-tier ablation.
`experiments/e7_lossy_fix.py` — the fix for the diffusion tier's
training signal (a layer-local objective and a differentiable state).
`experiments/gpu_graft.py` — the GPU-scale driver for a real LLM.
`superfluid/bpe.py` — a byte-level BPE tokenizer in pure Python (no
downloads), so every measurement can be repeated at a realistic token
granularity.
Six self-verification groups covering the frozen register, the logit
override, the budget law, the frustration index, the Hodge participation
and the graft.
Changed
The selftest grew from 21 to 38 checks.
`TinyTransformer.forward` was split into `forward` + `forward_from_h` so a
frozen model can be driven from a modified residual stream.
The design documents moved to `docs/`.
Fixed (all found by running the code)
the `‖L·1‖∞` balance diagnostic was wrong for scalar signed sheaves (the
frustration index is the right one — now verified on 150/150 random cycles);
the per-anchor "last write wins" store was evicted by ordinary filler
tokens, so the store is now content-keyed;
an injection of a token's embedding is not selective when the readout is
tied — the top principal direction of the embedding matrix is now projected
out (`pc1`);
`logit_override` now accepts a framework `Tensor`, not only an ndarray;
the single-file bundle now includes the `graft` module.
Measured results (all reproducible, see `docs/GRAFT.md`)
attaching the layer: bit-identical logits (max diff 0.0);
cross-window verbatim recall: 0.050 → 0.800 (chance 0.125), delivered
entirely by the exact tier — the diffusion tier alone: 0.050;
the diffusion tier, trained through the base model's loss, degrades the
model it is attached to (+0.81 nats char level, +0.98 subword);
the fix removes the damage (+0.0000) but the tier still has no upside:
the optimal injection gain is zero.
[1.1.0]
the frozen register (20 B per fact, bit-exact, append-only, fail-loud);
the domain-prior artifact (`experiments/pretrain_prior.py`);
the E1/E2/E4/E5 experiments.
[1.0.0]
the first working framework: a pure-NumPy sheaf-diffusion LM with a
21-check self-verification harness.