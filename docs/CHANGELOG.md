# Changelog

Notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.4.0] - 2026-11-06

### Added

- The three "coming soon" features, implemented and CI-measured (e21,
  verification-results): `fluidlayer/quantization.py` (int8, 4.0x
  smaller, round-trip error 0.3936%), `fluidlayer/multi_gpu.py` (device
  detection + a sharded bounded KV cache, shards stay balanced on 4
  virtual devices), `fluidlayer/acceleration.py` (framework detection +
  a vectorized fast path, 40.34x faster than the loop path on CI CPU),
  plus their unit tests (11/11 PASS) and the features-test workflow that
  re-measures them on every push.
- e20: the short-budget fine-tune + KV wrapper experiment
  (`experiments/e20_finetune_kv.py` + workflow). Measured on CI: a
  5-minute LoRA fine-tune on tiny-gpt2 keeps the KV cache bounded
  (max 256, 1280 evictions) and the wrapper loss-neutral (attach delta
  0.0); the train loss stays flat on the ~100k-param model
  (noise-level, as the README notes).
- The e21 modules are importable from the package root via PEP 562
  lazy imports: from fluidlayer import quantize_int8, ShardedKVCache,
  detect_acceleration (the package stays numpy-only at import time;
  multi_gpu still needs torch, so it loads lazily).

### Changed

- README restructured for readability (44 KB -> ~13 KB): configuration,
  performance, security, GGUF, usage examples, advanced features and
  troubleshooting moved to docs/USAGE.md; the 5-step Colab guide moved
  to docs/COLAB_GUIDE.md. Dynamic CI + PyPI badges added, the install
  section now leads with the PyPI package, and the file layout is
  accurate again. Fixed 8 hard line-wraps that broke copy-paste code in
  the Colab cells (e.g. the SAABranchBulletproof import was split across
  lines) and one broken table row in the e19 results table.

## [2.3.0] - 2026-11-06

### Added

- The MAIN layer: base + bounded KV cache (`fluidlayer.base_kv_layer` →
  `BaseKVLayer` / `BaseKVConfig`). Digest-keyed exact tier (12-gram context
  keys) + a KV window that never grows past `kv_max_size` + long-term
  memory fed by every KV eviction (REAL hidden states). Measured on CI
  (e19, verification-results): replay recall 0.9945, KV max 256 while
  2000 tokens streamed, 1744 evictions all kept, worst-case ~21.4 MB,
  0 thread errors, tiny-gpt2 loss delta 0.0000.
- `tests/test_base_kv.py`: a CPU self-test for the main layer (skips
  when torch is absent, like tests/test_torch_parity.py).
- The optional layers (superfluid mamba, ultra, combined) ship in the
  wheel: the `experiments`, `kv_cache`, and `combined` companions are
  now packaged, so `from fluidlayer import BaseKVLayer` works from an
  installed wheel.
- `from fluidlayer import BaseKVLayer` via a PEP 562 lazy import (the
  package itself stays numpy-only at import time).
- A PyPI release trigger workflow: pushing a commit that touches
  `.github/workflows/pypi-dispatch.yml` (or "Run workflow") dispatches
  release.yml, which re-runs all checks and publishes with trusted
  publishing (no tag needed, no token in the repo).

### Changed

- verify-claims CI measures the new README claims with the main
  layer's own self-test instead of the retired e16 harness.


## [2.2.0] - 2026-11-03

### Added

- Include the optional deployment session-isolation companion in the
  published wheel and source distribution.

### Changed

- Add clean-install CI verification for the deployment package.

## [2.1.0] - 2026-11-03

### Added

- Added the optional PyTorch Qwen2 adapter, E15 scale-model experiment,
  archived measurements, and a runnable app-kit example.
- Categorized the new project-process, app-kit, experiment, and result files
  into `docs/`, `examples/`, `experiments/`, and `results/`.

### Changed

- Improved the Markdown documentation structure, added a documentation
  landing page, and corrected the getting-started/API examples.
- Clarified the distinction between the bounded majority-tracking branch
  store and the low-level append-only `FrozenRegister`.

## [2.0.0] - 2026-11-02

### Changed

- Renamed the installable distribution and Python package from
  the previous name to `fluidlayer`. Update installation commands to
  `python -m pip install fluidlayer` and imports to `from fluidlayer ...`.
  This is a breaking change; no legacy import namespace is provided.
- Renamed the standalone verification bundle and downloadable source
  archive to use the `fluidlayer` name.
- Enabled PyPI Trusted Publishing for tagged releases. Configure the PyPI
  publisher for this repository, `.github/workflows/release.yml`, and the
  `pypi` GitHub Actions environment before publishing.

## [1.2.0] - 2026-10-29

### Added

- `fluidlayer/graft.py`: NumPy attachable memory layer with an exact store
  and a lossy diffusion tier.
- `fluidlayer/torch_graft.py`: optional PyTorch adapter for pretrained
  language models.
- Graft benchmarks for attach behavior, perplexity, and cross-window recall.
- A byte-level BPE tokeniz
er and real-text experiments.
- Self-verification groups for the memory register, logit override, budget
  law, frustration index, Hodge participation, and graft.

### Changed

- Split `TinyTransformer.forward` into embedding and
  `forward_from_h` continuation paths.
- Moved the design documentation into `docs/`.
- Expanded the self-verification harness.

### Fixed

- Corrected the scalar signed-sheaf balance diagnostic.
- Fixed the content-keyed store and tied-readout injection selectivity.
- Added tensor support to `logit_override`.
- Included the graft module in the standalone bundle.

### Measured

- Attaching a newly initialized layer leaves logits bit-identical.
- The exact tier improves cross-window recall in the reported benchmark;
  the diffusion tier alone does not.
- The earlier diffusion-tier training setup degraded the attached model.
  The revised training signal removed that measured damage but did not show
  standalone upside on the reported real-text task.

## [1.1.0]

### Added

- The low-level frozen register, domain-prior artifact, and E1/E2/E4/E5
  experiments.

## [1.0.0]

### Added

- The first pure-NumPy sheaf-diffusion language-model framework and its
  self-verification harness.
