# Changelog

Notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- A byte-level BPE tokenizer and real-text experiments.
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
