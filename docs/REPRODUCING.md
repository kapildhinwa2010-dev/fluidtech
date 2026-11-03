# Reproducing tests and results

The commands in this guide are run from the repository root. Core checks use
CPU and NumPy; PyTorch is optional.

## Install development tools

```bash
python -m pip install -e ".[dev]"
```

For the optional Torch parity test:

```bash
python -m pip install -e ".[gpu,dev]"
```

## Run correctness checks

```bash
make test
make lint
bash tools/check_artifact.sh
```

Equivalent direct commands are:

```bash
python selftest.py
python fluidlayer_single.py --selftest
python run_without_pytest.py
python -m pytest -q tests
```

The last command requires pytest. When PyTorch is installed, the optional
parity test runs; otherwise that test is skipped.

## Run experiments

The quick mode is a resource-bounded smoke test. It deliberately skips the
full E2 scaling sweep:

```bash
make quick
# or
python experiments/run_all.py --quick
```

Run the full pipeline only when resources are available:

```bash
python experiments/run_all.py --full
```

The full pipeline can take tens of minutes on CPU. To omit selected steps,
use their names with `--skip`, for example:

```bash
python experiments/run_all.py --skip e2,prior
```

Each experiment accepts `--help` for individual options. A selection of
standalone commands:

| Command | Measures | Default result |
|---|---|---|
| `python experiments/e9_ten_million.py --phase stream`<br>`python experiments/e9_ten_million.py --phase performance` | 10M-token recall and performance sweep | `results/e9_ten_million.json` |
| `python experiments/e1_prior_perplexity.py` | Training-free prior on a frozen model | `results/e1_prior.json` |
| `python experiments/e2_scaling.py` | Anchor and exact-memory budget sweeps | `results/e2_scaling.json` |
| `python experiments/e4_gbalance.py` | Frustration index and spectral-gap relationship | `results/e4_gbalance.json` |
| `python experiments/e5_graft.py` | Exact-memory selection policies | `results/e5_graft.json` |
| `python experiments/e6_graft.py` | Graft behavior and cross-window recall | `results/e6_graft.json` |
| `python experiments/e6_real_corpus.py` | Graft on English text | `results/e6_real.json` |
| `python experiments/e6_real_corpus.py --tokenizer bpe` | Subword-tokenized repeat | `results/e6_real_bpe.json` |
| `python experiments/e7_lossy_fix.py` | Diffusion-tier training behavior | `results/e7_lossy_fix.json` |

The repository includes `corpus.txt`, the generated `prior/` artifacts, and
recorded outputs including `results/selftest.json`, `results/e2_scaling.json`,
`results/e4_gbalance.json`, `results/e9_ten_million.json`,
`results/e14_real_gpt2.json`, `results/mechanism.json`, and
`results/train_results.json`. Regenerate the tokenizer and prior with:

```bash
python -m fluidlayer.bpe --corpus corpus.txt --vocab 512 --out prior/bpe_512.json
python experiments/pretrain_prior.py --text corpus.txt --out prior
```

Experiments may overwrite their output JSON. Review `git status` and the
result diff after a run before replacing a recorded snapshot.
E9's default `--phase all` runs both portions in one process; the two listed
phases let the long stream and model evaluation run separately. The
performance phase requires the completed stream-phase JSON at the same
`--out` path and matching `--tokens`/`--capacity` values.

## Optional pretrained-model experiment

The GPU-oriented driver uses optional dependencies and downloads model
weights:

```bash
python -m pip install -e ".[gpu]"
python experiments/gpu_graft.py --help
python experiments/gpu_graft.py --model Qwen/Qwen2.5-0.5B-Instruct
```

This run can require substantial memory, network access, and compute. Its
cost and results depend on the selected hardware and model. A successful
small-model CPU test does not validate production-scale inference.

The E14 GPT-2 experiment uses the repository's downloader by default for
the checkpoint location (`models/gpt2`). The model weights are not tracked;
the default UTF-8 corpus, `corpus.txt`, is included in the repository:

```bash
python tools/download_gpt2.py --out models/gpt2
python experiments/e14_real_gpt2.py --corpus /path/to/corpus.txt
```

Override `--model`, `--corpus`, or `--out` when your files are elsewhere.

## Qwen2 scale experiment (E15)

The E15 run uses Qwen2.5-0.5B, the custom PyTorch forward pass, and the
checkpoint's Hugging Face tokenizer. The model weights are not included.
Install the optional dependencies, download the checkpoint, then run:

```bash
python -m pip install -e ".[gpu]"
hf download Qwen/Qwen2.5-0.5B --local-dir models/qwen05b
python experiments/e15_scale_model.py --model models/qwen05b
```

The default run evaluates the attachment, perplexity against the kNN-LM
baseline, and a planted-fact recall case. It can take several minutes and
requires enough memory for the model. The checked-in
[`results/e15_scale_model.json`](../results/e15_scale_model.json) is the
archived run; its original checkpoint and hardware environment are not
bundled, so exact reproduction is not guaranteed. `--help` works without
PyTorch or downloaded weights.

## Determinism and interpreting results

Experiments accept a `--seed` option and use fixed random generators. CPU
BLAS implementations may still produce small differences across machines.
Treat the committed JSON files as the recorded run outputs and report the
command and environment when publishing a regenerated value.

For claim context, read the [limitations](LIMITATION.md) and
[negative results](NEGATIVE-RESULTS.md), not only the experiment tables.
