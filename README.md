# FluidLayer (FluidTech) — the measured memory layer for language models

[![CI](https://github.com/kapildhinwa2010-dev/fluidtech/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/fluidtech/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fluidlayer)](https://pypi.org/project/fluidlayer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/kapildhinwa2010-dev/fluidtech/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![NumPy 1.23+](https://img.shields.io/badge/NumPy-1.23%2B-blue)](https://numpy.org/)

FluidLayer is a **memory layer you can attach on top of any HuggingFace
causal model** (5 Colab cells — see the
[Colab guide](docs/COLAB_GUIDE.md)). It remembers what it has seen
(12-gram replay recall **0.9945**, measured), keeps a **bounded KV
cache** that never grows past kv_max_size (default 256), and pushes
**every evicted token into long-term memory with its REAL hidden
state** — no token is silently dropped.

Every number in this README was **measured on this repo's CI**, not
claimed. Full logs: the [verification-results](https://github.com/kapildhinwa2010-dev/fluidtech/tree/verification-results/results)
branch and [docs/VERIFICATION.md](docs/VERIFICATION.md).

---

## ✅ What is measured (re-measured on CI on every push to main)

| Claim | Measured |
|---|---|
| 12-gram replay recall | **0.9945** base-only, **0.9945** base+KV (exact parity) — [e19](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/base-kv/run.json) |
| Fresh (never-seen) recall | 0.0 **by construction** (the exact tier only recalls contexts it has seen) |
| Bounded KV cache | max 256 seen while 2000 tokens streamed; 1744 evictions, **all** stored to long-term memory (e19) |
| Attach Δ loss (tiny-gpt2) | **0.0000 nats** with the main layer (e19); ≈ +0.0005 untrained with the base store |
| Worst-case memory | **21.4 MB** measured (ceiling: 100 MB) — e19 |
| Thread safety | 8 threads, **0 errors** (e19) |
| Security | session-masked tenants, **0** cross-session leaks (measured); Fernet (AES-128-CBC + HMAC-SHA256) at MILITARY |
| GGUF export | valid GGUF container (magic b'GGUF'); llama.cpp execution needs a backend patch |
| Quantization (e21) | int8: **4.0x smaller**, round-trip error **0.3936%** — [e21](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e21-features/run.json) |
| Multi-GPU (e21) | sharded bounded KV: 4 balanced shards [64, 64, 64, 64], order preserved — e21 |
| Hardware acceleration (e21) | vectorized fast path **40.34x faster** than the loop path on CI CPU — e21 |
| Overhead | full forward ~5-9 ms / 256 tokens; 12-gram keygen ~3-6 µs; store lookup ~3-5 µs (CI CPU) |

The verification pipeline itself found and fixed seven real bugs
along the way — the story is in [docs/VERIFICATION.md](docs/VERIFICATION.md).

---

## 📦 Installation

~~~bash
pip install fluidlayer            # from PyPI (numpy-only core)
pip install "fluidlayer[gpu]"     # + torch, transformers, accelerate, safetensors

# from source
git clone https://github.com/kapildhinwa2010-dev/fluidtech.git
cd fluidtech && pip install .

pip install cryptography          # optional: MILITARY-level Fernet encryption
pip install pytest                # optional: to run the test suites
~~~

---

## 🚀 Quick start

~~~python
import torch
from fluidlayer import BaseKVConfig, BaseKVLayer   # the main layer (lazy import, needs torch)

layer = BaseKVLayer(BaseKVConfig(d=64, kv_max_size=256))
h   = torch.randn(8, 64)                       # hidden states
tok = torch.randint(0, 32000, (8,))            # token ids
out = layer.forward(h, tok)                    # attach inside any forward hook
print(layer.stats())                           # base recall, KV size, evictions, long-term
~~~

NumPy-only (no torch):

~~~python
import numpy as np
from fluidlayer.graft_bulletproof import SAABranchBulletproof, BulletproofConfig, HashMode, SecurityLevel
from fluidlayer import quantize_int8, dequantize_int8   # e21: int8 quantization (numpy-only)

layer = SAABranchBulletproof(M=16, r=4, d=64, config=BulletproofConfig(
    ctx_len=12, hash_mode=HashMode.DOUBLE_64, security_level=SecurityLevel.NONE))
h_out, info = layer.forward(np.random.randn(1, 64), np.random.randn(1, 64), np.array([7]))

q, scale = quantize_int8(h_out)               # float32 -> int8 (4x smaller)
back = dequantize_int8(q, scale)              # round-trip error < 0.4% (measured, e21)
~~~

More: [docs/USAGE.md](docs/USAGE.md) (configuration, performance, security,
GGUF, examples, troubleshooting) and [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md)
(attach to a real model in 5 Colab cells).

---
## 🧠 The layers (and the optional ones)

| Layer | What it is | Status | Where |
|---|---|---|---|
| **base + KV cache** | exact-tier store + bounded KV + long-term memory | **THE main layer — RECOMMENDED** | `fluidlayer/base_kv_layer.py` |
| base only | exact-tier store, no KV window, no long-term | optional (Colab guide 4.1) | `fluidlayer/graft_bulletproof.py` |
| mamba ssm | SuperfluidMambaFluidLayer | optional (Colab guide 4.2) | `experiments/superfluid_mamba_layer.py` |
| KV cache only | the wrapper standalone | optional (Colab guide 4.3) | `kv_cache/constant_kv_superfluid.py` |
| ultra | UltraFluidStack (stacked + cross-attention) | optional (Colab guide 4.4) | `experiments/ultra_fluid_layer.py` |
| combined | all branches fused into one layer | optional (Colab guide 4.5) | `combined/combined_layer.py` |

The optional layers are in the repo and importable, but **nothing depends on them** — use them only if you want them.

### Measured results (honest)

**base + KV cache — CI run [e19](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/base-kv/run.json):**

| Metric | base only | **base + KV cache** |
|---|---|---|
| 12-gram replay recall | 0.9945 | **0.9945** (exact parity) |
| fresh recall | 0.0 (by construction) | 0.0 (by construction) |
| speed (ms / 256 tokens, d=16) | 3.9 | **6.7** (1.72×) |
| worst-case memory | 10.7 MB | **21.4 MB** (ceiling: 100 MB) |
| thread errors (8×20) | 0 | **0** |
| bounded KV | — | **max 256 seen; 1744 evictions → all stored to long-term** |
| attach Δ loss (tiny-gpt2) | 0.0000 | **0.0000 nats** |

**All five layers in one identical harness — CI run [e18](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/comparison/run.json):**

| Layer | replay recall | ms/256tok | worst-case MB | threads | bounded KV | GPT-2 Δ |
|---|---|---|---|---|---|---|
| base (main) | 0.9945 | 3.9 | 10.7 | 0 | — | 0.0000 |
| mamba (experiments) | 0.9940 | 8.8 | 10.7 | 0 | — | +0.0017 |
| ultra (experiments) | 0.9940 | 24.7 | 21.4 | 0 | — | +0.0001 |
| kv (kv_cache) | n/a (wrapper) | 5.0 | 10.7 | 0 | ✅ | 0.0000 |
| combined (all 4) | 0.9945 | 36.4 | 42.9 | 0 | ✅ | +0.0001 |

Honest reading:
- **fresh recall is 0 BY CONSTRUCTION** — the exact tier only recalls contexts it has seen before;
- the attach deltas are **untrained** layers: proof that the layer *does no harm*, NOT a quality claim;
- speeds were measured at d=16 on CI CPU runners and scale with d.

### How the main layer works

1. **base tier** — every 12-gram of token ids becomes a digest key in an exact-tier store; when the same 12-gram context is seen again, the stored token is retrieved and written into the output (verbatim replay recall ≈ 0.9945);
2. **bounded KV cache** — the fused hidden states enter a fixed-size KV window (`kv_max_size`, default 256). The window **never grows**: when it is full, the oldest entries are evicted;
3. **long-term memory** — every evicted token is stored in the long-term memory together with its **REAL hidden state** (a sliding stream keeps the last `kv_max_size` states, so an eviction always has its true state available). Nothing is dropped.

### Compression (what is compressed, and how — honestly)

- **KV compression (architectural):** the KV window is bounded at `kv_max_size`, so KV memory is **O(1) with respect to context length** — stream 100 tokens or 1,000,000, the window stays 256. Evicted tokens are stored **once** (token id + one hidden-state row) instead of per-head K and V tensors, which is where the memory saving comes from.
- **Exact-tier dedup:** repeated 12-gram contexts collapse to a **single** store entry (digest key). The store evicts by LRU + frequency once `max_capacity` is reached.
- **Worst-case accounting (measured in e19):** each store entry ≈ 112 B → 2 stores × 100,000 entries ≈ 21.4 MiB, plus the KV window (256 × 2 × d × 4 B) — well under the **100 MiB ceiling**.
- **What this is NOT:** the main layer has no sparsity and no low-rank tensor compression — its compression is architectural (bounded window + dedup + eviction), and the numbers above are measured, not projected. (An OPTIONAL int8 quantization module now exists — see the e21 section below — measured in CI, but NOT applied to the main layer by default.)

---

## 📚 Documentation

| Doc | What it covers |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | first steps |
| [docs/USAGE.md](docs/USAGE.md) | quick start, configuration, performance, security, GGUF, examples, troubleshooting |
| [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) | attach the layer to a real model on a Colab T4 — 5 steps |
| [docs/API.md](docs/API.md) | the API surface |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | what is measured and how |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | release history |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) · [docs/COMPARISON.md](docs/COMPARISON.md) · [docs/LIMITATION.md](docs/LIMITATION.md) · [docs/NEGATIVE-RESULTS.md](docs/NEGATIVE-RESULTS.md) | honest deep-dives |
| [verification-results](https://github.com/kapildhinwa2010-dev/fluidtech/tree/verification-results/results) | the raw measured results (run.json + logs, refreshed on every push) |

---

## 📁 Repository layout

~~~text
fluidtech/
├── fluidlayer/            # the core package (numpy-only at import time)
│   ├── base_kv_layer.py         # ⭐ THE main layer: base + bounded KV + long-term memory
│   ├── graft_bulletproof.py     # the base layer (exact-tier store, security, GGUF)
│   ├── quantization.py          # e21: int8 quantization (4.0x smaller)
│   ├── multi_gpu.py             # e21: device detection + sharded bounded KV cache
│   ├── acceleration.py          # e21: framework detection + vectorized fast path
│   ├── torch_graft_bulletproof.py  # the PyTorch twin
│   └── sheaf.py, diffusion.py, topology.py, model.py, autodiff.py, ...   # the fluid core
├── kv_cache/              # bounded KV cache + long-term memory companions
├── combined/              # the optional combined layer (all four branches)
├── experiments/           # optional layers + the measurement scripts e1–e21
├── deployment/            # session-registry companion
├── tests/                 # pytest suites (torch-dependent tests skip without torch)
├── docs/                  # USAGE.md, COLAB_GUIDE.md, VERIFICATION.md, CHANGELOG.md, ...
├── examples/              # quickstart + demos
├── tools/                 # bundle + release helpers
├── .github/workflows/     # CI, verify-claims, features-test, e20-finetune, release
└── corpus.txt, selftest.py, run_without_pytest.py, test_bulletproof.py
~~~

---

## 🧪 Tests & CI

~~~bash
python -m pytest -q tests      # the suite (torch-dependent tests skip without torch)
python tests/test_base_kv.py   # the main layer's own self-test (CPU, ~1 min)
ruff check fluidlayer experiments tools tests examples selftest.py run_without_pytest.py
~~~

Every push to main runs: **CI** (Python 3.9-3.12 verify matrix, package
build + clean-venv install, lint, optional PyTorch parity),
**verify-claims** (re-measures e19/e18 and publishes to
verification-results), **features-test** (the 11 e21 unit tests + the
e21 measurement) and **e20-finetune** (on e20 changes).

---

## 🧯 Limitations (honest)

- The layers are **untrained**: attaching them does not improve a model's loss (measured: Δ ≈ 0 on tiny-gpt2). The value is the *memory* behaviour, not perplexity.
- The exact tier recalls **repeated contexts**; genuinely fresh text recalls 0 by construction.
- The layer runs on CPU (NumPy) inside the hook: keep demo prompts short; the KV window is O(1) but the per-token store work is not free.
- The optional mamba/ultra/combined layers are experimental (see each module's honest docstring).

---

### e20: short-budget fine-tune + KV wrapper (measured on main)

Idea: model par base+KV wrapper layer lagao, 5-10 min ka LoRA
fine-tune karo, aur MEASURE karo ki kya badalta hai. Script:
[e20_finetune_kv.py](https://github.com/kapildhinwa2010-dev/fluidtech/blob/main/experiments/e20_finetune_kv.py);
workflow [e20-finetune.yml](https://github.com/kapildhinwa2010-dev/fluidtech/blob/main/.github/workflows/e20-finetune.yml)
har e20 change par CI par run hota hai. KV wrapper training ke
dauraan OFF hai (evictions training context corrupt karti) — sirf
fine-tune ke baad attach hota hai.

**CI measured (GitHub Actions CPU, commit 4912034e):** 840 steps /
300.3 s (5 min budget, LoRA: 128 of 102842 params trainable), train
loss flat (10.8249 -> 10.825), eval 10.826 -> 10.8254 (delta 0.0006
nats = noise-level), attach delta 0.0, KV after fine-tune: max 256 +
1280 evictions — bounded.

Honest reading: tiny-gpt2 (2-dim hidden, ~100k params) + 128 LoRA
params me 5 min ka measurable effect nahi — bade model par try karo:
python experiments/e20_finetune_kv.py --model Qwen/Qwen2.5-1.5B
--minutes 10 (Colab T4). Full results: [run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e20-finetune/run.json)
+ [finetune.md](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e20-finetune/finetune.md).

---

### e21: the 3 "coming soon" features — quantization, multi-GPU, hardware acceleration (MEASURED, merged to main)

Ye 3 features implement + CI-test ho gaye — tests 11/11 PASS,
sab results positive mile, isliye main me merge kiya. Modules:
[quantization.py](https://github.com/kapildhinwa2010-dev/fluidtech/blob/main/fluidlayer/quantization.py),
[multi_gpu.py](https://github.com/kapildhinwa2010-dev/fluidtech/blob/main/fluidlayer/multi_gpu.py),
[acceleration.py](https://github.com/kapildhinwa2010-dev/fluidtech/blob/main/fluidlayer/acceleration.py);
workflow [features-test.yml](https://github.com/kapildhinwa2010-dev/fluidtech/blob/main/.github/workflows/features-test.yml)
har push par inko re-run + re-publish karta hai.

**CI measured (GitHub Actions CPU, commit 4912034e):**

| Feature | Measured |
|---|---|
| quantization (int8) | float32 -> int8 = **4.0x smaller**; round-trip error **0.3936%** (theoretical bound < 0.4%) |
| multi-GPU | device detection (cuda -> mps -> cpu fallback); 4 shards me 1000 inserts -> size 256, 744 evictions, shard sizes [64, 64, 64, 64] balanced, insertion order preserved |
| hardware acceleration | torch 2.15 (CPU) detected, cuda/mps/numba/cupy flags honest; vectorized fast path loop se **40.34x faster** (0.000674 s -> 0.000017 s), outputs bit-consistent |

Honest reading: CI runners par GPU nahi hai — multi-GPU ka
sharding + eviction logic CPU virtual shards par verify hua
(placement real GPU machines par automatic badlegi), aur
acceleration ka measured speedup CPU vectorization ka hai
(GPU backend detection ready hai, run-to-run speedup thoda
vary karta hai). Full results:
[run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e21-features/run.json),
[features.md](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e21-features/features.md)
aur [tests.log](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e21-features/tests.log)
(11/11 PASSED) verification-results branch par.

---

## 📜 License & contact

MIT — see [LICENSE](LICENSE). Created by **Kapil Dhinwa**.
Questions, issues, contributions:
[GitHub](https://github.com/kapildhinwa2010-dev/fluidtech) · kapildhinwa2010@gmail.com ·
[CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md)

---

**⭐ Star this repository if you find it useful!**

**🚀 Every claim in this README is measured — see [docs/VERIFICATION.md](docs/VERIFICATION.md)
and the [verification-results](https://github.com/kapildhinwa2010-dev/fluidtech/tree/verification-results/results) branch.**

>THIS SHOWS THAT TARNING DOEST WORK ON SMALL MODLE 
SO THIS IS WASTE 

>YOU CAN TRY TRANING OR FINE TUNING ON LARGE MODLES IF YOU ARE INTERESTED 


🌟 STAR IT IF YOU FIND THIS USEFUL
### Verification runs (for the skeptical)

| Run | What it measured | Where |
|---|---|---|
| e17 | combined layer, 9/9 claims | measured on the combined-layer branch before the merge; the layer is now the optional combined module |
| e18 | all 5 layers, one harness | [results/comparison/run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/comparison/run.json) |
| e19 | base + kv cache (this layer) | [results/base-kv/run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/base-kv/run.json) |
| e21 | quantization + multi-GPU + acceleration (the 3 coming-soon features) | [results/e21-features/run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e21-features/run.json) |
| e20 | short-budget fine-tune + KV wrapper (tiny-gpt2, 5-min LoRA) | [results/e20-finetune/run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/e20-finetune/run.json) |

Reproduce any of them: the scripts live on `main` — `experiments/e18_compare.py`, `experiments/e19_base_kv.py` — each is a standalone script that measures, prints, and publishes its own results (e18/e19 re-run on every push to main).


