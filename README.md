# FluidTech — FluidLayer (base + KV cache)

A **memory layer** you can attach on top of any HuggingFace causal model in **5 Colab cells**.

The layer remembers what it has seen (12-gram replay recall **0.9945**, measured), keeps a **bounded KV cache** that never grows past `kv_max_size` (default 256), and pushes **every evicted token into long-term memory with its REAL hidden state** — no token is silently dropped.

Every number in this README was **measured on this repo's CI**, not claimed. Full logs: [verification-results](https://github.com/kapildhinwa2010-dev/fluidtech/tree/verification-results/results).

---

## 1. The layer (and the optional ones)

| Layer | What it is | Status | Where |
|---|---|---|---|
| **base + KV cache** | exact-tier store + bounded KV + long-term memory | **THE main layer — RECOMMENDED** | `fluidlayer/base_kv_layer.py` |
| base only | exact-tier store, no KV window, no long-term | optional (§4.1) | `fluidlayer/graft_bulletproof.py` |
| mamba ssm | SuperfluidMambaFluidLayer | optional (§4.2) | `experiments/superfluid_mamba_layer.py` |
| KV cache only | the wrapper standalone | optional (§4.3) | `kv_cache/constant_kv_superfluid.py` |
| ultra | UltraFluidStack (stacked + cross-attention) | optional (§4.4) | `experiments/ultra_fluid_layer.py` |
| combined | all branches fused into one layer | optional (§4.5) | `combined/combined_layer.py` |

The optional layers are in the repo and importable, but **nothing depends on them** — use them only if you want them.

## 2. Measured results (honest)

**base + KV cache — CI run [e19](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/base-kv/run.json):**

| Metric | base only | **base + KV cache** |
|---|---|---|
| 12-gram replay recall | 0.9945 | **0.9945** (exact parity) |
| fresh recall | 0.0 (by construction) | 0.0 (by construction) |
| speed (ms / 256 tokens, d=16) | 5.3 | **8.9** (1.68×) |
| worst-case memory | 10.7 MB | **21.4 MB** (ceiling: 100 MB) |
| thread errors (8×20) | 0 | **0** |
| bounded KV | — | **max 256 seen; 1744 evictions → all stored to long-term** |
| attach Δ loss (tiny-gpt2) | 0.0000 | **0.0000 nats** |

**All five layers in one identical harness — CI run [e18](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/comparison/run.json):**

| Layer | replay recall | ms/256tok | worst-case MB | threads | bounded KV | GPT-2 Δ |
|---|---|---|---|---|---|---|
| base (main) | 0.9945 | 2.9 | 10.7 | 0 | — | 0.0000 |
| mamba (experiments) | 0.9940 | 6.4 | 10.7 | 0 | — | +0.0017 |
| ultra (experiments) | 0.9940 | 17.7 | 21.4 | 0 | — | +0.0001 |
| kv (kv_cache) | n/a (wrapper) | 3.7 | 10.7 | 0 | ✅ | 0.0000 |
| combined (all 4) | 0.9945 | 26.4 | 42.9 | 0 | ✅ | +0.0001 |

Honest reading:
- **fresh recall is 0 BY CONSTRUCTION** — the exact tier only recalls contexts it has seen before;
- the attach deltas are **untrained** layers: proof that the layer *does no harm*, NOT a quality claim;
- speeds were measured at d=16 on CI CPU runners and scale with d.

## 3. How the main layer works

1. **base tier** — every 12-gram of token ids becomes a digest key in an exact-tier store; when the same 12-gram context is seen again, the stored token is retrieved and written into the output (verbatim replay recall ≈ 0.9945);
2. **bounded KV cache** — the fused hidden states enter a fixed-size KV window (`kv_max_size`, default 256). The window **never grows**: when it is full, the oldest entries are evicted;
3. **long-term memory** — every evicted token is stored in the long-term memory together with its **REAL hidden state** (a sliding stream keeps the last `kv_max_size` states, so an eviction always has its true state available). Nothing is dropped.

## 4. Compression (what is compressed, and how — honestly)

- **KV compression (architectural):** the KV window is bounded at `kv_max_size`, so KV memory is **O(1) with respect to context length** — stream 100 tokens or 1,000,000, the window stays 256. Evicted tokens are stored **once** (token id + one hidden-state row) instead of per-head K and V tensors, which is where the memory saving comes from.
- **Exact-tier dedup:** repeated 12-gram contexts collapse to a **single** store entry (digest key). The store evicts by LRU + frequency once `max_capacity` is reached.
- **Worst-case accounting (measured in e19):** each store entry ≈ 112 B → 2 stores × 100,000 entries ≈ 21.4 MiB, plus the KV window (256 × 2 × d × 4 B) — well under the **100 MiB ceiling**.
- **What this is NOT:** there is no quantization, no sparsity and no low-rank tensor compression here. The compression is architectural (bounded window + dedup + eviction), and the numbers above are measured, not projected.

## 5. Repository layout

`text
fluidlayer/base_kv_layer.py      # THE main layer (base + kv cache) — recommended
fluidlayer/graft_bulletproof.py  # the base layer (exact-tier store)
kv_cache/                        # bounded KV + long-term memory (merged from the kv_cache branch)
experiments/                     # OPTIONAL layers (mamba ssm, ultra) + measurement scripts e1–e19
combined/                        # OPTIONAL combined layer (all branches fused)
tests/test_base_kv.py            # quick self-test for the main layer (CPU, ~1 min)
`

## 6. Tests (updated)

- **Quick self-test (CPU, no GPU):**
  `bash
  python tests/test_base_kv.py
  `
  6 checks: output finite · KV bounded · evictions→long-term · replay recall ~0.99 (fresh ~0) · thread safety · worst-case memory < 100 MB.
- **CI verification history** (all runs publish to the `verification-results` branch):
  - [e17 combined-layer verification](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/combined/run.json) — 9/9 claims CONFIRMED;
  - [e18 five-layer comparison](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/comparison/run.json) — the table in §2;
  - [e19 base+kv run](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/base-kv/run.json) — all 7 claims TRUE (recall parity, bounded KV, evictions→long-term, memory ceiling, threads, finite attach, small attach delta).

---

# 7. Implementation: run it on a real model (Google Colab, 5 steps)

Open Colab, run the cells in order, and the layer is attached to the model with tests run automatically.

## Step 1 — open Google Colab and choose a T4 GPU

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with a Google account;
2. **File → New notebook**;
3. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**

The T4 (16 GB) is free-tier and is enough for a 1.5B model in fp32.

## Step 2 — load the layer from GitHub

`python
# Cell 1 — load the layer from GitHub
!git clone -q https://github.com/kapildhinwa2010-dev/fluidtech.git
%cd fluidtech
!git checkout -q unified-layer
!pip install -q "numpy>=1.23,<3" cryptography torch transformers

import sys, os
sys.path.insert(0, os.getcwd())

from fluidlayer.base_kv_layer import BaseKVConfig, BaseKVLayer   # the main layer
print("layer loaded OK")
`

## Step 3 — load the model (example: Qwen 1.5B)

`python
# Cell 2 — load the model (any HuggingFace causal LM works)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"        # example: Qwen 1.5B
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to("cuda").eval()

D = model.config.hidden_size
print("model loaded OK, hidden size:", D)
`

## Step 4 — (RECOMMENDED) the layer: base + KV cache — full code

Run this cell to build **the main layer**. It sets `LAYER_FWD`, which Step 5 attaches to the model.

`python
# Cell 3 — (RECOMMENDED) the layer: base + kv cache — FULL CODE
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass

from fluidlayer.graft_bulletproof import (
    BulletproofConfig, HashMode, SAABranchBulletproof, SecurityLevel,
)
from experiments.superfluid_mamba_layer import (
    SuperfluidMambaConfig, SuperfluidMambaFluidLayerNP,
)
from kv_cache.constant_kv_superfluid import (
    ConstantKVCache, KVCacheConfig, SuperfluidMemory,
)

@dataclass
class BaseKVConfig:
    d: int = 64                 # model width (set to D from Cell 2)
    M: int = 16                 # number of anchors
    r: int = 4                  # rank of each anchor
    ctx_len: int = 12           # n-gram context length for exact keys
    capacity: int = 25000       # initial store capacity
    max_capacity: int = 100000  # hard store ceiling (worst-case memory)
    kv_max_size: int = 256      # bounded KV window — never exceeded

class BaseKVLayer:
    """base (exact-tier store) + bounded KV cache + long-term memory.

    Every KV eviction is stored to the long-term memory with its REAL
    hidden state (sliding stream pattern) — no token is dropped.
    """
    def __init__(self, config: BaseKVConfig):
        self.config = config
        d = config.d
        # 1) base tier: digest-keyed 12-gram exact store (main branch)
        self.base = SAABranchBulletproof(
            M=config.M, r=config.r, d=d,
            config=BulletproofConfig(
                ctx_len=config.ctx_len, d=d,
                hash_mode=HashMode.DOUBLE_64,
                security_level=SecurityLevel.NONE,
                capacity=config.capacity,
                max_capacity=config.max_capacity,
                use_gated_sheaf=False,
            ),
        )
        # 2) long-term memory (kv_cache branch), paired with the NumPy
        #    fluid layer — the pairing that branch intended
        self.np_layer = SuperfluidMambaFluidLayerNP(
            SuperfluidMambaConfig(
                M=config.M, r=config.r, d=d, ctx_len=config.ctx_len,
                capacity=config.capacity, max_capacity=config.max_capacity,
                security_level="STANDARD",
            )
        )
        self.memory = SuperfluidMemory(self.np_layer)
        # 3) bounded KV cache — this window NEVER grows past kv_max_size
        self.kv_cache = ConstantKVCache(
            KVCacheConfig(max_size=config.kv_max_size, num_heads=1, head_dim=d)
        )
        self._stream: deque = deque()   # real hidden states for evictions

    def forward(self, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        h_np = h.detach().cpu().numpy().astype(np.float64)
        tid_np = token_ids.detach().cpu().numpy().astype(np.int64)
        out, _ = self.base.forward(h_np, h_np.copy(), tid_np)
        fused = torch.from_numpy(np.asarray(out, dtype=np.float32))

        # bounded KV update; evictions -> long-term memory, REAL states
        k = fused.detach().unsqueeze(0).unsqueeze(0)      # (1, 1, S, d)
        before = len(self.kv_cache.evicted_tokens)
        self.kv_cache.update(k, k.clone(), token_ids.unsqueeze(0))
        n_evict = len(self.kv_cache.evicted_tokens) - before
        for t in range(h.shape[0]):
            self._stream.append((fused[t].detach(), int(tid_np[t])))
        if n_evict > 0:
            rows, ids = [], []
            for _ in range(n_evict):
                if self._stream:
                    row, tok = self._stream.popleft()
                    rows.append(row); ids.append(tok)
            if rows:
                self.memory.store(
                    torch.stack(rows), torch.tensor(ids, dtype=torch.long)
                )
        while len(self._stream) > self.config.kv_max_size:
            self._stream.popleft()
        return fused

    def stats(self):
        return {
            "base": self.base.get_stats(),
            "kv": {"size": self.kv_cache.size(),
                   "max_size": self.config.kv_max_size,
                   "evictions": len(self.kv_cache.evicted_tokens)},
            "long_term": {"tokens": self.memory.total_tokens},
        }

layer = BaseKVLayer(BaseKVConfig(d=D))

def LAYER_FWD(h, tid):
    return layer.forward(h, tid)

print("base + kv cache layer ready (RECOMMENDED)")
`

### 4.1 — (OPTIONAL) only the base layer

Skip Cell 3 and run this instead if you do NOT want the KV window / long-term memory:

`python
# Cell 3a — (OPTIONAL) 4.1: only the base layer
from fluidlayer.graft_bulletproof import (
    BulletproofConfig, HashMode, SAABranchBulletproof, SecurityLevel,
)
import numpy as np, torch

base = SAABranchBulletproof(
    M=16, r=4, d=D,
    config=BulletproofConfig(
        ctx_len=12, d=D, hash_mode=HashMode.DOUBLE_64,
        security_level=SecurityLevel.NONE,
        capacity=25000, max_capacity=100000, use_gated_sheaf=False,
    ),
)

def LAYER_FWD(h, tid):
    h_np = h.detach().cpu().numpy().astype(np.float64)
    out, _ = base.forward(h_np, h_np.copy(), tid.detach().cpu().numpy().astype(np.int64))
    return torch.from_numpy(np.asarray(out, dtype=np.float32))

print("optional layer selected: base only")
`

### 4.2 — (OPTIONAL) with the mamba SSM layer

`python
# Cell 3b — (OPTIONAL) 4.2: mamba ssm
import torch
from experiments.superfluid_mamba_layer import SuperfluidMambaConfig, SuperfluidMambaFluidLayer

mamba = SuperfluidMambaFluidLayer(SuperfluidMambaConfig(
    M=16, r=4, d=D, ctx_len=12, capacity=25000,
    max_capacity=100000, security_level="STANDARD",
))

def LAYER_FWD(h, tid):
    out, _ = mamba(h, h.clone(), tid)
    return out.detach().float()

print("optional layer selected: mamba ssm")
`

### 4.3 — (OPTIONAL) with the KV cache wrapper only

`python
# Cell 3c — (OPTIONAL) 4.3: kv cache wrapper only
import numpy as np, torch
from collections import deque
from experiments.superfluid_mamba_layer import (
    SuperfluidMambaConfig, SuperfluidMambaFluidLayerNP,
)
from kv_cache.constant_kv_superfluid import (
    ConstantKVCache, KVCacheConfig, SuperfluidMemory,
)

np_layer = SuperfluidMambaFluidLayerNP(SuperfluidMambaConfig(
    M=16, r=4, d=D, ctx_len=12, capacity=25000,
    max_capacity=100000, security_level="STANDARD",
))
memory = SuperfluidMemory(np_layer)
cache = ConstantKVCache(KVCacheConfig(max_size=256, num_heads=1, head_dim=D))
stream = deque()

def LAYER_FWD(h, tid):
    fused = h + memory.retrieve(h, tid)
    k = fused.detach().unsqueeze(0).unsqueeze(0)
    before = len(cache.evicted_tokens)
    cache.update(k, k.clone(), tid.unsqueeze(0))
    n_evict = len(cache.evicted_tokens) - before
    for t in range(h.shape[0]):
        stream.append((fused[t].detach(), int(tid[t])))
    if n_evict > 0:
        rows, ids = [], []
        for _ in range(n_evict):
            if stream:
                row, tok = stream.popleft()
                rows.append(row); ids.append(tok)
        if rows:
            memory.store(torch.stack(rows), torch.tensor(ids, dtype=torch.long))
    while len(stream) > 256:
        stream.popleft()
    return fused

print("optional layer selected: kv cache wrapper")
`

### 4.4 — (OPTIONAL) with the ultra stack

`python
# Cell 3d — (OPTIONAL) 4.4: ultra stack
import torch
from experiments.ultra_fluid_layer import UltraFluidConfig, UltraFluidStack

ultra = UltraFluidStack(UltraFluidConfig(
    M=16, r=4, d=D, ctx_len=12, num_layers=2,
    capacity=25000, max_capacity=100000,
    use_cross_attention=(D >= 8 and D % 4 == 0),
))

def LAYER_FWD(h, tid):
    out, _ = ultra(h, h.clone(), tid)
    return out.detach().float()

print("optional layer selected: ultra stack")
`

### 4.5 — (OPTIONAL) with the combined layer (all four branches)

`python
# Cell 3e — (OPTIONAL) 4.5: combined layer
import torch
from combined.combined_layer import CombinedFluidConfig, CombinedFluidLayer

combined = CombinedFluidLayer(CombinedFluidConfig(
    d=D, ctx_len=12, M=16, r=4, kv_max_size=256, ultra_layers=2,
))

def LAYER_FWD(h, tid):
    out, _ = combined(h, h.clone(), tid)
    return out.detach().float()

print("optional layer selected: combined (all 4 branches)")
`

## Step 5 — attach the layer on the model + run the tests

`python
# Cell 4 — attach the layer on the model + run the tests
import torch, numpy as np

def attach(fwd):
    state = {"ids": None, "handle": None}
    def hook(module, args, output):
        hs = output                       # (B, S, d) from the embeddings
        with torch.no_grad():
            fused = fwd(hs[0].float(), state["ids"])
        return fused.unsqueeze(0)
    state["handle"] = model.get_input_embeddings().register_forward_hook(hook)
    return state

def detach(state):
    state["handle"].remove()

results = []
def check(name, ok, info=""):
    results.append(bool(ok))
    print(("PASS" if ok else "FAIL"), name, info)

# ── TEST 1: attach + generate ─────────────────────────────────────
state = attach(LAYER_FWD)
prompt = "Memorize this: the secret number is 4711."
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
detach(state)
print("sample generation:", tok.decode(out[0], skip_special_tokens=True)[:200])
check("1. attach + generate runs", True)

# ── TEST 2: exact-tier replay recall (same text twice) ────────────
text = open("corpus.txt", encoding="utf-8", errors="ignore").read(600)
tids = torch.tensor(tok(text, return_tensors="pt").input_ids[0][:400])
h = torch.randn(len(tids), D)
LAYER_FWD(h, tids)                       # fresh pass — nothing stored yet
lay = layer if hasattr(layer, "stats") else None
if lay is None:
    fresh = replay = None
else:
    fresh = lay.base.get_stats()["recall_rate"]
    lay.reset()
    LAYER_FWD(h, tids)                   # replay pass — 12-grams hit the store
    replay = lay.base.get_stats()["recall_rate"]
    check("2. replay recall high, fresh ~0", replay > 0.3 and fresh < 0.3,
          f"fresh={fresh:.3f} replay={replay:.3f}")

# ── TEST 3: bounded KV + long-term memory ─────────────────────────
if lay is not None:
    big = torch.randint(0, 30000, (1200,))
    for i in range(0, 1200, 128):
        LAYER_FWD(torch.randn(128, D), big[i:i+128])
    st = lay.stats()
    check("3a. KV stays bounded", st["kv"]["size"] <= st["kv"]["max_size"],
          f"kv size={st['kv']['size']}")
    check("3b. evictions -> long-term", st["kv"]["evictions"] > 0
          and st["long_term"]["tokens"] > 0, str(st))

# ── TEST 4: does the layer hurt the model? (loss A/B) ─────────────
sample = tok(text[:1200], return_tensors="pt", truncation=True,
             max_length=256).input_ids.to("cuda")
def fwd_loss(with_layer):
    s = attach(LAYER_FWD) if with_layer else None
    with torch.no_grad():
        loss = model(sample, labels=sample).loss.item()
    if s:
        detach(s)
    return loss

base_loss, with_loss = fwd_loss(False), fwd_loss(True)
check("4. attach does not blow up the loss", abs(with_loss - base_loss) < 0.5,
      f"baseline={base_loss:.4f} with-layer={with_loss:.4f}")

print()
print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED — read the lines above")
`

---

## 8. Limitations (honest)

- The layers are **untrained**: attaching them does not improve a model's loss (measured: Δ ≈ 0 on tiny-gpt2). The value is the *memory* behaviour, not perplexity.
- The exact tier recalls **repeated contexts**; genuinely fresh text recalls 0 by construction.
- The layer runs on CPU (NumPy) inside the hook: keep demo prompts short; the KV window is O(1) but the per-token store work is not free.
- The optional mamba/ultra/combined layers are experimental (see each module's honest docstring).

## 9. Verification runs (for the skeptical)

| Run | What it measured | Where |
|---|---|---|
| e17 | combined layer, 9/9 claims | [results/combined/run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/combined/run.json) |
| e18 | all 5 layers, one harness | [results/comparison/run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/comparison/run.json) |
| e19 | base + kv cache (this layer) | [results/base-kv/run.json](https://github.com/kapildhinwa2010-dev/fluidtech/blob/verification-results/results/base-kv/run.json) |

Reproduce any of them: check out the `combined-layer` branch and look at `experiments/e17_combined.py`, `e18_compare.py`, `e19_base_kv.py` — each is a standalone script that measures, prints, and publishes its own results.
