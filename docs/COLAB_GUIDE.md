# Colab guide: attach the layer to a real model (5 steps)

Moved from the README so it stays short. Open Colab, run the cells in
order, and the layer is attached to the model with tests run
automatically. The optional layer cells (4.1-4.5) are at the end.
See also: [../README.md](../README.md), [USAGE.md](USAGE.md).

---
## The 5 steps

Open Colab, run the cells in order, and the layer is attached to the model with tests run automatically.

### Step 1 — open Google Colab and choose a T4 GPU

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with a Google account;
2. **File → New notebook**;
3. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**

The T4 (16 GB) is free-tier and is enough for a 1.5B model in fp32.

### Step 2 — load the layer from GitHub

~~~python
# Cell 1 — load the layer from GitHub
!git clone -q https://github.com/kapildhinwa2010-dev/fluidtech.git
%cd fluidtech
!git checkout -q main
!pip install -q "numpy>=1.23,<3" cryptography torch transformers

import sys, os
sys.path.insert(0, os.getcwd())

from fluidlayer.base_kv_layer import BaseKVConfig, BaseKVLayer   # the main layer
print("layer loaded OK")
~~~

### Step 3 — load the model (example: Qwen 1.5B)

~~~python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

# Automatically picks "cuda" if available, otherwise "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, 
    dtype=torch.float32
).to(device).eval()

D = model.config.hidden_size
print(f"model loaded OK on {device}, hidden size: {D}")
~~~

### Step 4 — (RECOMMENDED) the layer: base + KV cache — full code

Run this cell to build **the main layer**. It sets `LAYER_FWD`, which Step 5 attaches to the model.

~~~python
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
~~~

#### 4.1 — (OPTIONAL) only the base layer

Skip Cell 3 and run this instead if you do NOT want the KV window / long-term memory:

~~~python
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
~~~

#### 4.2 — (OPTIONAL) with the mamba SSM layer

~~~python
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
~~~

#### 4.3 — (OPTIONAL) with the KV cache wrapper only

~~~python
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
~~~

#### 4.4 — (OPTIONAL) with the ultra stack

~~~python
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
~~~

#### 4.5 — (OPTIONAL) with the combined layer (all four branches)

~~~python
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
~~~

### Step 5 — attach the layer on the model + run the tests

~~~python
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
    lay.reset()                          # clean slate FIRST: clear TEST 1 leftovers
    LAYER_FWD(h, tids)                   # fresh pass — nothing stored yet
    fresh = lay.base.get_stats()["recall_rate"]
    LAYER_FWD(h, tids)                   # replay pass — 12-grams now hit the store
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
~~~

---
