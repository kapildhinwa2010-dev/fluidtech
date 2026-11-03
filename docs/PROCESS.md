# FluidLayer process — how the layer was built, attached, and verified

This is the whole process in one file: what the layer is, how it is
attached to an LLM, what I did to build it, every experiment that was
run, every number that came out, every bug that the code produced (and
was fixed), and every negative result that did not get hidden.

---

## 1. What the layer is

A **constant-size memory tier that lives next to your model, not inside
it.** It is not an attention replacement, not a new architecture, and not
a fine-tune. It is one extra object you hold in RAM and one extra line in
your forward pass.

Three sub-tiers, each with a different job and a different cost:

| tier | what it holds | size | job |
|---|---|---|---|
| **L1 — the exact tier** | a content-addressed store `context → token` | **20 bytes per fact** (16 384 facts = 16 KB) | bit-exact recall of what was actually said |
| **L2 — the fluid tier** | an `M × r` diffusion state | **16 KB** (M=64, r=8, fp32) | an *address space*: the router's fallback for contexts too old for the store |
| **L3 — the pointer** | an external handle | **6 bytes** | the fact may live outside the process |

**The fluid answers WHERE; the register answers WHAT.** The exact tier
stores the *token id* as an `int32`, so the recall is bit-exact by
construction — it is not a similarity score that happens to be close.

### The byte layout of one fact (L1)

```
key    uint64   # the content key (the context hash)
value  int32    # the token id to predict
sig    uint32   # the signature: the fail-loud check
anchor uint16   # the router's anchor
meta   uint16   # the corroboration count / flags
────────────────────
20 bytes
```

---

## 2. How it attaches (the whole mechanism)

### 2.1 The insertion — one residual-stream edit

```
h' = h + W_e · E(h) + W_l · S(h)
```

* `h` — the residual stream, `(T, d)`
* `E(h)` — the **exact** tier's read: for each position, the stored token
  for that content key, projected through a contrastive table and scaled
  by `read_gain`
* `S(h)` — the **lossy** tier's read: the diffusion state's response
* `W_e`, `W_l` — the output projections, **zero-initialised**

Because `W_e` and `W_l` start at zero, `h' = h` exactly. So:

> **Attaching the layer is a provable no-op.** The logits are
> bit-identical. Measured on GPT-2 124M: max abs diff `0.000e+00`.
> Measured on Qwen2.5-0.5B: max abs diff `0.000e+00`.

This is the property that makes the layer safe to ship: you attach it, you
measure no change, and only then do you let it contribute.

### 2.2 The code

```python
# 1) build the layer
emb = torch.nn.Parameter(model.wte, requires_grad=False)
branch = SAABranchTorch(M=64, r=8, d=model.n_embd, emb=emb,
                        ctx_len=1, capacity=8192, read_gain=300.0,
                        transform="pc1", use_exact=True, use_lossy=False)
branch.refresh_contrast()

# 2) in your forward, ONE insertion
h  = model.wte[ids]                                  # (T, d)
h1, info = branch(h, model.wte[ids].float(), ids)   # (T, d)
logits = model.forward(ids, h_override=h1)          # the model continues

# 3) at generation time, the retrieved token is GATED, not forced
logits = branch.gated_logits(logits, info, gain=8.0)
```

### 2.3 The gate — why the memory does not ruin the model

The obvious way to use a retrieval is to overwrite the argmax. That is
why retrieval-augmented systems break generation. Here the retrieved
token is boosted **in proportion to the base model's own uncertainty**:

```
u_t  = − log p_max          (nats)          # the model's uncertainty
boost = gain · u_t
```

* where the model has no idea (`u ≈ 1`) the memory decides → **recall**
* where the model is confident (`u ≈ 0`) the memory stays quiet →
  **generation quality is untouched**

Measured (E10): the recall is *unchanged* at 0.975 and the base model's
perplexity *improves* by 0.145 nats, where the ungated override would
have cost +0.11 nats.

**The gate lives in logit space** (`u = −log p_max`). This was forced by
the real model: GPT-2's logits span ±2, Qwen's ±20, so a probability-space
gate would need a per-model constant. The logit-space gate is
scale-free.

---

## 3. The process — what I actually did, in order

### Step 1 — the theory (and its first failures)

The starting idea was a **cellular-sheaf diffusion** that replaces
attention: messages flow along a sheaf's simplicial complex, and the
Chebyshev approximation of `exp(−tL)` gives a cheap diffusion operator
over the token graph. Three propositions were written and costed
(`docs/` + the report): the FLOPs per token, the training cost, and a
comparison against attention.

**It did not work.** The first experiments measured:

* the topological (Hodge/landmark) selection scored **0.000** against a
  semantic baseline of **0.970** → the topology was not selecting
  anything useful (negative result **N1**);
* the spectral quantity `λ₂` correlated with quality at **pearson 0.199**
  → it is a diagnostic, not a driver (negative result **N3**);
* as a *replacement* for attention in a transformer it was worse at every
  scale (negative result **N4**).

Those three are published, with numbers, in `docs/NEGATIVE-RESULTS.md`.

### Step 2 — the framework (pure NumPy, with an autodiff)

Instead of replacing the transformer, the sheaf machinery became a
**library**: a small NumPy autodiff, a `SuperfluidBlock` with 7 stages
(Chebyshev `exp(−tL)` + a band-filter bank + Kruskal persistence), a
`TinyTransformer`, and a **50-check self-verification** (`selftest.py`).

The rule for the whole project became: *if a number is quoted, it is
produced by a script that writes a JSON*.

### Step 3 — the memory layer (the part that survived)

The **frozen register** (`fluidlayer/memory.py`): append-only, 20 B per
slot, fail-loud signature, versioning, a quorum, `to_bytes`/`from_bytes`.
And the **topology** module (Hodge embedding, relevance, edge
participation, persistence landmarks, local density).

Then the decisive design change: the memory became something you **graft
onto** a model instead of something you train into one.

### Step 4 — the graft (`fluidlayer/graft.py` + `torch_graft.py`)

The numpy twin and the torch twin, with a **parity test** between them
(`tests/test_torch_parity.py`) — which caught **four real port bugs**
(see §6).

### Step 5 — the training-signal fix (E7)

The lossy (diffusion) tier had **no usable training signal**: training
through the base model's loss diverged (**+1.1431 nats, DIVERGED**).
Three fixes were measured:

| variant | the base model's loss | the layer's own objective |
|---|---|---|
| the control (train through the base) | **+1.1431 → diverged** | — |
| FIX1 (a layer-local objective, persistent state) | +0.0000 | 4.343 → **6.801 (rises)** |
| FIX2 (a layer-local objective + a differentiable *rebuild* state) | +0.0000 | 4.343 → **3.807 (converges)** |

FIX2 is the answer: the state's *read* must be differentiable and the
state must be **rebuilt** from the store on every step, not carried
across steps.

### Step 6 — the router gives the diffusion tier a job (E12)

The lossy tier was a worse predictor than the base model on its own. So
it was given a job it can do: **the fluid's M anchors became an address
space**, each holding the majority continuation of the contexts routed
to it. Then:

* when the exact store is too small, the recall goes **0.100 → 0.625**
  (base: 0.000);
* on the *hard* positions (the base model's top-1 < 0.1), the perplexity
  goes **+0.0534 → −0.0707**;
* the routing cost is **0.0005 %–0.0075 %** of one transformer layer.

### Step 7 — the real LLM (E14, E15)

Everything up to here was on a toy model. Then:

* **E14** — GPT-2 124M, the real 50 257-token BPE, a GPT-2 forward pass
  written from scratch (no `transformers`): the attach is
  **bit-identical**, the base perplexity 56.20 → **54.70** with the
  layer, and a fact planted **2 947 tokens** back: the base predicts
  464, **the layer predicts the fact**.
* **E15** — Qwen2.5-0.5B, a *different architecture* (RoPE + RMSNorm +
  GQA + SwiGLU), again a forward written from scratch
  and **verified against HuggingFace** (max abs logit diff **0.34**, i.e.
  bf16 rounding). See `results/e15_scale_model.json`.

  *The scale ladder's answer:* the attach is bit-identical
  (`0.000e+00`), the capability works (a fact planted **1 411 tokens**
  back: the base predicts garbage, **the layer predicts the fact**), and
  the perplexity picture repeats E14's honest negative: the kNN-LM
  interpolation wins (−0.274 nats) while the layer's best gain is 1.0
  (−0.041 nats). **The gate's advantage shrinks as the model gets
  stronger** — the tiny model: the layer won by 7.4×; GPT-2 124M: the
  interpolation won by 8×; Qwen 0.5B: the interpolation won by 6.7×.

  *And the gain is not one knob for both jobs:* the capability needs a
  gain ≥ 20 (the model's uncertainty at the query is only 1.3–3.8 nats),
  while the perplexity needs a gain ≤ 1 (at gain 8 the perplexity
  explodes: 32.6 → 1377). **That is the sharpest statement of the
  trade-off in the whole project, and it is measured, not asserted.**

---

## 4. The experiment ledger (every number in the repo)

| # | experiment | the question | the answer |
|---|---|---|---|
| E1 | the prior's perplexity | does a learned prior help? | **no**: −0.0059 nats (the unigram prior: −0.0382). And the benefit *shrinks* as the model improves (0.0482 → 0.0065 from 30 → 250 steps). The 0.02-nat pre-registered bar is **not met**. |
| E2 | the budget law | anchors or pins? | **pins**: 64× the pins → +0.69 recall for 20 KB; 16× the anchors → +0.013 for 256× the compute. |
| E4 | the phase transition | is there a geometric signal? | frustration ⟺ a global section **150/150**; the contraction certificate **36/36**; `pearson(λ₂, quality) = 0.199` → a diagnostic, not a driver. |
| E5 | the graft, v1 | does the memory work? | the gist cosine 0.968 (local); semantic precision 0.970 / recall 0.469; **the topology selects 0.000**; `exact|promoted = 1.000`; the KV/ours ratio **6 554×**. |
| E6 | the graft on real text | does it survive a real corpus? | the recall **0.050 → 0.800** (chance 0.125); the lossy tier alone 0.050; the state 6 316 B. On a real corpus the lossy tier *diverges* (char +0.8094; BPE +0.9831) while the exact tier is neutral (−0.0320 / −0.0017). |
| E7 | the training signal | can the diffusion tier be trained? | **no, not through the base's loss** (+1.1431, diverged). FIX2 (a layer-local objective + a rebuild state) converges 4.343 → 3.807 with the base untouched (+0.0000). The prior-injection gain sweep is monotone **harm** ⇒ the optimum is **zero**. |
| E9 | ten million tokens | does the store scale? | **10 M tokens in 12.1 s** (0.83 M tok/s, no model forward); the same tokens as a KV cache = **1.31 TB** (8B) / **3.28 TB** (70B). The token-level store recalls **6/6** deep facts; a common-key control **0/3**; a 12-gram key recalls **0/6** ⇒ **the reach is the last ~10 k tokens (0.1 %)**. |
| E10 | the gate | can one knob do both? | **yes**: the recall stays 0.975 and the base's perplexity *improves* −0.1447 (vs −0.1219 ungated). At gain 20 the harm is 4× smaller. |
| E11 | the exact memory | is it really exact? | the recall **1.000 (40/40)** in every configuration; the majority tracker fixes the common-key case (0/3 → 40/40); the evidence-based eviction keeps the corroborated facts (40/40 survive a flood of 5 000 one-off contexts, capacity 64); the soft gate gives **−0.1756 nats** (ppl 13.61 → 11.42). |
| E12 | the router | can the diffusion tier do something? | **yes**: the recall 0.100 → 0.625 when the store is too small; the hard positions +0.0534 → −0.0707; the cost 0.0005 % of a layer. It cannot recover a *specific* fact (0/4). |
| E13 | the missing baseline | how does it compare to kNN-LM? | 16 KB store: **the layer −0.1134 vs kNN −0.0154 (7.4×)**. 1 MB store: the layer −0.2048 vs kNN **−1.3013** (kNN wins 6.4×). ⇒ the defensible novelty is **the gate under a hard memory budget**. |
| E14 | real GPT-2 124M | does it work on a real LLM? | the attach **bit-identical**; the base ppl **56.20** → **54.70** with the layer; kNN-LM −0.2259 (better!); the capability: the fact **2 947 tokens** back → the base predicts 464, **the layer predicts the fact**. |
| E15 | real Qwen2.5-0.5B | does it work on a *bigger*, different-architecture LLM? | see `results/e15_scale_model.json` — the attach **bit-identical**, and the scale-ladder answer for the gate's advantage. |

---

## 5. The negative results (published, not hidden)

`docs/NEGATIVE-RESULTS.md` carries all of them with the numbers. The
short form:

* **N1** — the topology-based selection is worthless (0.000 vs 0.970).
* **N2** — the prior does not beat a unigram baseline.
* **N3** — `λ₂` is a diagnostic (pearson 0.199), not a driver.
* **N4** — replacing attention with sheaf diffusion is worse.
* **N5** — the domain-prior product is superseded by the exact register.
* **N6** — the lossy tier alone is a worse predictor than the base model.
* **N7** — the prior-injection gain sweep is monotone harm; the optimum
  is zero.
* **N8** — the store's reach is the last ~10 k tokens, not 10 M.
* **N9** — the router cannot recover a *specific* fact (0/4).
* **N10** — the kNN-LM baseline beats the layer once the budget grows.
* **N11** — **the gate's advantage reverses with model quality**: on the
  tiny model the layer won by 7.4×; on real GPT-2 the interpolation won
  by 8×. The reason is structural — interpolation is *bounded* (a wrong
  retrieval can never dominate), the gate is not. **The capability
  (exact recall) works on both.**

**The honest positioning** follows from N11: *the gate wins where the
base model is weak; on a strong model the plain retrieval interpolation
is the better perplexity tool — but the exact capability is the same.*
That is a narrower claim than "we beat retrieval", and it is the one the
data supports.

---

## 6. The bugs the code produced (found by running it, then fixed)

Every one of these was found by *running* the thing, not by reading it:

1. the Laplacian's sign convention;
2. the Chebyshev expansion's factor 2;
3. an inert `index_add` (the gradient never reached the tensor);
4. the gradients zeroed after `backward`;
5. `exp(+α)` blow-up in the diffusion update;
6. `mean()` clobbering `sum()`'s backward;
7. no gradient seeding at the start of a backward pass;
8. a `newaxis` broadcasting bug;
9. `λmax` underestimation;
10. the fact-position initialisation order;
11. the pair-novelty off-by-one;
12. the filler overwriting the fact values;
13. the noise vectors not being norm-relative;
14. the Markov self-loops;
15. the k-means empty-centroid collapse;
16. the candidate-index slice bug;
17. `ridge_fit`'s return shape;
18. `budget_split` subtracting the codebook;
19. E1's OOM from the graph retention;
20. `cross_entropy` expecting a Tensor;
21. `bias2`'s shape/broadcast;
22. the advanced-indexing shape;
23. **the nats/KB arithmetic 1000× off** (the earlier "40.2 nats/KB" was wrong — the rank-8 unigram is the most byte-efficient at 0.0092 nats/KB);
24. the **route scale** (`/sqrt(dh)` vs `/sqrt(d)`) — found by the
    numpy↔torch parity test;
25. the **routing content** (the residual stream instead of the
    position-free embedding) — the same test;
26. the **anchor-confidence formula** — the same test;
27. the **read/write order** (the snapshot semantics) — the same test;
28. the **nats/KB** in the KV comparison;
29. the **Qwen2 q/k/v biases** — my forward assumed Qwen2 has no biases;
    it does, and dropping them turns the model into noise (caught by
    comparing against HuggingFace: max abs diff 23.1 → **0.34**);
30. the **downloader's truncated read** (a dropped connection looks like
    a clean EOF — the checkpoint was 12.4 MB short and silently
    corrupt);
31. the **logit cache** in E15 (40 windows × 152 k vocabulary = 3.1 GB
    → OOM on a 3 GB machine);
32. the **dtype seam** (a bf16 checkpoint against an fp32 layer).

---

## 7. The six uploaded files

The upload files are categorized by purpose:

| file | what it is |
|---|---|
| `examples/appkit/load_model.py` | local GPT-2/Qwen2 checkpoint loader |
| `examples/appkit/attach_layer.py` | app-kit integration and verification |
| `experiments/e15_scale_model.py` | E15 experiment source |
| `results/e15_scale_model.json` | archived E15 measurements |
| `docs/PROCESS.md` | this file |
| `fluidtech-quickstart.zip` | upload archive, intentionally kept at the repository root |

---

## 8. How to reproduce

```bash
pip install numpy                     # the core: no other dependency
python selftest.py                    # 50/50
python fluidlayer_single.py --selftest

# the real model and app kit (optional dependencies)
python -m pip install "fluidlayer[gpu]"
hf download Qwen/Qwen2.5-0.5B --local-dir models/qwen05b
python experiments/e15_scale_model.py --model models/qwen05b

# the app kit: load the model, attach the layer, run the tests
python examples/appkit/load_model.py
python examples/appkit/attach_layer.py --model models/qwen05b
```
