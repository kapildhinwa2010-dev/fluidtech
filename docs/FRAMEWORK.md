# Superfluid Framework v1.0 — Historical specification

> **Historical design note:** this is not a current implementation contract.
> For current behavior and limitations, see [the API reference](API.md) and
> [limitations](LIMITATION.md).

A topological fluid architecture for sequence models: replace the
transformer's quadratic attention with a cellular-sheaf diffusion whose
spectrum does the work, and whose geometry is recomputed in every forward pass.
The architecture below records an earlier design; its claims and numbers
must not be treated as current results without checking the implementation
and experiment outputs.

## 0. Why this exists

A transformer layer spends its budget on two things: dense weight matrices
(the FFN) and quadratic attention (the context mixing). The attention is
what makes long context expensive and what makes retrieval possible.
Superfluid replaces the attention with a geometric operator stack:

```text
X(t) = exp(-t L_F) X(0),        L_F = δᵀδ      (sheaf Laplacian)
```

This operator has three properties attention does not:

| property|statement|where it comes from|
| ---|---|---|
| jamming|the field converges exponentially to  ker(L_F)  — a frozen geometry|spectral theorem, Hansen  & Ghrist 2019|
| phase transition|balanced complexes freeze onto a non-trivial sign pattern; unbalanced ones cannot|G-balance, Hansen  & Ghrist 2019|
| certified contraction|‖e^{-tL}‖₂ = e^{-tλ₂} < 1  on  ker(L)⊥|spectral theorem|

…and one property attention has and it does not: a smooth diffusion is a
low-pass filter, so it cannot select a single neighbour sharply. Retrieval must
therefore be re-introduced explicitly. That trade — global geometry from the
fluid, sharp selection from a sparse route head — is the whole architecture.

## 1. Module map

```text
SuperfluidLM
 │
 ├── Embedding  E : ℝ^|V| → ℝ^d            (tied readout)
 ├── Positional  P : ℝ^n_ctx → ℝ^d         (learned absolute; order also enters
 │                                           the topology via the chain edges)
 │
 ├── × L  SuperfluidBlock
 │      │
 │      ├── ① ROUTER      top-k content routing → semantic edges
 │      │                Q,K = hW_Q, hW_K ; scores = QKᵀ/√d ; masked ; top-k
 │      │                (hard selection → this is the *needle* path)
 │      │
 │      ├── ② TOPOLOGY    K = semantic edges ∪ temporal chain
 │      │                chain edges (i, i+1) carry a learned weight β
 │      │                → ORDER BECOMES 1-DIMENSIONAL TOPOLOGY
 │      │
 │      ├── ③ SHEAF       signed restriction maps  α_uv = MLP([x_u ; x_v])
 │      │                tanh-bounded → |α| ≤ 1 → L bounded → stable gradients
 │      │                sign is learnable → G-balance is learnable
 │      │
 │      ├── ④ FREEZE      band filter bank  band_j = e^{-t_j L} − e^{-t_{j+1} L}
 │      │                B bands, log-spaced in SPECTRAL time α = t·λmax/2
 │      │                Chebyshev basis T_k(L̃)X computed ONCE for all bands
 │      │
 │      ├── ⑤ HARMONIC    the FROZEN state: exact Hodge projection
 │      │                = per-component mean (union-find + scatter, O(n+m))
 │      │                ← without this the block cannot represent a global
 │      │                  aggregate at all: the bands cancel the harmonic
 │      │                  component by construction (measured: 49.3% = chance)
 │      │
 │      ├── ⑥ READ        z = Σ_j softplus(g_j)·band_j  +  softplus(g_H)·harmonic
 │      │                z ← z W_mix            (channel mixing)
 │      │                z ← gelu(z)            (THE single nonlinearity)
 │      │                out = x + z W_o        (residual)
 │      │
 │      └── ⑦ ROUTE HEAD  hard top-k gather of V, softmax over the k logits
 │                       out += (Σ_k softmax_k ⊙ V[idx]) W_r
 │                       (winner-take-all = sharp retrieval)
 │
 ├── Final RMSNorm
 └── Readout  logits = h Eᵀ   (tied)
```

### The three paths (spectral decomposition)

The block computes a decomposition of the identity operator:

```text
I  =  P_ker(L)        ← harmonic branch: the FROZEN geometry (component means)
   +  Σ_j band_j      ← transient: the multi-scale relaxation
   +  (I − P_ker − Σ) ← residual: the identity path through the residual stream
```

Every architectural path has a spectral meaning: the harmonic branch is the
frozen state, the bands are the transient, the route head is sharp
selection, and the residual is the identity. Omitting any of them loses a
capability — and the harmonic one is not optional (measured below).

### Pseudocode (historical)

```python
def block(h):                                  # h: (n, d)
     xn = rmsnorm(h)
     scores = (xn @ Wq) @ (xn @ Wk).T / sqrt(d)  # (n, n)
     idx    = topk_masked(scores, k)             # (n, k) hard selection
     sem   = unique_undirected(idx)              # semantic edges
     chain = [(i, i+1) for i in range(n-1)]       # temporal edges
     alpha = tanh_mlp([xn[u] ; xn[v]])            # (m,) signed, content-adaptive
     L     = D_abs(alpha_sem) + D_abs(beta*chain) # signed sheaf Laplacian, PSD
     lam   = power_iteration_lmax(L)             # spectral scale
     ts    = 2*geomspace(a_lo, a_hi, B+1)/lam    # spectral-time band placement
     Y     = chebyshev_basis(L, lam, xn, m)      # m matvecs, shared by all bands
     bands = [sum_k (c_j[k]-c_j1[k]) * Y[k] for j in range(B)]
     harmonic = component_means(xn)             # the frozen state, O(n+m)
     z  = sum_j softplus(g_j) * bands[j] + softplus(g_H) * harmonic
     z  = gelu(z @ Wmix)
     out = h + z @ Wo
     v  = xn @ Wv
     out = out + (softmax_k(scores[idx]) * v[idx]).sum(1) @ Wr
     return out
```

## 2. The five lemmas (historical checks)

| #|statement|verification|status|
| ---|---|---|---|
| L1|‖e^{-tL}u‖ ≤ e^{-λ₂t}‖u‖  for  u ⊥ ker(L)|selftest.py  check 1|✅|
| L2|t_jam = ln(1/ε)/λ₂  is an  upper bound  on the freezing time|check 2|✅|
| L3|Chebyshev degree  m = O(√κ · ln(1/ε))  suffices; measured  m=16 → 9.4e-7  error at κ=14.8|check 3|✅|
| L4|a purely linear Superfluid  cannot  fit XOR (0.50 accuracy); adding one pointwise nonlinearity  can  (1.00)|checks 4–5|✅|
| L5|the frozen state has rank = #components; the band bank restores the signal (2.8e-6 vs 0.962 error)|checks 6–7|✅|

Plus: G-balance phase transition (check 8), contraction certificate (9),
frame property (10), and end-to-end gradient checks against finite differences
(11–12: max relative error 3e-7 on the op-level test, 0.0e0 on the
full model).

## 3. Improvement ledger — v0.1 weaknesses and v1.0 design responses

| v0.1 weakness|v1.0 module|mechanism|measured effect|
| ---|---|---|---|
| W1  literal  "remove the NN " ⇒ linear model|⑤ one pointwise  gelu|Hornik/Cybenko: σ + a fixed feature map is still a universal approximator|XOR 0.50 → 1.00 (check 5)|
| W2  frozen state has rank  c|④ band filter bank|multi-scale readout recovers the transient|err 0.962 → 2.8e-6 (check 7)|
| W3  no exact recall|⑥ route head + ① router|hard top-k winner-take-all = sharp selection|see §6 ablations|
| W4  permutation blindness|② temporal chain edges|order enters as 1-dimensional topology, not just an embedding|—|
| W5  fixed topology is content-blind|③ content-adaptive  α_uv|restriction maps are an MLP of the endpoint features → geometry changes with content|—|
| W6  edge-weight collapse / exploding gradients|③  tanh -bounded  α ,  softplus  β|‖L‖  stays bounded ⇒ Chebyshev regime preserved, gradients bounded|—|
| W7  conditioning cliff ( m ~ √κ )|④  spectral-time band placement|bands live in  α = t·λmax/2 ≤ 0.75m , so the degree is sufficient  by construction  for any graph scale|cheb_coeffs_exp  raises instead of silently overflowing|
| W8  persistence is 60% of runtime (2026 evidence)|② exact O(m log m) 0-D barcode via Kruskal + cycle rank|computed per forward pass, not cached; no heavy TDA needed|topology.py|
| W9  no scaling law exists|§5 harness|same protocol as Hoffmann et al. + FLOP-matched baselines|—|
| W10  topology can be redundant|④ band decorrelation regulariser|penalise  |cos|  between band responses|losses.band_decorrelation|
| W11  the band bank cannot carry the frozen state (found by experiment)|⑤ harmonic branch|the bands  cancel  the kernel component by construction; add the exact Hodge projection (per-component mean)|global-mean task: 49.3% →  97.3%|
| W12  open band gates swamp the frozen state|④ band gates init at softplus(−3) ≈ 0.05|the transient is orders of magnitude larger than the harmonic term|74.1% →  97.3%|

## 4. Objectives

L_total = CE(next-token)
        + w_E · E_sheaf            (smoothness / consistency on routed edges)
        + w_D · band decorrelation  (keep the filter bank a frame)
        + λ_clip · ‖α‖₂             (keep restriction maps near the unit ball)
`E_sheaf = Σ_e α_e² (x_u − x_v)²` — annealed from 0: start free, end smooth.
`band_decorrelation = mean_{j≠k} |⟨band_j, band_k⟩| / (‖band_j‖‖band_k‖)` —
stops all bands collapsing onto one spectral mode (the rank-c trap in
disguise).
The topological sufficiency term is implemented as a probe, not a
gradient: the routing decision is a hard top-k selection and carries no
gradient path, so a loss on the persistence descriptor would train nothing.
Probe instead: can a linear head on the descriptor alone predict the label?
If not, the geometry is decoration.

## 4b. The decisive mechanism test (historical measurement)

Input: a sequence of scalars in fixed features. Target: the mean — a
global, order-invariant statistic. Only the aggregation mechanism differs
between arms; nothing else is trainable.

| aggregator|sign accuracy|RMSE|vs constant baseline|wall clock|
| ---|---|---|---|---|
| Superfluid (the fluid)|97.3%|0.015|0.126|95 s|
| Attention (causal)|100.0%|0.000|0.126|4 s|
| Local, R = 4|65.6%|0.113|0.126|1 s|
| Local, R = 8|71.0%|0.102|0.126|1 s|

Conclusion: the fluid does compute a global statistic in one pass — it
beats both strictly-local aggregators and nearly matches attention. The first
version of this experiment scored 49.3% (chance) and exposed W11/W12 above.
The wall-clock gap (24×) is an implementation artefact of dense NumPy, not an
architectural property; a sparse Torch implementation removes most of it.

## 5. Instrumentation (historical design)

Every block exposes, per forward pass:

| diagnostic|meaning|why it matters|
| ---|---|---|
| λ₂  (spectral gap)|how fast the geometry jams|predicts  the freezing time before training finishes|
| λ_max|spectral scale|sets the band placement|
| balance_res = ‖L1‖∞|G-balance of the learned sheaf|the phase-transition indicator|
| α_neg_frac|fraction of negative restriction maps|how  "signed "/heterophilic the geometry is|
| band_energy[B]|energy per timescale|detects band collapse|
| topo[9]|β₀, β₁, persistence lifetimes, density|the persistent signature of the data|
| n_edges|size of the routed complex|routing entropy|

The spectral gap is a training-time early-warning system: if `λ₂ → 0` the
model is about to over-smooth into a constant, and the fix (raise the routing
temperature, raise the chain weight, or reduce depth) is visible before the
loss curve tells you.

## 6. Ablation plan

| #|ablation|expected|kills which hypothesis|
| ---|---|---|---|
| A1|SF-0: no nonlinearity|catastrophic failure|confirms L4 empirically|
| A2|SF-1: no route head|collapses on retrieval tasks|W3|
| A3|single exponential instead of band bank|catastrophic information loss|L5|
| A4|no temporal chain|order sensitivity collapses|W4|
| A5|static vs content-adaptive  α|content-adaptive wins on heterophilic data|W5|
| A6|band count B ∈ {1,2,4,8,16}|monotone up to ~8, then flat|filter-bank design|
| A7|k (neighbours) ∈ {4,8,16,32}|saturates ~16|routing cost/benefit|
| A8|m (Chebyshev degree) ∈ {4,8,12,16,24}|error ~1e-1 → 1e-12|L3|
| A9|SF vs Transformer at  equal FLOPs|?|the central question|
| A10|SF vs Transformer at  equal params|?|ditto|
| A11|context length 4K → 128K|SF advantage grows|the cost thesis|
| A12|band decorrelation weight = 0|band collapse|W10|

## 6b. Character-level LM on real text (historical measurement)

Project Gutenberg (Alice in Wonderland, 120 k chars, char-level), identical
tokens / optimiser / schedule, 400 steps, `d = 64`, `L = 2`, `ctx = 64`,
batch 16, NumPy.

| model|params|final val (nats/char)|best val|wall clock|
| ---|---|---|---|---|
| Superfluid|71,408|2.7397|2.6181|340 s|
| Transformer|116,544|2.6331|2.5401|95 s|

Read this honestly. At this scale the transformer is ~0.08–0.10 nats
better despite the Superfluid having 39 % fewer parameters, and the
Superfluid is 3.6× slower in wall clock. Two things to keep separate:
the wall-clock gap is an implementation artefact — dense `O(n²)`
Laplacians, a Python loop over the batch, no fused kernels. A sparse Torch
implementation removes most of it;
the quality gap is real at this scale and is exactly what ablation A9/A10
exist to measure. 71 k parameters, 400 steps and a 76-symbol vocabulary is
not a place where a topological prior can show its value — the prior has to
beat a prior-working, well-tuned baseline, and it does not yet.
The mechanism test (§4b) is the more informative result: there the fluid
does deliver the capability it claims (97.3% on a global statistic vs 65–71%
for local aggregators).

## 7. Cost model (per token, per layer)

| component|Superfluid|Transformer|
| ---|---|---|
| projections|8d²|8d²|
| context mixing|4kd  (k=16)|4nd|
| the fluid itself|2mk  (m=12,k=16 →  384 flops )|—|
| FFN|16d²|16d²|
| total (d=4096, n=4096)|3.02e8|4.70e8|

The fluid is 0.00016 % of a transformer layer. Two consequences, both
important and both honest:
the fluid is essentially free — so it cannot be the reason the model works;
the capacity still lives in the FFN and the mixing matrices;
the real win is not FLOPs but memory: with a static corpus-level
topology the decode state is constant (16 KB) instead of a growing KV cache
(488 GB at 1M context).
Param-matched totals (D ≈ 44B, 470B tokens, 512×H100 @ 40 % MFU):

| context|Transformer|Superfluid|speed-up|
| ---|---|---|---|
| 8K|188.9 h|172.3 h|1.10×|
| 32K|238.7 h|172.3 h|1.39×|
| 64K|305.1 h|172.3 h|1.77×|
| 128K|438.0 h|172.3 h|2.54×|

## 8. Scaling-law harness (for W9)

Follow Hoffmann et al. exactly, or the comparison is worthless:
sweep `D ∈ {6e6 … 1e9}` × `N ∈ {6e7 … 4e9}` (≥ 3 points per axis);
FLOP-matched baselines: the transformer is run at the same FLOPs, i.e.
its `n` is reduced to compensate — report both FLOP- and param-matched;
fit `L(D,N) = E + A/D^α + B/N^β`, report α, β, E with bootstrap CIs;
the hypothesis: Superfluid should show a smaller`B` (data term) if the
topological prior is real, and possibly a larger`α` if the parameter
efficiency is real. Either would be a publishable result; if `α`, `β` match
the transformer, the architecture is merely cheaper, which is still a
result but a smaller one;
report the gap collapse curve `λ₂(step)` alongside — if the gap closes
as the model gets better, that is the mechanism.

## 9. Proposed deployment target (historical)

The strongest use case is not replacing the whole transformer. It is:
a static, corpus-level sheaf used as a memory substrate with zero KV
cache, sitting next to a small attention stack that handles needle retrieval.
[ tokens ] → [ Superfluid memory layer (static topology, 0-byte KV cache) ]
           → [ small dense-attention head (sharp retrieval) ]
           → [ readout ]
For 1M-token context this is 488 GB → ~16 KB of state. The cost model says the
memory layer is ~free in FLOPs (the fluid is 384 flops/token/layer); the only
real cost is that the topology is fixed, so it carries distributional memory
(statistical geometry of the corpus), not instance-specific memory.

## 10. Roadmap

v1.0 (this repo) — verified mathematics, trainable model, end-to-end
autodiff, LM A/B, global-aggregation mechanism demo, full instrumentation.
v1.1 (2–4 weeks)
sparse Laplacian backend (currently dense `O(n²)` — the hard scaling limit)
Torch backend (the NumPy engine is verified; a torch port is mechanical and
unlocks GPU + batched sparse ops)
band-energy regulariser wired into the training loop
ablation runner (A1–A12) producing a single results table
checkpoint/resume, data loader with multi-worker streaming
v1.2 (1–2 months)
scaling-law harness (§8) on a 1B+ parameter run
learned band placement: place the bands around the measured `1/λ₂` instead
of the whole spectrum — the "critical temperature" as a hyperparameter
topology cache: amortise the complex across steps (the routing entropy
monitor decides when to rebuild)
the probe-based topological-sufficiency head, wired as a diagnostic
v2.0 (research)
multi-sheaf superposition: several sheaves (syntax / semantics /
topology) with learned mixing — the "complementarity, not substitution" pattern
the 2026 literature keeps finding
learned topology: differentiable complex construction (soft simplicial
complexes) so the geometry itself is trained end-to-end
physical priors: the G-balance phase transition as a regulariser
(encourage balance → stronger contraction certificate)
domain port: molecules / meshes / proteins, where the topological prior
is not a gimmick but the correct inductive bias

## 11. Repository map

superfluid/
  autodiff.py    reverse-mode autodiff (Tensor, ops, gradient-checked)
  sheaf.py       signed sheaf Laplacian, G-balance, TV energy
  diffusion.py   λmax power iteration, Chebyshev exp(-tL), band bank,
                 spectral gap, jam time, spectral-time band placement
  topology.py    top-k routing, unique edges, Kruskal 0-D barcode,
                 Betti numbers, persistence descriptor
  blocks.py      SuperfluidBlock (the 6-stage stack)
  model.py       SuperfluidLM + TinyTransformer baseline
  losses.py      CE (+ tensor variant), sheaf energy, band decorrelation
  optim.py       Adam (decoupled WD, clipping, cosine schedule)
  data.py        Gutenberg corpus + char tokenizer + synthetic fallback
selftest.py      15 numerical checks of the mathematics   -> results/selftest.json
train.py         LM A/B + global-aggregation demo             -> results/train_results.json

### Run it

python3 selftest.py                 # verify the maths (15/15)
python3 train.py --steps 400 --d 64 --layers 2 --ctx 64 --batch 16
Requirements: numpy only (no torch, no scipy). The whole point of the
NumPy engine is that every claim can be re-verified on any laptop in under a
minute.

## 11b. The engineering ledger (historical bug list)

Each of these silently turned a claim into a no-op, and each was caught by a
test rather than by reading the code:

| bug|symptom|
| ---|---|
| Laplacian built as  A − D|non-PSD operator, overflow in exp(−tL)|
| Chebyshev recurrence missing the factor 2|expansion plateaued at 78% error, independent of degree|
| index_add  never scattered in the forward pass|the entire fluid was inert  — the block's Laplacian was identically zero|
| index_add  backward never reached  src|restriction maps received no gradient|
| gradients zeroed  after   backward()|both models frozen at ln(vocab) for 400 steps|
| exp(+α)  instead of  exp(−α)|bands multiplied by e^(2α) ≈ 10⁶ → divergence|
| mean()  overwrote the sum's backward|gradients silently never reached the inputs|
| no gradient seeding in  backward()|every gradient exactly zero|
| newaxis ( [:, :, None] ) unhandled in the backward|scatter with mismatched index arrays|
| λmax underestimated by power iteration|Chebyshev recurrence outside its stable region → band energies of 10¹⁵|

Regression tests now exist for each class of failure (selftest checks 11–15).

## 12. What would falsify this

Stated up front, because a research proposal without kill criteria is a wish:
SF-2 trails the transformer by > 0.15 nats at matched FLOPs on a 400M-token
run → the fluid contributes nothing and the architecture is a slower
attention. (Cost of this test: < 1 GPU-hour.)
The band filter bank provides no gain over a single exponential after
the fix for the rank-c trap → the multi-scale story is decorative.
`λ₂` collapses during training and the model degrades to a constant →
the geometry is not learnable and over-smoothing dominates.
The content-adaptive sheaf never becomes G-balanced → the learned
geometry has no frozen state, and the "jamming" narrative does not apply to
the learned model.
Ablation A2 shows the route head does all the work → the fluid is
decoration and the honest conclusion is "sparse attention with extra steps".
Any of these is a publishable negative result, and the framework is built so
that each one is cheap to test.
Appendix (v1.1) — the memory module
`superfluid/memory.py`
`FrozenRegister` — the exact tier
An append-only, write-once table of exact facts.  20 bytes per slot:

| field|type|bytes|meaning|
| ---|---|---|---|
| key|uint64|8|64-bit context hash (the occurrence identity)|
| val|int32|4|the exact token id|
| sig|uint32|4|4×uint8 embedding signature (collision detector)|
| anchor|uint16|2|the anchor the fact routed to|
| meta|uint16|2|doc_id (12 b) | trust (2 b) | reserved|

API:
reg = FrozenRegister(capacity=819)          # 819 x 20 B = 16 KB
slot = reg.promote(token_id, ctx_key, sig, anchor, doc_id, trust)
out   = reg.recall(ctx_key, sig)             # -> (slot, token_id) or None
reg.set_trust(slot, 1)                      # corroboration gate
reg.quorum(key)                             # distinct sources that promoted it
reg.stats(); reg.to_bytes(); FrozenRegister.from_bytes(blob, capacity)

## Guarantees in the historical design

Invariance — a slot's value never changes; the register is append-only.
Integer exactness — the value is an `int32` token id, so a successful
recall returns a value that is equal, not approximately equal.
Fail-loud — the recall verifies the 64-bit key and the 4-byte
signature; on a mismatch it returns `None` rather than another fact's value.

### Versioning

Promoting the same key twice appends a second slot and the
the attachable store tracks the majority continuation and evicts the
least-corroborated entry. This is not a tamper-evident audit log.
`SheafMemoryAdapter` — the geometry-addressed tier
ad = SheafMemoryAdapter(M=1024, r=8, d=64, capacity=819)
ad.set_prototypes(protos)            # (M, d) codebook
ad.write(X, t=0.1, cheb_m=12)        # streaming write, O(1) memory
idx, scores, content = ad.read(q)     # geometry read
ad.promote(vec, value_id, window)     # exact write
ad.recall(window, vec)                # exact read
ad.state_bytes(); ad.codebook_bytes()

### Write path

The write path scatters the chunk into its routed anchors and then applies
`exp(-tL)`.  That is exact for a linear operator, so chunking is not an
approximation.
`logit_override(logits, token_ids)`
Forces the argmax to be the recalled token.  This is the difference between
"the model was told" and "the model cannot disagree" — a tool that returns
a token can be outvoted by the model's own logits; an override cannot.
`budget_split(total_bytes, M, r, ...)`
The budget law.  Naming one anchor out of `M` needs `log2(M)` bits while an
anchor holds `r×16` bits, so `r` is over-provisioned for addressing and a
bigger budget should buy more pins, not more anchors.  The function never
over-commits: it reduces `M` instead.
`superfluid/topology.py` (v1.1 additions)

| function|what it computes|
| ---|---|
| hodge_embedding(n, edges, alpha)|Ψ = (L + (1/n)11ᵀ)^{-1}  for the signed sheaf Laplacian|
| hodge_relevance(n, edges, alpha)|per-node harmonic participation  Σ_j |Ψ_ij||
| hodge_edge_participation(Ψ, a, b)|normalised participation of a  pair  of cells, in [0,1]|
| local_density(protos, radius=None, k=8)|how crowded each cell's neighbourhood is|
| persistence_landmarks(n, edges, weights, k)|nodes whose topological class survives longest in the Kruskal filtration|

Prior art note (important). Hodge-based topological relevance for
selecting points is not new — see "Topologically Relevant Landmark
Selection on Point Clouds", arXiv 2610.00019, which scores points by
harmonic participation to choose PH landmarks.  What is new here is the
application: allocating the exact slots of a constant-size neural memory.
And the measured result is that it does not work (0.000 precision vs
0.970 for the plain geometric affinity) — see `docs/DESIGN_pinned_memory.md`
§9.5.2.

### Corrected balance diagnostic

`balance_residual(L) = ‖L·1‖∞` does not detect the frozen state of a
scalar signed sheaf.  An all-negative EVEN cycle has `‖L·1‖∞ = 4` yet
freezes onto the alternating sign pattern.  The right diagnostic is the
frustration index, and the right statement is
frustration == 0   <=>   a non-trivial global section exists
verified on 150/150 random signed cycles and checked in the selftest.
`‖L·1‖∞` measures the stricter property "the CONSTANT field is a global
section".
