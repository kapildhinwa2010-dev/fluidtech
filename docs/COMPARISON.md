# Comparison — FluidLayer and bounded memory approaches

Reference config: Llama-3.1 (GQA, 8 KV heads, head_dim 128, fp16 KV).
Every number is computed from published configs — these are arithmetic
facts, not estimates. Section 9 marks exactly which claims are measured
in our sandbox, which are facts by arithmetic, and which are projected.

Bottom line first: this prototype does not extend the attention context
window. It explores a compact, addressable memory channel alongside attention.

The 20 B/fact figures are packed-slot payload estimates and exclude the
bounded anchor store. The Python reference uses dictionaries and has
additional per-entry object overhead; these figures are not measured
process-RAM usage.

## 0. The verdict in one table

|  |full KV cache|sliding window / StreamingLLM|KV quant (int4)|MLA (DeepSeek-V2)|Infini-attention|Titans|Memory Layers (Meta)|RAG|Ours: fluid + register|
| ---|---|---|---|---|---|---|---|---|---|
| memory @128K (8B)|16.0 GB|0.25 GB|4.0 GB|~0.3 GB|16 KB|~16 KB|128B  trainable  params|external DB|16 KB + 20 B/fact|
| growth with context|linear|constant|linear|linear|constant|constant|constant|n/a|constant|
| exact token recall|✅|✅ (recent only)|✅|❌|❌|❌|❌|✅|✅ (pinned) / ✅ (L3) / ❌ (fluid)|
| global gist|✅|❌|✅|✅|✅|✅|❌|❌|✅ (harmonic branch)|
| infinite stream|❌|✅|❌|❌|✅|✅|n/a|✅|✅|
| needs training|–|–|–|yes (full)|yes (full)|yes (full)|yes (pretrain)|–|no (G0 protocol)|
| 70B @128K on 2×H100|0 sessions|260|~10|~40|45|45|n/a|n/a|45|
| retained exact token IDs|❌|❌|❌|❌|❌|❌|❌|❌|✅ (bounded register)|
| attach to existing LLM|❌|❌|❌|❌|~|~|❌|✅|✅ (one residual insert)|

## 0b. Comparison with kNN-LM

The original comparison omitted the closest prior art. Experiment E13 now
includes a kNN-LM (Khandelwal et al., 2020, Retrieval Augmented Language Model):
a datastore of (context → next token) plus an LM, combined by interpolating
their next-token distributions:

```text
p(v) = λ · p_LM(v) + (1 − λ) · p_knn(v)
```

Functionally that is this project's layer, with three differences: the
datastore is bounded here, the combination is a gated override
(the model's own uncertainty scales the retrieval) instead of a flat
interpolation, and retained token IDs are stored exactly. The register is
not a tamper-evident audit trail.
`experiments/e13_knn_baseline.py` measures both on the same corpus, the same
datastore and the same base model:

|  |val loss|ppl|Δ vs base|token accuracy|
| ---|---|---|---|---|
| the base model alone|2.6111|13.61|–|0.2839|
| kNN-LM, λ = 0.95|2.5647|13.00|−0.0464|0.2893|
| kNN-LM, λ = 0.9|2.5610|12.95|−0.0501|0.2894|
| kNN-LM, λ = 0.8|2.5951|13.40|−0.0159|0.2454|
| kNN-LM, λ = 0.5|2.8600|17.46|+0.2489|0.2014|
| the layer (gated), gain = 2|2.5312|12.57|−0.0799|0.3043|
| the layer (gated), gain = 8|2.4690|11.81|−0.1421|0.3177|
| the layer (gated), gain = 20|2.7133|15.08|+0.1022|0.2737|

With the 16 KB datastore the gate beats the interpolation by 2.8× on the
loss and by 8× on the token accuracy.
But the datastore size is decisive, and the honest reading is sharper than
"we beat kNN-LM":

| the datastore|the layer (gated)|kNN-LM (λ=0.8)|
| ---|---|---|
| 16 KB  (the constant-memory regime)|−0.1134|−0.0154 ( 7.4× worse )|
| 1 MB  (the unbounded regime)|−0.2048|−1.3013  ( 6.4× better )|

The contribution is the gate under a hard memory budget.  With a
bounded store, the interpolation dilutes a mostly-wrong vote with the
model's distribution and barely helps; the gate can act decisively exactly
where the model has no idea.  With an unbounded store the classic kNN-LM
wins by a lot, because its votes are mostly right and it can never overrule
the model.  Anyone who wants the last 1 % of perplexity should use kNN-LM
with a big datastore; anyone who needs a 16 KB memory should use this.

### The other closest relatives

|  |what it is|how it differs|
| ---|---|---|
| RETRO  (Borsa et al., 2022)|retrieves  chunks  from a corpus, encodes them, and cross-attends to them|a chunk-level, trained, retrieval-augmented architecture; ours is a per-token, training-free, constant-size layer|
| REALM  (Guu et al., 2020)|a retriever + a frozen LM, marginalised over documents|trained end-to-end, an external index; ours is training-free and constant-size|
| Infini-attention|compresses the KV cache into a fixed set of  "registers "|compresses the KV, ours stores  exact token ids  alongside|
| Memory Layers  (Meta)|learns to store long-range information in the residual stream|trainable, parameter-heavy; ours is 20 B a fact and training-free|

## 1. The KV cache reality

`bytes/token = 2 (K,V) × kv_dim × 2 (fp16) × L`

| model|L|kv_dim|weights fp16|KV/token|@32K|@128K|@1M|
| ---|---|---|---|---|---|---|---|
| Llama-3.1-8B|32|1024|15.0 GB|0.131 MB|4.0 GB|16.0 GB|128 GB|
| Llama-3.1-70B|80|1024|131.5 GB|0.328 MB|10.0 GB|40.0 GB|320 GB|
| 405B (MHA)|126|16384|754.4 GB|8.258 MB|252 GB|1008 GB|8064 GB|

Correction to earlier v0.1 numbers: the previously cited 2.62 MB/token
assumed MHA (no GQA). Real GQA configs are 8× smaller. We state the
smaller number here because it makes our own relative win smaller — the
honest direction.

### 1.1 The memory-budget example

70B on 2× H100 80GB:
  weights            131.5 GB
  free VRAM           28.5 GB
  KV budget      93,380 tokens  (ALL sessions combined)
  @ 32K ctx  ->  2 concurrent sessions
  @128K ctx  ->  0 sessions   <<< PHYSICALLY IMPOSSIBLE
A 70B model cannot serve 128K context on 2×80GB. Not "slow" — impossible.
This is the single fact that makes the whole project interesting, and it is
pure arithmetic, not a claim.

## 2. Context window: what actually changes

Honest answer: the fluid does NOT increase the context window. Attention
still sees what attention sees. The fluid is a constant-size long-range
channel that sits beside attention.

|  |attention window|+ fluid channel|total  "reachable "|
| ---|---|---|---|
| full KV|up to trained ctx|–|128K|
| sliding window|2–8K|16 KB (gist of everything)|2–8K exact + everything lossy|
| ours (P+G)|2K exact window|16 KB fluid + register|2K exact + all-of-it gist + 819 exact facts|

The win, stated correctly:
With a 2K local window + a 16 KB fluid, a 70B serves 128K-context
sessions on hardware where the full KV cache cannot hold a single one.
That is not "more effective context". It is context that fits in the
hardware, at 45 concurrent sessions instead of 0.
The two-memory split is the honest form of the claim:
P-memory (small sliding window of K,V) → recent context, exact
G-memory (fluid, 16 KB) → everything else, lossy gist
Register (frozen, 20 B/fact) → the chosen facts, bit-exact

## 3. Prefill: the quadratic tax

Llama-3.1-8B, H100 bf16 (~400 TFLOPs). `attn ≈ 2n²d`, `linear ≈ 2nP`.

| ctx|attn TFLOP|linear TFLOP|attn share|prefill|fluid-only|
| ---|---|---|---|---|---|
| 8,192|0.5|106.5|0.5 %|0.3 s|0.000003 s|
| 32,768|8.8|426.0|2.0 %|1.1 s|0.000013 s|
| 131,072|140.7|1703.9|7.6 %|4.6 s|0.000050 s|
| 1,048,576|9007.2|13631.5|39.8 %|56.6 s|0.000403 s|
| 4,194,304|144115|54526|72.6 %|496.6 s|0.001612 s|

Crossover (attn == linear, i.e. `n·d = P`): 8B at 1.59M tokens,
70B at 7.81M tokens.
Honest reading: the quadratic tax is not the near-term problem — at
128K attention is only 7.6 % of prefill. It bites past ~1.5M (8B) / ~8M (70B).
Do not oversell this. The near-term problem is memory, not FLOPs —
and the memory problem bites at 32K.
The fluid's per-token cost is 384 flops/layer (measured in our framework),
i.e. 0.00016 % of one transformer layer. Prefill through the fluid is
free; the expensive part of long context is the local attention window you
still keep, and that is linear.

## 4. Decode: bandwidth is the wall

Every decode step must read the whole KV cache. H100 HBM 1800 GB/s.
8B (weights 15.0 GB):

| ctx|KV read|full-KV step|tok/s|2K-window+fluid step|tok/s|speedup|
| ---|---|---|---|---|---|---|
| 32K|4.0 GB|11.31 ms|88.4|9.07 ms|110.2|1.25×|
| 128K|16.0 GB|18.47 ms|54.2|9.07 ms|110.2|2.04×|

70B (weights 131.5 GB):

| ctx|KV read|full-KV step|tok/s|2K-window+fluid step|tok/s|speedup|
| ---|---|---|---|---|---|---|
| 32K|10.0 GB|84.41 ms|11.8|78.82 ms|12.7|1.07×|
| 128K|40.0 GB|102.31 ms|9.8|78.82 ms|12.7|1.30×|

The fluid's own read: 16 KB = 0.009 microseconds — 0.0 % of the step.
Reading: at 8B/128K the KV read is more than the weights, so removing it
gives ~2× per-session speed. At 70B the weights dominate the step, so the
per-session speedup is modest (1.07–1.30×) — the real 70B win is
concurrency, not latency.

## 5. Concurrency — where the money is

| model|ctx|full KV|2K win + fluid|fluid only|ratio (window kept)|
| ---|---|---|---|---|---|
| 8B, 1×80GB|32K|16|260|4,262,655|16×|
| 8B, 1×80GB|128K|4|260|4,262,655|65×|
| 70B, 2×80GB|32K|2|45|1,867,595|22×|
| 70B, 2×80GB|128K|0|45|1,867,595|∞ (was impossible)|

Multi-tenant, 10,000 users, 8B, 32K ctx:

|  |total KV/state|
| ---|---|
| full KV|40,000 GB (40 TB)|
| 2K window + fluid|2,500 GB|
| fluid only|0.153 GB (153 MB)|

Per-user memory: 40 MB → 0.25 MB → 0.015 MB. The fluid-only number is
what makes per-user persistent memory economically possible at all.

## 6. Capability matrix

| capability|full KV|sliding win|quant|MLA|Infini|Titans|Mem-Layers|RAG|ours|
| ---|---|---|---|---|---|---|---|---|---|
| exact needle at a position|✅|✅ recent|✅|❌|❌|❌|❌|✅|✅ pinned / L3|
| exact  specific  fact|✅|✅ recent|✅|❌|❌|❌|❌|✅|✅ pinned (20 B)|
| global gist / aggregate|✅|❌|✅|✅|✅|✅|❌|❌|⚠️ historical 97.3 % claim; raw result unavailable|
| positional recall ( "3rd paragraph ")|✅|✅|✅|✅|❌|❌|❌|❌|❌|
| infinite stream / never re-encode|❌|✅|❌|❌|✅|✅|n/a|✅|✅|
| no  "lost in the middle "|❌|❌|❌|❌|~ (no positions)|~|n/a|~|✅ (no positional decay)|
| per-user memory at 10k scale|❌|❌|❌|❌|✅|✅|❌|✅|✅ 153 MB total|
| training-free attach|n/a|✅|✅|❌|❌|❌|❌|✅|✅ (G0 protocol)|
| tamper-evident audit log|❌|❌|❌|❌|❌|❌|❌|❌|❌|
| poisoning blast-radius bound|❌|❌|❌|❌|~|~|❌|❌|❌ (not provided)|
| inspectable memory|❌|❌|❌|❌|❌|❌|❌|❌|✅ bounded register contents|
| raw perplexity at scale|✅ best|❌|~|~|❌|✅|✅|✅|❌ (transformer wins — measured)|

The one thing only the full-KV system does better: raw fidelity over
everything, and positional recall. Everything else is a trade.

## 7. Per-fact storage estimates

| system|bytes per exactly-retrievable fact|capacity in 16 KB|
| ---|---|---|
| KV cache (8B, GQA)|131,072|0.12 facts|
| KV cache (70B, GQA)|327,680|0.05 facts|
| RAG chunk (512 tok + index)|2,176|7.5 facts|
| Larimar (512×768 fp32)|1,573|10 facts|
| Frozen register (ours)|20|819 facts|
| Landmark codebook (ours, exact spans)|64|256 spans|

Larimar's own number: 100 % recall at N ≤ K = 1000, 82 % at N = 1024 — a
hard cliff. Ours: retained token IDs stay bit-exact in the bounded register, and
the unpinned remainder degrades continuously instead of falling off a
cliff. Cliff → knee.
Measured (v1.1, 1M tokens,  `results/e5_graft.json` ):  when a fact is
promoted, the recall is  100 % bit-exact  ( `exact | promoted = 1.000` ),
at  20 B/fact  —  6,554×  cheaper per exactly-retrievable fact than the
8B GQA KV cache.  The  selection  criterion is a separate story and an
honest negative: the plain geometric affinity reaches 0.970 pin precision
while every global-topological criterion reaches 0.000 (see
 `docs/DESIGN_pinned_memory.md`  §9.5.2).

## 8. Prior-art positioning

| work|closest thing to ours|the gap we claim|
| ---|---|---|
| Titans|fixed-size, learned, long-context, does needles|lossy everywhere; no exactness; no audit; per-token gradient cost|
| Larimar|episodic memory, one-shot edits, exact recall ≤K|1,573 B/fact;  hard cliff ; no graceful degradation|
| Infini-attention|constant-size compressive memory|no exact tier, no audit, no selection policy|
| Landmark attention|fixed KV set, fixed size|learned weights , not written content; lossy; no selection policy|
| Memory Layers  (Meta)|trainable KV lookup, exact-ish|needs pretraining; GB-scale; approximate top-k; not append-only|
| MLA|compresses KV|still O(n); lossy; needs full training|
| RAG|exact retrieval, training-free|2,176 B/fact; needs a vector DB; consumes context; chunking loses identity|
| LLMLingua/TokenGate|token importance via perplexity/MI|deletes  tokens; no exact tier; shrinks the  input , not a state|

Our claim: a fixed-size state that is exact for a chosen subset,
gracefully lossy for the rest, with a certified bound and an
bounded majority-tracking exact store — and selection by global topology did not work;
the measured semantic/local-geometric policy performed better.

## 9. Facts versus projections

✅ FACT — arithmetic on published configs, or measured in our sandbox
KV bytes/token, KV @ctx, sessions that fit in VRAM, prefill FLOPs, crossover point
fluid read cost 16 KB = 0.009 µs; fluid compute = 384 flops/token/layer
our `selftest.py` → 50/50
mechanism test: historical report claims 97.3 % sign accuracy, but source and raw result are unavailable; unverified
LM A/B: historical report says transformer wins (2.6331 vs 2.7397), but source and raw result are unavailable; unverified
param-matched FLOPs: 1.10× cheaper @8K → 2.54× @128K
⚠️ PROJECTED — needs E1/E5 to become fact
that the 16 KB fluid carries long-range information at usable quality
that a real LLM actually routes to the fluid (the historical 97.3 % synthetic-task result is unverified)
that topology picks the right facts to freeze
that pinned facts survive a real model's distribution shift
Every table above is a CONDITIONAL: "IF the fluid is usable, THEN this".
The conditional is what E1 and E5 buy. Until then, say "projected", not "we
get 65× more sessions". Say "the KV cache is 40 TB for 10k tenants, and our
state would be 153 MB if the fluid carries its share" — the first half is
fact, the second half is conditional.

## 10. Decision guide

| your situation|use|
| ---|---|
| single user, ≤32K ctx, quality is everything|full KV  (transformer wins — we measured it)|
| multi-tenant SaaS, 10k+ users|fluid + register  (153 MB vs 40 TB)|
| 70B @128K on 2×80GB|fluid + register  (45 sessions vs 0)|
| long legal/medical docs, need exact clause text|fluid + register + L3 tool  (20 B/fact, exact token ids)|
| infinite stream / live agent|fluid  (state never grows)|
| you want to inspect the current cached entries|fluid + bounded exact store (not an audit log)|
| you can afford training|MLA / Memory Layers / Titans  (they are trained; we are not)|
| you need raw perplexity SOTA|transformer  — do not use us|

## 11. The verdict

The FluidLayer layer is not a better attention. It is a different memory
tier, and the comparison is not "fluidlayer vs transformer" but
"which tier for which part of the context".
The honest, defensible, arithmetic-backed claim:
KV cache is the binding constraint on long-context serving from 32K
upward. A 70B model cannot hold a single 128K session on 2×80GB. A
16 KB sheaf state — plus a bounded exact store with a 20-byte packed-slot
payload estimate for the
facts that must stay exact — replaces that constraint with a constant,
and costs 0.009 microseconds and 384 flops per token.
The unproven half: whether the fluid carries its share of the information
well enough. That is E1. Everything else is arithmetic.
